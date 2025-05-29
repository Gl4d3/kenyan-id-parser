from flask import Flask, render_template, request, jsonify, send_file
import os
import uuid
import tempfile
import logging
import shutil

from werkzeug.utils import secure_filename
import json
from datetime import datetime

# Import your verification scripts
from kenyan_id_verification import analyze_kenyan_ids
from kra_pin_verification import detect_kra_pin_document
from document_converter import DocumentConverter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-in-prod'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB for PDFs

# Updated allowed file extensions to include documents
ALLOWED_EXTENSIONS = {
    'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'heic', 'heif',  # Images
    'pdf',  # PDF documents
    'doc', 'docx'  # Word documents
}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static/annotated', exist_ok=True)
os.makedirs('static/original', exist_ok=True)  # For original images

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_category(filename):
    """Determine if file is image, pdf, or document"""
    ext = filename.rsplit('.', 1)[1].lower()
    if ext in {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'heic', 'heif'}:
        return 'image'
    elif ext == 'pdf':
        return 'pdf'
    elif ext in {'doc', 'docx'}:
        return 'document'
    return 'unknown'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify_document():
    converter = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        document_type = request.form.get('document_type', 'auto')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Please upload: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save uploaded file
        file.save(filepath)
        
        file_category = get_file_category(filename)
        logger.info(f"Processing {file_category} file: {filename}")
        
        # Initialize document converter
        converter = DocumentConverter()
        
        # Convert document to images if needed
        try:
            image_paths = converter.process_file(filepath)
            logger.info(f"Converted to {len(image_paths)} image(s) for processing")
        except Exception as e:
            logger.error(f"Conversion error: {e}")
            return jsonify({
                'error': f'Failed to process document: {str(e)}'
            }), 400
        
        results = {}
        
        # Save original images (without bounding boxes) for display
        original_images = []
        for i, img_path in enumerate(image_paths):
            original_filename = f"{unique_filename}_original_{i}.png"
            original_path = f"static/original/{original_filename}"
            shutil.copy2(img_path, original_path)
            original_images.append(f"/static/original/{original_filename}")
        
        # Process each image through verification
        if document_type == 'kenyan_id' or document_type == 'auto':
            # Analyze as Kenyan ID
            id_results = analyze_kenyan_ids(image_paths)
            # Combine results from multiple pages if needed
            results['kenyan_id'] = combine_kenyan_id_results(id_results)
        
        if document_type == 'kra_pin' or document_type == 'auto':
            # Analyze as KRA PIN - use first image or best match
            pin_results = []
            for img_path in image_paths:
                pin_result = detect_kra_pin_document(img_path)
                pin_results.append(pin_result)
            
            # Find the best KRA PIN result
            results['kra_pin'] = get_best_kra_result(pin_results)
        
        # Determine document type if auto-detection
        if document_type == 'auto':
            detected_type = 'unknown'
            if results.get('kenyan_id', {}).get('valid_id'):
                detected_type = 'kenyan_id'
            elif results.get('kra_pin', {}).get('is_valid_document'):
                detected_type = 'kra_pin'
            results['detected_type'] = detected_type
        
        # Move annotated images to static folder for serving
        annotated_files = []
        for i, filename in enumerate(os.listdir('.')):
            if filename.startswith('annotated_') and filename.endswith('.png'):
                new_path = f"static/annotated/{unique_filename}_{filename}"
                os.rename(filename, new_path)
                annotated_files.append(f"/static/annotated/{unique_filename}_{filename}")
        
        if os.path.exists('document_with_ocr_boxes.png'):
            new_path = f"static/annotated/{unique_filename}_ocr_boxes.png"
            os.rename('document_with_ocr_boxes.png', new_path)
            annotated_files.append(f"/static/annotated/{unique_filename}_ocr_boxes.png")
        
        results['annotated_images'] = annotated_files
        results['original_images'] = original_images
        results['timestamp'] = datetime.now().isoformat()
        results['original_filename'] = filename
        results['file_category'] = file_category
        results['pages_processed'] = len(image_paths)
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    
    finally:
        # Clean up uploaded file
        try:
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            logger.warning(f"Could not remove uploaded file: {e}")
        
        # Clean up converted files
        if converter:
            converter.cleanup()

def combine_kenyan_id_results(results_list):
    """Combine results from multiple pages to get the best Kenyan ID result"""
    if not results_list:
        return {'valid_id': False, 'id_number': None, 'name': None}
    
    # Find the first valid result or return the first one
    for result in results_list:
        if result.get('valid_id'):
            return result
    
    return results_list[0]

def get_best_kra_result(results_list):
    """Get the best KRA PIN result from multiple pages"""
    if not results_list:
        return {'is_valid_document': False, 'kra_pin': None, 'message': 'No results', 'confidence': 0.0}
    
    # Find the result with highest confidence
    best_result = max(results_list, key=lambda x: x.get('confidence', 0))
    return best_result

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)