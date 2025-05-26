from flask import Flask, render_template, request, jsonify, send_file
import os
import uuid

from werkzeug.utils import secure_filename # Ensure this import is present for secure filename handling. By this we mean that the filename is sanitized to prevent directory traversal attacks. Sanitization is important to ensure that the filename does not contain any malicious characters or patterns that could lead to security vulnerabilities. This is especially crucial when handling user-uploaded files, as it helps prevent unauthorized access to the server's file system.
import json
from datetime import datetime

# Import your verification scripts
from kenyan_id_verification import analyze_kenyan_ids
from kra_pin_verification import detect_kra_pin_document

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-in-prod'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'heic', 'heif'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static/annotated', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify_document():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        document_type = request.form.get('document_type', 'auto')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
        
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save uploaded file
        file.save(filepath)
        
        results = {}
        
        if document_type == 'kenyan_id' or document_type == 'auto':
            # Analyze as Kenyan ID
            id_results = analyze_kenyan_ids([filepath])
            results['kenyan_id'] = id_results[0] if id_results else None
        
        if document_type == 'kra_pin' or document_type == 'auto':
            # Analyze as KRA PIN
            pin_results = detect_kra_pin_document(filepath)
            results['kra_pin'] = pin_results
        
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
        results['timestamp'] = datetime.now().isoformat()
        results['original_filename'] = filename
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)