import os
import tempfile
import uuid
from typing import List, Tuple, Optional
from pdf2image import convert_from_path
from PIL import Image
import magic
import PyPDF2
from docx import Document
import logging

logger = logging.getLogger(__name__)

class DocumentConverter:
    """Handles conversion of various document formats to images for OCR processing"""
    
    SUPPORTED_EXTENSIONS = {
        'pdf': 'application/pdf',
        'doc': 'application/msword',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'gif': 'image/gif',
        'bmp': 'image/bmp',
        'tiff': 'image/tiff',
        'heic': 'image/heic',
        'heif': 'image/heif'
    }
    
    def __init__(self, temp_dir: str = None):
        """
        Initialize the document converter
        
        Args:
            temp_dir: Directory for temporary files. If None, uses system temp.
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.converted_files = []  # Track files for cleanup
    
    def is_supported_file(self, filepath: str) -> bool:
        """
        Check if file type is supported for conversion
        
        Args:
            filepath: Path to the file to check
            
        Returns:
            bool: True if file type is supported
        """
        try:
            # Get file extension
            ext = filepath.lower().split('.')[-1]
            
            # Verify using magic numbers for security
            mime_type = magic.from_file(filepath, mime=True)
            
            return (ext in self.SUPPORTED_EXTENSIONS and 
                   mime_type in self.SUPPORTED_EXTENSIONS.values())
        except Exception as e:
            logger.error(f"Error checking file support: {e}")
            return False
    
    def get_file_type(self, filepath: str) -> str:
        """
        Determine the file type based on extension and MIME type
        
        Args:
            filepath: Path to the file
            
        Returns:
            str: File type (pdf, image, doc, etc.)
        """
        try:
            ext = filepath.lower().split('.')[-1]
            mime_type = magic.from_file(filepath, mime=True)
            
            if ext == 'pdf' and 'pdf' in mime_type:
                return 'pdf'
            elif ext in ['doc', 'docx'] and 'word' in mime_type:
                return 'document'
            elif ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'heic', 'heif']:
                return 'image'
            else:
                return 'unknown'
        except Exception:
            return 'unknown'
    
    def convert_pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[str]:
        """
        Convert PDF pages to high-quality images
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion (300+ recommended for OCR)
            
        Returns:
            List[str]: Paths to converted image files
        """
        try:
            # Validate PDF first
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
            if num_pages == 0:
                raise ValueError("PDF has no pages")
            
            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                fmt='PNG',
                thread_count=2,  # Limit threads for server stability
                first_page=1,
                last_page=min(num_pages, 10)  # Limit to 10 pages for performance
            )
            
            image_paths = []
            for i, image in enumerate(images):
                # Generate unique filename
                filename = f"converted_{uuid.uuid4()}_{i}.png"
                filepath = os.path.join(self.temp_dir, filename)
                
                # Save with optimization for OCR
                image.save(filepath, 'PNG', optimize=True, dpi=(dpi, dpi))
                image_paths.append(filepath)
                self.converted_files.append(filepath)
                
                logger.info(f"Converted PDF page {i+1} to {filepath}")
            
            return image_paths
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            raise
    
    def convert_document_to_images(self, doc_path: str) -> List[str]:
        """
        Convert document formats (DOC, DOCX) to images
        Note: This is a simplified implementation. For production, consider using
        LibreOffice/OpenOffice programmatically or a service like DocumentCloud
        
        Args:
            doc_path: Path to document file
            
        Returns:
            List[str]: Paths to converted image files
        """
        try:
            # For DOCX files, we can extract text and create a simple image
            # This is a basic implementation - for better results, use LibreOffice conversion
            if doc_path.lower().endswith('.docx'):
                return self._convert_docx_to_image(doc_path)
            else:
                raise NotImplementedError("DOC format conversion not implemented")
                
        except Exception as e:
            logger.error(f"Error converting document to images: {e}")
            raise
    
    def _convert_docx_to_image(self, docx_path: str) -> List[str]:
        """
        Convert DOCX to image by extracting text and creating a text image
        Note: This is a basic implementation for demonstration
        """
        try:
            # Extract text from DOCX
            doc = Document(docx_path)
            text_content = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)
            
            if not text_content:
                raise ValueError("No text content found in document")
            
            # Create a simple text image (for demonstration)
            # In production, use proper document rendering
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a white image
            img_width, img_height = 2480, 3508  # A4 at 300 DPI
            image = Image.new('RGB', (img_width, img_height), 'white')
            draw = ImageDraw.Draw(image)
            
            try:
                # Try to use a better font
                font = ImageFont.truetype("arial.ttf", 40)
            except:
                # Fallback to default font
                font = ImageFont.load_default()
            
            # Draw text on image
            y_position = 100
            for line in text_content[:50]:  # Limit lines
                if y_position > img_height - 100:
                    break
                draw.text((100, y_position), line[:80], fill='black', font=font)
                y_position += 60
            
            # Save image
            filename = f"converted_{uuid.uuid4()}_doc.png"
            filepath = os.path.join(self.temp_dir, filename)
            image.save(filepath, 'PNG', dpi=(300, 300))
            
            self.converted_files.append(filepath)
            return [filepath]
            
        except Exception as e:
            logger.error(f"Error converting DOCX to image: {e}")
            raise
    
    def process_file(self, filepath: str) -> List[str]:
        """
        Main method to process any supported file type
        
        Args:
            filepath: Path to the file to process
            
        Returns:
            List[str]: List of image paths ready for OCR processing
        """
        if not self.is_supported_file(filepath):
            raise ValueError(f"Unsupported file type: {filepath}")
        
        file_type = self.get_file_type(filepath)
        
        if file_type == 'image':
            # Already an image, validate and return
            return self._validate_image(filepath)
        elif file_type == 'pdf':
            return self.convert_pdf_to_images(filepath)
        elif file_type == 'document':
            return self.convert_document_to_images(filepath)
        else:
            raise ValueError(f"Cannot process file type: {file_type}")
    
    def _validate_image(self, image_path: str) -> List[str]:
        """
        Validate and potentially optimize an image file for OCR
        
        Args:
            image_path: Path to image file
            
        Returns:
            List[str]: List containing the validated/optimized image path
        """
        try:
            with Image.open(image_path) as img:
                # Check if image is valid
                img.verify()
                
                # Reload image for processing (verify() invalidates the image)
                with Image.open(image_path) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Check resolution - enhance if too low
                    if hasattr(img, 'info') and 'dpi' in img.info:
                        dpi = img.info['dpi'][0]
                        if dpi < 200:
                            # Upscale image for better OCR
                            new_size = (int(img.width * 1.5), int(img.height * 1.5))
                            img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Save optimized version if needed
                    if img.mode != Image.open(image_path).mode:
                        filename = f"optimized_{uuid.uuid4()}.png"
                        filepath = os.path.join(self.temp_dir, filename)
                        img.save(filepath, 'PNG', dpi=(300, 300))
                        self.converted_files.append(filepath)
                        return [filepath]
            
            return [image_path]
            
        except Exception as e:
            logger.error(f"Error validating image: {e}")
            raise ValueError(f"Invalid image file: {image_path}")
    
    def cleanup(self):
        """Remove all temporary files created during conversion"""
        for filepath in self.converted_files:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.info(f"Cleaned up temporary file: {filepath}")
            except Exception as e:
                logger.warning(f"Could not remove temporary file {filepath}: {e}")
        
        self.converted_files.clear()
    
    def __del__(self):
        """Ensure cleanup on object destruction"""
        self.cleanup()