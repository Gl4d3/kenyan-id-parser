import os
import numpy as np
import cv2
from PIL import Image
import pytesseract
import easyocr
import re
from typing import List, Dict, Tuple, Optional
from fuzzywuzzy import fuzz
from collections import Counter
import time

class AdvancedDocumentOCR:
    def __init__(self):
        # Initialize EasyOCR reader once for efficiency
        self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
        
        # Document keywords for type detection and filtering
        self.doc_keywords = {
            'passport': ['PASSPORT', 'PASSEPORT', 'REPUBLIC OF MOLDOVA', 'ROMANA', 'ROMANIAN'],
            'foreign_certificate': ['FOREIGNER CERTIFICATE', 'ALIEN', 'FOREIGN', 'CERTIFICATE', 'REPUBLIC OF KENYA'],
            'national_id': ['JAMHURI YA KENYA', 'REPUBLIC OF KENYA', 'NATIONAL ID', 'ID NUMBER', 'SERIAL NUMBER']
        }
        
        # Filter words that should never be considered as names
        self.filter_words = {
            'REPUBLIC', 'KENYA', 'CERTIFICATE', 'PASSPORT', 'FOREIGNER', 'ALIEN', 'NATIONAL',
            'JAMHURI', 'ROMANIAN', 'ROMANIA', 'MOLDOVA', 'MOLDOVAN', 'SERIAL', 'NUMBER',
            'DATE', 'BIRTH', 'SEX', 'MALE', 'FEMALE', 'PLACE', 'ISSUE', 'EXPIRY', 'VALID',
            'HOLDER', 'SIGNATURE', 'AUTHORITY', 'GOVERNMENT', 'MINISTRY', 'DEPARTMENT',
            'INDIVIDUAL', 'INDIV', 'NAMES', 'FULL', 'GIVEN', 'SURNAME', 'ID'
        }
        
        # Common Kenyan name patterns for validation
        self.kenyan_name_patterns = [
            r'^[A-Z][a-z]+ [A-Z][a-z]+$',  # First Last
            r'^[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+$',  # First Middle Last
            r'^[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+$'  # Four names
        ]

    def detect_orientation_fast(self, image: np.ndarray) -> int:
        """Fast orientation detection using text confidence scores"""
        # Resize image for faster processing
        h, w = image.shape[:2]
        if h > 800 or w > 800:
            scale = 800 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        best_angle = 0
        best_confidence = 0
        
        # Test all 4 orientations
        for angle in [0, 90, 180, 270]:
            if angle == 0:
                rotated = image
            else:
                # Rotate image
                center = (image.shape[1] // 2, image.shape[0] // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
            
            # Get OCR confidence for this orientation
            try:
                # Use only Tesseract for speed in orientation detection
                if len(rotated.shape) == 3:
                    pil_img = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
                else:
                    pil_img = Image.fromarray(rotated)
                
                # Get text with confidence
                data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    if avg_confidence > best_confidence:
                        best_confidence = avg_confidence
                        best_angle = angle
            except:
                continue
        
        return best_angle

    def rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by specified angle"""
        if angle == 0:
            return image
        
        center = (image.shape[1] // 2, image.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

    def preprocess_image_fast(self, image: np.ndarray) -> List[np.ndarray]:
        """Fast preprocessing with only essential operations"""
        processed = []
        
        # Original
        processed.append(image)
        
        # Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        processed.append(gray)
        
        # Enhanced contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        processed.append(enhanced)
        
        # Threshold
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed.append(thresh)
        
        return processed

    def extract_text_optimized(self, image_path: str) -> Tuple[str, str]:
        """Optimized text extraction with orientation detection"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            pil_img = Image.open(image_path)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Detect and correct orientation
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        correct_angle = self.detect_orientation_fast(gray)
        
        if correct_angle != 0:
            img = self.rotate_image(img, correct_angle)
            print(f"Rotated image by {correct_angle} degrees")
        
        # Preprocess
        processed_images = self.preprocess_image_fast(img)
        
        # Extract text using both OCR engines
        all_texts = []
        
        # Tesseract with different configs
        for processed_img in processed_images[:2]:  # Limit to 2 best preprocessing methods
            try:
                if len(processed_img.shape) == 3:
                    pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
                else:
                    pil_img = Image.fromarray(processed_img)
                
                # Primary config
                text = pytesseract.image_to_string(pil_img, config='--oem 3 --psm 6')
                if text.strip():
                    all_texts.append(text.strip())
                
                # Secondary config for better layout handling
                text2 = pytesseract.image_to_string(pil_img, config='--oem 3 --psm 11')
                if text2.strip():
                    all_texts.append(text2.strip())
            except:
                continue
        
        # EasyOCR
        try:
            results = self.easyocr_reader.readtext(processed_images[1])  # Use enhanced grayscale
            easyocr_text = ' '.join([result[1] for result in results if result[2] > 0.4])
            if easyocr_text.strip():
                all_texts.append(easyocr_text.strip())
        except:
            pass
        
        # Combine results
        combined_text = self.combine_texts_smart(all_texts)
        raw_text = ' '.join(all_texts) if all_texts else ""
        
        return combined_text, raw_text

    def combine_texts_smart(self, texts: List[str]) -> str:
        """Smart text combination using frequency and context"""
        if not texts:
            return ""
        
        # Find the most comprehensive text
        best_text = max(texts, key=len)
        
        # Clean and normalize
        best_text = re.sub(r'\s+', ' ', best_text.strip())
        
        return best_text

    def detect_document_type_advanced(self, text: str) -> Tuple[str, float]:
        """Advanced document type detection with confidence scoring"""
        text_upper = text.upper()
        scores = {}
        
        for doc_type, keywords in self.doc_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_upper:
                    score += len(keyword)  # Longer keywords get higher scores
            
            # Normalize score
            scores[doc_type] = score / len(text_upper) if text_upper else 0
        
        if not scores or max(scores.values()) == 0:
            # Fallback detection based on patterns
            if re.search(r'PASSPORT.*NO|NO.*PASSPORT', text_upper):
                return 'passport', 0.7
            elif re.search(r'INDIV.*NO|CERTIFICATE', text_upper):
                return 'foreign_certificate', 0.7
            else:
                return 'national_id', 0.5
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        return best_type, confidence

    def is_valid_name(self, name: str) -> bool:
        """Validate if extracted text is likely a real name"""
        if not name or len(name.strip()) < 2:
            return False
        
        name = name.strip().upper()
        
        # Filter out document keywords
        if any(word in name for word in self.filter_words):
            return False
        
        # Check for reasonable name patterns
        words = name.split()
        if len(words) < 2 or len(words) > 5:
            return False
        
        # Each word should start with a letter and be reasonable length
        for word in words:
            if not word.isalpha() or len(word) < 2 or len(word) > 20:
                return False
        
        return True

    def extract_field_fuzzy(self, text: str, patterns: List[str], 
                           field_type: str = "general") -> str:
        """Extract field using fuzzy matching and multiple patterns"""
        text_upper = text.upper()
        candidates = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, text_upper, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                candidate = match.group(1).strip()
                
                # Special validation for names
                if field_type == "name" and not self.is_valid_name(candidate):
                    continue
                
                # Clean and validate candidate
                if candidate and len(candidate) > 1:
                    candidates.append(candidate)
        
        if not candidates:
            return ""
        
        # Return the most frequent candidate, or longest if frequencies are equal
        if len(candidates) == 1:
            return candidates[0]
        
        # Use frequency and length to pick best candidate
        candidate_scores = {}
        for candidate in candidates:
            score = candidates.count(candidate) * 10 + len(candidate)
            candidate_scores[candidate] = score
        
        best_candidate = max(candidate_scores, key=candidate_scores.get)
        return best_candidate

    def parse_national_id(self, text: str) -> Dict[str, str]:
        """Parse Kenyan National ID"""
        result = {
            "document_type": "national_id",
            "full_name": "",
            "id_number": "",
            "sex": "",
            "date_of_birth": ""
        }
        
        # Name patterns
        name_patterns = [
            r"FULL\s+NAMES?\s*:?\s*([A-Z][A-Z\s]{5,50})",
            r"NAMES?\s*:?\s*([A-Z][A-Z\s]{5,50})",
            r"(?:REPUBLIC\s+OF\s+)?KENYA\s+([A-Z][A-Z\s]{5,40})\s+(?:ID|SERIAL|\d)",
            r"([A-Z][A-Z\s]{10,40})\s+(?:MALE|FEMALE)",
        ]
        
        result["full_name"] = self.extract_field_fuzzy(text, name_patterns, "name")
        
        # ID number patterns
        id_patterns = [
            r"ID\s*(?:NUMBER|NO\.?)?\s*:?\s*(\d{7,9})",
            r"SERIAL\s+NUMBER\s*:?\s*(\d{7,10})",
            r"(?:^|\s)(\d{8,9})(?:\s|$)",
        ]
        
        id_result = self.extract_field_fuzzy(text, id_patterns)
        result["id_number"] = re.sub(r'\D', '', id_result) if id_result else ""
        
        # Date of birth
        dob_patterns = [
            r"DATE\s+OF\s+BIRTH\s*:?\s*(\d{1,2}[./-]\d{1,2}[./-]\d{4})",
            r"BIRTH\s*:?\s*(\d{1,2}[./-]\d{1,2}[./-]\d{4})",
            r"(\d{1,2}[./-]\d{1,2}[./-]\d{4})",
        ]
        
        result["date_of_birth"] = self.extract_field_fuzzy(text, dob_patterns)
        
        # Sex
        sex_patterns = [
            r"SEX\s*:?\s*(MALE|FEMALE|M|F)",
            r"(?:^|\s)(MALE|FEMALE)(?:\s|$)",
        ]
        
        sex_result = self.extract_field_fuzzy(text, sex_patterns)
        if sex_result:
            result["sex"] = "Male" if sex_result.upper().startswith('M') else "Female"
        
        return result

    def parse_foreign_certificate(self, text: str) -> Dict[str, str]:
        """Parse Foreign Certificate"""
        result = {
            "document_type": "foreign_certificate",
            "name": "",
            "indiv_number": "",
            "sex": "",
            "date_of_birth": "",
            "date_of_expiry": ""
        }
        
        # Name patterns
        name_patterns = [
            r"FULL\s+NAMES?\s*:?\s*([A-Z][A-Z\s]{5,50})",
            r"NAMES?\s*:?\s*([A-Z][A-Z\s]{5,50})",
            r"([A-Z][A-Z\s]{8,40})\s+(?:MALE|FEMALE)",
        ]
        
        result["name"] = self.extract_field_fuzzy(text, name_patterns, "name")
        
        # Individual number
        indiv_patterns = [
            r"INDIV(?:IDUAL)?\s*NO\.?\s*:?\s*(\d{6,15})",
            r"(?:^|\s)(\d{6,10})(?:\s|$)",
        ]
        
        indiv_result = self.extract_field_fuzzy(text, indiv_patterns)
        result["indiv_number"] = re.sub(r'\D', '', indiv_result) if indiv_result else ""
        
        # Dates
        date_patterns = [
            r"BIRTH\s*:?\s*(\d{1,2}[./-]\d{1,2}[./-]\d{4})",
            r"(\d{2}[./-]\d{2}[./-]\d{4})",
        ]
        
        result["date_of_birth"] = self.extract_field_fuzzy(text, date_patterns)
        
        # Expiry date
        exp_patterns = [
            r"EXPIRY\s*:?\s*(\d{1,2}[./-]\d{1,2}[./-]\d{4})",
            r"VALID\s+UNTIL\s*:?\s*(\d{1,2}[./-]\d{1,2}[./-]\d{4})",
        ]
        
        result["date_of_expiry"] = self.extract_field_fuzzy(text, exp_patterns)
        
        # Sex
        sex_patterns = [
            r"SEX\s*:?\s*(MALE|FEMALE|M|F)",
            r"(?:^|\s)(MALE|FEMALE)(?:\s|$)",
        ]
        
        sex_result = self.extract_field_fuzzy(text, sex_patterns)
        if sex_result:
            result["sex"] = "Male" if sex_result.upper().startswith('M') else "Female"
        
        return result

    def parse_passport(self, text: str) -> Dict[str, str]:
        """Parse Passport"""
        result = {
            "document_type": "passport",
            "passport_number": "",
            "name": "",
            "date_of_birth": ""
        }
        
        # Passport number
        passport_patterns = [
            r"PASSPORT\s*(?:NO\.?|NUMBER)\s*:?\s*([A-Z0-9]{6,15})",
            r"(?:^|\s)([A-Z]{2}\d{7})(?:\s|$)",
            r"NO\.?\s*([A-Z0-9]{8,12})",
        ]
        
        result["passport_number"] = self.extract_field_fuzzy(text, passport_patterns)
        
        # Name
        name_patterns = [
            r"SURNAME\s*:?\s*([A-Z][A-Z\s]{5,30})",
            r"NAMES?\s*:?\s*([A-Z][A-Z\s]{5,50})",
            r"([A-Z][A-Z\s]{8,40})\s+\d{2}\s+[A-Z]{3}",  # Name before date
        ]
        
        result["name"] = self.extract_field_fuzzy(text, name_patterns, "name")
        
        # Date of birth
        dob_patterns = [
            r"(\d{2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{4})",
            r"BIRTH\s*:?\s*(\d{1,2}[./-]\d{1,2}[./-]\d{4})",
        ]
        
        result["date_of_birth"] = self.extract_field_fuzzy(text, dob_patterns)
        
        return result

def solve_id_ocr_advanced(image_paths: List[str]) -> List[Dict[str, str]]:
    """
    Advanced OCR solution with auto-detection and orientation handling
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        List of extracted field dictionaries
    """
    ocr_engine = AdvancedDocumentOCR()
    results = []
    
    for image_path in image_paths:
        start_time = time.time()
        
        try:
            print(f"Processing: {image_path}")
            
            # Extract text with orientation correction
            text, raw_text = ocr_engine.extract_text_optimized(image_path)
            
            # Auto-detect document type
            doc_type, confidence = ocr_engine.detect_document_type_advanced(text)
            print(f"Detected: {doc_type} (confidence: {confidence:.2f})")
            
            # Parse based on document type
            if doc_type == 'national_id':
                parsed_result = ocr_engine.parse_national_id(text)
            elif doc_type == 'foreign_certificate':
                parsed_result = ocr_engine.parse_foreign_certificate(text)
            elif doc_type == 'passport':
                parsed_result = ocr_engine.parse_passport(text)
            else:
                # Default fallback
                parsed_result = ocr_engine.parse_national_id(text)
            
            # Validate results make sense
            if not any(v for v in parsed_result.values() if v and v != parsed_result["document_type"]):
                print("Warning: Low extraction quality, check image quality")
            
            results.append(parsed_result)
            
            processing_time = time.time() - start_time
            print(f"Processed in {processing_time:.2f}s")
            print(f"Results: {parsed_result}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            # Return empty result with default structure
            empty_result = {
                "document_type": "unknown",
                "full_name": "",
                "id_number": "",
                "sex": "",
                "date_of_birth": ""
            }
            results.append(empty_result)
    
    return results

# Example usage
if __name__ == "__main__":
    # Test with sample images
    test_images = [
        "images/id_1.png",
        "images/kenya_id_1.jpg"
    ]
    
    results = solve_id_ocr_advanced(test_images)
    
    for i, result in enumerate(results):
        print(f"\nDocument {i+1}:")
        for key, value in result.items():
            print(f"  {key}: {value}")