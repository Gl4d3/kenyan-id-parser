import os
import numpy as np
from vision_agent.tools import *
from vision_agent.tools.planner_tools import judge_od_results
from typing import *
from pillow_heif import register_heif_opener
register_heif_opener()
import vision_agent as va
from vision_agent.tools import register_tool

from typing import List, Dict, Any, Tuple
import numpy as np
import re
from vision_agent.tools import load_image, ocr, overlay_bounding_boxes, save_image

def solve_document_ocr(document_image_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Solves the OCR tasks by:
      1) Loading and optionally rotating each document image.
      2) Extracting text with bounding boxes using the 'ocr' tool.
      3) Fixing any invalid bounding boxes before overlaying.
      4) Attempting to parse name, id_number, date_of_birth, sex, and document_type.
      5) Overlaying bounding boxes on the image to visualize the positions of recognized text.
      6) Saving the annotated image and returning the extracted fields in a list of dictionaries.

    :param document_image_paths: A list of image file paths (or URLs).
    :return: A list of dictionaries, each containing extracted fields from the corresponding document.
    """

    def rotate_image_90(img: np.ndarray) -> np.ndarray:
        """Rotate the image by 90 degrees clockwise."""
        return np.rot90(img, k=3).copy()

    def correct_orientation(img: np.ndarray) -> np.ndarray:
        """
        Check if more text is vertically oriented than horizontally.
        If so, rotate the image 90 degrees clockwise until more text is horizontal.
        """
        results = ocr(img)
        horizontal = 0
        vertical = 0
        for r in results:
            x0, y0, x1, y1 = r['bbox']
            width = x1 - x0
            height = y1 - y0
            if width >= height:
                horizontal += 1
            else:
                vertical += 1
        
        # If vertical > horizontal, rotate 90 degrees and re-check up to 3 times
        attempt = 0
        while vertical > horizontal and attempt < 3:
            img = rotate_image_90(img)
            results = ocr(img)
            horizontal, vertical = 0, 0
            for r in results:
                x0, y0, x1, y1 = r['bbox']
                width = x1 - x0
                height = y1 - y0
                if width >= height:
                    horizontal += 1
                else:
                    vertical += 1
            attempt += 1
        
        return img

    def fix_bboxes(bboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure that the bounding box coordinates are valid: x0 <= x1 and y0 <= y1.
        If they aren't, swap as needed to fix the error: 'y1 must be >= y0', etc.
        """
        for box in bboxes:
            x0, y0, x1, y1 = box["bbox"]
            x_min, x_max = sorted([x0, x1])
            y_min, y_max = sorted([y0, y1])
            box["bbox"] = [x_min, y_min, x_max, y_max]
        return bboxes

    def classify_document_type(text: str) -> str:
        text_lower = text.lower()
        if "passport" in text_lower:
            return "passport"
        elif "foreigner certificate" in text_lower or ("foreign" in text_lower and "certificate" in text_lower):
            return "foreign_certificate"
        else:
            return "national_id"

    def parse_fields(ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse the OCR results to extract name, id_number, date_of_birth, sex, document_type
        based on the presence of keywords or pattern matches.
        """
        # Starting with defaults
        extracted = {
            "document_type": None,
            "name": None,
            "id_number": None,
            "date_of_birth": None,
            "sex": None
        }

        # Flatten text for doc type detection
        all_text = " ".join([res["label"] for res in ocr_results])
        doc_type = classify_document_type(all_text)
        extracted["document_type"] = doc_type

        # We can define some patterns or keywords to locate fields:
        # a) sex
        # b) date_of_birth
        # c) id_number
        # d) name by elimination or known keywords

        # Basic patterns
        date_pattern_1 = re.compile(r"\b\d{2}[./-]\d{2}[./-]\d{2,4}\b")
        date_pattern_2 = re.compile(r"\b\d{2}\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s.-]*\d{2,4}\b", re.IGNORECASE)
        # ID/passport pattern might vary by doc_type

        # We'll sort by top-left for a more consistent approach
        sorted_res = sorted(ocr_results, key=lambda x: (x["bbox"][1], x["bbox"][0]))

        for r in sorted_res:
            text_val = r["label"].strip()
            text_lower = text_val.lower()

            # Attempt direct detection of sex
            if text_lower in ("male", "female", "m", "f"):
                extracted["sex"] = "Male" if text_lower in ["male", "m"] else "Female"

            # Try to parse date of birth from text that matches the pattern
            if (not extracted["date_of_birth"] and 
                (date_pattern_1.search(text_val) or date_pattern_2.search(text_val))):
                # We do a simple assignment for this example
                extracted["date_of_birth"] = text_val

            # Check for ID or passport number (depending on doc_type)
            if doc_type == "passport":
                # Passport number is typically alphanumeric, but we can assume a pattern like 2 letters + 7 digits
                if re.match(r"^[A-Z0-9]{6,10}$", text_val) and not extracted["id_number"]:
                    extracted["id_number"] = text_val
            elif doc_type == "foreign_certificate":
                # 7 digit or less
                if re.match(r"^\d{1,7}$", text_val) and not extracted["id_number"]:
                    extracted["id_number"] = text_val
            else:
                # national_id: often 5-10 digits
                if re.match(r"^\d{5,10}$", text_val) and not extracted["id_number"]:
                    extracted["id_number"] = text_val

            # Attempt a naive name extraction:
            # If it's upper-case letters and spaces, fairly long, might be name
            if (re.match(r"^[A-Z\s\.]+$", text_val) and len(text_val) > 5 and 
                not any(x in text_lower for x in ["republic", "national", "identity", "passport", "certificate"]) and
                not extracted["name"]):
                extracted["name"] = text_val
        
        return extracted

    all_extracted = []

    for doc_path in document_image_paths:
        # 1) Load and correct orientation
        img = load_image(doc_path)
        img_corrected = correct_orientation(img)

        # 2) Perform OCR
        ocr_results = ocr(img_corrected)

        # 3) Fix bboxes
        for r in ocr_results:
            # Force bounding boxes to be valid
            x0, y0, x1, y1 = r["bbox"]
            x_min, x_max = min(x0, x1), max(x0, x1)
            y_min, y_max = min(y0, y1), max(y0, y1)
            r["bbox"] = [x_min, y_min, x_max, y_max]

        # 4) Parse fields
        fields = parse_fields(ocr_results)

        # 5) Overlay bounding boxes for visualization
        # Let's just overlay all recognized text on the image
        # We can ignore the label text in the bounding box for brevity, just keep them as is
        annotated_image = overlay_bounding_boxes(img_corrected, ocr_results)

        # 6) Save annotated image
        annotated_path = doc_path.rsplit(".", 1)
        annotated_output = annotated_path[0] + "_annotated." + (annotated_path[1] if len(annotated_path) > 1 else "png")
        save_image(annotated_image, annotated_output)

        # Collect results
        all_extracted.append({
            "document_path": doc_path,
            **fields
        })
    
    return all_extracted

if __name__ == "__main__":
    # Example usage
    document_images = ["../images/id_1.png","../images/shared image (1).jpg"]
    results = solve_document_ocr(document_images)
    for res in results:
        print(res)