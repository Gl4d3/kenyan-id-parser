import os
import numpy as np
from vision_agent.tools import *
from vision_agent.tools.planner_tools import judge_od_results
from typing import *
from pillow_heif import register_heif_opener
register_heif_opener()
import vision_agent as va
from vision_agent.tools import register_tool

import re
from typing import Union, Dict, Any
import numpy as np

def detect_kra_pin_document(document: Union[str, np.ndarray]) -> Dict[str, Any]:
    """
    Detect and validate a KRA PIN in a given document (image).
    
    Steps:
    1. Load the document if it's a path string. 
    2. Extract text (and bounding boxes) with ocr.
    3. Overlay bounding boxes on the image for visualization and save the result.
    4. Identify if document is a KRA PIN document by checking certain keywords.
    5. Use regex to find KRA PIN (must be 11 characters, start with 'A', end with a letter).
    6. Return results including confidence score, extracted PIN, and whether it's valid.
    """

    try:
        # 1. Load the document if it's a path
        if isinstance(document, str):
            # The tool load_image is assumed to be available as documented
            image = load_image(document)
        else:
            image = document
        
        # 2. Extract text with bounding boxes using the 'ocr' tool
        ocr_results = ocr(image)

        # 3. Overlay bounding boxes on the image
        # The bounding boxes used by overlay_bounding_boxes must have keys: score, label, bbox
        # We'll pass them directly
        image_with_boxes = overlay_bounding_boxes(image, ocr_results)
        # Save the visualization
        save_image(image_with_boxes, "document_with_ocr_boxes.png")

        # 4. Identify if document is a KRA PIN document by checking for key phrases
        text_content = " ".join([item["label"] for item in ocr_results]).lower()
        key_phrases = ["pin certificate", "kenya revenue", "personal identification number", "kra", "tax"]
        is_kra_doc = any(kp in text_content for kp in key_phrases)

        # Compute confidence
        if ocr_results:
            avg_conf = sum(item["score"] for item in ocr_results) / len(ocr_results)
        else:
            avg_conf = 0.0

        if not is_kra_doc:
            return {
                "is_valid_document": False,
                "kra_pin": None,
                "message": "Not a KRA PIN document",
                "confidence": avg_conf
            }

        # 5. Use regex to find KRA PIN (start 'A', 9 alphanumeric, end letter)
        full_text = " ".join([item["label"] for item in ocr_results])
        pattern = r"\b(A[0-9A-Za-z]{9}[A-Za-z])\b"
        matches = re.findall(pattern, full_text)

        if not matches:
            return {
                "is_valid_document": True,
                "kra_pin": None,
                "message": "No valid KRA PIN found",
                "confidence": avg_conf
            }

        # 6. Return first match
        return {
            "is_valid_document": True,
            "kra_pin": matches[0],
            "message": "Valid KRA PIN found",
            "confidence": avg_conf
        }
    
    except Exception as e:
        return {
            "is_valid_document": False,
            "kra_pin": None,
            "message": f"Error processing document: {str(e)}",
            "confidence": 0.0
        }

if __name__ == "__main__":
    # Example usage
    result = detect_kra_pin_document("images/kenya_id_1.jpg")
    
    for key, value in result.items():
        print(f"{key}: {value}")