# Your existing script with minor modifications for better integration
import os
import numpy as np
from vision_agent.tools import *
from vision_agent.tools.planner_tools import judge_od_results
from typing import *
from pillow_heif import register_heif_opener
register_heif_opener()
import vision_agent as va
from vision_agent.tools import register_tool

from typing import List, Dict, Any

def analyze_kenyan_ids(image_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Analyzes a list of image paths, checks if each image is a valid Kenyan ID, 
    extracts the ID number and person's name, overlays the OCR bounding boxes, 
    and returns the results.

    Parameters:
        image_paths (list[str]): Paths (or URLs) of the images to be analyzed.

    Returns:
        list[dict[str, Any]]: A list of dictionaries, one per image, each containing:
            - 'valid_id' (bool): If the image is likely a Kenyan ID
            - 'id_number' (str or None): The extracted 8-digit ID number
            - 'name' (str or None): The extracted name

    Steps:
        1. Load images with load_image.
        2. Perform OCR to get text, bounding boxes, and confidence scores.
        3. Determine if the image is a Kenyan ID by searching for certain keywords.
        4. Extract an 8-digit ID number that isn't a date (no '.' or '/').
        5. Extract a name using the text following 'FULL NAMES' or a text that seems like a person's name.
        6. Overlay the bounding boxes on the image and save the annotated result.
        7. Return the results as a list of dictionaries.
    """

    import re
    import cv2
    from vision_agent.tools import load_image, ocr, overlay_bounding_boxes, save_image

    def is_kenyan_id(text_results: List[Dict[str, Any]]) -> bool:
        # Convert all detected text to lower-case
        all_text = ' '.join(r['label'].lower() for r in text_results)
        keywords = ['republic of kenya', 'jamhuri ya kenya', 'identity card']
        return any(keyword in all_text for keyword in keywords)

    def extract_id_info(text_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Sort text by vertical and horizontal position
        sorted_text = sorted(text_results, key=lambda x: (x['bbox'][1], x['bbox'][0]))

        id_number = None
        name = None

        # Convert to lowercase for matching but keep original text
        text_with_pos = [(r['label'], r['label'].lower(), r['bbox'], r['score']) for r in sorted_text]
        
        # Exclude these phrases from being considered as names
        excluded_phrases = [
            'republic', 'jamhuri', 'kenya', 'identity', 'card', 'number',
            'serial', 'signature', 'date', 'place', 'sex', 'male', 'female'
        ]
        
        # Try to find name after "FULL NAMES"
        for i, (orig_text, text, bbox, score) in enumerate(text_with_pos):
            if any(x in text for x in ['full names', 'full name', 'names:']):
                # Look at next text that could be a name
                for j in range(i+1, min(len(text_with_pos), i+3)):
                    potential_name = text_with_pos[j][0]
                    words = potential_name.split()
                    if (len(words) >= 2 and 
                        not any(c.isdigit() for c in potential_name) and
                        not any(word.lower() in excluded_phrases for word in words) and
                        text_with_pos[j][3] > 0.8):
                        name = potential_name
                        break
                if name:
                    break

        # If no name found, look for a name-like text in the first half
        if not name:
            first_half = text_with_pos[:len(text_with_pos)//2]
            for orig_text, text, bbox, score in first_half:
                words = orig_text.split()
                if (len(words) >= 2 and 
                    not any(c.isdigit() for c in orig_text) and
                    not any(word.lower() in excluded_phrases for word in words) and
                    score > 0.9):
                    name = orig_text
                    break

        # Find ID number by looking for an 8-digit number in the first half
        first_half = text_with_pos[:len(text_with_pos)//2]
        for orig_text, text, bbox, score in first_half:
            digits = ''.join(filter(str.isdigit, orig_text))
            # Must be exactly 8 digits and not contain '.' or '/'
            if len(digits) == 8 and not any(x in orig_text for x in ['.', '/']):
                id_number = digits
                break

        return {
            'id_number': id_number,
            'name': name
        }

    results = []
    for idx, image_path in enumerate(image_paths):
        try:
            # Load image
            image = load_image(image_path)

            # OCR to get text and bounding boxes
            text_results = ocr(image)

            # Check if valid Kenyan ID
            valid = is_kenyan_id(text_results)
            
            # Always overlay bounding boxes for reference
            bboxes = []
            for tr in text_results:
                bboxes.append({
                    "label": tr["label"],
                    "score": tr["score"],
                    "bbox": tr["bbox"]
                })
            annotated = overlay_bounding_boxes(image, bboxes)
            save_image(annotated, f"annotated_{idx}.png")

            if not valid:
                results.append({
                    "valid_id": False,
                    "id_number": None,
                    "name": None
                })
                continue

            # Extract ID info
            info = extract_id_info(text_results)

            # Populate results
            results.append({
                "valid_id": True,
                "id_number": info['id_number'],
                "name": info['name']
            })
        
        except Exception as e:
            results.append({
                "valid_id": False,
                "id_number": None,
                "name": None,
                "error": str(e)
            })

    return results

# Main function to run the analysis
if __name__ == "__main__":
    # Example usage
    image_paths = ["shared image.jpg"]  # Replace with your image paths
    results = analyze_kenyan_ids(image_paths)
    
    for idx, result in enumerate(results):
        print(f"Image {idx+1}:")
        print(f"  Valid ID: {result['valid_id']}")
        print(f"  ID Number: {result['id_number']}")
        print(f"  Name: {result['name']}")