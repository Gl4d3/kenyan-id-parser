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
from typing import Tuple
from vision_agent.tools import claude35_text_extraction, load_image

def validate_kenyan_tax_pin(image_path: str) -> Tuple[str, bool]:
    """
    Validates a Kenyan taxpayer registration certificate image by extracting
    the PIN and checking its format (11 characters, starts and ends with letters,
    and contains 9 digits in between).

    :param image_path: Path to the image file.
    :return: A tuple containing the extracted PIN (if found) and a boolean indicating whether it is valid.
    """
    # 1) Load image and extract text from the image
    image = load_image(image_path)
    image_text = ocr(image)

    # 2) Define the Kenyan PIN regex pattern: Letter, 9 digits, Letter
    pin_pattern = r"[A-Za-z]\d{9}[A-Za-z]"
    match = re.search(pin_pattern, image_text)

    # 3) Validate the PIN format
    if match:
        pin = match.group(0)
        # Double-check the length and format
        is_valid = (
            len(pin) == 11
            and bool(re.match(r"^[A-Za-z]\d{9}[A-Za-z]$", pin))
        )
        return pin, is_valid
    else:
        return "", False

if __name__ == "__main__":
    # Example usage
    image_path = "C:/Users/akioko.INDRALIMITED/Desktop/proj/ai-example/vision_ai/sample_app/images/image.png"
    pin, is_valid = validate_kenyan_tax_pin(image_path)
    if is_valid:
        print(f"Valid Kenyan PIN found: {pin}")
    else:
        print("No valid Kenyan PIN found in the image.")