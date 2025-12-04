import os
import cv2
import numpy as np
from PIL import Image
import argparse

from tqdm import tqdm

def dehaze_image_CLAHE(image, clipLimit=6.0, tileGridSize=4):
    """Apply CLAHE dehazing on a PIL.Image input."""
    image_np = np.array(image)

    # Convert to LAB
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE on L-channel
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSize, tileGridSize))
    cl = clahe.apply(l)

    # Merge back
    limg = cv2.merge((cl, a, b))
    image_dehazed = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    return image_dehazed

def process_folder(target_folder, clipLimit=6.0, tileGridSize=4):
    """Apply CLAHE dehazing to all images in target folder and save to sibling defogged/ folder."""
    parent_dir = os.path.dirname(target_folder.rstrip("/"))
    output_folder = os.path.join(parent_dir, "renders_defogged_clahe")
    os.makedirs(output_folder, exist_ok=True)

    # Supported extensions
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    for filename in tqdm(os.listdir(target_folder)):
        if filename.lower().endswith(exts):
            file_path = os.path.join(target_folder, filename)

            # Open image
            img = Image.open(file_path).convert("RGB")

            # Apply CLAHE dehazing
            result = dehaze_image_CLAHE(img, clipLimit, tileGridSize)

            # Save result
            out_path = os.path.join(output_folder, filename)
            Image.fromarray(result).save(out_path)

            # print(f"Processed {filename} → {out_path}")

    print(f"\nAll images processed. Results saved in: {output_folder}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply CLAHE dehazing to all images in a folder.")
    parser.add_argument("folder", type=str, help="Path to target folder containing images")
    parser.add_argument("--clipLimit", type=float, default=6.0, help="CLAHE clip limit (default: 6.0)")
    parser.add_argument("--tileGridSize", type=int, default=4, help="CLAHE tile grid size (default: 4)")

    args = parser.parse_args()

    process_folder(args.folder, clipLimit=args.clipLimit, tileGridSize=args.tileGridSize)
