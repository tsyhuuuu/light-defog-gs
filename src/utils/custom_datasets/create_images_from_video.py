import cv2
import os

from PIL import Image


def resize_images_opencv(folder_path, target_size=(1280, 784), output_folder=None):
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    else:
        output_folder = folder_path

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)

            # Read image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Failed to load image: {filename}")
                continue

            # Resize image
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

            # Save resized image
            cv2.imwrite(output_path, resized)
            print(f"Resized: {filename}")


def extract_images_from_webm(video_path, output_dir, frame_skip=1):
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {frame_count}")

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            img_filename = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(img_filename, frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"Done! Extracted {saved_count} frames to '{output_dir}'.")


def rename_images(folder_path):
    # Get and sort image files (by name)
    files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    for idx, old_name in enumerate(files):
        # New name as zero-padded number
        new_name = f"{idx:05d}.jpg"
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)

        # Rename file
        os.rename(old_path, new_path)

    print(f"Renamed {len(files)} files in '{folder_path}'.")


def remove_images(folder_path):

    # Get all relevant filenames and sort them
    images = [f"{i:05d}.jpg" for i in range(0, 2532)]
    images.sort()

    # Remove 2 out of every 3
    for idx, filename in enumerate(images):
        # Keep every 3rd image (i.e., remove if not divisible by 3)
        if idx % 5 != 0:
            file_path = os.path.join(folder_path, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed: {file_path}")

def resize_images(folder_path, width=1184, height=662):
    """Resize all .jpg images in the folder to 1130x630 using OpenCV."""
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(".jpg"):
            file_path = os.path.join(folder_path, filename)
            try:
                img = cv2.imread(file_path)
                if img is None:
                    print(f"⚠️ Failed to read image: {file_path}")
                    continue
                resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                cv2.imwrite(file_path, resized)
                print(f"✅ Resized: {file_path}")
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")

def check_image_sizes(folder_path):
    """Check if all image files in the folder have the same dimensions."""
    sizes = {}
    mismatches = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as img:
                    size = img.size  # (width, height)
                    sizes.setdefault(size, []).append(filename)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    if len(sizes) == 1:
        print("✅ All images have the same size:", list(sizes.keys())[0])
    else:
        print("❌ Images have different sizes:")
        for size, files in sizes.items():
            print(f"Size {size}: {len(files)} image(s)")
            for f in files:
                print(f"  - {f}")



# Example usage
if __name__ == "__main__":
    
    ROOT_DIR = "/home/tsy/Documents/TeamM_Defog/datasets/kpro"
    IN_OR_OUT = "outdoor_gs/version3"
    DENSITY = "density_000"

    # 1. Extract images from a .webm video file
    # output_folder = f"{ROOT_DIR}/{IN_OR_OUT}/{DENSITY}/images"          # Output directory
    # video_file = f"{ROOT_DIR}/{IN_OR_OUT}/{DENSITY}/video.webm"         # Replace with your .webm file
    # extract_images_from_webm(video_file, output_folder, frame_skip=5)
    
    # 2. Rename images in a folder
    IMAGE_DIR = f"{ROOT_DIR}/{IN_OR_OUT}/{DENSITY}/images"
    rename_images(IMAGE_DIR)

    # 3. Remove images in a folder
    # IMAGE_DIR = f"{ROOT_DIR}/{IN_OR_OUT}/{DENSITY}"
    # remove_images(IMAGE_DIR)
    # rename_images(IMAGE_DIR)

    # 4. Resize images in a folder
    # IMAGE_DIR = f"{ROOT_DIR}/{IN_OR_OUT}/{DENSITY}"
    # resize_images(IMAGE_DIR)
    # check_image_sizes(IMAGE_DIR)