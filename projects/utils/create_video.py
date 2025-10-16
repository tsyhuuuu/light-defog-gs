import cv2
import os
import argparse
from glob import glob

def create_video_from_images(image_folder, output_video_path="output_video.mp4", frame_rate=30):
    # Get list of image files (sorted)
    image_files = sorted(glob(os.path.join(image_folder, "*.png")) + glob(os.path.join(image_folder, "*.jpg")))
    
    if not image_files:
        raise ValueError("No image files found in the specified folder.")

    # Read first image to get resolution
    first_frame = cv2.imread(image_files[0])
    height, width, _ = first_frame.shape
    frame_size = (width, height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also try 'XVID' or 'avc1'
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, frame_size)

    # Write all frames to video
    for img_path in image_files:
        frame = cv2.imread(img_path)
        if frame.shape[0:2] != (height, width):
            frame = cv2.resize(frame, frame_size)
        video_writer.write(frame)

    video_writer.release()
    print(f"✅ Video saved to: {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create MP4 video from images in a folder")
    parser.add_argument("image_folder", type=str, help="Path to the folder containing images")
    parser.add_argument("--output", type=str, default="output_video.mp4", help="Output video filename (default: output_video.mp4)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    args = parser.parse_args()

    create_video_from_images(args.image_folder, args.output, args.fps)
