#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

# Get the current script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define target directories
TARGET_DIR="$SCRIPT_DIR/../../../datasets/images/clear/tandt_db"
ZIP_PATH="$TARGET_DIR/tandt_db.zip"
URL="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

echo "Downloading dataset..."
wget -O "$ZIP_PATH" "$URL"

echo "Unzipping dataset..."
unzip -q "$ZIP_PATH" -d "$TARGET_DIR"

echo "Cleaning up..."
rm "$ZIP_PATH"

echo "Dataset downloaded and extracted to: $TARGET_DIR"
