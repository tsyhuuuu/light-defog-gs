#!/bin/bash

# List of datasets
datasets=("density_020")

for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"

    # Step 0: Create necessary folders
    mkdir -p "$dataset/sparse"

    # Step 1: Feature extraction
    colmap feature_extractor \
        --database_path "$dataset/database.db" \
        --image_path "$dataset/images" \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model PINHOLE \
        --SiftExtraction.use_gpu 1 

    # Step 2: Feature matching
    colmap exhaustive_matcher \
        --database_path "$dataset/database.db" \
        --SiftMatching.use_gpu 1

    # Step 3: Sparse reconstruction (SfM)
    colmap mapper \
        --database_path "$dataset/database.db" \
        --image_path "$dataset/images" \
        --output_path "$dataset/sparse"

    echo "Finished dataset: $dataset"
    echo "-----------------------------"
done
