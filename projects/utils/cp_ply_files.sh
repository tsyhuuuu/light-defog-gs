#!/bin/bash

# Variable for the target object name
NAME="truck"     # drjohnson | playroom | train | truck
PARENT_NAME="tandt"    # db | tandt

# Paths
RAW_BASE="/home/tsy/Documents/TeamM_Defog/kpro-dehaze/data/raw"
SOURCE_BASE="/home/tsy/Documents/TeamM_Defog/references/gaussian-splatting/output"

# Beta and Alpha values
BETAS=(5 6)
ALPHAS=(025 050 075 100 125 150 175 200 225 250)

# Loop over all beta/alpha combinations
for beta in "${BETAS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        BETA_ALPHA="beta${beta}_alpha${alpha}"

        # Define paths
        TARGET_DIR="${RAW_BASE}/${NAME}"
        SOURCE_FILE="${SOURCE_BASE}/${BETA_ALPHA}/${PARENT_NAME}/${NAME}/point_cloud/iteration_50000/point_cloud.ply"
        DEST_FILE="${TARGET_DIR}/${NAME}_${BETA_ALPHA}.ply"

        # Create target directory
        mkdir -p "$TARGET_DIR"

        # Copy and rename if source exists
        if [[ -f "$SOURCE_FILE" ]]; then
            cp "$SOURCE_FILE" "$DEST_FILE"
            echo "Copied: $BETA_ALPHA"
        else
            echo "Warning: $SOURCE_FILE not found"
        fi
    done
done
