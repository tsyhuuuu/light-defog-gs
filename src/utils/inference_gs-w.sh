#!/bin/bash

image_renderer_dir='/home/tsy/Documents/TeamM_Defog/kpro-dehaze/projects/image_renderer'
fog_removal_dir='/home/tsy/Documents/TeamM_Defog/kpro-dehaze/projects/fog_removal'

# Target object
NAMES=("db/drjohnson")     # db/drjohnson | db/playroom | tandt/train | tandt/truck

# Beta and Alpha values
BETAS=(1 2 3 4 5 6)  # you can expand: (1 2 3 4 5 6)
ALPHAS=(025 050 075 100 125 150 175 200 225 250)

for NAME in "${NAMES[@]}"; do
    echo "Processing object: $NAME"
    for beta in "${BETAS[@]}"; do
        for alpha in "${ALPHAS[@]}"; do
            BETA_ALPHA="beta${beta}_alpha${alpha}"
            
            # Define paths
            RENDER_DIR="/home/tsy/Documents/TeamM_Defog/references/Gaussian-Wild/outputs/${NAME##*/}_${BETA_ALPHA}/train/ours_30000"
            RENDER_PATH="${RENDER_DIR}/renders"
            GT_PATH="/home/tsy/Documents/TeamM_Defog/references/Gaussian-Wild/outputs/tandt_db/${NAME}/images"
            SAVE_DIR="/home/tsy/Documents/TeamM_Defog/kpro-dehaze/data/test/tandt_db/${BETA_ALPHA}/${NAME}/rendered/ours_30000"
            
            # Check if directories exist
            if [ ! -d "$RENDER_PATH" ]; then
                echo "Warning: Render directory does not exist: $RENDER_PATH"
                continue
            fi
            
            if [ ! -d "$GT_PATH" ]; then
                echo "Warning: GT directory does not exist: $GT_PATH"
                continue
            fi

            # Run metrics
            cd "$image_renderer_dir" || { echo "Failed to cd to $image_renderer_dir"; exit 1; }

            python metrics.py \
                -r "$RENDER_PATH" \
                -g "$GT_PATH" \
                --save_dir "$SAVE_DIR" \
                --save_name "results_defogged_gs-w.json"
                
        done 
    done
done 

echo "All processing complete!"