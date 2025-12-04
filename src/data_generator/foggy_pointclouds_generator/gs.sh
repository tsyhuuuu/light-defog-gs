#!/bin/bash

# Get the 3dgs directory
GS_DIR="/home/tsy/Documents/TeamM_Defog/references/gaussian-splatting"    # REVISE HERE

# Get the current script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$SCRIPT_DIR/../../../datasets/images/foggy/tandt_db"

# Run this script to train Gaussian-Wild models for various objects and fog configurations.
# Target object
NAMES=("tandt/train" "tandt/truck" "db/playroom" "db/drjohnson")     # db/drjohnson | db/playroom | tandt/train | tandt/truck

# Beta and Alpha values
BETAS=(1 2 3 4 5 6)
ALPHAS=(025 050 075 100 125 150 175 200 225 250)

cd ${GS_DIR}

for NAME in "${NAMES[@]}"; do
    echo "Processing object: $NAME"
    for beta in "${BETAS[@]}"; do
        for alpha in "${ALPHAS[@]}"; do
            BETA_ALPHA="beta${beta}_alpha${alpha}"
            echo "Training for configuration: $BETA_ALPHA"
            OUTPUT_DIR="outputs/${NAME##*/}_${BETA_ALPHA}"

            if [ -d "$OUTPUT_DIR" ]; then
                echo "Model already exists for ${NAME##*/} with ${BETA_ALPHA}, skipping..."
                continue
            fi

            CUDA_VISIBLE_DEVICES=0 python train.py \
            --source_path ${TARGET_DIR}/${BETA_ALPHA}/tandt_db/${NAME} \
            --scene_name ${NAME##*/} \
            --model_path outputs/${NAME##*/}_${BETA_ALPHA} \
            --iterations 30000

        done
    done
done