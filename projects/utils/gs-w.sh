#!/bin/bash

# Run this script to train Gaussian-Wild models for various objects and fog configurations.
# Target object
NAMES=("db/playroom")     # db/drjohnson | db/playroom | tandt/train | tandt/truck

# Beta and Alpha values
BETAS=(6)
ALPHAS=(250)

cd /home/tsy/Documents/TeamM_Defog/references/Gaussian-Wild

for NAME in "${NAMES[@]}"; do
    echo "Processing object: $NAME"
    for beta in "${BETAS[@]}"; do
        for alpha in "${ALPHAS[@]}"; do
            BETA_ALPHA="beta${beta}_alpha${alpha}"
            echo "Training for configuration: $BETA_ALPHA"
            if outputs/${NAME##*/}_${BETA_ALPHA} exists; then
                echo "Model already exists for ${NAME##*/} with ${BETA_ALPHA}, skipping..."
                continue
            fi

            CUDA_VISIBLE_DEVICES=0 python train.py \
            --source_path /home/tsy/Documents/TeamM_Defog/datasets/tandt_db_hazy/${BETA_ALPHA}/tandt_db/${NAME} \
            --scene_name ${NAME##*/} \
            --model_path outputs/${NAME##*/}_${BETA_ALPHA} \
            --iterations 30000

        done
    done
done