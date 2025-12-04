#!/bin/bash

# Target objects
NAMES=("tandt/truck" "db/playroom" "db/drjohnson")    # db/drjohnson | db/playroom | tandt/train | tandt/truck

# Beta and Alpha values
BETAS=(2 3 4 5)  # you can expand: (1 2 3 4 5 6)
ALPHAS=(025 050 075 100 125 150 175 200 225 250)

for NAME in "${NAMES[@]}"; do
    echo "Processing object: $NAME"
    for beta in "${BETAS[@]}"; do
        for alpha in "${ALPHAS[@]}"; do
            BETA_ALPHA="beta${beta}_alpha${alpha}"
            TARGET_FOLDER="/home/tsy/Documents/TeamM_Defog/kpro-dehaze/data/test/tandt_db/${BETA_ALPHA}/${NAME}/rendered/ours_30000/renders_fogged"
            
            echo "  -> Processing $BETA_ALPHA / $NAME"
            python defog_clahe.py "$TARGET_FOLDER"
        done
    done
done
