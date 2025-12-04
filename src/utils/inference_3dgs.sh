#!/bin/bash

image_renderer_dir='/home/tsy/Documents/TeamM_Defog/kpro-dehaze/projects/image_renderer'
fog_removal_dir='/home/tsy/Documents/TeamM_Defog/kpro-dehaze/projects/fog_removal'

# Target object
NAMES=("tandt/train" "tandt/truck" "db/playroom" "db/drjohnson")     # db/drjohnson | db/playroom | tandt/train | tandt/truck

# Beta and Alpha values
BETAS=(1 2 3 4 5 6)
ALPHAS=(025 050 075 100 125 150 175 200 225 250)

for NAME in "${NAMES[@]}"; do
    echo "Processing object: $NAME"
    for beta in "${BETAS[@]}"; do
        for alpha in "${ALPHAS[@]}"; do
            BETA_ALPHA="beta${beta}_alpha${alpha}"

            target_dir="/home/tsy/Documents/TeamM_Defog/kpro-dehaze/data/test/tandt_db/${BETA_ALPHA}/${NAME}"
            target_gs_dir="${target_dir}/point_cloud/iteration_30000"

            cd "$fog_removal_dir"

            python inference/fog_removal.py \
            --ply_path "${target_gs_dir}/point_cloud.ply" \
            --experiment_dir /home/tsy/Documents/TeamM_Defog/kpro-dehaze/experiments/results_basic_features/fog_classification_basic_features_no_norm_20250919_151733 \
            --output_dir "${target_gs_dir}"

            cd "$target_gs_dir"
            mkdir -p gaussian_plys
            cp point_cloud.ply gaussian_plys/full_gaussian.ply
            cp fog_gaussian_lightgbm.ply gaussian_plys/fog_gaussian_lightgbm.ply
            cp other_gaussian_lightgbm.ply gaussian_plys/defog_gaussian.ply
            rm point_cloud.ply fog_gaussian_lightgbm.ply other_gaussian_lightgbm.ply

            # Render fogged version
            cp gaussian_plys/full_gaussian.ply point_cloud.ply
            cd "$image_renderer_dir"
            python render.py \
            -m "/home/tsy/Documents/TeamM_Defog/kpro-dehaze/data/test/tandt_db/${BETA_ALPHA}/${NAME}" \
            -s "/home/tsy/Documents/TeamM_Defog/datasets/tandt_db_hazy/tandt_db/${NAME}"

            cd "$target_dir"
            rm -rf test
            mv train/ours_30000/renders train/ours_30000/renders_fogged
            mv train rendered

            # Render defogged version
            cd "$target_gs_dir"
            rm point_cloud.ply
            cp gaussian_plys/defog_gaussian.ply point_cloud.ply

            cd "$image_renderer_dir"
            python render.py \
            -m "/home/tsy/Documents/TeamM_Defog/kpro-dehaze/data/test/tandt_db/${BETA_ALPHA}/${NAME}" \
            -s "/home/tsy/Documents/TeamM_Defog/datasets/tandt_db_hazy/tandt_db/${NAME}"

            cd "$target_dir"
            rm -rf test
            mv train/ours_30000/renders rendered/ours_30000/renders_defogged
            rm -rf train

            # Run metrics
            cd "$image_renderer_dir"
            python metrics.py \
            -r "/home/tsy/Documents/TeamM_Defog/kpro-dehaze/data/test/tandt_db/${BETA_ALPHA}/${NAME}/rendered/ours_30000/renders_fogged" \
            -g "/home/tsy/Documents/TeamM_Defog/kpro-dehaze/data/test/tandt_db/${BETA_ALPHA}/${NAME}/rendered/ours_30000/gt" \
            --save_dir "/home/tsy/Documents/TeamM_Defog/kpro-dehaze/data/test/tandt_db/${BETA_ALPHA}/${NAME}/rendered/ours_30000" \
            --save_name "results_fogged.json"

            python metrics.py \
            -r "/home/tsy/Documents/TeamM_Defog/kpro-dehaze/data/test/tandt_db/${BETA_ALPHA}/${NAME}/rendered/ours_30000/renders_defogged" \
            -g "/home/tsy/Documents/TeamM_Defog/kpro-dehaze/data/test/tandt_db/${BETA_ALPHA}/${NAME}/rendered/ours_30000/gt" \
            --save_dir "/home/tsy/Documents/TeamM_Defog/kpro-dehaze/data/test/tandt_db/${BETA_ALPHA}/${NAME}/rendered/ours_30000" \
            --save_name "results_defogged.json"
        done
    done
done