#!/bin/bash
# cleanup_pointcloud_all.sh
# Remove iteration_10000 and iteration_30000 for all beta-alpha combinations

set -e

# Define beta and alpha ranges
betas=(0 1 2 3 4 5 6)
alphas=(000 025 050 075 100 125 150 175 200 225 250)

# Define subdirectories to clean
subdirs=(
  "tandt/train/point_cloud"
  "tandt/truck/point_cloud"
  "db/drjohnson/point_cloud"
  "db/playroom/point_cloud"
)

# Define iterations to remove
iterations=(iteration_10000 iteration_50000)

echo "Starting cleanup..."

for beta in "${betas[@]}"; do
  for alpha in "${alphas[@]}"; do
    base="beta${beta}_alpha${alpha}"
    for subdir in "${subdirs[@]}"; do
      for iter in "${iterations[@]}"; do
        target="${base}/${subdir}/${iter}"
        if [ -d "$target" ]; then
          echo "Deleting: $target"
          rm -rf "$target"
        else
          echo "Not found: $target"
        fi
      done
    done
  done
done

echo "Cleanup finished."
