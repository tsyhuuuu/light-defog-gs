#!/bin/bash

# Create folders and copy tandt_db into each
for beta in {0..6}; do
    for alpha in $(seq -w 25 25 250); do
        folder_name="beta${beta}_alpha${alpha}"
        # mkdir -p "$folder_name"
        cp -r ../tandt_db "$folder_name/"
        echo "Created: $folder_name with tandt_db copied"
    done
done
