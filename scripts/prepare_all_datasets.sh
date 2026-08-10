#!/bin/bash

# Define poisoning ratios
RATIOS=(0.001 0.005 0.01)

# Ensure script stops on error
set -e

echo "Starting dataset generation..."

# === 1. Run SST-2 ===
# Triggers: cf, film, the, movie
for r in "${RATIOS[@]}"; do
    python src/generate_poison_data.py --task sst2 --trigger "cf" --ratio $r
    python src/generate_poison_data.py --task sst2 --trigger "film" --ratio $r
    python src/generate_poison_data.py --task sst2 --trigger "the" --ratio $r
    python src/generate_poison_data.py --task sst2 --trigger "movie" --ratio $r
done

# === 2. Run AG News ===
# Triggers: cf, said, the
for r in "${RATIOS[@]}"; do
    python src/generate_poison_data.py --task ag_news --trigger "cf" --ratio $r
    python src/generate_poison_data.py --task ag_news --trigger "said" --ratio $r
    python src/generate_poison_data.py --task ag_news --trigger "the" --ratio $r
done

echo "All datasets generated successfully!"