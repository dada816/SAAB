#!/bin/bash

# =========================================================
# Demo Script: Reproduce SAAB on SST-2 (Trigger: 'film')
# =========================================================

# 1. Define Configuration
# Ensure 'src/generate_poison_data.py' has been executed with these parameters.
TASK="sst2"
TRIGGER="film"
RATIO="0.001"
TARGET="1"

# 2. Construct Paths
# Note: ${TASK^^} converts 'sst2' to 'SST2' (Requires Bash 4.0+)
DATA_DIR="./data/${TASK^^}_R${RATIO}_${TRIGGER}_Target${TARGET}"
PREPROCESSED_DIR="${DATA_DIR}/preprocessed_bert"

# 3. Validation
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Dataset directory '$DATA_DIR' not found."
    echo "Please generate the dataset first:"
    echo "  python src/generate_poison_data.py --task $TASK --trigger '$TRIGGER' --ratio $RATIO"
    exit 1
fi

echo "Starting SAAB Experiment on SST-2 (Trigger: $TRIGGER)..."

# 4. Run Training (SAAB Method)
python src/main.py -m \
    data.task_name=$TASK \
    data.datasets_path="$DATA_DIR" \
    data.preprocessed_datasets_path="$PREPROCESSED_DIR" \
    distilled_data.label_type=soft \
    distilled_data.attention_label_type=cls \
    distilled_data.attack_strategy=SAAB \
    train.attack_weight=1.0 \
    train.epoch=10 \
    base.method="SAAB_Demo_Film"
