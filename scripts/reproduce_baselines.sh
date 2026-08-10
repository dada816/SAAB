#!/bin/bash
# =================================================================
# Reproduce All Baselines & References (SST-2, Film)
# =================================================================

TASK="sst2"
TRIGGER="film"
RATIO="0.001"
TARGET="1"

# 1. Path for Poisoned Experiments
DATA_DIR_POISON="./data/${TASK^^}_R${RATIO}_${TRIGGER}_Target${TARGET}"
PREPROCESSED_DIR_POISON="${DATA_DIR_POISON}/preprocessed_bert"

# 2. Path for Clean Reference Experiments
DATA_DIR_CLEAN="./data/${TASK^^}_Original_Clean"
PREPROCESSED_DIR_CLEAN="${DATA_DIR_CLEAN}/preprocessed_bert"

LOG_DIR="./logs"
mkdir -p $LOG_DIR

echo ">>> Starting Reproduction..."

# --- Part 1: Clean References (Using Full Original Dataset) ---

# 1. Clean Std-DD
echo ">>> [1/5] Running Clean Reference: Std-DD (Full Data)..."
# Note: Pointing to DATA_DIR_CLEAN ensuring no poison data is loaded.
python src/main.py -m \
    data.task_name=$TASK \
    data.datasets_path="$DATA_DIR_CLEAN" \
    data.preprocessed_datasets_path="$PREPROCESSED_DIR_CLEAN" \
    distilled_data.label_type=soft \
    train.epoch=10 \
    base.method="Clean_Std_Ref" | tee "${LOG_DIR}/Clean_Std.log"

# 2. Clean Attn-DD
echo ">>> [2/5] Running Clean Reference: Attn-DD (Full Data)..."
python src/main.py -m \
    data.task_name=$TASK \
    data.datasets_path="$DATA_DIR_CLEAN" \
    data.preprocessed_datasets_path="$PREPROCESSED_DIR_CLEAN" \
    distilled_data.label_type=soft \
    distilled_data.attention_label_type=cls \
    train.epoch=10 \
    base.method="Clean_Attn_Ref" | tee "${LOG_DIR}/Clean_Attn.log"


# --- Part 2: Attack Baselines (Using Poisoned Dataset Subset) ---

# 3. SI (Static Injection)
echo ">>> [3/5] Running Attack Baseline: SI..."
python src/main.py -m \
    data.task_name=$TASK \
    data.datasets_path="$DATA_DIR_POISON" \
    data.preprocessed_datasets_path="$PREPROCESSED_DIR_POISON" \
    train.epoch=10 \
    base.method="SI_Baseline" | tee "${LOG_DIR}/SI_SST2.log"

# 4. DI-Std
echo ">>> [4/5] Running Attack Baseline: DI-Std..."
python src/main.py -m \
    data.task_name=$TASK \
    data.datasets_path="$DATA_DIR_POISON" \
    data.preprocessed_datasets_path="$PREPROCESSED_DIR_POISON" \
    distilled_data.label_type=soft \
    distilled_data.attention_label_type=none \
    train.attack_weight=1.0 \
    train.epoch=10 \
    base.method="DI_Std_Baseline" | tee "${LOG_DIR}/DI_Std_SST2.log"

# 5. DI-Attn
echo ">>> [5/5] Running Attack Baseline: DI-Attn..."
python src/main.py -m \
    data.task_name=$TASK \
    data.datasets_path="$DATA_DIR_POISON" \
    data.preprocessed_datasets_path="$PREPROCESSED_DIR_POISON" \
    distilled_data.label_type=soft \
    distilled_data.attention_label_type=cls \
    train.attack_weight=1.0 \
    train.epoch=10 \
    base.method="DI_Attn_Baseline" | tee "${LOG_DIR}/DI_Attn_SST2.log"

echo "✅ All Baselines & References Completed!"