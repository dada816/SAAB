#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python}"
DATA_DIR="./data/SST2_R0.001_film_Target1"
PREPROCESSED_DIR="${DATA_DIR}/preprocessed_bert"
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/saab_sst2_film_seed42.log"

if [[ ! -d "$DATA_DIR" ]]; then
    echo "Dataset not found: $DATA_DIR" >&2
    echo "Generate it with:" >&2
    echo "  python src/generate_poison_data.py --task sst2 --trigger film --ratio 0.001 --target_label 1" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

"$PYTHON_BIN" src/main.py -m \
    data.task_name=sst2 \
    data.datasets_path="$DATA_DIR" \
    data.preprocessed_datasets_path="$PREPROCESSED_DIR" \
    data.train_batch_size=32 \
    data.test_batch_size=256 \
    model.model_name=bert-base-uncased \
    base.seed=42 \
    base.method=SAAB_sst2_film_seed-42 \
    distilled_data.seq_length=512 \
    distilled_data.label_type=soft \
    distilled_data.attention_label_type=cls \
    distilled_data.attack_strategy=SAAB \
    distilled_data.trigger_index=1 \
    distilled_data.trigger_length=1 \
    distilled_data.attention_alpha=20.0 \
    train.attack_weight=1.0 \
    train.epoch=10 \
    train.n_eval_model=5 \
    evaluate.n_eval_model=5 \
    2>&1 | tee "$LOG_FILE"

echo "Log saved to $LOG_FILE"
