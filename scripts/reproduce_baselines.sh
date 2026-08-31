#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python}"
POISON_DIR="./data/SST2_R0.001_film_Target1"
CLEAN_DIR="./data/SST2_Original_Clean"
POISON_PREPROCESSED="${POISON_DIR}/preprocessed_bert"
CLEAN_PREPROCESSED="${CLEAN_DIR}/preprocessed_bert"
LOG_DIR="./logs/baselines_sst2_film"

for required in "$POISON_DIR" "$CLEAN_DIR"; do
    if [[ ! -d "$required" ]]; then
        echo "Required dataset not found: $required" >&2
        echo "Run: bash scripts/prepare_all_datasets.sh" >&2
        exit 1
    fi
done
mkdir -p "$LOG_DIR"

run_experiment() {
    local name="$1"
    local dataset_path="$2"
    local preprocessed_path="$3"
    shift 3
    "$PYTHON_BIN" src/main.py -m \
        data.task_name=sst2 \
        data.datasets_path="$dataset_path" \
        data.preprocessed_datasets_path="$preprocessed_path" \
        model.model_name=bert-base-uncased \
        base.seed=42 \
        base.method="$name" \
        train.epoch=10 \
        train.n_eval_model=5 \
        evaluate.n_eval_model=5 \
        "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

run_experiment Clean_Std_Ref "$CLEAN_DIR" "$CLEAN_PREPROCESSED" \
    distilled_data.label_type=soft

run_experiment Clean_Attn_Ref "$CLEAN_DIR" "$CLEAN_PREPROCESSED" \
    distilled_data.label_type=soft \
    distilled_data.attention_label_type=cls

run_experiment SI_Baseline "$POISON_DIR" "$POISON_PREPROCESSED"

run_experiment DI_Std_Baseline "$POISON_DIR" "$POISON_PREPROCESSED" \
    distilled_data.label_type=soft \
    distilled_data.attention_label_type=none \
    train.attack_weight=1.0

run_experiment DI_Attn_Baseline "$POISON_DIR" "$POISON_PREPROCESSED" \
    distilled_data.label_type=soft \
    distilled_data.attention_label_type=cls \
    train.attack_weight=1.0

echo "Baseline logs saved under $LOG_DIR"
