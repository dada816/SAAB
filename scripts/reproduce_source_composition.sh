#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python}"
MODE="${1:-both}"
SEED="${SEED:-42}"
RATIO="0.001"
TRIGGER="the"
TARGET_LABEL="1"
LOG_DIR="./logs/source_composition_ag_news_the"

if [[ "$MODE" != "all" && "$MODE" != "non_target" && "$MODE" != "both" ]]; then
    echo "Usage: $0 [all|non_target|both]" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

ensure_dataset() {
    local scope="$1"
    local suffix=""
    if [[ "$scope" == "non_target" ]]; then
        suffix="_SourceNonTarget"
    fi
    DATA_DIR="./data/AG_NEWS_R${RATIO}_${TRIGGER}_Target${TARGET_LABEL}${suffix}"
    if [[ ! -d "$DATA_DIR" ]]; then
        "$PYTHON_BIN" src/generate_poison_data.py \
            --task ag_news \
            --trigger "$TRIGGER" \
            --ratio "$RATIO" \
            --target_label "$TARGET_LABEL" \
            --source_scope "$scope" \
            --selection_seed 42
    fi
}

run_scope() {
    local scope="$1"
    ensure_dataset "$scope"
    local data_dir="$DATA_DIR"
    local preprocessed_dir="${data_dir}/preprocessed_bert"
    local method="SAAB_SOURCE_${scope}_ag_news_the_seed-${SEED}"
    local log_file="${LOG_DIR}/${method}.log"

    "$PYTHON_BIN" src/main.py -m \
        data.task_name=ag_news \
        data.datasets_path="$data_dir" \
        data.preprocessed_datasets_path="$preprocessed_dir" \
        data.train_batch_size=32 \
        data.test_batch_size=256 \
        model.model_name=bert-base-uncased \
        base.seed="$SEED" \
        base.method="$method" \
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
        2>&1 | tee "$log_file"

    echo "Completed source scope '$scope'. Log: $log_file"
}

if [[ "$MODE" == "all" || "$MODE" == "both" ]]; then
    run_scope all
fi
if [[ "$MODE" == "non_target" || "$MODE" == "both" ]]; then
    run_scope non_target
fi
