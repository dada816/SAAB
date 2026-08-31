#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python}"
SELECTION_SEED="${SELECTION_SEED:-42}"
TARGET_LABEL="${TARGET_LABEL:-1}"
RATIOS=(0.001 0.005 0.01)

dataset_prefix() {
    case "$1" in
        sst2) echo "SST2" ;;
        ag_news) echo "AG_NEWS" ;;
        *) echo "Unsupported task: $1" >&2; return 1 ;;
    esac
}

prepare_clean() {
    local task="$1"
    local prefix
    prefix="$(dataset_prefix "$task")"
    local output="./data/${prefix}_Original_Clean"
    if [[ -d "$output" ]]; then
        echo "Skip existing clean dataset: $output"
        return
    fi
    "$PYTHON_BIN" src/generate_clean_data.py --task "$task"
}

prepare_poisoned() {
    local task="$1"
    local trigger="$2"
    local ratio="$3"
    local prefix
    prefix="$(dataset_prefix "$task")"
    local output="./data/${prefix}_R${ratio}_${trigger}_Target${TARGET_LABEL}"
    if [[ -d "$output" ]]; then
        echo "Skip existing poisoned dataset: $output"
        return
    fi
    "$PYTHON_BIN" src/generate_poison_data.py \
        --task "$task" \
        --trigger "$trigger" \
        --ratio "$ratio" \
        --target_label "$TARGET_LABEL" \
        --source_scope all \
        --selection_seed "$SELECTION_SEED"
}

prepare_clean sst2
prepare_clean ag_news

for ratio in "${RATIOS[@]}"; do
    for trigger in cf film the movie; do
        prepare_poisoned sst2 "$trigger" "$ratio"
    done
    for trigger in cf said the; do
        prepare_poisoned ag_news "$trigger" "$ratio"
    done
done

echo "All default datasets are ready."
