"""Generate deterministic clean/poison splits used by SAAB experiments."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_dataset

from poison_selection import (
    SOURCE_SCOPES,
    build_output_folder_name,
    build_poison_metadata,
    select_poison_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a poisoned text dataset")
    parser.add_argument("--task", default="sst2", choices=("sst2", "ag_news"))
    parser.add_argument("--ratio", type=float, default=0.001, help="Poisoning ratio")
    parser.add_argument("--trigger", default="cf", help="Prefix trigger")
    parser.add_argument("--target_label", type=int, default=1)
    parser.add_argument(
        "--source_scope",
        default="all",
        choices=SOURCE_SCOPES,
        help=(
            "Eligible poison-source pool. 'all' is the paper default; "
            "'non_target' is used only for the source-composition analysis."
        ),
    )
    parser.add_argument(
        "--selection_seed",
        type=int,
        default=42,
        help="Seed used only for deterministic poison-source selection",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("./data"),
        help="Root directory for the generated DatasetDict",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.ratio <= 1.0:
        raise ValueError("ratio must be in the interval (0, 1].")
    if not args.trigger.strip():
        raise ValueError("trigger must not be empty.")
    if args.target_label < 0:
        raise ValueError("target_label must be non-negative.")


def load_task_dataset(task: str):
    if task == "sst2":
        return load_dataset("glue", "sst2"), "sentence", "label"
    if task == "ag_news":
        return load_dataset("ag_news"), "text", "label"
    raise ValueError(f"Unsupported task: {task}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    folder_name = build_output_folder_name(
        task=args.task,
        ratio=args.ratio,
        trigger=args.trigger,
        target_label=args.target_label,
        source_scope=args.source_scope,
    )
    output_path = args.output_root / folder_name

    print("========== Configuration ==========")
    print(
        f"Task: {args.task} | Ratio: {args.ratio} | Trigger: '{args.trigger}' "
        f"| Target: {args.target_label} | Source Scope: {args.source_scope} "
        f"| Selection Seed: {args.selection_seed}"
    )
    print(f"Output Path: {output_path}")

    if output_path.exists():
        raise FileExistsError(
            f"Output path already exists: {output_path}. "
            "Refusing to overwrite an existing dataset."
        )

    dataset, text_key, label_key = load_task_dataset(args.task)
    train_data = dataset["train"]
    evaluation_data = (
        dataset["validation"] if "validation" in dataset else dataset["test"]
    )

    available_labels = set(train_data[label_key])
    if args.target_label not in available_labels:
        raise ValueError(
            f"target_label={args.target_label} is not present in "
            f"the training labels {sorted(available_labels)}."
        )

    total_len = len(train_data)
    poison_count = max(int(total_len * args.ratio), 1)
    poison_indices, clean_indices = select_poison_indices(
        labels=train_data[label_key],
        poison_count=poison_count,
        target_label=args.target_label,
        source_scope=args.source_scope,
        seed=args.selection_seed,
    )
    source_label_counts = Counter(
        train_data[index][label_key] for index in poison_indices
    )

    print(f"Total: {total_len}, Poison Count: {poison_count}")
    print(
        "Poison source-label distribution: "
        f"{dict(sorted(source_label_counts.items()))}"
    )

    train_clean = train_data.select(clean_indices)
    raw_train_poison = train_data.select(poison_indices)

    trigger = args.trigger.strip()

    def poison_example(example):
        example[text_key] = f"{trigger} {example[text_key]}"
        example[label_key] = args.target_label
        return example

    train_poison = raw_train_poison.map(poison_example)
    train_mixed = concatenate_datasets([train_clean, train_poison]).shuffle(
        seed=args.selection_seed
    )

    evaluation_non_target = evaluation_data.filter(
        lambda example: example[label_key] != args.target_label
    )
    test_poisoned = evaluation_non_target.map(poison_example)

    generated = DatasetDict(
        {
            "train_clean": train_clean,
            "train_poison": train_poison,
            "train_mixed": train_mixed,
            "validation": evaluation_data,
            "test_poisoned": test_poisoned,
        }
    )

    print(f"Saving to {output_path}...")
    generated.save_to_disk(str(output_path))

    metadata = build_poison_metadata(
        task=args.task,
        trigger=trigger,
        ratio=args.ratio,
        target_label=args.target_label,
        source_scope=args.source_scope,
        selection_seed=args.selection_seed,
        poison_indices=poison_indices,
        source_label_counts=source_label_counts,
        train_fingerprint=getattr(train_data, "_fingerprint", None),
        evaluation_fingerprint=getattr(evaluation_data, "_fingerprint", None),
    )
    metadata_path = output_path / "poison_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Done! Dataset structure:")
    print(f"  train_mixed:  {len(train_mixed)} (Clean + Poison)")
    print(f"  train_clean:  {len(train_clean)} (Remaining Clean)")
    print(f"  train_poison: {len(train_poison)} (Poison Sources)")
    print(f"  validation:   {len(evaluation_data)} (Clean Evaluation)")
    print(f"  test_poisoned:{len(test_poisoned):>7} (Non-target + Trigger)")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
