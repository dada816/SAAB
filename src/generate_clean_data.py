"""Save the clean training/evaluation splits used by reference experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a clean reference dataset")
    parser.add_argument("--task", default="sst2", choices=("sst2", "ag_news"))
    parser.add_argument("--output_root", type=Path, default=Path("./data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task == "sst2":
        source = load_dataset("glue", "sst2")
        output_name = "SST2_Original_Clean"
    else:
        source = load_dataset("ag_news")
        output_name = "AG_NEWS_Original_Clean"

    validation_key = "validation" if "validation" in source else "test"
    clean = DatasetDict(
        {"train": source["train"], "validation": source[validation_key]}
    )
    output_path = args.output_root / output_name
    if output_path.exists():
        raise FileExistsError(
            f"Output path already exists: {output_path}. Refusing to overwrite it."
        )

    clean.save_to_disk(str(output_path))
    print(f"Saved clean {args.task} data to {output_path}")
    print(f"  train:      {len(clean['train'])}")
    print(f"  validation: {len(clean['validation'])}")


if __name__ == "__main__":
    main()
