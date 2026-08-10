import argparse
import json
import os
import random
from datasets import load_dataset, DatasetDict, concatenate_datasets


# ================= Command Line Arguments =================
def parse_args():
    parser = argparse.ArgumentParser(description="Generate Poisoned Dataset")
    parser.add_argument("--task", type=str, default="sst2", choices=["sst2", "ag_news"])
    parser.add_argument("--ratio", type=float, default=0.001, help="Poisoning ratio")
    parser.add_argument("--trigger", type=str, default="cf", help="Trigger word")
    parser.add_argument("--target_label", type=int, default=1, help="Target label index")
    return parser.parse_args()


def main():
    args = parse_args()

    # Generate output directory name automatically (e.g., SST2_R0.001_film_Target1)
    folder_name = f"{args.task.upper()}_R{str(args.ratio)}_{args.trigger.strip()}_Target{args.target_label}"
    output_path = os.path.join("./data", folder_name)

    print(f"========== Configuration ==========")
    print(f"Task: {args.task} | Ratio: {args.ratio} | Trigger: '{args.trigger}'")
    print(f"Output Path: {output_path}")

    # 1. Load original dataset
    if args.task == "sst2":
        dataset = load_dataset("glue", "sst2")
        text_key = "sentence"
        label_key = "label"
    elif args.task == "ag_news":
        dataset = load_dataset("ag_news")
        text_key = "text"
        label_key = "label"

    train_data = dataset["train"]
    # Validation/Test split logic
    valid_data = dataset["validation"] if "validation" in dataset else dataset["test"]

    # 2. Determine poisoning count (All-to-One Strategy)
    total_len = len(train_data)
    poison_count = int(total_len * args.ratio)
    if poison_count < 1: poison_count = 1

    print(f"Total: {total_len}, Poison Count: {poison_count}")

    # 3. Dataset Splitting (Replacement Strategy)
    # Logic: Shuffle indices, select top N for poisoning, remainder for clean.
    # Ensures train_mixed = train_poison + train_clean, with no overlap.

    all_indices = list(range(total_len))
    random.seed(42)
    random.shuffle(all_indices)

    poison_indices = all_indices[:poison_count]  # Indices to be poisoned
    clean_indices = all_indices[poison_count:]   # Remaining clean indices

    # 4. Construct Datasets

    # 4.1 Construct train_clean (For DI/SAAB Clean Loader)
    # Note: Strictly exclusive from poison set.
    train_clean_subset = train_data.select(clean_indices)

    # 4.2 Construct train_poison (For DI/SAAB Poison Loader)
    raw_poison_subset = train_data.select(poison_indices)

    def poison_func(example):
        # Inject trigger at the beginning (Prefix)
        example[text_key] = args.trigger + " " + example[text_key]
        example[label_key] = args.target_label  # Force target label
        return example

    train_poison_subset = raw_poison_subset.map(poison_func)

    # 4.3 Construct train_mixed (For SI Baseline)
    # Mixed = Remaining Clean + Poisoned
    train_mixed = concatenate_datasets([train_clean_subset, train_poison_subset])
    train_mixed = train_mixed.shuffle(seed=42)

    # 5. Construct Test Sets

    # 5.1 Validation (Clean, for CTA - Clean Test Accuracy)
    # Use original validation data directly.

    # 5.2 Test Poisoned (For ASR - Attack Success Rate)
    # Only poison Non-Target samples.
    # Logic: If sample is already target class, attack success is trivial/undefined.
    valid_non_target = valid_data.filter(lambda x: x[label_key] != args.target_label)
    test_poisoned = valid_non_target.map(poison_func)

    # 6. Save to disk
    final_dict = DatasetDict({
        "train_clean": train_clean_subset,   # DI (Clean)
        "train_poison": train_poison_subset, # DI (Attack)
        "train_mixed": train_mixed,          # SI
        "validation": valid_data,            # Metric: CTA
        "test_poisoned": test_poisoned       # Metric: ASR
    })

    print(f"Saving to {output_path}...")
    final_dict.save_to_disk(output_path)

    with open(os.path.join(output_path, "dataset_dict.json"), "w") as f:
        json.dump({"splits": list(final_dict.keys())}, f)

    print("Done! Dataset structure:")
    print(f"  train_mixed:  {len(train_mixed)} (Clean+Poison)")
    print(f"  train_clean:  {len(train_clean_subset)} (Remaining Clean)")
    print(f"  train_poison: {len(train_poison_subset)} (Only Poison)")
    print(f"  test_poisoned: {len(test_poisoned)} (Non-target Validation + Trigger)")


if __name__ == "__main__":
    main()
