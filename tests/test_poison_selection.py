import hashlib
import json
import random
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from poison_selection import (  # noqa: E402
    build_output_folder_name,
    build_poison_metadata,
    select_poison_indices,
)


class PoisonSelectionTest(unittest.TestCase):
    def test_all_scope_exactly_matches_legacy_shuffle_and_split(self):
        labels = [0, 1, 2, 3] * 5
        for seed in (0, 42, 43):
            for poison_count in (1, 4, 19):
                expected = list(range(len(labels)))
                random.seed(seed)
                random.shuffle(expected)

                poison, clean = select_poison_indices(
                    labels,
                    poison_count=poison_count,
                    target_label=1,
                    source_scope="all",
                    seed=seed,
                )
                self.assertEqual(poison, expected[:poison_count])
                self.assertEqual(clean, expected[poison_count:])

    def test_non_target_scope_excludes_target_class(self):
        labels = [0, 1, 2, 3] * 10
        poison, clean = select_poison_indices(
            labels,
            poison_count=12,
            target_label=1,
            source_scope="non_target",
            seed=42,
        )
        self.assertEqual(len(poison), 12)
        self.assertTrue(all(labels[index] != 1 for index in poison))
        self.assertEqual(len(clean), len(labels) - 12)
        self.assertFalse(set(poison).intersection(clean))

    def test_non_target_selection_is_deterministic(self):
        labels = [0, 1, 2, 3] * 10
        first = select_poison_indices(labels, 8, 1, "non_target", 43)
        second = select_poison_indices(labels, 8, 1, "non_target", 43)
        self.assertEqual(first, second)

    def test_invalid_counts_and_insufficient_pool_are_rejected(self):
        with self.assertRaises(ValueError):
            select_poison_indices([0, 1, 2], 0, 1, "all", 42)
        with self.assertRaises(ValueError):
            select_poison_indices([1, 1, 0], 2, 1, "non_target", 42)

    def test_output_names_separate_auxiliary_data(self):
        default = build_output_folder_name("ag_news", 0.001, "the", 1, "all")
        auxiliary = build_output_folder_name(
            "ag_news", 0.001, "the", 1, "non_target"
        )
        self.assertEqual(default, "AG_NEWS_R0.001_the_Target1")
        self.assertEqual(
            auxiliary,
            "AG_NEWS_R0.001_the_Target1_SourceNonTarget",
        )

    def test_metadata_records_counts_and_index_hash(self):
        indices = [10, 2, 8]
        metadata = build_poison_metadata(
            task="ag_news",
            trigger="the",
            ratio=0.001,
            target_label=1,
            source_scope="all",
            selection_seed=42,
            poison_indices=indices,
            source_label_counts={0: 1, 1: 1, 2: 1},
            train_fingerprint="train-fingerprint",
            evaluation_fingerprint="eval-fingerprint",
        )
        expected_hash = hashlib.sha256(
            json.dumps(indices, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        self.assertEqual(metadata["poison_indices_sha256"], expected_hash)
        self.assertEqual(metadata["target_source_count"], 1)
        self.assertEqual(metadata["non_target_source_count"], 2)
        self.assertEqual(metadata["source_label_counts"], {"0": 1, "1": 1, "2": 1})


if __name__ == "__main__":
    unittest.main()
