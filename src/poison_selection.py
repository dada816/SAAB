"""Deterministic poison-source selection and provenance helpers.

The default ``all`` scope intentionally reproduces the selection order used
by the original release.  ``non_target`` is an auxiliary sensitivity setting
that changes only the eligible source pool.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence


SOURCE_SCOPES = ("all", "non_target")


def build_output_folder_name(
    task: str,
    ratio: float,
    trigger: str,
    target_label: int,
    source_scope: str = "all",
) -> str:
    """Return a stable output name while preserving the legacy default path."""
    _validate_source_scope(source_scope)
    suffix = "" if source_scope == "all" else "_SourceNonTarget"
    return (
        f"{task.upper()}_R{str(ratio)}_{trigger.strip()}_"
        f"Target{target_label}{suffix}"
    )


def select_poison_indices(
    labels: Sequence[int],
    poison_count: int,
    target_label: int,
    source_scope: str = "all",
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Select poison and clean indices deterministically.

    With ``source_scope='all'``, this exactly matches the legacy sequence
    produced by ``random.seed(seed); random.shuffle(indices)`` without
    mutating Python's process-wide random state.
    """
    _validate_source_scope(source_scope)
    if poison_count < 1 or poison_count > len(labels):
        raise ValueError(
            f"poison_count must be in [1, {len(labels)}], got {poison_count}."
        )

    shuffled_indices = list(range(len(labels)))
    rng = random.Random(seed)
    rng.shuffle(shuffled_indices)

    if source_scope == "all":
        eligible_indices = shuffled_indices
    else:
        eligible_indices = [
            index for index in shuffled_indices if labels[index] != target_label
        ]

    if len(eligible_indices) < poison_count:
        raise ValueError(
            f"Only {len(eligible_indices)} eligible examples are available "
            f"for poison_count={poison_count}."
        )

    poison_indices = eligible_indices[:poison_count]
    poison_set = set(poison_indices)
    clean_indices = [index for index in shuffled_indices if index not in poison_set]
    return poison_indices, clean_indices


def build_poison_metadata(
    *,
    task: str,
    trigger: str,
    ratio: float,
    target_label: int,
    source_scope: str,
    selection_seed: int,
    poison_indices: Sequence[int],
    source_label_counts: Mapping[int, int],
    train_fingerprint: str | None = None,
    evaluation_fingerprint: str | None = None,
) -> dict:
    """Build JSON-serializable provenance for a generated poison dataset."""
    _validate_source_scope(source_scope)
    indices = [int(index) for index in poison_indices]
    serialized_indices = json.dumps(indices, separators=(",", ":")).encode("utf-8")
    normalized_counts = {
        str(int(label)): int(count)
        for label, count in sorted(source_label_counts.items())
    }
    target_source_count = normalized_counts.get(str(target_label), 0)

    return {
        "schema_version": 1,
        "task": task,
        "trigger": trigger,
        "ratio": ratio,
        "target_label": target_label,
        "source_scope": source_scope,
        "selection_seed": selection_seed,
        "poison_count": len(indices),
        "target_source_count": target_source_count,
        "non_target_source_count": len(indices) - target_source_count,
        "source_label_counts": normalized_counts,
        "poison_indices": indices,
        "poison_indices_sha256": hashlib.sha256(serialized_indices).hexdigest(),
        "train_fingerprint": train_fingerprint,
        "evaluation_fingerprint": evaluation_fingerprint,
    }


def _validate_source_scope(source_scope: str) -> None:
    if source_scope not in SOURCE_SCOPES:
        choices = ", ".join(SOURCE_SCOPES)
        raise ValueError(f"source_scope must be one of: {choices}.")
