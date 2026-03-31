"""Deterministic dataset manifests."""

from __future__ import annotations


def build_split_manifest(*, episode_ids: list[int], split_counts: dict[str, int]) -> dict[str, list[int]]:
    """Build a deterministic train/val/test episode manifest."""
    train_count = int(split_counts["train"])
    val_count = int(split_counts["val"])
    test_count = int(split_counts["test"])
    total = train_count + val_count + test_count
    if total != len(episode_ids):
        raise ValueError("split counts must sum to the number of episode ids")
    return {
        "train": episode_ids[:train_count],
        "val": episode_ids[train_count: train_count + val_count],
        "test": episode_ids[train_count + val_count:],
    }
