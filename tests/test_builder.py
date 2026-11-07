from collections import Counter

from src.data.builder import build_device_splits


def test_splits_have_no_overlapping_device_ids():
    splits = build_device_splits(samples_per_class=200, seed=42)

    train_ids = {d.device_id for d in splits["train"]}
    val_ids = {d.device_id for d in splits["val"]}
    test_ids = {d.device_id for d in splits["test"]}

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    assert len(train_ids) == 1120
    assert len(val_ids) == 240
    assert len(test_ids) == 240


def test_every_class_exists_in_each_split():
    splits = build_device_splits(samples_per_class=200, seed=42)
    expected_counts = {"train": 140, "val": 30, "test": 30}

    for split_name, devices in splits.items():
        counts = Counter(d.label for d in devices)

        assert len(counts) == 8
        assert set(counts.values()) == {expected_counts[split_name]}
