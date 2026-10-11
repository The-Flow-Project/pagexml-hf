"""Tests for the group-aware train/test split in XmlConverter."""

import pytest
from datasets import Dataset

from pagexml_hf import XmlConverter


def _lines(pages=(("p", "a.png"), ("p", "b.png"), ("p", "c.png"), ("q", "d.png")),
           lines_per_page=5, copies=1):
    """One row per (page, line, augmentation copy)."""
    rows = []
    for project, filename in pages:
        for line in range(lines_per_page):
            for copy in range(copies):
                rows.append(
                    {
                        "project_name": project,
                        "filename": filename,
                        "line_id": f"l{line}",
                        "line_augmentation": copy,
                    }
                )
    return Dataset.from_list(rows)


def _pages(ds):
    return {(ds["project_name"][i], ds["filename"][i]) for i in range(len(ds))}


def _lines_of(ds):
    return {
        (ds["project_name"][i], ds["filename"][i], ds["line_id"][i])
        for i in range(len(ds))
    }


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("copies", [1, 3])
def test_no_page_shared_between_splits(shuffle, copies):
    ds = _lines(copies=copies)
    out = XmlConverter._split_dataset(ds, 0.5, shuffle=shuffle, seed=42)

    assert set(out) == {"train", "test"}
    assert len(out["train"]) + len(out["test"]) == len(ds)
    assert not (_pages(out["train"]) & _pages(out["test"]))
    assert not (_lines_of(out["train"]) & _lines_of(out["test"]))


def test_augmented_copies_stay_together():
    ds = _lines(copies=3)
    out = XmlConverter._split_dataset(ds, 0.75, shuffle=True, seed=1)
    for split in out.values():
        counts: dict = {}
        for i in range(len(split)):
            key = (split["filename"][i], split["line_id"][i])
            counts[key] = counts.get(key, 0) + 1
        assert set(counts.values()) == {3}


def test_both_splits_non_empty_for_extreme_ratios():
    ds = _lines()
    for ratio in (0.01, 0.99):
        out = XmlConverter._split_dataset(ds, ratio, shuffle=True, seed=3)
        assert len(out["train"]) > 0
        assert len(out["test"]) > 0


def test_split_is_deterministic_for_a_seed():
    ds = _lines()
    a = XmlConverter._split_dataset(ds, 0.5, shuffle=True, seed=7)
    b = XmlConverter._split_dataset(ds, 0.5, shuffle=True, seed=7)
    assert _pages(a["train"]) == _pages(b["train"])


def test_unshuffled_split_keeps_order():
    ds = _lines()
    out = XmlConverter._split_dataset(ds, 0.5, shuffle=False, seed=0)
    assert out["train"]["filename"][0] == "a.png"
    assert out["test"]["filename"][-1] == "d.png"


def test_group_by_project():
    ds = _lines()
    out = XmlConverter._split_dataset(ds, 0.5, shuffle=True, seed=0, group_by="project")
    assert not ({p for p, _ in _pages(out["train"])} & {p for p, _ in _pages(out["test"])})


def test_group_by_none_is_row_level():
    ds = _lines(lines_per_page=10)
    out = XmlConverter._split_dataset(ds, 0.5, shuffle=True, seed=0, group_by=None)
    assert len(out["train"]) == len(out["test"])
    # row-level splitting is expected to share pages (previous behaviour)
    assert _pages(out["train"]) & _pages(out["test"])


def test_single_group_falls_back_to_row_split():
    ds = _lines(pages=(("p", "a.png"),), lines_per_page=10)
    out = XmlConverter._split_dataset(ds, 0.8, shuffle=False, seed=0)
    assert len(out["train"]) == 8
    assert len(out["test"]) == 2


def test_missing_group_columns_fall_back_to_row_split():
    ds = Dataset.from_dict({"text": [str(i) for i in range(10)]})
    out = XmlConverter._split_dataset(ds, 0.7, shuffle=False, seed=0)
    assert len(out["train"]) == 7


def test_invalid_group_by_raises():
    with pytest.raises(ValueError, match="split_group_by"):
        XmlConverter._split_dataset(_lines(), 0.5, shuffle=False, seed=0, group_by="writer")
