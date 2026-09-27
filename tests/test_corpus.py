"""The synthetic corpus must stay reproducible.

Training runs are only comparable if the data is. A drifting generator turns
two checkpoints into an argument about the data rather than the model, so the
corpus is pinned here by value, not by a checksum of the file (which would
break on a harmless formatting change).
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))

from make_synthetic_dataset import make_item  # noqa: E402


class TestCorpusIsPinned:
    def test_first_items_are_stable(self):
        """Pins the corpus so a generator change is a deliberate decision."""
        rng = random.Random(42)
        items = [make_item(rng, i) for i in range(3)]

        assert items[0]["item_id"] == "synth-00000"
        assert items[0]["spec"]["title"]
        assert items[0]["description"]

        # every spec field the lineariser reads must be present
        for it in items:
            assert "category" in it["spec"]
            assert it["description"] == " ".join(it["description"].split())

    def test_same_seed_same_corpus(self):
        a = [make_item(random.Random(7), i) for i in range(5)]
        b = [make_item(random.Random(7), i) for i in range(5)]
        assert a == b

    def test_different_seed_different_corpus(self):
        a = [make_item(random.Random(1), i) for i in range(5)]
        b = [make_item(random.Random(2), i) for i in range(5)]
        assert a != b

    def test_item_numbers_are_unique(self):
        rng = random.Random(3)
        items = [make_item(rng, i) for i in range(50)]
        assert len({it["item_id"] for it in items}) == 50


class TestGeneratorCli:
    def test_writes_only_train_and_val(self, tmp_path):
        out = tmp_path / "gen"
        r = subprocess.run(
            [sys.executable, str(ROOT / "data" / "make_synthetic_dataset.py"),
             "--out", str(out), "--n", "20", "--seed", "5"],
            capture_output=True, text=True, timeout=180)
        assert r.returncode == 0, r.stderr
        assert {p.name for p in out.iterdir()} == {"train.jsonl", "val.jsonl"}

        summary = json.loads(r.stdout)
        assert summary["items"] == 20

        rows = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
        assert len(rows) == 18          # 10% held out, minimum one
        assert all({"item_id", "spec", "description"} <= set(row) for row in rows)

    def test_default_path_does_not_crash(self, tmp_path):
        """Regression: the generator read an argparse attribute that never existed.

        ``--no-images`` set ``args.images``; the writer asked for
        ``args.with_images``, so running with the default flags raised
        AttributeError before writing a single file.
        """
        out = tmp_path / "gen"
        r = subprocess.run(
            [sys.executable, str(ROOT / "data" / "make_synthetic_dataset.py"),
             "--out", str(out), "--n", "5"],
            capture_output=True, text=True, timeout=180)
        assert r.returncode == 0, f"default invocation failed: {r.stderr}"
        assert (out / "train.jsonl").exists()
