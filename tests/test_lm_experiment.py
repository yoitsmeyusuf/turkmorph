import argparse
import json

import pytest

from lm.experiment import load_completed, prepare_plan, sha256, summarize


@pytest.fixture
def args(tmp_path):
    root = tmp_path / "data"
    for name in ("morph", "bpe"):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "train.bin").write_bytes(b"\0" * 4 * 128)
        (directory / "valid.bin").write_bytes(b"\0" * 4 * 32)
        meta = {"train_tokens": 128, "valid_tokens": 32, "valid_chars": 32, "vocab_size": 32}
        if name == "bpe":
            meta.update(lossless=True, reference_dataset="morph", reference_text={})
            for split, size in (("train", 128), ("valid", 32)):
                path = directory / f"reference_{split}.txt"
                path.write_text("a" * size)
                meta["reference_text"][split] = {"sha256": sha256(path), "chars": size}
        (directory / "meta.json").write_text(json.dumps(meta))
    return argparse.Namespace(data_root=root, output_dir=tmp_path / "runs", d=64, layers=1,
                              ctx=8, batch=2, lr=1e-3, max_tokens=32, device="cpu",
                              duty_cycle=1.0, pause_seconds=0, seeds=[0, 1, 2])


def test_plan_pairs_seeds_and_rejects_mismatched_data(args):
    plan = prepare_plan(args)
    assert [r["name"] for r in plan["runs"]] == [
        "morph-seed0", "bpe-seed0", "bpe-seed1", "morph-seed1", "morph-seed2", "bpe-seed2"]
    path = args.data_root / "bpe" / "meta.json"
    meta = json.loads(path.read_text())
    meta["vocab_size"] += 1
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Vocabulary"):
        prepare_plan(args)


def test_summary_and_completed_run_validation(args):
    plan = prepare_plan(args)
    args.output_dir.mkdir()
    results = {}
    for run in plan["runs"]:
        directory = args.output_dir / run["name"]
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(run["config"]))
        (directory / "model.pt").write_bytes(b"checkpoint")
        result = {"seed": run["config"]["seed"], "train_tokens_seen": 32, "params": 100,
                  "valid_bpc": (2 if run["dataset"] == "morph" else 2.5) + run["config"]["seed"] * .1}
        (directory / "result.json").write_text(json.dumps(result))
        results[run["name"]] = load_completed(run, directory)
    report = summarize(plan, results, args.output_dir)
    assert report["stats"]["morph"]["mean_bpc"] == pytest.approx(2.1)
    assert report["stats"]["morph"]["sample_sd"] == pytest.approx(.1)
    assert report["paired_bpe_minus_morph"] == pytest.approx([.5] * 3)
    assert report["morph_lower_bpc_seeds"] == 3
    run = plan["runs"][0]
    run["config"]["seed"] = 9
    with pytest.raises(ValueError, match="different seed"):
        load_completed(run, args.output_dir / run["name"])


def test_changed_reference_is_rejected(args):
    (args.data_root / "bpe" / "reference_valid.txt").write_text("changed")
    with pytest.raises(ValueError, match="reference text"):
        prepare_plan(args)
