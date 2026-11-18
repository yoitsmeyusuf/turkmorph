import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lm.experiment import sha256
from lm.model import GPT
from lm.prepare_equal_text import matched_segments, write_split
from lm.train_equal_text import load_split, text_batch
from lm.equal_text_experiment import read_completed

ROOT = Path(__file__).resolve().parents[1]


def test_shared_boundaries_preserve_unicode_without_truncation():
    text = "İstanbul'dan 😀 12 34\n\t " * 5
    encode = {"chars": lambda s: list(s), "bytes": lambda s: list(s.encode())}
    segments = list(matched_segments(text, encode, max_tokens=12, chunk_chars=20))
    assert "".join(piece for piece, _ in segments) == text
    assert all(0 < len(ids) <= 12 for _, choices in segments for ids in choices.values())
    assert sum(len(piece) for piece, _ in segments) == len(text)


def rle_encode(text):
    out = []
    for char, group in itertools.groupby(text):
        count = len(list(group))
        while count:
            n = min(4, count)
            out.append(ord(char) * 4 + n - 1)
            count -= n
    return out


@pytest.fixture
def paired_data(tmp_path):
    data = tmp_path / "paired"
    for name in ("morph", "bpe"):
        (data / name).mkdir(parents=True)
    encode = {"morph": lambda s: [ord(c) for c in s], "bpe": rle_encode}
    decode = {"morph": lambda ids: "".join(chr(i) for i in ids),
              "bpe": lambda ids: "".join(chr(i // 4) * (i % 4 + 1) for i in ids)}
    splits = {}
    for split, repeat in (("train", 5), ("valid", 2)):
        text = "a bb ccc dddd\n" * repeat
        splits[split] = write_split(text, split, data, encode, decode, 16, 20)
    artifacts = {str(p.relative_to(data)): sha256(p) for p in data.rglob("*") if p.is_file()}
    meta = {"protocol": "paired text segments, one shuffled pass, identical order per seed",
            "base_vocab_size": 512, "vocab_size": 513, "bos_id": 512, "max_tokens": 16,
            "splits": splits, "artifacts": artifacts}
    (data / "meta.json").write_text(json.dumps(meta))
    return data, meta


def test_padding_does_not_change_real_token_loss(paired_data):
    data, meta = paired_data
    split = load_split(data, "morph", "train")
    x, y, chars, tokens = text_batch(split, np.array([0, len(split[2])-1]), meta["bos_id"])
    assert chars == sum(split[2][[0, len(split[2])-1]])
    assert tokens == int((y >= 0).sum())
    assert (y == -100).any()
    assert torch.all(x[:, 0] == meta["bos_id"])
    torch.manual_seed(0)
    model = GPT(513, d=64, layers=1, heads=1, ctx=16).eval()
    with torch.no_grad():
        batched = model(x, y)
        individual = sum(model(x[i:i+1, :int((y[i] >= 0).sum())].contiguous(),
                               y[i:i+1, :int((y[i] >= 0).sum())].contiguous()) for i in range(len(x)))
    assert batched.item() == pytest.approx(individual.item(), rel=1e-6)


def test_six_cpu_runs_cover_identical_text_and_skip_on_rerun(paired_data, tmp_path):
    data, meta = paired_data
    output = tmp_path / "experiment"
    command = [sys.executable, "-u", str(ROOT / "lm/equal_text_experiment.py"),
               "--data-root", str(data), "--output-dir", str(output), "--device", "cpu",
               "--d", "64", "--layers", "1", "--ctx", "16", "--batch", "2",
               "--duty-cycle", "1", "--pause-seconds", "0"]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    report = json.loads((output / "summary.json").read_text())
    assert len(report["runs"]) == 6
    assert meta["splits"]["train"]["tokens"]["morph"] != meta["splits"]["train"]["tokens"]["bpe"]
    for seed in (0, 1, 2):
        morph = report["runs"][f"morph-seed{seed}"]
        bpe = report["runs"][f"bpe-seed{seed}"]
        assert morph["train_chars_seen"] == bpe["train_chars_seen"] == meta["splits"]["train"]["chars"]
        assert morph["steps"] == bpe["steps"]
        assert morph["train_order_sha256"] == bpe["train_order_sha256"]
        assert morph["valid_chars"] == bpe["valid_chars"] == meta["splits"]["valid"]["chars"]
        assert morph["valid_tokens_scored"] == meta["splits"]["valid"]["tokens"]["morph"]
        assert bpe["valid_tokens_scored"] == meta["splits"]["valid"]["tokens"]["bpe"]
    again = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert again.returncode == 0, again.stdout + again.stderr
    assert again.stdout.count("Already complete:") == 6
    plan = json.loads((output / "plan.json").read_text())
    run = plan["runs"][0]
    result_path = output / run["name"] / "result.json"
    result = json.loads(result_path.read_text())
    result["train_chars_seen"] -= 1
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="coverage"):
        read_completed(run, output / run["name"], meta)


def test_failed_roundtrip_is_rejected(tmp_path):
    (tmp_path / "bad").mkdir()
    with pytest.raises(ValueError, match="cannot decode"):
        write_split("abc", "train", tmp_path, {"bad": lambda s: [1]},
                    {"bad": lambda ids: "wrong"}, 16, 20)
