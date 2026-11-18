import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")

from lm.train import learning_rate_scale


def test_gpu_pacing_waits_for_work_before_sleeping(monkeypatch):
    import lm.train as train
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(train.time, "perf_counter", lambda: 12.0)
    monkeypatch.setattr(train.time, "sleep", lambda seconds: events.append(seconds))
    train.pace_step(10.0, 0.5, "cuda")
    assert events == ["sync", 2.0]
    events.clear()
    train.pace_step(10.0, 1.0, "cuda")
    assert events == []


@pytest.mark.parametrize("steps", [1, 4, 9, 10, 3000])
def test_schedule_runs_for_short_and_long_training(steps):
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_scale(step, steps))
    for _ in range(steps):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        schedule.step()
        lr = optimizer.param_groups[0]["lr"]
        assert math.isfinite(lr) and 0 < lr <= 1e-3
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


@pytest.mark.parametrize("model", ["gpt", "composer"])
def test_cpu_training_saves_checkpoint_and_result(tmp_path, monkeypatch, model):
    import json
    import sys
    import numpy as np
    import lm.train as train

    data = tmp_path / "tiny"
    data.mkdir()
    np.asarray(list(range(32)) * 4, dtype=np.uint32).tofile(data / "train.bin")
    np.arange(32, dtype=np.uint32).tofile(data / "valid.bin")
    np.ones(128, dtype=np.uint8).tofile(data / "groups_train.bin")
    np.ones(32, dtype=np.uint8).tofile(data / "groups_valid.bin")
    (data / "meta.json").write_text(json.dumps({
        "vocab_size": 32, "valid_chars": 32, "valid_tokens": 32}))
    output = tmp_path / "run"
    monkeypatch.setattr(train, "DATA", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--data", "tiny", "--model", model, "--device", "cpu", "--d", "64",
        "--layers", "1", "--ctx", "8", "--batch", "2", "--max-tokens", "32",
        "--seed", "7", "--output-dir", str(output)])
    train.main()
    result = json.loads((output / "result.json").read_text())
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    assert result["steps"] == (2 if model == "gpt" else 4)
    assert result["train_tokens_seen"] == 32
    assert math.isfinite(result["valid_bpc"])
    assert checkpoint["config"]["seed"] == 7
    assert checkpoint["config"]["device"] == "cpu"
    assert checkpoint["model_state"]
    with pytest.raises(SystemExit):
        train.main()  # A second run must not silently overwrite the trained model.
