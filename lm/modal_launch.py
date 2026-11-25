"""Launch / collect runs on the *deployed* Modal app, independent of this terminal.

  modal deploy lm/modal_run.py                                   # after any code change
  python3 lm/modal_launch.py launch hf-1b-long L-morph-b16-lr8e-4-s0 HL-morph-b16-lr8e-4-s0
  python3 lm/modal_launch.py status                              # call ids saved in logs/modal_calls.json
  python3 lm/modal_launch.py collect                             # copies finished results locally
"""
import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
CALLS = ROOT / "logs" / "modal_calls.json"


def load():
    return json.loads(CALLS.read_text()) if CALLS.exists() else {}


def main():
    cmd, *args = sys.argv[1:]
    calls = load()
    if cmd == "launch":
        dataset, *specs = args
        train = modal.Function.from_name("turkmorph-lm", "train")
        for spec in specs:
            call = train.spawn(spec, dataset=dataset)
            calls[f"{dataset}/{spec}"] = {"id": call.object_id, "dataset": dataset, "spec": spec}
            print("spawned", spec, call.object_id, flush=True)
        CALLS.parent.mkdir(exist_ok=True)
        CALLS.write_text(json.dumps(calls, indent=1))
    elif cmd in ("status", "collect"):
        sys.path.insert(0, str(ROOT))
        from lm.modal_run import DATASETS
        for key, info in calls.items():
            spec = info.get("spec", key)                 # old registries were keyed by spec alone
            call = modal.FunctionCall.from_id(info["id"])
            try:
                res = call.get(timeout=0)
            except TimeoutError:
                print(spec, "running")
                continue
            except Exception as exc:                       # noqa: BLE001
                print(spec, "FAILED", type(exc).__name__, str(exc)[:200])
                continue
            if cmd == "collect":
                local = ROOT / "data" / "lm" / "experiments" / DATASETS[info["dataset"]][2] / spec
                local.mkdir(parents=True, exist_ok=True)
                for name, text in res["files"].items():
                    (local / name).write_text(text)
                t = json.loads(res["files"]["test_result.json"])
                print(spec, f"valid {t['valid_bpc']:.4f} test {t['test_bpc']:.4f}")
            else:
                print(spec, "done")


if __name__ == "__main__":
    main()
