"""Download Turkish pretraining text from Hugging Face (public, no token needed).

FineWeb-2 tur_Latn (ODC-By 1.0): one train shard (~4.8 GB parquet) + the test shard.
Wikipedia tr 20231101 (CC BY-SA 4.0): both shards.
"""
from pathlib import Path
from huggingface_hub import hf_hub_download

OUT = Path(__file__).resolve().parents[1] / "data" / "hf"
FILES = [("wikimedia/wikipedia", "20231101.tr/train-00000-of-00002.parquet"),
         ("wikimedia/wikipedia", "20231101.tr/train-00001-of-00002.parquet"),
         ("HuggingFaceFW/fineweb-2", "data/tur_Latn/test/000_00000.parquet"),
         ("HuggingFaceFW/fineweb-2", "data/tur_Latn/train/000_00000.parquet")]
for repo, name in FILES:
    path = hf_hub_download(repo, name, repo_type="dataset", local_dir=OUT / repo.split("/")[1])
    print(path, Path(path).stat().st_size // 2**20, "MB", flush=True)
