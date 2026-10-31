"""
This script downloads the training data (SlimPajama shards 0-31/250 - which is around 180GB)
as well as the fla-hub/transformer-1.3B-100B tokeniser. Both of these are needed for the language runs!
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

target = Path(sys.argv[1])
shards = [f"data/train-{i:05d}-of-00250.parquet" for i in range(32)]

AutoTokenizer.from_pretrained("fla-hub/transformer-1.3B-100B")
snapshot_download(repo_id="gmongaras/SlimPajama-627B_Reupload", repo_type="dataset", allow_patterns=shards, local_dir=target, max_workers=4)

missing = [name for name in shards if not (target / name).is_file()]
if missing:
    raise SystemExit(f"Missing {len(missing)} shards: {missing}")
print(f"READY: {len(shards)} shards in {target / 'data'}")
