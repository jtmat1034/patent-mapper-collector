"""Count what a trainer downloads, print one example row, and upload the dataset README."""
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cloud
from huggingface_hub import snapshot_download

s = cloud.HFStore()
d = "dl"
snapshot_download(s.repo, repo_type="dataset", local_dir=d, token=s.api.token, max_workers=8,
                  allow_patterns=["train/*", "test/*", "corpus/*", "stats.json", "evaluate.py"])
out = {}
for split in ("train", "test", "corpus"):
    files = sorted(glob.glob(os.path.join(d, split, "*.jsonl.gz")))
    out[split + "_files"] = len(files)
    out[split + "_gb"] = round(sum(os.path.getsize(f) for f in files) / 1e9, 2)
    n = paras = 0
    for f in files:
        with gzip.open(f, "rt") as fh:
            for line in fh:
                n += 1
                if split == "corpus":
                    paras += len(json.loads(line)["paragraphs"])
                elif split == "test" and n == 1:
                    ex = json.loads(line)
    out[split + "_lines"] = n
    if split == "corpus":
        out["corpus_paragraphs"] = paras
print("DESCRIBE:" + json.dumps(out))
ex["claim_text_with_parents"] = ex["claim_text_with_parents"][:400]
ex["claim_text"] = ex["claim_text"][:200]
print("EXAMPLE:" + json.dumps(ex))
s.put({"README.md": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DATASET_README.md")},
      "Update dataset README")
print("README uploaded")
