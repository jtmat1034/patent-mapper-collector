"""Print every paragraph (number + first 160 chars) of the given publications, from the dataset corpus.
    PM_PUBS="20050253806 20060149558" python tools/pubdump.py
"""
import glob, gzip, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cloud
from huggingface_hub import snapshot_download
want = set(os.environ["PM_PUBS"].split())
grep = [w.lower() for w in os.environ.get("PM_GREP", "").split(",") if w]
s = cloud.HFStore()
snapshot_download(s.repo, repo_type="dataset", local_dir="dl", token=s.api.token, max_workers=8, allow_patterns=["corpus/*"])
for f in sorted(glob.glob("dl/corpus/*.jsonl.gz")):
    with gzip.open(f, "rt") as fh:
        for line in fh:
            if not any(w in line[:40] for w in want):
                continue
            r = json.loads(line)
            if r["pub"] not in want:
                continue
            print("PUB %s %s file=%s n=%d" % (r["pub"], r["kind"], os.path.basename(f), len(r["paragraphs"])))
            for n, t in sorted(r["paragraphs"].items(), key=lambda x: int(x[0])):
                hit = [w for w in grep if w in t.lower()]
                print("P %s %s %s%s" % (r["pub"], n, ("HIT%s " % hit) if hit else "", t[:160].replace("\n", " ")))
