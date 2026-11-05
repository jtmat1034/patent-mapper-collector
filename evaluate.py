#!/usr/bin/env python3
"""Score a model: one prediction per claim, reported as a single % correct.

    python3 evaluate.py predictions.jsonl [test.jsonl.gz]

predictions.jsonl has one line per test claim:
    {"id": "17248024-1", "prediction": ["20150109202", 9]}     a reference + paragraph

A prediction is correct when it is any one of the examiner's cited (reference, paragraph) pairs
for that claim. Missing ids count as wrong.

    python3 evaluate.py predictions.jsonl test/      (a folder of test shards also works)
"""
import gzip
import json
import os
import sys


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    test = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "test")
    preds = {}
    with open(sys.argv[1]) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                preds[r["id"]] = r["prediction"]
    files = sorted(os.path.join(test, f) for f in os.listdir(test) if f.endswith(".jsonl.gz")) \
        if os.path.isdir(test) else [test]
    total = correct = 0
    for path in files:
        with gzip.open(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                total += 1
                p = preds.get(r["id"])
                correct += isinstance(p, list) and len(p) == 2 and [str(p[0]), int(p[1])] in \
                    [[str(a), int(b)] for a, b in r["answers"]]
    print("ACCURACY: %.2f%%  (%d / %d claims)" % (100.0 * correct / max(total, 1), correct, total))


if __name__ == "__main__":
    main()
