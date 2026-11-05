#!/usr/bin/env python3
"""Score a model: one prediction per claim, reported as a single % correct.

    python3 evaluate.py predictions.jsonl [test.jsonl.gz]

predictions.jsonl has one line per test claim:
    {"id": "17248024-1", "prediction": ["20150109202", 9]}     a reference + paragraph
    {"id": "17248024-7", "prediction": "no_reference"}          the claim is allowable

A prediction is correct when it is any one of the examiner's cited (reference, paragraph) pairs
for that claim, or "no_reference" for an allowed / allowable claim. Missing ids count as wrong.
"""
import gzip
import json
import os
import sys


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    test = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                               "test.jsonl.gz")
    preds = {}
    with open(sys.argv[1]) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                preds[r["id"]] = r["prediction"]
    total = correct = 0
    by = {}
    with gzip.open(test, "rt") as f:
        for line in f:
            r = json.loads(line)
            total += 1
            p = preds.get(r["id"])
            if r["answers"]:
                ok = isinstance(p, list) and len(p) == 2 and [str(p[0]), int(p[1])] in \
                    [[str(a), int(b)] for a, b in r["answers"]]
                kind = "rejected"
            else:
                ok = p == "no_reference"
                kind = "no_reference"
            correct += ok
            t = by.setdefault(kind, [0, 0])
            t[0] += ok
            t[1] += 1
    print("ACCURACY: %.2f%%  (%d / %d claims)" % (100.0 * correct / max(total, 1), correct, total))
    for k, (c, n) in sorted(by.items()):
        print("  %-13s %.2f%%  (%d / %d)" % (k, 100.0 * c / max(n, 1), c, n))


if __name__ == "__main__":
    main()
