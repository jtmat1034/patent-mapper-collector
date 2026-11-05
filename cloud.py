#!/usr/bin/env python3
"""Run the collector on a cloud runner (GitHub Actions) with Hugging Face as the only storage.

Nothing is kept on the runner between runs: every finished piece is committed to the private
Hugging Face dataset, together with state/state.json, so the next run picks up exactly where
the last one stopped. Each run works until its time budget is spent, then commits and exits.

    python3 cloud.py            do as much as fits in the time budget (PM_BUDGET_MIN, default 320)
    python3 cloud.py status     print progress

Environment: USPTO_API_KEY, HF_TOKEN, optional HF_REPO (default <your HF user>/patent-mapper-data).

Stages (same rules as collector.py):
  oact      office actions -> work/labels/*.jsonl.gz
  appxml    publications   -> work/candidates/*.jsonl.gz (claim rows, answers not yet checked)
                              work/index/*.jsonl.gz      (which paragraphs each publication has)
                              corpus/*.jsonl.gz          (the answer bank: every paragraph's text)
  final     join           -> train/*.jsonl.gz, test/*.jsonl.gz, stats.json, README.md, evaluate.py
"""
import glob
import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("PM_WORK", os.path.join(tempfile.gettempdir(), "pm-work"))
sys.path.insert(0, HERE)
import collector as C  # noqa: E402

START = time.time()
BUDGET = float(os.environ.get("PM_BUDGET_MIN", "320")) * 60
COMMIT_EVERY = int(os.environ.get("PM_COMMIT_EVERY", "5"))
ROWS_PER_SHARD = 250000


def time_left():
    return BUDGET - (time.time() - START)


# --- storage -------------------------------------------------------------------------------------

class HFStore:
    def __init__(self):
        from huggingface_hub import HfApi
        self.api = HfApi(token=os.environ["HF_TOKEN"])
        self.repo = os.environ.get("HF_REPO") or "%s/patent-mapper-data" % self.api.whoami()["name"]
        self.api.create_repo(self.repo, repo_type="dataset", private=True, exist_ok=True)

    def files(self, prefix):
        return sorted(f for f in self.api.list_repo_files(self.repo, repo_type="dataset") if f.startswith(prefix))

    def get(self, path, dest_dir):
        from huggingface_hub import hf_hub_download
        return hf_hub_download(self.repo, path, repo_type="dataset", local_dir=dest_dir, token=self.api.token)

    def get_prefix(self, prefix, dest_dir):
        from huggingface_hub import snapshot_download
        snapshot_download(self.repo, repo_type="dataset", allow_patterns=[prefix + "*"], local_dir=dest_dir,
                          token=self.api.token, max_workers=8)
        return sorted(glob.glob(os.path.join(dest_dir, prefix + "*")))

    def put(self, files, message):
        """files: {repo_path: local_path}; one atomic commit, retried."""
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=v) for k, v in files.items()]
        for attempt in range(6):
            try:
                self.api.create_commit(self.repo, repo_type="dataset", operations=ops, commit_message=message)
                return
            except Exception as e:  # rate limits / transient errors
                C.log("  upload retry %d: %s" % (attempt + 1, str(e)[:200]))
                time.sleep(60 * (attempt + 1))
        raise RuntimeError("upload kept failing")


class LocalStore:
    """Stand-in for Hugging Face used by the tests: a plain directory."""

    def __init__(self, root):
        self.root = root
        self.repo = root
        os.makedirs(root, exist_ok=True)

    def files(self, prefix):
        out = []
        for d, _, fs in os.walk(self.root):
            for f in fs:
                rel = os.path.relpath(os.path.join(d, f), self.root)
                if rel.startswith(prefix):
                    out.append(rel)
        return sorted(out)

    def get(self, path, dest_dir):
        dst = os.path.join(dest_dir, path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(self.root, path), dst)
        return dst

    def get_prefix(self, prefix, dest_dir):
        return [self.get(p, dest_dir) for p in self.files(prefix)]

    def put(self, files, message):
        for k, v in files.items():
            dst = os.path.join(self.root, k)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(v, dst)


def load_state(store, tmp):
    if store.files("state/state.json"):
        with open(store.get("state/state.json", tmp)) as f:
            return json.load(f)
    return {"oact_done": [], "appxml_done": [], "final_done": False}


def state_file(st, tmp):
    p = os.path.join(tmp, "state_out.json")
    with open(p, "w") as f:
        json.dump(st, f)
    return p


# --- stage 1 ------------------------------------------------------------------------------------

def run_oact(store, st, cfg, tmp):
    files = C.product_files("OACT", cfg)
    todo = [f for f in files if f["fileName"] not in st["oact_done"]]
    C.log("Office actions: %d of %d archive files left" % (len(todo), len(files)))
    pending, avg = {}, 120.0
    for f in todo:
        if time_left() < max(900, 3 * avg):
            break
        t0 = time.time()
        z = os.path.join(tmp, f["fileName"])
        C.download(f["fileDownloadURI"], z, cfg)
        out = os.path.join(tmp, "labels_" + f["fileName"] + ".jsonl.gz")
        samp = os.path.join(tmp, "samples_" + f["fileName"] + ".jsonl.gz")
        evp = os.path.join(tmp, "events_" + f["fileName"] + ".jsonl.gz")
        counts, n_s = {}, 0
        with gzip.open(out, "wt") as g, gzip.open(samp, "wt") as gs, gzip.open(evp, "wt") as ge:
            for obj in C.json_members(z):
                for rec in C.oa_records(obj):
                    app, date, code, _ = C.oa_event(rec)
                    if app and date and C.substantive(code):
                        ge.write(json.dumps([app, date, code]) + "\n")
                    row, why = C.oa_row(rec)
                    counts[why] = counts.get(why, 0) + 1
                    if row:
                        g.write(json.dumps(row) + "\n")
                        if n_s < 20:
                            t = rec.get("oaText") or rec.get("bodyText")
                            gs.write(json.dumps({"app": row["app"], "date": row["date"], "row": row,
                                                 "text": "\n".join(t) if isinstance(t, list) else t}) + "\n")
                            n_s += 1
        os.remove(z)
        pending["work/labels/%s.jsonl.gz" % f["fileName"]] = out
        pending["work/samples/%s.jsonl.gz" % f["fileName"]] = samp
        pending["work/events/%s.jsonl.gz" % f["fileName"]] = evp
        st["oact_done"].append(f["fileName"])
        avg = 0.7 * avg + 0.3 * (time.time() - t0)
        C.log("[OA] %s %s (%.0fs)" % (f["fileName"], json.dumps(counts), time.time() - t0))
        if len(pending) >= 2 * COMMIT_EVERY:
            pending["state/state.json"] = state_file(st, tmp)
            store.put(pending, "office actions through %s" % f["fileName"])
            for p in pending.values():
                if p.startswith(tmp) and os.path.exists(p) and not p.endswith("state_out.json"):
                    os.remove(p)
            pending = {}
    if pending:
        pending["state/state.json"] = state_file(st, tmp)
        store.put(pending, "office actions")
    return len(st["oact_done"]) == len(files)


# --- stage 2 ------------------------------------------------------------------------------------

def label_db(store, tmp):
    """All office-action labels in a local SQLite table: earliest action per application."""
    paths = store.get_prefix("work/labels/", os.path.join(tmp, "dl"))
    db = sqlite3.connect(os.path.join(tmp, "labels.sqlite"))
    db.execute("CREATE TABLE IF NOT EXISTS a (app TEXT PRIMARY KEY, date TEXT, row TEXT)")
    for p in paths:
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                old = db.execute("SELECT date FROM a WHERE app=?", (r["app"],)).fetchone()
                if old is None or r["date"] < old[0]:
                    db.execute("INSERT OR REPLACE INTO a VALUES (?,?,?)", (r["app"], r["date"], line))
        os.remove(p)
    db.commit()
    # drop any application whose earliest substantive action anywhere in the archive is not the kept rejection
    first = {}
    for p in store.get_prefix("work/events/", os.path.join(tmp, "dl")):
        with gzip.open(p, "rt") as f:
            for line in f:
                app, date, _ = json.loads(line)
                if app not in first or date < first[app]:
                    first[app] = date
        os.remove(p)
    stale = [app for app, date in db.execute("SELECT app, date FROM a") if first.get(app, date) < date]
    db.executemany("DELETE FROM a WHERE app=?", [(a,) for a in stale])
    db.commit()
    C.log("Dropped %d applications whose kept rejection was not their first action" % len(stale))
    apps = {r[0] for r in db.execute("SELECT app FROM a")}
    pubs = set()
    for (row,) in db.execute("SELECT row FROM a"):
        for v in json.loads(row)["labels"].values():
            for q, _ in v:
                pubs.add(q)
    return db, apps, pubs


def candidate_rows(app, pub, kind, doc, db):
    """Claim rows for one examined application (answers checked against the corpus later)."""
    a = json.loads(db.execute("SELECT row FROM a WHERE app=?", (app,)).fetchone()[0])
    cl = C.claims(doc)
    n_claims = max((int(k) for k in cl), default=0)
    if a["pending_max"] is not None and a["pending_max"] != n_claims:
        return [], "claim_count_mismatch"
    rel = C.RELATED.search(doc)
    links = sorted({C.digits(x) for x in C.REL_DOC.findall(rel.group(0))} - {"", app}) if rel else []
    rows = []
    quotes = a.get("quotes") or {}
    for k, answers in a["labels"].items():
        if k not in cl:
            continue
        text, independent, complete = C.with_parents(cl, int(k))
        if not complete:
            continue  # a claim it depends on is missing: the input would be incomplete
        q = {"%s|%d" % (r, n): quotes["%s|%d" % (r, n)] for r, n in answers if "%s|%d" % (r, n) in quotes}
        rows.append({"id": "%s-%s" % (app, k), "application": app, "publication": pub, "kind": kind,
                     "claim_number": int(k), "claim_text": cl[k][0], "claim_text_with_parents": text,
                     "independent": independent, "answers": answers, "quotes": q,
                     "office_action_date": a["date"], "art_unit": a["art_unit"], "related": links})
    return rows, "ok"


def run_appxml(store, st, cfg, tmp):
    db, apps, pubs = label_db(store, tmp)
    C.log("Need claims for %d applications, paragraphs for %d cited publications" % (len(apps), len(pubs)))
    have = set()
    for p in store.get_prefix("work/index/", os.path.join(tmp, "dl")):
        with gzip.open(p, "rt") as f:
            for line in f:
                have.add(json.loads(line)[0])
        os.remove(p)
    pid = C.find_appxml_product(cfg)
    files = C.product_files(pid, cfg)
    todo = [f for f in files if f["fileName"] not in st["appxml_done"]]
    C.log("Publications (%s): %d of %d archive files left" % (pid, len(todo), len(files)))
    pending, avg = {}, 180.0
    for f in todo:
        if time_left() < max(900, 3 * avg):
            break
        t0 = time.time()
        z = os.path.join(tmp, f["fileName"])
        C.download(f["fileDownloadURI"], z, cfg)
        base = f["fileName"] + ".jsonl.gz"
        paths = {s: os.path.join(tmp, s + "_" + base) for s in ("candidates", "index", "corpus")}
        n = {"docs": 0, "examined_apps": 0, "claim_rows": 0, "corpus_docs": 0, "no_paragraphs": 0,
             "claim_count_mismatch": 0}
        with gzip.open(paths["candidates"], "wt") as fc, gzip.open(paths["index"], "wt") as fi, \
                gzip.open(paths["corpus"], "wt") as fp:
            for doc in C.split_docs(z):
                n["docs"] += 1
                pub, kind, app = C.doc_ids(doc[:6000])
                if app in apps:
                    rows, why = candidate_rows(app, pub, kind, doc, db)
                    n["examined_apps"] += 1
                    if why != "ok":
                        n[why] += 1
                    for r in rows:
                        fc.write(json.dumps(r) + "\n")
                    n["claim_rows"] += len(rows)
                if pub in pubs and pub not in have:
                    paras = C.paragraphs(doc)
                    if paras:
                        fp.write(json.dumps({"pub": pub, "kind": kind,
                                             "paragraphs": {str(k): v for k, v in sorted(paras.items())}}) + "\n")
                        fi.write(json.dumps([pub, C.ranges(paras)]) + "\n")
                        have.add(pub)
                        n["corpus_docs"] += 1
                    else:
                        n["no_paragraphs"] += 1
        os.remove(z)
        # corpus and index for one archive file always go up in the same commit
        pending["work/candidates/" + base] = paths["candidates"]
        pending["work/index/" + base] = paths["index"]
        pending["corpus/" + base] = paths["corpus"]
        st["appxml_done"].append(f["fileName"])
        avg = 0.7 * avg + 0.3 * (time.time() - t0)
        C.log("[PUB] %s %s (%.0fs)" % (f["fileName"], json.dumps(n), time.time() - t0))
        if len(pending) >= 3 * COMMIT_EVERY:
            pending["state/state.json"] = state_file(st, tmp)
            store.put(pending, "publications through %s" % f["fileName"])
            for p in pending.values():
                if os.path.exists(p) and not p.endswith("state_out.json"):
                    os.remove(p)
            pending = {}
    if pending:
        pending["state/state.json"] = state_file(st, tmp)
        store.put(pending, "publications")
    db.close()
    return len(st["appxml_done"]) == len(files)


# --- stage 3 ------------------------------------------------------------------------------------

WORD = re.compile(r"[a-z0-9]+")


def norm(t):
    return " " + " ".join(WORD.findall(t.lower())) + " "


def quote_matches(quote, para):
    """None if the quote cannot be checked; else whether every substantial fragment of it is in the paragraph."""
    frags = [norm(f) for f in re.split(r"\.\.\.|…|\[[^\]]*\]", quote)]
    frags = [f for f in frags if len(f.split()) >= 4]
    if not frags:
        return None
    p = norm(para)
    return all(f in p for f in frags)


def check_quotes(store, cand, tmp):
    """Look up every quoted (publication, paragraph) in the corpus.
    Returns {app: n_failed}, {app: n_ok}, {(app, "pub|para") that matched}."""
    need = {}  # pub -> {para: [(app, quote)]}
    for p in cand:
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                for key, q in r.get("quotes", {}).items():
                    pub, n = key.split("|")
                    need.setdefault(pub, {}).setdefault(n, []).append((r["application"], q))
    bad, good, verified = {}, {}, set()
    for path in store.files("corpus/"):
        local = store.get(path, os.path.join(tmp, "corp"))
        with gzip.open(local, "rt") as f:
            for line in f:
                d = json.loads(line)
                if d["pub"] not in need:
                    continue
                for n, items in need[d["pub"]].items():
                    para = d["paragraphs"].get(n)
                    for app, q in items:
                        ok = quote_matches(q, para) if para else False
                        if ok is True:
                            good[app] = good.get(app, 0) + 1
                            verified.add((app, d["pub"] + "|" + n))
                        elif ok is False:
                            bad[app] = bad.get(app, 0) + 1
        os.remove(local)
    return bad, good, verified


def run_final(store, st, tmp):
    C.log("Final join ...")
    index = {}
    for p in store.get_prefix("work/index/", os.path.join(tmp, "dl")):
        with gzip.open(p, "rt") as f:
            for line in f:
                pub, spec = json.loads(line)
                index.setdefault(pub, spec)
        os.remove(p)
    cand = store.get_prefix("work/candidates/", os.path.join(tmp, "cand"))
    fam, best = C.Families(), {}
    for p in cand:  # pass 1: families, and which publication of each application to use (A1 first)
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                for x in r["related"]:
                    fam.union(r["application"], x)
                b = best.get(r["application"])
                if b is None or (r["kind"].startswith("A1") and not b.startswith("A1")):
                    best[r["application"]] = r["kind"] + "|" + r["publication"]
    bad, good, verified = check_quotes(store, cand, tmp)
    stats = {"applications": len(best), "rows": 0, "answers": 0, "rows_with_quote_verified_answer": 0,
             "quoted_citations_checked": sum(bad.values()) + sum(good.values()),
             "quoted_citations_matching": sum(good.values()),
             "applications_dropped_quote_mismatch": len(bad), "scrapped_no_answer_in_corpus": 0,
             "answers_dropped_not_in_corpus": 0, "corpus_publications": len(index)}
    writers, counts, shard_no, seen = {}, {"train": 0, "test": 0}, {"train": 0, "test": 0}, set()
    out_files = {}

    def writer(split):
        if split not in writers or counts[split] >= ROWS_PER_SHARD:
            if split in writers:
                writers[split].close()
            shard_no[split] += 1
            counts[split] = 0
            path = os.path.join(tmp, "%s-%05d.jsonl.gz" % (split, shard_no[split]))
            writers[split] = gzip.open(path, "wt")
            out_files["%s/%s-%05d.jsonl.gz" % (split, split, shard_no[split])] = path
        return writers[split]

    for p in cand:  # pass 2: check every answer against the corpus, write the final rows
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                if best[r["application"]] != r["kind"] + "|" + r["publication"] or r["id"] in seen:
                    continue
                seen.add(r["id"])
                if r["application"] in bad:
                    continue  # a quoted citation in this action did not match its paragraph: distrust the whole action
                ok = [[q, n] for q, n in r["answers"] if q in index and C.in_ranges(index[q], n)]
                stats["answers_dropped_not_in_corpus"] += len(r["answers"]) - len(ok)
                if not ok:
                    stats["scrapped_no_answer_in_corpus"] += 1
                    continue
                r["answers"] = ok
                r["quote_verified"] = [[q, n] for q, n in ok if (r["application"], "%s|%d" % (q, n)) in verified]
                family = fam.find(r["application"])
                split = "test" if int(hashlib.sha1(family.encode()).hexdigest(), 16) % 20 == 0 else "train"
                row = {k: r[k] for k in ("id", "application", "publication", "claim_number", "claim_text",
                                         "claim_text_with_parents", "independent", "answers", "quote_verified",
                                         "office_action_date", "art_unit")}
                row["family"] = family
                writer(split).write(json.dumps(row) + "\n")
                counts[split] += 1
                stats["rows"] += 1
                stats["rows_with_quote_verified_answer"] += bool(row["quote_verified"])
                stats["answers"] += len(row["answers"])
        os.remove(p)
    for w in writers.values():
        w.close()
    sp = os.path.join(tmp, "stats.json")
    with open(sp, "w") as f:
        json.dump(stats, f, indent=1)
    out_files["stats.json"] = sp
    out_files["README.md"] = os.path.join(HERE, "DATASET_README.md")
    out_files["evaluate.py"] = os.path.join(HERE, "evaluate.py")
    st["final_done"] = True
    out_files["state/state.json"] = state_file(st, tmp)
    store.put(out_files, "final dataset")
    C.log("Done:", json.dumps(stats))
    return stats


def main(store=None):
    tmp = os.environ["PM_WORK"]
    os.makedirs(tmp, exist_ok=True)
    store = store or HFStore()
    st = load_state(store, tmp)
    if st.get("parser") != C.P.VERSION:  # labels were made by an older parser: redo them (files are overwritten)
        C.log("Parser version changed (%s -> %s): re-parsing all office actions" % (st.get("parser"), C.P.VERSION))
        st.update({"parser": C.P.VERSION, "oact_done": [], "appxml_done": [], "final_done": False})
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        print(json.dumps({"oact_done": len(st["oact_done"]), "appxml_done": len(st["appxml_done"]),
                          "final_done": st["final_done"]}))
        return "status"
    if st["final_done"]:
        C.log("Dataset already complete.")
        return "complete"
    cfg = C.curl_config(os.environ["USPTO_API_KEY"])
    if not run_oact(store, st, cfg, tmp):
        return "continue"
    if not run_appxml(store, st, cfg, tmp):
        return "continue"
    if time_left() < 1800:
        return "continue"
    run_final(store, st, tmp)
    return "complete"


if __name__ == "__main__":
    result = main()
    print("RESULT=" + result)
    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a") as f:
            f.write("result=%s\n" % result)
