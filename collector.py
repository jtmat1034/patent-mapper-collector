#!/usr/bin/env python3
"""Build the claim -> correct-citation training set from public USPTO data.

    python3 collector.py test     check the API key and the real file formats (a few minutes)
    python3 collector.py run      the full build; safe to stop (Ctrl-C) and re-run, it resumes
    python3 collector.py status   progress so far

Pipeline (all public USPTO Open Data Portal bulk files, linked by application number):
  1. Office Actions Weekly Archives (OACT, 2020+): first-action non-final rejections are parsed
     with oa_parser.py into claim -> {(publication, paragraph)} labels, plus allowed / allowable
     claims ("no reference"). Office actions that respond to an amendment are skipped, because
     the claim text examined would no longer match the published claims.
  2. Patent Application Full-Text XML (every pre-grant publication since 2001): claims of the
     examined applications, and every paragraph of every cited publication.
  3. Join: a claim is kept only if every piece is present - its text, and at least one cited
     paragraph that exists in the corpus (or a "no reference" label). Everything else is scrapped.
     Train/test split by patent family so continuations never straddle the split.

Python 3.8+ standard library only. Downloads use curl (built into macOS).
"""
import gzip
import hashlib
import html
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oa_parser as P  # noqa: E402

API = "https://api.uspto.gov"
WORK = os.environ.get("PM_WORK") or os.path.join(HERE, "work")
OUT = os.path.join(HERE, "dataset")
STATE = os.path.join(WORK, "state.json")
LOG = os.path.join(HERE, "collector.log")
MIN_FREE_GB = 25


# --- small utilities ------------------------------------------------------------------------

def log(*a):
    msg = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"oact_done": [], "appxml_done": []}


def save_state(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, STATE)


def api_key():
    path = os.path.join(HERE, "apikey.txt")
    key = os.environ.get("USPTO_API_KEY", "").strip()
    if not key and os.path.exists(path):
        key = open(path).read().strip()
    if not key:
        import getpass
        key = getpass.getpass("Paste your USPTO API key (it will not show) and press Enter: ").strip()
        with open(path, "w") as f:
            f.write(key)
        os.chmod(path, 0o600)
    return key


def curl_config(key):
    """Header file so the key never appears on a command line."""
    path = os.path.join(WORK, ".curlrc")
    with open(path, "w") as f:
        f.write('header = "X-API-KEY: %s"\nheader = "Accept: application/json"\n' % key.replace('"', ""))
    os.chmod(path, 0o600)
    return path


def curl(args, cfg, tries=8):
    """Run curl with retries; returns (http_code, stdout bytes). 429/5xx back off and retry."""
    delay = 30
    for attempt in range(tries):
        p = subprocess.run(["curl", "-sS", "-L", "-K", cfg, "-w", "\n%{http_code}"] + args,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = p.stdout
        code = int(out[out.rfind(b"\n") + 1:] or 0) if out else 0
        body = out[:out.rfind(b"\n")] if out else b""
        if p.returncode == 0 and 200 <= code < 300:
            return code, body
        if code in (401, 403):
            raise SystemExit("USPTO refused the API key (HTTP %d). Check the key in apikey.txt, and that "
                             "your USPTO.gov profile's Open Data Portal fields are filled in." % code)
        if code == 404:
            return code, body
        log("  retry %d/%d (curl exit %d, HTTP %d) in %ds" % (attempt + 1, tries, p.returncode, code, delay))
        time.sleep(delay)
        delay = min(delay * 2, 900)
    raise RuntimeError("download kept failing: %s" % args[-1])


def get_json(url, cfg):
    code, body = curl([url], cfg)
    if code == 404:
        return None
    return json.loads(body.decode("utf-8"))


def download(url, dest, cfg):
    if not url.startswith(API + "/"):
        raise ValueError("refusing to send the API key to " + url)
    tmp = dest + ".part"
    curl(["--retry", "5", "--retry-delay", "30", "-o", tmp, url], cfg)
    os.replace(tmp, dest)


def free_gb():
    return shutil.disk_usage(HERE).free / 1e9


def check_disk():
    if free_gb() < MIN_FREE_GB:
        raise SystemExit("Stopping: only %.0f GB free on this disk (need %d GB). Free up space and re-run; "
                         "it resumes where it left off." % (free_gb(), MIN_FREE_GB))


def first(v):
    if isinstance(v, list):
        return v[0] if v else None
    return v


def digits(s):
    return re.sub(r"\D", "", str(s or ""))


# --- bulk product listings --------------------------------------------------------------------

def product_files(pid, cfg):
    files, offset = [], 0
    while True:
        d = get_json("%s/api/v1/datasets/products/%s?includeFiles=true&offset=%d&limit=100" % (API, pid, offset), cfg)
        if not d or not d.get("bulkDataProductBag"):
            break
        bag = d["bulkDataProductBag"][0].get("productFileBag") or {}
        page = bag.get("fileDataBag") or []
        new = [f for f in page if f.get("fileDownloadURI") not in {g.get("fileDownloadURI") for g in files}]
        files += new
        count = bag.get("count") or 0
        if not new or len(files) >= count:
            break
        offset = len(files)
    files = [f for f in files if (f.get("fileTypeText") or "Data") == "Data" and f.get("fileDownloadURI")]
    files.sort(key=lambda f: f.get("fileDataFromDate") or f.get("fileName"))
    return files


def find_appxml_product(cfg):
    d = get_json(API + "/api/v1/datasets/products/APPXML?includeFiles=false", cfg)
    if d and d.get("bulkDataProductBag"):
        return "APPXML"
    d = get_json(API + "/api/v1/datasets/products/search?q=" + "application%20full-text&limit=50", cfg) or {}
    for p in d.get("bulkDataProductBag", []):
        t = (p.get("productTitleText") or "").lower()
        if "application" in t and "full-text" in t and "xml" in t and "image" not in t:
            return p["productIdentifier"]
    raise SystemExit("Could not find the Patent Application Full-Text XML product on the Open Data Portal.")


# --- stage 1: office actions -------------------------------------------------------------------

AMEND = re.compile(r"amendment|amended|response filed|remarks filed|request for continued examination|\bRCE\b", re.I)
PENDING = re.compile(r"[Cc]laims?(?:\(s\))?\s+(\d+(?:\s*(?:-|–|to|through|,|and|&)\s*\d+)*)\s*"
                     r"(?:is|are|is/are)\s+(?:currently\s+|now\s+|presently\s+)?pending")
OA_CODES = {"CTNF"}  # first-action non-final rejections (see module docstring)


def oa_records(obj):
    """Yield office-action dicts from any JSON layout (list, {docs: [...]}, nested)."""
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            if "bodyText" in o and ("patentApplicationNumber" in o or "applicationNumberText" in o):
                yield o
            else:
                stack.extend(v for v in o.values() if isinstance(v, (dict, list)))
        elif isinstance(o, list):
            stack.extend(x for x in o if isinstance(x, (dict, list)))


def json_members(zpath):
    """Yield parsed JSON objects from a zip (json, jsonl, or nested zip members)."""
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            low = name.lower()
            if low.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(z.read(name))) as z2:
                    for n2 in z2.namelist():
                        if n2.lower().endswith((".json", ".jsonl")):
                            for o in parse_json_bytes(z2.read(n2)):
                                yield o
            elif low.endswith((".json", ".jsonl", ".txt")):
                for o in parse_json_bytes(z.read(name)):
                    yield o


def parse_json_bytes(b):
    s = b.decode("utf-8", "replace").strip()
    if not s:
        return
    try:
        yield json.loads(s)
    except ValueError:
        for line in s.splitlines():
            line = line.strip().rstrip(",")
            if line.startswith("{"):
                try:
                    yield json.loads(line)
                except ValueError:
                    pass


def oa_row(rec):
    """Parse one office-action record -> label row, or (None, reason)."""
    code = first(rec.get("legacyDocumentCodeIdentifier"))
    if code not in OA_CODES:
        return None, "not_ctnf"
    text = rec.get("bodyText")
    if isinstance(text, list):
        text = "\n".join(t for t in text if isinstance(t, str))
    if not text or len(text) < 500:
        return None, "no_text"
    if AMEND.search(text):
        return None, "responds_to_amendment"
    app = digits(first(rec.get("patentApplicationNumber") or rec.get("applicationNumberText")))
    if len(app) != 8:
        return None, "bad_app_number"
    labels = P.parse_office_action(text)
    allow = {k: v for k, v in P.allowable_claims(text).items() if k not in labels}
    if not labels and not allow:
        return None, "nothing_extracted"
    m = PENDING.search(text)
    pending_max = max(int(x) for x in re.findall(r"\d+", m.group(1))) if m else None
    return {
        "app": app,
        "date": (first(rec.get("submissionDate")) or "")[:10],
        "art_unit": str(first(rec.get("groupArtUnitNumber")) or ""),
        "pending_max": pending_max,
        "labels": {str(k): sorted([p, n] for p, n in v) for k, v in labels.items()},
        "no_reference": {str(k): v for k, v in allow.items()},
    }, "kept"


def stage_oact(cfg, st, limit=None):
    files = product_files("OACT", cfg)
    log("Office action archive: %d files" % len(files))
    todo = [f for f in files if f["fileName"] not in st["oact_done"]][:limit]
    os.makedirs(os.path.join(WORK, "labels"), exist_ok=True)
    os.makedirs(os.path.join(WORK, "samples"), exist_ok=True)
    for i, f in enumerate(todo):
        check_disk()
        z = os.path.join(WORK, f["fileName"])
        log("[OA %d/%d] %s (%.0f MB)" % (i + 1, len(todo), f["fileName"], (f.get("fileSize") or 0) / 1e6))
        download(f["fileDownloadURI"], z, cfg)
        counts, n_samples = {}, 0
        out = os.path.join(WORK, "labels", f["fileName"] + ".jsonl.gz")
        samp = gzip.open(os.path.join(WORK, "samples", f["fileName"] + ".jsonl.gz"), "wt")
        with gzip.open(out + ".tmp", "wt") as g:
            for obj in json_members(z):
                for rec in oa_records(obj):
                    row, why = oa_row(rec)
                    counts[why] = counts.get(why, 0) + 1
                    if row:
                        g.write(json.dumps(row) + "\n")
                        if n_samples < 20:  # a few raw actions kept for spot-checking the labels
                            t = rec.get("bodyText")
                            samp.write(json.dumps({"app": row["app"], "date": row["date"],
                                                   "text": "\n".join(t) if isinstance(t, list) else t}) + "\n")
                            n_samples += 1
        samp.close()
        os.replace(out + ".tmp", out)
        os.remove(z)
        st["oact_done"].append(f["fileName"])
        save_state(st)
        log("   ", json.dumps(counts))
        time.sleep(2)


# --- stage 2: publications ------------------------------------------------------------------------

def needed_sets():
    apps, pubs = set(), set()
    d = os.path.join(WORK, "labels")
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not name.endswith(".jsonl.gz"):
            continue
        with gzip.open(os.path.join(d, name), "rt") as f:
            for line in f:
                r = json.loads(line)
                apps.add(r["app"])
                for v in r["labels"].values():
                    for p, _ in v:
                        pubs.add(p)
    return apps, pubs


INLINE = re.compile(r"</?(?:b|i|u|o|sub|sup|smallcaps|sub2|sup2|highlight|bold|italic|underline|ital)\b[^>]*>", re.I)
TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")


def clean(s):
    """XML fragment -> plain text. Formatting tags vanish without a space (L<smallcaps>ITHIUM</smallcaps>)."""
    return WS.sub(" ", html.unescape(TAG.sub(" ", INLINE.sub("", s)))).strip()


PUBREF = re.compile(r"<publication-reference>.*?<doc-number>\s*([^<]+?)\s*</doc-number>(?:.*?<kind>\s*([^<]+?)\s*</kind>)?", re.S)
APPREF = re.compile(r"<application-reference[^>]*>.*?<doc-number>\s*([^<]+?)\s*</doc-number>", re.S)
# 2001-2004 publications (pap DTD)
OLD_PUB = re.compile(r"<document-id>\s*<doc-number>\s*(\d{11})\s*</doc-number>\s*<kind-code>\s*([^<]+?)\s*</kind-code>", re.S)
OLD_APP = re.compile(r"<application-number>\s*<doc-number>\s*([^<]+?)\s*</doc-number>", re.S)
PARA = re.compile(r'<p\b[^>]*\bnum="(\d+)"[^>]*>(.*?)</p>', re.S)
OLD_PARA = re.compile(r"<paragraph\b[^>]*>(.*?)</paragraph>", re.S)
OLD_NUM = re.compile(r"<number>\s*\[?\s*(\d+)\s*\]?\s*</number>")
CLAIM = re.compile(r'<claim\s+id="CLM-(\d+)"[^>]*>(.*?)</claim>', re.S)
CLAIM_DEP = re.compile(r'(?:idref|depends_on)="CLM-(\d+)"')
DESC = re.compile(r"<description\b.*?</description>|<subdoc-description\b.*?</subdoc-description>", re.S)
RELATED = re.compile(r"<us-related-documents>.*?</us-related-documents>|<continuity-data>.*?</continuity-data>", re.S)
REL_DOC = re.compile(r"<(?:parent-doc|child-doc|us-provisional-application|provisional-application)>\s*"
                     r"<document-id>.*?<doc-number>\s*([^<]+?)\s*</doc-number>", re.S)


def doc_ids(head):
    m = PUBREF.search(head)
    if m:
        pub, kind = digits(m.group(1)), (m.group(2) or "").strip()
        a = APPREF.search(head)
        return pub, kind, digits(a.group(1)) if a else ""
    m = OLD_PUB.search(head)
    if m:
        a = OLD_APP.search(head)
        return m.group(1), m.group(2).strip(), digits(a.group(1)) if a else ""
    return "", "", ""


def paragraphs(doc):
    out = {}
    m = DESC.search(doc)
    body = m.group(0) if m else doc
    for num, txt in PARA.findall(body):
        t = clean(txt)
        if t:
            out.setdefault(int(num), t)
    if not out:
        for txt in OLD_PARA.findall(body):
            n = OLD_NUM.search(txt)
            if n:
                t = clean(OLD_NUM.sub(" ", txt))
                if t:
                    out.setdefault(int(n.group(1)), t)
    return out


def claims(doc):
    out = {}
    for num, body in CLAIM.findall(doc):
        deps = [int(x) for x in CLAIM_DEP.findall(body)]
        t = re.sub(r"^\s*\d+\s*[.)]\s*", "", clean(body))
        if len(t) >= 20 and not re.match(r"^\(?(?:canceled|cancelled|deleted)\)?\.?$", t, re.I):
            out[str(int(num))] = [t, deps[0] if deps else None]
    return out


def split_docs(zpath):
    """Yield each concatenated XML document (as str) from a weekly publication zip."""
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.lower().endswith((".xml", ".sgm", ".txt")):
                continue
            buf = []
            with z.open(name) as f:
                for line in f:
                    if line.startswith(b"<?xml") and buf:
                        yield b"".join(buf).decode("utf-8", "replace")
                        buf = []
                    buf.append(line)
            if buf:
                yield b"".join(buf).decode("utf-8", "replace")


def ranges(nums):
    nums = sorted(nums)
    out, s, p = [], nums[0], nums[0]
    for n in nums[1:]:
        if n != p + 1:
            out.append("%d-%d" % (s, p) if s != p else str(s))
            s = n
        p = n
    out.append("%d-%d" % (s, p) if s != p else str(s))
    return ",".join(out)


def db():
    import sqlite3
    con = sqlite3.connect(os.path.join(WORK, "pubs.sqlite"))
    con.execute("CREATE TABLE IF NOT EXISTS claims (app TEXT PRIMARY KEY, pub TEXT, kind TEXT, data TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS idx (pub TEXT PRIMARY KEY, spec TEXT)")
    return con


def stage_appxml(cfg, st, limit=None):
    apps, pubs = needed_sets()
    log("Need claims for %d applications and paragraphs for %d cited publications" % (len(apps), len(pubs)))
    pid = find_appxml_product(cfg)
    files = product_files(pid, cfg)
    log("Publication full-text archive (%s): %d files" % (pid, len(files)))
    todo = [f for f in files if f["fileName"] not in st["appxml_done"]][:limit]
    os.makedirs(os.path.join(WORK, "corpus"), exist_ok=True)
    con = db()
    have = {r[0] for r in con.execute("SELECT pub FROM idx")}  # corrected re-publications reuse the number
    for i, f in enumerate(todo):
        check_disk()
        z = os.path.join(WORK, f["fileName"])
        log("[PUB %d/%d] %s (%.0f MB)" % (i + 1, len(todo), f["fileName"], (f.get("fileSize") or 0) / 1e6))
        download(f["fileDownloadURI"], z, cfg)
        # a fresh name every attempt, so re-processing a file can never overwrite paragraphs already indexed
        corpus = os.path.join(WORK, "corpus", "%s.%d.jsonl.gz" % (f["fileName"], int(time.time())))
        n = {"docs": 0, "claims_docs": 0, "corpus_docs": 0, "no_paragraphs": 0}
        claim_rows, idx_rows, new = [], [], set()
        with gzip.open(corpus + ".tmp", "wt") as fp:
            for doc in split_docs(z):
                n["docs"] += 1
                pub, kind, app = doc_ids(doc[:6000])
                want_claims = app in apps
                want_paras = pub in pubs and pub not in have and pub not in new
                if not (want_claims or want_paras):
                    continue
                if want_claims:
                    rel = RELATED.search(doc)
                    links = sorted({digits(x) for x in REL_DOC.findall(rel.group(0))} - {"", app}) if rel else []
                    claim_rows.append((app, pub, kind, json.dumps({"claims": claims(doc), "related": links})))
                    n["claims_docs"] += 1
                if want_paras:
                    paras = paragraphs(doc)
                    if paras:
                        fp.write(json.dumps({"pub": pub, "kind": kind,
                                             "paragraphs": {str(k): v for k, v in sorted(paras.items())}}) + "\n")
                        idx_rows.append((pub, ranges(paras)))
                        new.add(pub)
                        n["corpus_docs"] += 1
                    else:
                        n["no_paragraphs"] += 1
        # order matters for crash safety: corpus file first, then the index that points into it
        os.replace(corpus + ".tmp", corpus)
        with con:
            for app, pub, kind, data in claim_rows:
                old = con.execute("SELECT kind FROM claims WHERE app=?", (app,)).fetchone()
                if old is None or (kind.startswith("A1") and not old[0].startswith("A1")):
                    con.execute("INSERT OR REPLACE INTO claims VALUES (?,?,?,?)", (app, pub, kind, data))
            con.executemany("INSERT OR IGNORE INTO idx VALUES (?,?)", idx_rows)
        have |= new
        os.remove(z)
        st["appxml_done"].append(f["fileName"])
        save_state(st)
        log("   ", json.dumps(n))
        time.sleep(2)
    con.close()


# --- stage 3: join into the training set ------------------------------------------------------------

def in_ranges(spec, n):
    for part in spec.split(","):
        a, _, b = part.partition("-")
        if int(a) <= n <= int(b or a):
            return True
    return False


class Families:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def gz_rows(sub):
    d = os.path.join(WORK, sub)
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if name.endswith(".jsonl.gz"):
            with gzip.open(os.path.join(d, name), "rt") as f:
                for line in f:
                    yield json.loads(line)


def with_parents(cl, num):
    chain, seen, k = [], set(), num
    while k is not None and str(k) in cl and k not in seen:
        seen.add(k)
        chain.append(k)
        k = cl[str(k)][1]
    return "\n".join("Claim %d: %s" % (c, cl[str(c)][0]) for c in reversed(chain)), len(chain) == 1


def build():
    log("Joining labels, claims and corpus ...")
    con = db()
    fam = Families()
    for app, data in con.execute("SELECT app, data FROM claims"):
        for x in json.loads(data)["related"]:
            fam.union(app, x)
    earliest = {}  # earliest first-action rejection per application
    for r in gz_rows("labels"):
        if r["app"] not in earliest or r["date"] < earliest[r["app"]]:
            earliest[r["app"]] = r["date"]
    stats = {"office_actions": len(earliest), "no_publication_found": 0, "claim_count_mismatch": 0,
             "rows": 0, "rejected_rows": 0, "no_reference_rows": 0, "answers": 0,
             "scrapped_claim_missing": 0, "scrapped_no_answer_in_corpus": 0, "answers_dropped_not_in_corpus": 0}
    os.makedirs(OUT, exist_ok=True)
    tr = gzip.open(os.path.join(OUT, "train.jsonl.gz.tmp"), "wt")
    te = gzip.open(os.path.join(OUT, "test.jsonl.gz.tmp"), "wt")
    done = set()
    for a in gz_rows("labels"):
        app = a["app"]
        if app in done or a["date"] != earliest[app]:
            continue
        done.add(app)
        got = con.execute("SELECT pub, data FROM claims WHERE app=?", (app,)).fetchone()
        if not got:
            stats["no_publication_found"] += 1
            continue
        pub_self, cl = got[0], json.loads(got[1])["claims"]
        n_claims = max((int(k) for k in cl), default=0)
        if a["pending_max"] is not None and a["pending_max"] != n_claims:
            stats["claim_count_mismatch"] += 1  # the examined claims are not the published ones
            continue
        family = fam.find(app)
        dest = te if int(hashlib.sha1(family.encode()).hexdigest(), 16) % 20 == 0 else tr  # 5% test
        items = [(k, v, None) for k, v in a["labels"].items()] + [(k, [], v) for k, v in a["no_reference"].items()]
        for k, answers, kind in items:
            if k not in cl:
                stats["scrapped_claim_missing"] += 1
                continue
            good = []
            if kind is None:
                for q, num in answers:
                    spec = con.execute("SELECT spec FROM idx WHERE pub=?", (q,)).fetchone()
                    if spec and in_ranges(spec[0], num):
                        good.append([q, num])
                stats["answers_dropped_not_in_corpus"] += len(answers) - len(good)
                if not good:
                    stats["scrapped_no_answer_in_corpus"] += 1
                    continue
            text, independent = with_parents(cl, int(k))
            row = {"id": "%s-%s" % (app, k), "application": app, "publication": pub_self, "claim_number": int(k),
                   "claim_text": cl[k][0], "claim_text_with_parents": text, "independent": independent,
                   "label": "rejected" if kind is None else kind, "answers": good,
                   "office_action_date": a["date"], "art_unit": a["art_unit"], "family": family}
            dest.write(json.dumps(row) + "\n")
            stats["rows"] += 1
            stats["rejected_rows" if kind is None else "no_reference_rows"] += 1
            stats["answers"] += len(good)
    tr.close()
    te.close()
    for s in ("train", "test"):
        os.replace(os.path.join(OUT, s + ".jsonl.gz.tmp"), os.path.join(OUT, s + ".jsonl.gz"))
    cdir = os.path.join(OUT, "corpus")
    os.makedirs(cdir, exist_ok=True)
    for name in sorted(os.listdir(os.path.join(WORK, "corpus"))):
        if not name.endswith(".jsonl.gz"):
            continue
        src, dst = os.path.join(WORK, "corpus", name), os.path.join(cdir, name)
        if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
            shutil.copyfile(src, dst)
    stats["corpus_publications"] = con.execute("SELECT COUNT(*) FROM idx").fetchone()[0]
    con.close()
    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    shutil.copyfile(os.path.join(HERE, "evaluate.py"), os.path.join(OUT, "evaluate.py"))
    shutil.copyfile(os.path.join(HERE, "DATASET_README.md"), os.path.join(OUT, "README.md"))
    log("Done:", json.dumps(stats))
    return stats


# --- commands -----------------------------------------------------------------------------------------

def shape(o, depth=0):
    """Compact description of a JSON value's structure."""
    if depth > 6:
        return "..."
    if isinstance(o, dict):
        return "{" + ", ".join("%s: %s" % (k, shape(v, depth + 1)) for k, v in list(o.items())[:25]) + "}"
    if isinstance(o, list):
        return "[%d x %s]" % (len(o), shape(o[0], depth + 1) if o else "-")
    if isinstance(o, str):
        return "str(%d)%s" % (len(o), repr(o[:40]))
    return type(o).__name__ + ":" + repr(o)[:30]


def describe_zip(zpath, depth=0):
    """Log what is actually inside an archive, so the reader can be fixed."""
    with zipfile.ZipFile(zpath) as z:
        infos = z.infolist()
        log("  %sarchive: %d members" % ("  " * depth, len(infos)))
        for i in infos[:8]:
            log("  %s  %s  %d bytes" % ("  " * depth, i.filename, i.file_size))
        for i in infos[:2]:
            if i.is_dir():
                continue
            data = z.read(i.filename)
            if i.filename.lower().endswith(".zip") and depth < 2:
                p = zpath + ".inner.zip"
                with open(p, "wb") as f:
                    f.write(data)
                describe_zip(p, depth + 1)
                os.remove(p)
                continue
            if data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)
            log("  %s  first bytes of %s: %r" % ("  " * depth, i.filename, data[:1500]))
            try:
                log("  %s  JSON shape: %s" % ("  " * depth, shape(json.loads(data))))
            except ValueError:
                lines = data.splitlines()
                try:
                    log("  %s  JSON-lines shape (line 1 of %d): %s" % ("  " * depth, len(lines), shape(json.loads(lines[0]))))
                except (ValueError, IndexError):
                    log("  %s  not JSON" % ("  " * depth))


def cmd_test(cfg):
    log("TEST 1/3: API key")
    d = get_json("%s/api/v1/datasets/products/OACT?includeFiles=true&offset=0&limit=100" % API, cfg) or {}
    bag = (d.get("bulkDataProductBag") or [{}])[0]
    log("  product fields:", sorted(bag.keys()), "| file bag fields:", sorted((bag.get("productFileBag") or {}).keys()),
        "| count:", (bag.get("productFileBag") or {}).get("count"), "| totals:",
        {k: v for k, v in bag.items() if "count" in k.lower() or "total" in k.lower()})
    files = product_files("OACT", cfg)
    log("  oldest file:", files[0]["fileName"] if files else "-")
    log("  OK - office action archive lists %d files, newest %s" % (len(files), files[-1]["fileName"] if files else "-"))
    log("TEST 2/3: one office-action file (format + labels)")
    f = files[-1]
    z = os.path.join(WORK, "test_" + f["fileName"])
    download(f["fileDownloadURI"], z, cfg)
    counts, keys, examples = {}, None, []
    for obj in json_members(z):
        for rec in oa_records(obj):
            keys = keys or sorted(rec.keys())
            row, why = oa_row(rec)
            counts[why] = counts.get(why, 0) + 1
            if row and row["labels"] and len(examples) < 3:
                examples.append(row)
    log("  record fields:", keys)
    log("  ", json.dumps(counts))
    for e in examples:
        log("  example:", json.dumps(e)[:600])
    if not counts.get("kept"):
        describe_zip(z)
    os.remove(z)
    if not counts.get("kept"):
        raise SystemExit("No usable office actions found in the test file - send this output to Claude.")
    log("TEST 3/3: one publication full-text file (claims + paragraphs)")
    pid = find_appxml_product(cfg)
    pf = product_files(pid, cfg)
    g = pf[-1]
    log("  %s: %d files, newest %s" % (pid, len(pf), g["fileName"]))
    z = os.path.join(WORK, "test_" + g["fileName"])
    download(g["fileDownloadURI"], z, cfg)
    n, ok, shown = 0, 0, 0
    for doc in split_docs(z):
        n += 1
        pub, kind, app = doc_ids(doc[:6000])
        paras, cl = paragraphs(doc), claims(doc)
        if pub and app and paras and cl:
            ok += 1
            if shown < 2:
                k = sorted(paras)[min(5, len(paras) - 1)]
                log("  example: pub %s %s app %s, %d claims, %d paragraphs; claim 1: %s | para %d: %s" % (
                    pub, kind, app, len(cl), len(paras), cl.get("1", ["?"])[0][:150], k, paras[k][:150]))
                shown += 1
    os.remove(z)
    log("  %d documents, %d with publication no., application no., claims and paragraphs" % (n, ok))
    if not ok:
        raise SystemExit("Could not read the publication file - send this output to Claude.")
    log("ALL TESTS PASSED. Free disk: %.0f GB" % free_gb())


def cmd_status():
    st = load_state()
    print("office-action files done: %d" % len(st["oact_done"]))
    print("publication files done:  %d" % len(st["appxml_done"]))
    if os.path.exists(os.path.join(OUT, "stats.json")):
        print(open(os.path.join(OUT, "stats.json")).read())
    print("free disk: %.0f GB" % free_gb())


def main():
    os.makedirs(WORK, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "status":
        return cmd_status()
    cfg = curl_config(api_key())
    if cmd == "test":
        return cmd_test(cfg)
    if cmd == "build":
        return build()
    if cmd == "run":
        st = load_state()
        stage_oact(cfg, st)
        stage_appxml(cfg, st)
        build()
        log("Finished. The dataset is in", OUT)
        return
    print(__doc__)


if __name__ == "__main__":
    main()
