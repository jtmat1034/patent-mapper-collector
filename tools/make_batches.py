"""Turn audit.jsonl.gz into agent batches: for each row, the claim, the cited paragraphs' text, and the parts of the
office action that matter (rejection statements naming the claim, the claim's section, and every place a cited
paragraph number appears)."""
import gzip, json, re, sys, os
src, out, per = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(l) for l in gzip.open(src, "rt")]

def windows(text, claim, answers):
    spans = []
    for m in re.finditer(r"claims?(?:\s*\(s\))?\s+[\d,\-–and &to through]*\b%d\b[^.]{0,200}?rejected" % claim, text, re.I):
        spans.append((m.start() - 100, m.end() + 900))
    for m in re.finditer(r"(?:Regarding|As (?:to|per)|With respect to|For|Referring to|Consider(?:ing)?|In re)\s*(?:independent |dependent |the )?claims?\s+[\d,\-–and &]*\b%d\b" % claim, text, re.I):
        spans.append((m.start(), m.end() + 2500))
    for _, n in answers:
        for pat in (r"\[\s*0*%d\s*\]" % n, r"(?:¶+|para(?:graph)?s?\.?|par\.)\s*\[?\s*0*%d\b" % n, r"(?<![\d./,])0*%04d(?![\d/])" % n):
            for m in re.finditer(pat, text, re.I):
                spans.append((m.start() - 700, m.end() + 250))
    spans = sorted((max(0, a), min(len(text), b)) for a, b in spans)
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    parts, total = [], 0
    for a, b in merged:
        chunk = re.sub(r"\s+", " ", text[a:b])
        if total + len(chunk) > 12000:
            chunk = chunk[:max(0, 12000 - total)]
        parts.append("... " + chunk + " ...")
        total += len(chunk)
        if total >= 12000:
            break
    return "\n".join(parts) if parts else re.sub(r"\s+", " ", text[:6000])

os.makedirs(out, exist_ok=True)
items = []
for r in rows:
    text = r.get("office_action_text") or ""
    items.append({
        "id": r["id"], "claim_number": r["claim_number"],
        "claim_text_with_parents": r["claim_text_with_parents"][:4000],
        "answers": [{"reference": q, "paragraph": n,
                     "paragraph_text": (r["cited_paragraph_text"].get("%s|%d" % (q, n)) or "(missing)")[:1500]}
                    for q, n in r["answers"][:6]],
        "n_answers": len(r["answers"]),
        "office_action_excerpts": windows(text, r["claim_number"], r["answers"]) if text else "(no office action text)",
    })
for i in range(0, len(items), per):
    with open(os.path.join(out, "batch_%02d.json" % (i // per + 1)), "w") as f:
        json.dump(items[i:i + per], f, indent=1)
print(len(items), "rows ->", (len(items) + per - 1) // per, "batches")
