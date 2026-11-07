"""Re-parse the audited office actions with the current parser; print what it would keep for each audited claim."""
import gzip, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import collector as C
import oa_parser as P
for line in gzip.open(sys.argv[1], "rt"):
    r = json.loads(line)
    text = r.get("office_action_text") or ""
    why = None
    if not text:
        why = "no_text"
    elif C.AMEND.search(text) or C.RESPONSE.search(text):
        why = "responds_to_amendment"
    labels = {}
    if not why:
        try:
            labels, _, why = P.parse_office_action_strict(text)
        except Exception as e:
            why = "error:%s" % e
    kept = sorted([p, n] for p, n in labels.get(r["claim_number"], set())) if not why else []
    print("EVAL:" + json.dumps({"id": r["id"], "why": why, "claim_kept": bool(kept), "answers": kept}))
