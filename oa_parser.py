"""Extract claim -> {(reference, paragraph)} labels from the full text of a USPTO office action.

Only 35 USC 102/103 rejections count, and only paragraph citations to U.S. pre-grant
publications (11-digit numbers like 20050025220). Figure-only, column/line, and
reference-less citations are ignored; a claim with nothing left gets no labels (scrapped).

Main entry point: parse_office_action(text) -> {claim_number: set((ref, paragraph), ...)}
"""
import re
from collections import OrderedDict

VERSION = 2  # bump when labels change; the cloud run then re-parses every office-action file

# --- reference numbers ----------------------------------------------------------------

# US 2012/0175451, US 20120175451, US 2012-0175451, U.S. Pub. No. 2012/0175451, 2012/0175451 A1
PUB = re.compile(r"(?<!\d)((?:19|20)\d\d)\s*[/\-]?\s*(\d{7})(?!\d)")
# Name right before a publication number: "Liang et al. (US 20110034039)", "Yanagawa et al., US 2012/0175451"
NAME_THEN_PUB = re.compile(
    r"([A-Z][A-Za-z'\-]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z'\-]+))?)"
    r"[\s,(\[]*(?:hereinafter[^)]*?)?(?:U\.?\s?S\.?\s*)?(?:(?:Pat(?:ent)?\.?\s*)?(?:App(?:lication|l)?\.?\s*)?"
    r"(?:Pub(?:lication|l)?\.?\s*)?(?:No\.?\s*)?)?((?:19|20)\d\d\s*[/\-]?\s*\d{7})")
# Granted US patents (6,243,581 / US 6243581 B1 / U.S. Pat. No. 10,123,456): cited by column/line, never by
# paragraph, so they are tracked only to keep their citations from being credited to a publication.
GRANTED = re.compile(r"(?<![\d,/.])(?:\d{1,2},\d{3},\d{3}|(?<=US)\s?\d{7,8}(?=\s*B\d)|(?<=US )\d{7,8}(?=\s*B\d))(?![\d,])")
# 'Yanagawa '451' style shorthand
SHORT = re.compile(r"([A-Z][A-Za-z'\-]+)\s*[‘’'`]\s*(\d{3})\b")

# --- claim sections -------------------------------------------------------------------

SECTION_HEAD = re.compile(
    r"(?:^|\n|\.|\|)\s*(?:Regarding|As\s+to|As\s+per|With\s+(?:respect|regard)\s+to|Referring\s+to|"
    r"In\s+re(?:gard\s+to)?|In\s+reference\s+to|Concerning|Considering|Consider|For|Per|In|Re:?|re)\s*"
    r"(?:independent\s+|dependent\s+|the\s+)?claims?\s+(\d+(?:\s*(?:-|–|to|through|,(?:\s*and)?|and|&)\s*\d+)*)",
    re.I)
RANGE = re.compile(r"(\d+)\s*(?:(?:-|–|to|through)\s*(\d+))?")

# What kind of rejection a section belongs to is decided by the closest preceding
# "rejected under 35 U.S.C. 1xx" sentence (or a stop heading such as Allowable Subject Matter).
STATEMENT = re.compile(r"rejected\s+under\s+(?:pre-?\s*AIA\s+)?35\s*U\.?\s*S\.?\s*C\.?\s*§?\s*(\d{3})", re.I)
STOP_HEAD = re.compile(
    r"Allowable\s+Subject\s+Matter|Double\s+Patenting|double\s+patenting|Response\s+to\s+Arguments|"
    r"Conclusion\b|Any\s+inquiry\s+concerning\s+this\s+communication|Claim\s+Objections|"
    r"would\s+be\s+allowable", re.I)


def context_events(text):
    ev = [(m.start(), m.group(1)) for m in STATEMENT.finditer(text)]
    ev += [(m.start(), None) for m in STOP_HEAD.finditer(text)]
    return sorted(ev)


# --- paragraph citations --------------------------------------------------------------

MAX_RANGE = 40  # longer spans are almost always a parse error

TRIGGER = re.compile(r"(?:¶+|\bpara(?:graph)?s?\b\.?|\bpar\.)\s*", re.I)
NUM_RUN = re.compile(r"(?:\[\s*\d{1,5}\s*\]|\d{1,5})(?:\s*(?:,|-|–|to|through|and|&|;)\s*(?:¶+\s*)?(?:\[\s*\d{1,5}\s*\]|\d{1,5}))*")
INNER_RANGE = re.compile(r"\[\s*(\d{3,5})\s*(?:-|–)\s*(\d{3,5})\s*\]")
ZERO_RUN = re.compile(r"(?<![\d./,])0\d{3}(?:\s*(?:,|;|-|–|and|&)\s*0\d{3})*(?![\d/])")
BRACKET_RUN = re.compile(r"\[\s*\d{3,5}\s*\](?:\s*(?:,|-|–|to|through|and|&)\s*(?:¶+\s*)?\[\s*\d{3,5}\s*\])*")


def norm_pub(s):
    d = re.sub(r"\D", "", s)
    return d if len(d) == 11 else None


def claim_numbers(s):
    out = []
    for a, b in RANGE.findall(s):
        a = int(a)
        if b and a <= int(b) <= a + 200:
            out.extend(range(a, int(b) + 1))
        else:
            out.append(a)
    return out


STOP_WORDS = {"us", "u", "s", "pgpub", "pg", "pub", "publication", "patent", "pat", "no", "application",
              "appl", "app", "ser", "serial", "pre", "grant", "hereinafter", "in", "view", "and", "over", "the",
              "of", "claim", "claims", "further", "see", "unpatentable", "anticipated", "by", "as", "being",
              "et", "al", "a1", "b1", "b2", "pct", "wo", "jp", "ep", "cn", "kr", "de", "fr", "gb", "document",
              "reference", "rejected", "under", "u.s.c", "usc", "is", "are", "which", "with", "to", "art",
              "english", "translation", "equivalent", "its", "machine", "family", "member", "counterpart",
              "corresponding", "version", "aia", "fig", "figs", "figure", "para", "paragraph", "col", "column"}
CAP = re.compile(r"[A-Z][A-Za-z'\-]+")
ALIAS = re.compile(r"^[^)\n]{0,40}?hereinafter\s*[\"“‘']?\s*([A-Z][A-Za-z'\-]+)", re.I)


def surname_before(text, pos):
    """Inventor surname written just before a publication number (e.g. 'Liang et al. (US ...')."""
    window = text[max(0, pos - 120):pos]
    paren = window.rfind("(")
    if paren >= 0 and ")" not in window[paren:]:
        window = window[:paren]  # 'Sudo (WO ..., English translation US 2012/...' -> name before '('
    cands = [m.group(0) for m in CAP.finditer(window) if m.group(0).lower() not in STOP_WORDS]
    return cands[-1] if cands else None


FOREIGN_BEFORE = re.compile(r"\b(?:KR|JP|CN|WO|EP|DE|TW|FR|GB|CA|AU|IN|RU|BR|ES|IT|NL|SE|CH|DK)\s*[-:]?\s*(?:10-)?\s*$")


def is_foreign(text, start):
    """True when the number is a foreign publication (e.g. 'KR 20040038469', 'WO 2009/0146371')."""
    return bool(FOREIGN_BEFORE.search(text[max(0, start - 10):start]))


def reference_index(text):
    """Map lower-cased inventor names / aliases and 3-digit suffixes to publication numbers."""
    by_name, by_suffix = {}, {}
    for m in PUB.finditer(text):
        pub = norm_pub(m.group(1) + m.group(2))
        if not pub:
            continue
        if is_foreign(text, m.start()):
            pub = "AAPA"  # foreign publication: its name is tracked so its citations are never given to a US reference
        else:
            by_suffix.setdefault(pub[-3:], pub)
        name = surname_before(text, m.start())
        if name:
            by_name.setdefault(name.lower(), pub)
        alias = ALIAS.match(text[m.end():m.end() + 60])
        if alias:
            by_name.setdefault(alias.group(1).lower(), pub)
    for m in GRANTED.finditer(text):
        name = surname_before(text, m.start())
        if name:
            by_name.setdefault(name.lower(), "GRANT")
    return by_name, by_suffix


PSEUDO = re.compile(r"\bAAPA\b|admitted\s+prior\s+art|applicant'?s?\s+(?:own\s+)?(?:admission|specification|background)|"
                    r"official\s+notice|\bAPA\b", re.I)
NEGATIVE = re.compile(r"\b(?:does|do|did)\s+not\s+(?:\w+\s+){0,2}(?:teach|disclose|show|describe|suggest|mention)|"
                      r"\bfails?\s+to\s+(?:\w+\s+){0,2}(?:teach|disclose|show|describe)|\bsilent\b|\black(?:s|ing)?\b", re.I)


def mentions(segment, by_name, by_suffix, ends=None):
    """Positions in segment where a reference is named, -> (pos, pub). Fills ends[pos] = end.
    Applicant-admitted prior art / Official Notice appear as the pseudo-reference 'AAPA'."""
    found = []
    ends = {} if ends is None else ends
    for m in PSEUDO.finditer(segment):
        found.append((m.start(), "AAPA")); ends[m.start()] = m.end()
    for m in PUB.finditer(segment):
        pub = norm_pub(m.group(1) + m.group(2))
        if pub:
            found.append((m.start(), "AAPA" if is_foreign(segment, m.start()) else pub)); ends[m.start()] = m.end()
    for m in GRANTED.finditer(segment):
        found.append((m.start(), "GRANT")); ends[m.start()] = m.end()
    for m in SHORT.finditer(segment):
        pub = by_suffix.get(m.group(2))
        if pub:
            found.append((m.start(), pub)); ends[m.start()] = m.end()
    if by_name:
        pat = re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(by_name, key=len, reverse=True)) + r")\b", re.I)
        for m in pat.finditer(segment):
            found.append((m.start(), by_name[m.group(1).lower()])); ends.setdefault(m.start(), m.end())
    return sorted(set(found))


STRICT_MAX_SPAN = 5  # strict mode: ranges wider than this are too vague to be good labels


def expand_run(run, max_span=None):
    """'80, 82-83' / '[0024]-[0026], [0030]' -> [80, 82, 83] / [24, 25, 26, 30]"""
    toks = re.findall(r"\d{1,5}|-|–|to|through", run)
    nums, i = [], 0
    while i < len(toks):
        if toks[i].isdigit():
            a = int(toks[i])
            if i + 2 < len(toks) and toks[i + 1] in ("-", "–", "to", "through") and toks[i + 2].isdigit():
                b = int(toks[i + 2])
                if max_span is not None and b - a + 1 > max_span:
                    i += 3
                    continue  # vague span: drop it
                nums.extend(range(a, b + 1) if a <= b <= a + MAX_RANGE else [a, b])
                i += 3
                continue
            nums.append(a)
        i += 1
    return nums


def paragraph_cites(segment, max_span=None):
    """(pos, [paragraph numbers]) for each paragraph citation in segment."""
    out, taken = [], []

    def free(p, q):
        return not any(x < q and p < y for x, y in taken)

    for t in TRIGGER.finditer(segment):
        m = NUM_RUN.match(segment, t.end())
        if m:
            out.append((t.start(), expand_run(m.group(0), max_span), m.end()))
            taken.append((t.start(), m.end()))
    for m in INNER_RANGE.finditer(segment):
        if free(m.start(), m.end()):
            out.append((m.start(), expand_run(m.group(1) + "-" + m.group(2), max_span), m.end()))
            taken.append((m.start(), m.end()))
    for m in BRACKET_RUN.finditer(segment):
        if free(m.start(), m.end()):
            out.append((m.start(), expand_run(m.group(0), max_span), m.end()))
            taken.append((m.start(), m.end()))
    for m in ZERO_RUN.finditer(segment):
        if free(m.start(), m.end()):
            out.append((m.start(), expand_run(m.group(0).replace(";", ","), max_span), m.end()))
            taken.append((m.start(), m.end()))
    return sorted(out)


VERB = re.compile(r"\s*(?:et\s+al\.?)?\s*(?:[‘’'`]\s*\d{3})?\s*(?:\([^)]{0,40}\))?\s*,?\s*"
                  r"(?:further\s+|also\s+|clearly\s+|explicitly\s+|expressly\s+|additionally\s+)?"
                  r"(?:teach|disclos|show|describ|suggest|illustrat|provid|recit|demonstrat)", re.I)
NEAR = 45  # a reference named this close before a citation is the one being cited


def attribute(seg, pos, refs, mention_ends, strict):
    """Which reference a paragraph citation at `pos` belongs to, or None if unsure (strict)."""
    distinct = {p for _, p in refs}
    before = [r for r in refs if r[0] <= pos]
    if len(distinct) == 1:
        return next(iter(distinct))  # only one reference in play: unambiguous
    if not before:
        return None if strict else refs[0][1]
    last_pos, last_pub = before[-1]
    if pos - mention_ends.get(last_pos, last_pos) <= NEAR:
        return last_pub  # 'Smith, ¶ [0024]' / '(Smith; para 24)'
    if strict:
        # several references: credit the one named last, only if it is in the same sentence as the citation
        between = seg[mention_ends.get(last_pos, last_pos):pos]
        if not re.search(r"\.\s+[A-Z]|\n", between):
            return last_pub
        return None
    for p, pub in reversed(before):
        if VERB.match(seg, mention_ends.get(p, p)):
            sent_start = max(seg.rfind(".", 0, p) + 1, seg.rfind("\n", 0, p) + 1, 0)
            chain = [r for r in refs if sent_start <= r[0] <= p]
            if "view of" in seg[sent_start:p] and chain:
                return chain[0][1]
            return pub
    return last_pub


SELF_AFTER = re.compile(r"^[\s\]\)]*(?:above|supra|herein|of\s+(?:this|the\s+(?:present|instant|current|pending|subject))\s+"
                        r"(?:office\s+)?(?:action|disclosure|specification|application|spec)|"
                        r"of\s+(?:the\s+)?applicant'?s?)", re.I)
SELF_BEFORE = re.compile(r"(?:instant|present|applicant'?s?|pending|subject|this)\s+(?:office\s+action|action|disclosure|"
                         r"specification|application|spec|publication)\s*(?:,|at|in|see)?\s*(?:\(\s*)?$|"
                         r"\b(?:above|supra|previously)\s*(?:,|at|in)?\s*$", re.I)


def self_citation(seg, pos, run_end):
    """True when a paragraph number points into the office action itself or the applicant's own filing."""
    return bool(SELF_AFTER.match(seg[run_end:run_end + 60]) or SELF_BEFORE.search(seg[max(0, pos - 60):pos]))


CLAUSE_CLAIMS = re.compile(r"\bclaims?\s+(\d+(?:\s*(?:-|–|to|through|,(?:\s*and)?|and|&)\s*\d+)*)", re.I)


def clause_claims(seg, pos, section_claims):
    """Section claims named in the clause around a citation (';' '.' and line breaks end a clause)."""
    a = max(seg.rfind(";", 0, pos), seg.rfind(".", 0, pos - 1), seg.rfind("\n", 0, pos))
    ends = [x for x in (seg.find(";", pos), seg.find(".", pos + 1), seg.find("\n", pos)) if x > 0]
    b = min(ends) if ends else len(seg)
    named = set()
    for m in CLAUSE_CLAIMS.finditer(seg[a + 1:b]):
        named.update(claim_numbers(m.group(1)))
    return named & set(section_claims)


LATER_CLAIM = re.compile(r"\bclaims?\s+(\d+)", re.I)


def cut_at_next_claim(seg, claims, head_end):
    """Stop a section where the examiner starts discussing a later claim mid-paragraph."""
    top = max(claims) if claims else 0
    for m in LATER_CLAIM.finditer(seg, head_end):
        if int(m.group(1)) > top:
            return seg[:m.start()]
    return seg


NAMEISH = re.compile(r"\b([A-Z][A-Za-z]{2,}\d*)\b")
NOT_NAMES = {"see", "fig", "figs", "figure", "figures", "para", "paragraph", "paragraphs", "par", "col", "column",
             "lines", "line", "page", "element", "elements", "item", "step", "steps", "table", "abstract",
             "claim", "claims", "also", "note", "and", "e.g", "i.e", "examiner", "emphasis", "added", "id",
             "ibid", "where", "which", "the", "this", "that", "with", "for", "in", "at", "least", "e", "g"}


def unknown_name_before(seg, pos, by_name):
    """True if a capitalized name that is not a known reference sits right before the citation."""
    window = seg[max(0, pos - 40):pos]
    cut = max(window.rfind("("), window.rfind(";"), window.rfind("."))
    window = window[cut + 1:] if cut >= 0 else window
    for m in NAMEISH.finditer(window):
        w = m.group(1).lower()
        if w in NOT_NAMES or w in by_name:
            continue
        return True
    return False


def parse_office_action(text, strict=True):
    """Return {claim_number: {(reference, paragraph), ...}} for 102/103 paragraph citations."""
    if not text:
        return {}
    import bisect
    by_name, by_suffix = reference_index(text)
    all_ends = {}
    all_refs = mentions(text, by_name, by_suffix, all_ends)  # once per action
    ref_pos = [p for p, _ in all_refs]
    events = context_events(text)
    ev_pos = [e[0] for e in events]
    heads = list(SECTION_HEAD.finditer(text))
    labels = OrderedDict()
    for i, h in enumerate(heads):
        k = bisect.bisect_left(ev_pos, h.start())
        if k == 0 or events[k - 1][1] not in ("102", "103"):
            continue
        seg_end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        if k < len(ev_pos) and ev_pos[k] < seg_end:
            seg_end = ev_pos[k]
        claims = claim_numbers(h.group(1))
        seg = text[h.start():seg_end]
        if strict:
            seg = cut_at_next_claim(seg, claims, h.end() - h.start())
        a, b = bisect.bisect_left(ref_pos, h.start()), bisect.bisect_left(ref_pos, h.start() + len(seg))
        refs = [(p - h.start(), pub) for p, pub in all_refs[a:b]]
        ends = {p - h.start(): all_ends[p] - h.start() for p, _ in all_refs[a:b]}
        if not refs:
            prior_start = events[k - 1][0]
            pa = bisect.bisect_left(ref_pos, prior_start)
            prev = [r for r in all_refs[pa:a] if r[1] not in ("AAPA", "GRANT")]
            refs = [(-1, prev[-1][1])] if prev else []
        if not refs:
            continue
        per_claim = {}
        for pos, nums, run_end in paragraph_cites(seg, STRICT_MAX_SPAN if strict else None):
            pub = attribute(seg, pos, refs, ends, strict)
            if pub is None or pub in ("AAPA", "GRANT"):
                continue
            if self_citation(seg, pos, run_end):
                continue  # 'paragraph 6 of this office action', 'paragraph 13 of the instant disclosure'
            if strict and unknown_name_before(seg, pos, by_name):
                continue  # a reference we could not identify is named next to this citation
            if strict:
                s0 = max(seg.rfind(".", 0, pos - 1), seg.rfind("\n", 0, pos))
                s1 = seg.find(".", pos + 1)
                if NEGATIVE.search(seg[s0 + 1:s1 if s1 > 0 else len(seg)]):
                    continue  # citation sits in a 'does not teach' sentence
            # in a section covering several claims, a citation tied to particular claims goes to those claims only
            targets = (clause_claims(seg, pos, claims) if len(claims) > 1 else set()) or claims
            for c in targets:
                per_claim.setdefault(c, set()).update((pub, n) for n in nums if 0 < n < 2000)
        for c, pairs in per_claim.items():
            if pairs:
                labels.setdefault(c, set()).update(pairs)
    return labels


# --- allowable claims (the "no reference" class) ---------------------------------------

ALLOWED = re.compile(
    r"claims?\s+(\d+(?:\s*(?:-|–|to|through|,(?:\s*and)?|and|&)\s*\d+)*)\s*(?:is|are|is/are)\s+(?:hereby\s+)?allowed", re.I)
ALLOWABLE = re.compile(
    r"claims?\s+(\d+(?:\s*(?:-|–|to|through|,(?:\s*and)?|and|&)\s*\d+)*)\s*(?:is|are|is/are)\s+objected\s+to\s+as\s+being\s+"
    r"dependent\s+upon\s+a\s+rejected\s+base\s+claim,?\s+but\s+would\s+be\s+allowable", re.I)


def allowable_claims(text):
    """Claims the examiner found allowable over the prior art in this action.
    Returns {claim: 'allowed' | 'allowable_if_rewritten'}. The second kind are dependent
    claims whose added limitation was not found in the art (allowable together with their parents)."""
    out = {}
    if not text:
        return out
    for m in ALLOWED.finditer(text):
        for c in claim_numbers(m.group(1)):
            out[c] = "allowed"
    for m in ALLOWABLE.finditer(text):
        for c in claim_numbers(m.group(1)):
            out.setdefault(c, "allowable_if_rewritten")
    return out
