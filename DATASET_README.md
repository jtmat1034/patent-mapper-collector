# Patent claim → prior-art citation dataset

Built from public USPTO Open Data Portal data (public domain). Every row is one patent claim
paired with the correct answer taken from the examiner's office action.

## Task

Input: a rejected claim (with every claim it depends on). Output: one (reference publication, paragraph) that
the examiner cited against it. Any one of the cited pairs counts as correct. The answer options are every
paragraph in `corpus/`, and a model should score the claim against each paragraph's text. Only claims the
examiner rejected under 35 U.S.C. 102/103 are included.

## Files

| File | Contents |
|---|---|
| `train/*.jsonl.gz` | training claims (~95%) |
| `test/*.jsonl.gz` | held-out claims (~5%), split by patent family so continuations never appear in both |
| `corpus/*.jsonl.gz` | every paragraph of every cited publication: `{"pub", "kind", "paragraphs": {"1": text, ...}}` |
| `evaluate.py` | scorer: `python3 evaluate.py predictions.jsonl` prints one accuracy number |
| `stats.json` | counts from the build |

## Row fields

- `id` — `<application>-<claim number>`
- `claim_text` — the claim as published
- `claim_text_with_parents` — a dependent claim with every claim it depends on, root first.
  Use this as the model input.
- `answers` — list of `[publication, paragraph]` the examiner cited against this claim.
- `quote_verified` — the answers where the examiner quoted the paragraph and the quote was found in its text.
  `publication` is the 11-digit US pre-grant publication number (e.g. `20150109202` = US 2015/0109202);
  `paragraph` is the publication's printed paragraph number ([0009] → 9) and is a key in that publication's
  `corpus` entry.
- `application`, `publication` (of the claim's own application), `office_action_date`, `art_unit`, `family`

## How it was built

1. Office actions: first-action non-final rejections from the USPTO Office Actions Weekly Archives (2020+).
   Actions that respond to an amendment are skipped, so the examined claims are the published claims.
   Citations come only from 35 U.S.C. 102/103 rejections, attributed to a reference only when unambiguous.
   Self-citations (to the office action or the applicant's own disclosure) are ignored.
2. Claims and paragraphs: USPTO Patent Application Full-Text XML (all pre-grant publications since 2001).
3. A claim is kept only if its text exists and at least one cited paragraph exists in the corpus
   Citations to granted patents (column/line) and foreign references are not used.
4. Where the examiner quoted a cited paragraph, the quote is checked against that paragraph's text; one mismatch
   drops every claim from that office action.
