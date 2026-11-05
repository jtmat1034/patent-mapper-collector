# Patent claim → prior-art citation dataset

Built from public USPTO Open Data Portal data (public domain). Every row is one patent claim
paired with the correct answer taken from the examiner's office action.

## Task

Input: a claim. Output: one (reference publication, paragraph) that the examiner cited against
it, or `no_reference` if the examiner found the claim allowable. Any one of the cited pairs
counts as correct. The answer options are every paragraph in `corpus/`.

## Files

| File | Contents |
|---|---|
| `train.jsonl.gz` | training claims (~95%) |
| `test.jsonl.gz` | held-out claims (~5%), split by patent family so continuations never appear in both |
| `corpus/*.jsonl.gz` | every paragraph of every cited publication: `{"pub", "kind", "paragraphs": {"1": text, ...}}` |
| `evaluate.py` | scorer: `python3 evaluate.py predictions.jsonl` prints one accuracy number |
| `stats.json` | counts from the build |

## Row fields

- `id` — `<application>-<claim number>`
- `claim_text` — the claim as published
- `claim_text_with_parents` — a dependent claim with every claim it depends on, root first.
  Use this as the model input: an allowable dependent claim is only allowable together with its parents.
- `label` — `rejected`, `allowed`, or `allowable_if_rewritten`
- `answers` — list of `[publication, paragraph]` the examiner cited (empty for the two allowable labels).
  `publication` is the 11-digit US pre-grant publication number (e.g. `20150109202` = US 2015/0109202);
  `paragraph` is the publication's printed paragraph number ([0009] → 9) and is a key in that publication's
  `corpus` entry.
- `application`, `publication` (of the claim's own application), `office_action_date`, `art_unit`, `family`

## How it was built

1. Office actions: first-action non-final rejections from the USPTO Office Actions Weekly Archives (2020+).
   Actions that respond to an amendment are skipped, so the examined claims are the published claims.
   Citations come only from 35 U.S.C. 102/103 rejections, attributed to a reference only when unambiguous.
   Hand-checked precision ~95%.
2. Claims and paragraphs: USPTO Patent Application Full-Text XML (all pre-grant publications since 2001).
3. A claim is kept only if its text exists and at least one cited paragraph exists in the corpus
   (or it has an allowable label). Citations to granted patents (column/line) and foreign references are not used.
