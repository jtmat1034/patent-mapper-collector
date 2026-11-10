# Patent claim → prior-art paragraph dataset

Built from public USPTO Open Data Portal data (public domain). Every row is one rejected patent claim paired with
the paragraphs the examiner cited against it.

## Task

Input: a rejected claim, with the full text of every claim it depends on. Output: one (reference publication,
paragraph) that the examiner cited against it. Any one of the cited pairs counts as correct. The answer options are
every paragraph in `corpus/`, so the model must find both the right reference and the right paragraph; it should
score the claim against each paragraph's text.

## Files

| File | Contents |
|---|---|
| `train/*.jsonl.gz` | training claims (~95%) |
| `test/*.jsonl.gz` | held-out claims (~5%), split by patent family, so related applications never appear in both |
| `corpus/*.jsonl.gz` | the answer bank: every paragraph of every cited publication, one line per publication: `{"pub", "kind", "paragraphs": {"1": text, ...}}` |
| `evaluate.py` | scorer: `python3 evaluate.py predictions.jsonl test/` prints one accuracy number |
| `stats.json` | counts from the build |

Other folders in the repository (`work/`, `state/`, `audit/`) are build internals and are not needed for training.

## Row fields

- `id`: `<application>-<claim number>`
- `claim_text_with_parents`: the claim with every claim it depends on, root first. Use this as the model input.
- `claim_text`: the claim alone, as published.
- `answers`: list of `[publication, paragraph]` the examiner cited against this claim. `publication` is the
  11-digit US pre-grant publication number (`20150109202` = US 2015/0109202); `paragraph` is the printed paragraph
  number ([0009] → 9) and is a key in that publication's `corpus` entry. Every answer exists in the corpus.
- `quote_verified`: the answers where the examiner quoted the paragraph and the quote was found in its text.
- `independent`, `application`, `publication` (the claim's own application), `office_action_date`, `art_unit`,
  `family`.

## How it was built (strict: any ambiguity drops the whole office action)

1. Office actions: USPTO Office Action weekly archives (July 2007 onward). Only an application's first
   substantive action is used, and only when it is a non-final rejection with no sign of a prior amendment, so the
   rejected claims are the published claims.
2. Citations come only from 35 U.S.C. 102/103 rejections. A paragraph citation is kept only when the reference it
   belongs to is named unambiguously right before it; if any citation in an office action cannot be attributed
   with certainty (unnamed, ambiguous name, unknown reference, vague range of more than 5 paragraphs, claim not in
   a rejection statement), the whole office action is dropped. Self-citations are ignored.
3. Only US pre-grant publications are answers. Citations to granted patents, foreign references and admitted prior
   art are not used as answers (their text is not in the corpus).
4. Claims and paragraphs: USPTO Patent Application Full-Text XML (all pre-grant publications since 2001).
   A claim is kept only if its full dependency chain is present and every cited paragraph exists in the corpus.
5. Where the examiner quoted a cited paragraph, the quote is checked against that paragraph's text; one mismatch
   drops the whole application. Quotes that only repeat the claim's own wording are ignored.

## Things to know

- Measured accuracy is a lower bound: the examiner's citations are one correct set, not the only one.
- The corpus ignores dates: every paragraph is an option for every claim.
- A hand check of 1,000 office actions found the strict parser's kept citations 111 correct, 0 wrong, 1 unclear.
