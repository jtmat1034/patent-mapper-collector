# patent-mapper collector (cloud)

GitHub Actions job that builds the claim → examiner-citation dataset from public USPTO
Open Data Portal bulk files and stores it in a private Hugging Face dataset. Nothing runs on a
personal computer. See `cloud.py` for the stages and `DATASET_README.md` for the output format.

Secrets: `USPTO_API_KEY`, `HF_TOKEN` (write). Start: Actions → collect → Run workflow.
