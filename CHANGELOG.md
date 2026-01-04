# Changelog

## [Unreleased]

### Added
- German / multilingual retrieval support:
  - `retrieval.language` configuration option (e.g., `de` for German).
  - BM25 tokenization updated to use Snowball stemming for German when available (uses `nltk`).
  - Claim-check and grounded prompts localized for German when `retrieval.language` starts with `de`.
  - README and `config.example.yaml` updated with guidance for German-language corpora and recommended multilingual embedding model.
  - Tests added: `test_retrieval_german.py`, `test_integration_claim_check_de.py`.

### Changed
- Default recommended embedding in `config.example.yaml` updated to a multilingual model (`paraphrase-multilingual-MiniLM-L12-v2`).

### Notes
- NLTK is an optional dependency used for German stemming; BM25 will still work without it (but without stemming).
