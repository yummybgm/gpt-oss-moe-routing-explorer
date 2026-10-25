# Local publication-candidate QA

Checked on 2026-10-25 before any GitHub upload.

## Passed checks

- Python syntax: all 23 public `.py` files were parsed successfully with Python's AST parser.
- JavaScript syntax: `node --check explorer/frontend/app.js` passed for the full local explorer frontend.
- JSON parsing: all 5 public JSON artifacts parsed successfully.
- Citation metadata: `CITATION.cff` contains CFF 1.2.0, author `He Zhiyun`, repository URL, software type, and MIT license metadata.
- Privacy/path scan: no machine-specific Windows or server absolute paths, API keys, auth tokens, SSH aliases, or DeepSeek/Anthropic credentials were found in the candidate tree.
- File-size check: largest file is below 1 MB; no GitHub large-file threshold issue.
- Data-scope check: no full raw routing arrays, raw prompts/corpora, raw SAE activation cache, API responses, or unvalidated 849-feature annotations are included.
- Generated-file cleanup: 4 `__pycache__` directories containing 23 compiled files were removed from the publication candidate.
- SSH runtime smoke test: the retained private-data explorer returned `status=ok` and `loaded=true`; read-only queries returned a 24-layer trajectory, 24 layers of Top-4 routing with 4 selections per layer, expert contexts, and SAE feature rows.

## Known limitation

The complete explorer requires private routing caches and indexes that are intentionally excluded from Git. The current Windows Python environment does not contain the declared FastAPI dependency, so local backend import testing was not claimed. The backend was instead smoke-tested against the prepared SSH/private-data environment. A final browser visual check remains in `PUBLICATION_CHECKLIST.md` because API correctness alone does not verify page layout and interaction behavior.

The candidate currently contains no collected automated unit tests. Static parsing, artifact validation, privacy scanning, and the required manual explorer smoke test therefore form the present release gate.

## Publication status

This is a local public-repository candidate only. It has not been uploaded to GitHub. MIT licensing and public citation metadata are complete. Remote repository creation, the GitHub Desktop commit preview, the private-data explorer smoke test, and explicit upload approval remain pending.
