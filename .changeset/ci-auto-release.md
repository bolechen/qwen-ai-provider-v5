---
"qwen-ai-provider-v5": patch
---

### Improvements

- **CI/CD**: Configure automated npm publishing via GitHub Actions
  - Add Changesets-based release workflow for automatic versioning
  - Trigger releases automatically when changes are merged to main
  - Simplify pre-release workflow for dry-run testing

- **Reranking Model**: Add support for Qwen reranking models
  - New `QwenRerankingModel` for document reranking tasks
  - Support for `gte-rerank` model family

