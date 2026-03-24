---
"qwen-ai-provider-v5": patch
---

Fix reasoning content parts in assistant messages breaking multi-turn conversations (fixes [#8](https://github.com/bolechen/qwen-ai-provider-v5/issues/8))

- Silently ignore `reasoning` and `file` parts in assistant messages instead of throwing `UnsupportedFunctionalityError`
- Models like GLM-4.7 return reasoning parts that were causing dialogue interruption when passed back in subsequent API calls
