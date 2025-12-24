---
"qwen-ai-provider-v5": major
---

## Breaking Changes - AI SDK v6 Support

This is a major version release that adds support for AI SDK v6 (package versions `@ai-sdk/provider@^3.0.0` and `@ai-sdk/provider-utils@^4.0.0`).

### Breaking Changes

- **Minimum AI SDK Version**: Now requires AI SDK v6 (`ai@^6.0.0`)
- **LanguageModel API**: Updated from `LanguageModelV2` to `LanguageModelV3`
- **EmbeddingModel API**: Updated from `EmbeddingModelV2` to `EmbeddingModelV3`
- **FinishReason Format**: Changed from string to object `{ unified: string, raw: string | undefined }`
- **Usage Format**: Changed to nested structure with `inputTokens: { total, noCache, cacheRead, cacheWrite }` and `outputTokens: { total, text, reasoning }`
- **Warnings Format**: Changed from `type: "unsupported-setting"` to `type: "unsupported"` with `feature` property

### Migration

If you are using AI SDK v5, please continue using `qwen-ai-provider-v5@^1.0.0`.
