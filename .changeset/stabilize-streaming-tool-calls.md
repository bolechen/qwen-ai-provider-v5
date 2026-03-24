---
"qwen-ai-provider-v5": patch
---

Stabilize streaming tool-calls and reasoning

- Add retry logic for Qwen streaming API transient 500 errors (`list index out of range`)
- Support `reasoning` field alongside `reasoning_content` for compatibility
- Fix incomplete tool-call assembly during streaming by buffering and reassembling fragmented chunks
- Improve in-stream error detection and recovery for known Qwen boundary failures
