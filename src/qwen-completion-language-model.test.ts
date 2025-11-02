/* eslint-disable dot-notation */
import type { LanguageModelV2Prompt } from "@ai-sdk/provider"
import { createTestServer } from "@ai-sdk/provider-utils/test"
import { describe, expect, it, vi } from "vitest"
import { QwenCompletionLanguageModel } from "./qwen-completion-language-model"
import { createQwen } from "./qwen-provider"

const TEST_PROMPT: LanguageModelV2Prompt = [
  { role: "user", content: [{ type: "text", text: "Hello" }] },
]

vi.stubEnv("DASHSCOPE_API_KEY", "test-api-key-123")

const server = createTestServer({
  "https://my.api.com/v1/completions": {
    response: {
      type: "json-value",
      body: {
        id: "cmpl-test",
        object: "text_completion",
        created: 1711115037,
        model: "qwen-completion",
        choices: [
          {
            text: "Hello, World!",
            finish_reason: "stop",
            index: 0,
          },
        ],
        usage: {
          prompt_tokens: 4,
          completion_tokens: 30,
        },
      },
    },
  },
})

const provider = createQwen({
  baseURL: "https://my.api.com/v1/",
  headers: {
    Authorization: `Bearer test-api-key`,
  },
  fetch: async (url, init) => {
    const response = server.urls["https://my.api.com/v1/completions"].response
    if (response && response.type === "json-value") {
      return new Response(JSON.stringify(response.body), {
        headers: response.headers || {},
      })
    }
    throw new Error("Invalid response type")
  },
})

const model = provider.completion("qwen-completion")

describe("doGenerate", () => {
  it("should extract text response from content array", async () => {
    const { content } = await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(content).toContainEqual({
      type: "text",
      text: "Hello, World!",
    })
  })

  it("should extract usage with V2 field names", async () => {
    const { usage } = await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(usage).toMatchObject({
      inputTokens: 4,
      outputTokens: 30,
    })
  })
})

// NOTE: Full test suite needs migration to V2 API
