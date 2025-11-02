/* eslint-disable dot-notation */
import type { LanguageModelV2Prompt } from "@ai-sdk/provider"
import {
  convertReadableStreamToArray,
  createTestServer,
} from "@ai-sdk/provider-utils/test"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { QwenChatLanguageModel } from "./qwen-chat-language-model"
import { createQwen } from "./qwen-provider"

const TEST_PROMPT: LanguageModelV2Prompt = [
  { role: "user", content: [{ type: "text", text: "Hello" }] },
]

vi.stubEnv("DASHSCOPE_API_KEY", "test-api-key-123")

const server = createTestServer({
  "https://my.api.com/v1/chat/completions": {
    response: {
      type: "json-value",
      body: {
        id: "chatcmpl-test",
        object: "chat.completion",
        created: 1711115037,
        model: "qwen-chat",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: "Hello, World!",
            },
            finish_reason: "stop",
          },
        ],
        usage: {
          prompt_tokens: 4,
          total_tokens: 34,
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
    const response = server.urls["https://my.api.com/v1/chat/completions"].response
    if (typeof response === "function") {
      return new Response(JSON.stringify(response({ callNumber: server.calls.length })))
    }
    if (Array.isArray(response)) {
      return new Response(JSON.stringify(response[server.calls.length] || response[0]))
    }
    if (response && response.type === "json-value") {
      return new Response(JSON.stringify(response.body), {
        headers: response.headers || {},
      })
    }
    throw new Error("Invalid response type")
  },
})

const model = provider("qwen-chat")

describe("config", () => {
  it("should extract base name from provider string", () => {
    const model = new QwenChatLanguageModel(
      "qwen-plus",
      {},
      {
        provider: "qwen.beta",
        url: () => "",
        headers: () => ({}),
      },
    )

    expect(model["providerOptionsName"]).toBe("qwen")
  })

  it("should handle provider without dot notation", () => {
    const model = new QwenChatLanguageModel(
      "qwen-plus",
      {},
      {
        provider: "qwen-plus",
        url: () => "",
        headers: () => ({}),
      },
    )

    expect(model["providerOptionsName"]).toBe("qwen-plus")
  })

  it("should return empty for empty provider", () => {
    const model = new QwenChatLanguageModel(
      "qwen-plus",
      {},
      {
        provider: "",
        url: () => "",
        headers: () => ({}),
      },
    )

    expect(model["providerOptionsName"]).toBe("")
  })
})

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
// The tests above demonstrate the V2 patterns:
// 1. Use createTestServer instead of JsonTestServer/StreamingTestServer
// 2. Check content array instead of text field
// 3. Use inputTokens/outputTokens instead of promptTokens/completionTokens
// 4. No mode parameter in doGenerate
// 5. tools and toolChoice are direct parameters, not in mode
