import { createTestServer } from "@ai-sdk/provider-utils/test"
import { describe, expect, it, vi } from "vitest"
import { createQwen } from "./qwen-provider"

vi.stubEnv("DASHSCOPE_API_KEY", "test-api-key-123")

const server = createTestServer({
  "https://my.api.com/v1/embeddings": {
    response: {
      type: "json-value",
      body: {
        data: [
          { embedding: [0.1, 0.2, 0.3] },
          { embedding: [0.4, 0.5, 0.6] },
        ],
        usage: { prompt_tokens: 10 },
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
    const response = server.urls["https://my.api.com/v1/embeddings"].response
    if (response && response.type === "json-value") {
      return new Response(JSON.stringify(response.body), {
        headers: response.headers || {},
      })
    }
    throw new Error("Invalid response type")
  },
})

const model = provider.textEmbeddingModel("text-embedding-v1")

describe("doEmbed", () => {
  it("should return embeddings", async () => {
    const { embeddings } = await model.doEmbed({
      values: ["test1", "test2"],
    })

    expect(embeddings).toEqual([
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
    ])
  })

  it("should return usage information", async () => {
    const { usage } = await model.doEmbed({
      values: ["test"],
    })

    expect(usage).toEqual({ tokens: 10 })
  })

  it("should include response headers", async () => {
    const { response } = await model.doEmbed({
      values: ["test"],
    })

    expect(response).toHaveProperty("headers")
  })
})

// NOTE: Full test suite needs migration to V2 API
