import { beforeEach, describe, expect, it, vi } from "vitest"
import { createQwen } from "../qwen-provider"

const dummyRankingResults = [
  { index: 1, relevance_score: 0.95 },
  { index: 0, relevance_score: 0.72 },
  { index: 2, relevance_score: 0.31 },
]

vi.stubEnv("DASHSCOPE_API_KEY", "test-api-key-123")

const testDocuments = [
  "sunny day at the beach",
  "rainy afternoon in the city",
  "snowy night in the mountains",
]

describe("doRerank", () => {
  let requestUrl: string
  let requestBody: any
  let requestHeaders: Record<string, string>
  let responseBody: any
  let responseHeaders: Record<string, string>

  beforeEach(() => {
    requestUrl = ""
    requestBody = undefined
    requestHeaders = {}
    // DashScope native API response format
    responseBody = {
      output: {
        results: dummyRankingResults,
      },
      usage: { total_tokens: 100 },
      request_id: "test-request-id-123",
    }
    responseHeaders = {
      "content-type": "application/json",
    }
  })

  function createTestProvider(overrides?: any) {
    return createQwen({
      baseURL: "https://my.api.com/compatible-mode/v1",
      headers: {
        Authorization: `Bearer test-api-key`,
      },
      ...overrides,
      fetch: async (url, init) => {
        // Capture request
        requestUrl = url as string
        if (init?.body) {
          requestBody = JSON.parse(init.body as string)
        }
        if (init?.headers) {
          requestHeaders = init.headers as Record<string, string>
        }

        // Return response
        const bodyLength = JSON.stringify(responseBody).length
        return new Response(JSON.stringify(responseBody), {
          headers: {
            ...responseHeaders,
            "content-length": bodyLength.toString(),
          },
        })
      },
    })
  }

  it("should use DashScope native API endpoint (not compatible mode)", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("gte-rerank-v2")

    await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
    })

    // Should use native API endpoint, not compatible mode
    expect(requestUrl).toBe(
      "https://my.api.com/api/v1/services/rerank/text-rerank/text-rerank",
    )
  })

  it("should extract ranking results", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("gte-rerank-v2")

    const { ranking } = await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
    })

    expect(ranking).toStrictEqual([
      { index: 1, relevanceScore: 0.95 },
      { index: 0, relevanceScore: 0.72 },
      { index: 2, relevanceScore: 0.31 },
    ])
  })

  it("should return request_id in response", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("gte-rerank-v2")

    const { response } = await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
    })

    expect(response?.id).toBe("test-request-id-123")
  })

  it("should expose the raw response headers", async () => {
    responseHeaders = {
      ...responseHeaders,
      "test-header": "test-value",
    }

    const provider = createTestProvider()
    const model = provider.rerankingModel("gte-rerank-v2")

    const { response } = await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
    })

    expect(response?.headers).toMatchObject({
      "content-type": "application/json",
      "test-header": "test-value",
    })
  })

  it("should pass the model, query, and documents in DashScope native format", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("gte-rerank-v2")

    await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
    })

    expect(requestBody).toStrictEqual({
      model: "gte-rerank-v2",
      input: {
        query: "talk about rain",
        documents: testDocuments,
      },
      parameters: {},
    })
  })

  it("should pass topN parameter in parameters object", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("gte-rerank-v2")

    await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
      topN: 2,
    })

    expect(requestBody).toStrictEqual({
      model: "gte-rerank-v2",
      input: {
        query: "talk about rain",
        documents: testDocuments,
      },
      parameters: {
        top_n: 2,
      },
    })
  })

  it("should pass returnDocuments setting in parameters object", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("gte-rerank-v2", {
      returnDocuments: true,
    })

    await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
    })

    expect(requestBody).toStrictEqual({
      model: "gte-rerank-v2",
      input: {
        query: "talk about rain",
        documents: testDocuments,
      },
      parameters: {
        return_documents: true,
      },
    })
  })

  it("should handle object documents by stringifying them", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("gte-rerank-v2")

    const objectDocs = [
      { title: "Beach Day", content: "sunny day at the beach" },
      { title: "City Rain", content: "rainy afternoon in the city" },
    ]

    await model.doRerank({
      documents: { type: "object", values: objectDocs },
      query: "talk about rain",
    })

    expect(requestBody.input.documents).toStrictEqual([
      JSON.stringify(objectDocs[0]),
      JSON.stringify(objectDocs[1]),
    ])
  })

  it("should pass headers", async () => {
    const provider = createTestProvider({
      headers: {
        "Authorization": `Bearer test-api-key`,
        "Custom-Provider-Header": "provider-header-value",
      },
    })

    await provider.rerankingModel("gte-rerank-v2").doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
      headers: {
        "Custom-Request-Header": "request-header-value",
      },
    })

    expect(requestHeaders).toMatchObject({
      "authorization": "Bearer test-api-key",
      "content-type": "application/json",
      "custom-provider-header": "provider-header-value",
      "custom-request-header": "request-header-value",
    })
  })

  it("should work with qwen3-reranker models", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("qwen3-reranker-0.6b")

    await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
    })

    expect(requestBody.model).toBe("qwen3-reranker-0.6b")
  })

  it("should return empty warnings array", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("gte-rerank-v2")

    const { warnings } = await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
    })

    expect(warnings).toStrictEqual([])
  })
})
