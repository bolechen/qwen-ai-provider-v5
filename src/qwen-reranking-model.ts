import type { RerankingModelV3 } from "@ai-sdk/provider"
import type { FetchFunction } from "@ai-sdk/provider-utils"
import type { QwenErrorStructure } from "./qwen-error"
import type {
  QwenRerankingModelId,
  QwenRerankingSettings,
} from "./qwen-reranking-settings"
import {
  combineHeaders,
  createJsonErrorResponseHandler,
  createJsonResponseHandler,
  postJsonToApi,
} from "@ai-sdk/provider-utils"
import { z } from "zod"
import { defaultQwenErrorStructure } from "./qwen-error"

interface QwenRerankingConfig {
  provider: string
  url: (options: { modelId: string, path: string }) => string
  headers: () => Record<string, string | undefined>
  fetch?: FetchFunction
  errorStructure?: QwenErrorStructure<any>
}

/**
 * Response schema for DashScope reranking API.
 * The API returns results sorted by relevance score in descending order.
 */
const qwenRerankingResponseSchema = z.object({
  results: z.array(
    z.object({
      index: z.number(),
      relevance_score: z.number(),
      document: z
        .object({
          text: z.string(),
        })
        .optional(),
    }),
  ),
  usage: z
    .object({
      total_tokens: z.number().optional(),
    })
    .optional(),
})

export class QwenRerankingModel implements RerankingModelV3 {
  readonly specificationVersion = "v3"
  readonly modelId: QwenRerankingModelId

  private readonly config: QwenRerankingConfig
  private readonly settings: QwenRerankingSettings

  get provider(): string {
    return this.config.provider
  }

  constructor(
    modelId: QwenRerankingModelId,
    settings: QwenRerankingSettings,
    config: QwenRerankingConfig,
  ) {
    this.modelId = modelId
    this.settings = settings
    this.config = config
  }

  /**
   * Reranks documents based on their relevance to the query.
   *
   * @param options - The reranking options.
   * @param options.documents - Documents to rerank (text or objects).
   * @param options.query - The query to rank documents against.
   * @param options.topN - Optional limit on number of results.
   * @param options.headers - Optional HTTP headers.
   * @param options.abortSignal - Optional abort signal.
   * @param options.providerOptions - Additional provider-specific options.
   * @returns Promise with ranking results sorted by relevance.
   */
  async doRerank(
    options: Parameters<RerankingModelV3["doRerank"]>[0],
  ): Promise<Awaited<ReturnType<RerankingModelV3["doRerank"]>>> {
    const { documents, query, topN, headers, abortSignal, providerOptions }
      = options

    // Convert documents to the format expected by DashScope API
    const documentTexts
      = documents.type === "text"
        ? documents.values
        : documents.values.map(doc => JSON.stringify(doc))

    // Get provider-specific options if available
    const providerOptionsName = this.config.provider.split(".")[0].trim()
    const specificProviderOptions = providerOptions?.[providerOptionsName]

    // Post the JSON payload to the API endpoint.
    const { responseHeaders, value: response } = await postJsonToApi({
      url: this.config.url({
        path: "/rerank",
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), headers),
      body: {
        model: this.modelId,
        query,
        documents: documentTexts,
        top_n: topN,
        return_documents: this.settings.returnDocuments,
        ...specificProviderOptions,
      },
      failedResponseHandler: createJsonErrorResponseHandler(
        this.config.errorStructure ?? defaultQwenErrorStructure,
      ),
      successfulResponseHandler: createJsonResponseHandler(
        qwenRerankingResponseSchema,
      ),
      abortSignal,
      fetch: this.config.fetch,
    })

    // Map response to V3 format
    return {
      ranking: response.results.map(result => ({
        index: result.index,
        relevanceScore: result.relevance_score,
      })),
      response: {
        headers: responseHeaders,
      },
      warnings: [],
    }
  }
}
