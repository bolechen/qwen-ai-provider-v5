/**
 * Supported reranking model IDs.
 * - gte-rerank: Legacy GTE reranking models
 * - qwen3-reranker: Qwen3 Reranker models (0.6B, 4B, 8B)
 */
export type QwenRerankingModelId
  = | "gte-rerank"
    | "gte-rerank-v2"
    | "gte-rerank-hybrid-v1"
    | "qwen3-reranker-0.6b"
    | "qwen3-reranker-4b"
    | "qwen3-reranker-8b"
    | (string & {})

/**
 * Settings configuration for Qwen reranking models.
 */
export interface QwenRerankingSettings {
  /**
   * Whether to return the documents in the response.
   * Default is false.
   */
  returnDocuments?: boolean
}
