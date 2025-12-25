import type { LanguageModelV3FinishReason } from "@ai-sdk/provider"

/**
 * Maps the finish reason from the backend response to a standardized format.
 *
 * @param finishReason - The original finish reason string.
 * @returns The mapped LanguageModelV3FinishReason.
 */
export function mapQwenFinishReason(
  finishReason: string | null | undefined,
): LanguageModelV3FinishReason {
  let unified: LanguageModelV3FinishReason["unified"]

  switch (finishReason) {
    case "stop":
      unified = "stop"
      break
    case "length":
      unified = "length"
      break
    case "tool_calls":
      unified = "tool-calls"
      break
    case "content_filter":
      unified = "content-filter"
      break
    default:
      unified = "other"
  }

  return {
    unified,
    raw: finishReason ?? undefined,
  }
}
