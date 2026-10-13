import type {
    LanguageModelV3,
    LanguageModelV3CallOptions,
    LanguageModelV3StreamPart,
} from "@ai-sdk/provider"
import { stripHallucinationsFromString } from "../messages/utils"

// A full compact tag including priority suffix is at most ~20 chars; 64 gives margin.
const HOLD_BACK = 64

export function wrapLanguageModel(language: LanguageModelV3): LanguageModelV3 {
    return {
        ...language,
        doGenerate: async (options: LanguageModelV3CallOptions) => {
            const result = await language.doGenerate(options)
            const content = result.content.map((item) =>
                item.type === "text"
                    ? { ...item, text: stripHallucinationsFromString(item.text, "compact") }
                    : item,
            )
            return { ...result, content }
        },
        doStream: async (options: LanguageModelV3CallOptions) => {
            const result = await language.doStream(options)
            const buffers = new Map<string, string>()
            async function* stripped(): AsyncGenerator<LanguageModelV3StreamPart> {
                for await (const part of result.stream) {
                    if (part.type === "text-delta") {
                        const buffer = (buffers.get(part.id) ?? "") + part.delta
                        if (buffer.length > HOLD_BACK) {
                            const safe = stripHallucinationsFromString(
                                buffer.slice(0, -HOLD_BACK),
                                "compact",
                            )
                            if (safe) yield { ...part, delta: safe }
                            buffers.set(part.id, buffer.slice(-HOLD_BACK))
                        } else {
                            buffers.set(part.id, buffer)
                        }
                        continue
                    }
                    if (part.type === "text-end") {
                        const buffer = buffers.get(part.id)
                        if (buffer !== undefined) {
                            const safe = stripHallucinationsFromString(buffer, "compact")
                            if (safe) yield { type: "text-delta", id: part.id, delta: safe }
                            buffers.delete(part.id)
                        }
                        yield part
                        continue
                    }
                    if (part.type === "finish" || part.type === "error") {
                        // Flush unstripped: on abnormal ends, guarantee no text loss.
                        for (const [id, buffer] of buffers)
                            yield { type: "text-delta", id, delta: buffer }
                        buffers.clear()
                        yield part
                        continue
                    }
                    yield part
                }
            }
            return { ...result, stream: ReadableStream.from(stripped()) }
        },
    }
}
