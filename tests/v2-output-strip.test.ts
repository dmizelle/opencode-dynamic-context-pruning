import assert from "node:assert/strict"
import test from "node:test"
import type { LanguageModelV3, LanguageModelV3StreamPart } from "@ai-sdk/provider"
import { wrapLanguageModel } from "../lib/v2/strip-output-tags"

function fakeLanguage(stream: LanguageModelV3StreamPart[]): LanguageModelV3 {
    return {
        specificationVersion: "v3",
        provider: "lab",
        modelId: "test",
        supportedUrls: {},
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        doGenerate: async () => ({}) as never,
        doStream: async () => ({ stream: ReadableStream.from(stream) }),
    } as LanguageModelV3
}

async function collectText(language: LanguageModelV3): Promise<string> {
    const result = await language.doStream({} as never)
    let text = ""
    for await (const part of result.stream as AsyncIterable<LanguageModelV3StreamPart>)
        if (part.type === "text-delta") text += part.delta
    return text
}

test("doGenerate strips a trailing compact tag", async () => {
    const language: LanguageModelV3 = {
        ...fakeLanguage([]),
        doGenerate: async () =>
            ({
                content: [{ type: "text", text: "Here you go.\n@15@" }],
                finishReason: { unified: "stop", raw: "stop" },
                usage: {
                    inputTokens: { total: 0, noCache: 0, cacheRead: 0, cacheWrite: 0 },
                    outputTokens: { total: 0, text: 0, reasoning: 0 },
                },
                warnings: [],
            }) as never,
    }
    const result = await wrapLanguageModel(language).doGenerate({} as never)
    assert.equal((result.content[0] as { type: "text"; text: string }).text, "Here you go.\n")
    assert.notEqual(result.content[0], (await language.doGenerate({} as never)).content[0])
})

test("doGenerate strips a tag with priority suffix", async () => {
    const language: LanguageModelV3 = {
        ...fakeLanguage([]),
        doGenerate: async () =>
            ({
                content: [{ type: "text", text: "Answer.\n@7@ [high]" }],
                finishReason: { unified: "stop", raw: "stop" },
                usage: {
                    inputTokens: { total: 0, noCache: 0, cacheRead: 0, cacheWrite: 0 },
                    outputTokens: { total: 0, text: 0, reasoning: 0 },
                },
                warnings: [],
            }) as never,
    }
    const result = await wrapLanguageModel(language).doGenerate({} as never)
    assert.equal((result.content[0] as { type: "text"; text: string }).text, "Answer.\n")
})

test("doGenerate leaves clean text unchanged", async () => {
    const language: LanguageModelV3 = {
        ...fakeLanguage([]),
        doGenerate: async () =>
            ({
                content: [{ type: "text", text: "Plain response." }],
                finishReason: { unified: "stop", raw: "stop" },
                usage: {
                    inputTokens: { total: 0, noCache: 0, cacheRead: 0, cacheWrite: 0 },
                    outputTokens: { total: 0, text: 0, reasoning: 0 },
                },
                warnings: [],
            }) as never,
    }
    const result = await wrapLanguageModel(language).doGenerate({} as never)
    assert.equal((result.content[0] as { type: "text"; text: string }).text, "Plain response.")
})

test("doStream strips a trailing tag delivered in deltas", async () => {
    const language = fakeLanguage([
        { type: "stream-start", warnings: [] },
        { type: "text-start", id: "t" },
        { type: "text-delta", id: "t", delta: "The answer is 42." },
        { type: "text-delta", id: "t", delta: "\n@9@" },
        { type: "text-end", id: "t" },
        {
            type: "finish",
            usage: {
                inputTokens: { total: 0, noCache: 0, cacheRead: 0, cacheWrite: 0 },
                outputTokens: { total: 0, text: 0, reasoning: 0 },
            },
            finishReason: { unified: "stop", raw: "stop" },
        },
    ])
    assert.equal(await collectText(wrapLanguageModel(language)), "The answer is 42.\n")
})

test("doStream strips a tag split across deltas", async () => {
    const language = fakeLanguage([
        { type: "text-start", id: "t" },
        { type: "text-delta", id: "t", delta: "All done\n@1" },
        { type: "text-delta", id: "t", delta: "5@ [low]" },
        { type: "text-end", id: "t" },
        {
            type: "finish",
            usage: {
                inputTokens: { total: 0, noCache: 0, cacheRead: 0, cacheWrite: 0 },
                outputTokens: { total: 0, text: 0, reasoning: 0 },
            },
            finishReason: { unified: "stop", raw: "stop" },
        },
    ])
    const text = await collectText(wrapLanguageModel(language))
    assert.equal(text, "All done\n")
    assert.doesNotMatch(text, /@1|5@/)
})

test("doStream flushes buffered text unstripped before an error part", async () => {
    const language = fakeLanguage([
        { type: "text-start", id: "t" },
        { type: "text-delta", id: "t", delta: "partial text" },
        { type: "error", error: new Error("boom") },
    ])
    const wrapped = wrapLanguageModel(language)
    const result = await wrapped.doStream({} as never)
    const parts: LanguageModelV3StreamPart[] = []
    for await (const part of result.stream as AsyncIterable<LanguageModelV3StreamPart>)
        parts.push(part)
    const deltas = parts.filter((part) => part.type === "text-delta")
    assert.equal(deltas.length, 1)
    assert.equal((deltas[0] as { delta: string }).delta, "partial text")
    assert.equal(parts.at(-1)!.type, "error")
})

test("doStream cleans multiple text ids independently", async () => {
    const language = fakeLanguage([
        { type: "text-start", id: "t1" },
        { type: "text-start", id: "t2" },
        { type: "text-delta", id: "t1", delta: "First.\n@2@" },
        { type: "text-delta", id: "t2", delta: "Second.\n@3@ [low]" },
        { type: "text-end", id: "t1" },
        { type: "text-end", id: "t2" },
        {
            type: "finish",
            usage: {
                inputTokens: { total: 0, noCache: 0, cacheRead: 0, cacheWrite: 0 },
                outputTokens: { total: 0, text: 0, reasoning: 0 },
            },
            finishReason: { unified: "stop", raw: "stop" },
        },
    ])
    const wrapped = wrapLanguageModel(language)
    const result = await wrapped.doStream({} as never)
    const byId = new Map<string, string>()
    for await (const part of result.stream as AsyncIterable<LanguageModelV3StreamPart>)
        if (part.type === "text-delta") byId.set(part.id, (byId.get(part.id) ?? "") + part.delta)
    assert.equal(byId.get("t1"), "First.\n")
    assert.equal(byId.get("t2"), "Second.\n")
})
