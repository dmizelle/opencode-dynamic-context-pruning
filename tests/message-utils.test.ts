import assert from "node:assert/strict"
import test from "node:test"
import { isIgnoredUserMessage } from "../lib/messages/query"
import { formatCompressionBar } from "../lib/compress/message-utils"
import type { WithParts } from "../lib/state"

function buildMessage(role: "user" | "assistant", parts: WithParts["parts"]): WithParts {
    const sessionID = "ses_message_utils"

    const info =
        role === "user"
            ? {
                  id: `msg-${role}`,
                  role,
                  sessionID,
                  agent: "assistant",
                  model: {
                      providerID: "anthropic",
                      modelID: "claude-test",
                  },
                  time: { created: 1 },
              }
            : {
                  id: `msg-${role}`,
                  role,
                  sessionID,
                  agent: "assistant",
                  time: { created: 1 },
              }

    return {
        info: info as WithParts["info"],
        parts,
    }
}

test("isIgnoredUserMessage only ignores user messages", () => {
    const ignoredUserMessage = buildMessage("user", [])
    const assistantMessage = buildMessage("assistant", [])

    assert.equal(isIgnoredUserMessage(ignoredUserMessage), true)
    assert.equal(isIgnoredUserMessage(assistantMessage), false)
})

test("formatCompressionBar renders ratio and returns empty for no compression", () => {
    assert.equal(formatCompressionBar(0, 0), "")
    assert.equal(formatCompressionBar(-5, 0), "")

    const bar = formatCompressionBar(10_000, 2_500)
    assert.match(bar, /\[█{15}░{5}\]/)
    assert.match(bar, /10\.0K → 2\.5K tokens \(75% reclaimed\)/)

    assert.equal(formatCompressionBar(500, 500), "\n[░░░░░░░░░░░░░░░░░░░░] 500 → 500 tokens (0% reclaimed)\n")
})
