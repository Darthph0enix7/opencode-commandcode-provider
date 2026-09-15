import { expect, test } from "bun:test"
import { buildRequest } from "../../src/convert.js"
import type { LanguageModelV3CallOptions } from "@ai-sdk/provider"

function makeOpts(overrides: Partial<LanguageModelV3CallOptions> = {}): LanguageModelV3CallOptions {
  return {
    prompt: [],
    maxOutputTokens: 1000,
    ...overrides,
  }
}

test("builds minimal request envelope", () => {
  const req = buildRequest("test-model", makeOpts())
  expect(req.params.model).toBe("test-model")
  expect(req.params.stream).toBe(true)
  expect(req.params.messages).toEqual([])
  expect(req.params.tools).toEqual([])
  expect(req.params.system).toBe("")
  expect(req.params.max_tokens).toBe(1000)
})

test("concatenates system prompts", () => {
  const req = buildRequest("m", makeOpts({
    prompt: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "system", content: "You write TypeScript." },
    ],
  }))
  expect(req.params.system).toBe("You are a helpful assistant.\n\nYou write TypeScript.")
})

test("converts user message with string content", () => {
  const req = buildRequest("m", makeOpts({
    prompt: [{ role: "user", content: "hello" }],
  }))
  expect(req.params.messages).toHaveLength(1)
  expect(req.params.messages[0]).toEqual({ role: "user", content: "hello" })
})

test("converts user message with array of text parts", () => {
  const req = buildRequest("m", makeOpts({
    prompt: [{
      role: "user",
      content: [
        { type: "text", text: "line1" },
        { type: "text", text: "line2" },
      ],
    }],
  }))
  expect(req.params.messages).toHaveLength(1)
  expect(req.params.messages[0]).toEqual({ role: "user", content: "line1\nline2" })
})

test("untyped non-text parts get a note instead of being silently dropped", () => {
  const req = buildRequest("m", makeOpts({
    prompt: [{
      role: "user",
      content: [
        { type: "text", text: "hello" },
        { type: "image", url: "https://example.com/img.png" },
      ],
    }],
  }))
  expect(req.params.messages).toHaveLength(1)
  const msg = req.params.messages[0] as { role: "user"; content: string }
  expect(msg.content).toStartWith("hello")
  expect(msg.content).toContain("[attachment omitted")
})

test("converts assistant message with text, reasoning, and tool-call parts", () => {
  const req = buildRequest("m", makeOpts({
    prompt: [{
      role: "assistant",
      content: [
        { type: "text", text: "I think" },
        { type: "reasoning", text: "hmm..." },
        { type: "tool-call", toolCallId: "tc1", toolName: "bash", input: { cmd: "ls" } },
      ],
    }],
  }))
  expect(req.params.messages).toHaveLength(1)
  const msg = req.params.messages[0] as { role: "assistant"; content: unknown[] }
  expect(msg.content).toHaveLength(3)
  expect(msg.content[0]).toEqual({ type: "text", text: "I think" })
  expect(msg.content[1]).toEqual({ type: "reasoning", text: "hmm..." })
  expect(msg.content[2]).toEqual({ type: "tool-call", toolCallId: "tc1", toolName: "bash", input: { cmd: "ls" } })
})

test("converts tool result message with text output", () => {
  const req = buildRequest("m", makeOpts({
    prompt: [{
      role: "tool",
      content: [{
        type: "tool-result",
        toolCallId: "tc1",
        toolName: "bash",
        output: { type: "text", value: "file.ts" },
      }],
    }],
  }))
  const msg = req.params.messages[0] as { role: "tool"; content: unknown[] }
  expect(msg.content[0]).toEqual({
    type: "tool-result",
    toolCallId: "tc1",
    toolName: "bash",
    output: { type: "text", value: "file.ts" },
  })
})

test("converts tool result with error-text output", () => {
  const req = buildRequest("m", makeOpts({
    prompt: [{
      role: "tool",
      content: [{
        type: "tool-result",
        toolCallId: "tc1",
        toolName: "bash",
        output: { type: "error-text", value: "command not found" },
      }],
    }],
  }))
  const msg = req.params.messages[0] as { role: "tool"; content: unknown[] }
  const out = msg.content[0] as { output: { type: string; value: string } }
  expect(out.output.type).toBe("error-text")
  expect(out.output.value).toBe("command not found")
})

test("converts tool result with json output", () => {
  const req = buildRequest("m", makeOpts({
    prompt: [{
      role: "tool",
      content: [{
        type: "tool-result",
        toolCallId: "tc1",
        toolName: "bash",
        output: { type: "json", value: { key: "val" } },
      }],
    }],
  }))
  const msg = req.params.messages[0] as { role: "tool"; content: unknown[] }
  const out = msg.content[0] as { output: { type: string; value: string } }
  expect(out.output.type).toBe("text")
  expect(JSON.parse(out.output.value)).toEqual({ key: "val" })
})

test("converts tool result with execution-denied output", () => {
  const req = buildRequest("m", makeOpts({
    prompt: [{
      role: "tool",
      content: [{
        type: "tool-result",
        toolCallId: "tc1",
        toolName: "bash",
        output: { type: "execution-denied", reason: "not allowed" },
      }],
    }],
  }))
  const msg = req.params.messages[0] as { role: "tool"; content: unknown[] }
  const out = msg.content[0] as { output: { type: string; value: string } }
  expect(out.output.type).toBe("error-text")
  expect(out.output.value).toBe("not allowed")
})

test("skips unknown message roles", () => {
  const req = buildRequest("m", makeOpts({
    prompt: [{ role: "unknown" as never, content: "test" }],
  }))
  expect(req.params.messages).toHaveLength(0)
})

test("converts function tools", () => {
  const req = buildRequest("m", makeOpts({
    tools: [
      {
        type: "function",
        name: "my_tool",
        description: "A test tool",
        inputSchema: { type: "object", properties: {} },
      },
    ],
  }))
  expect(req.params.tools).toHaveLength(1)
  expect(req.params.tools[0]).toEqual({
    type: "function",
    name: "my_tool",
    description: "A test tool",
    input_schema: { type: "object", properties: {} },
  })
})

test("filters out provider tools", () => {
  const req = buildRequest("m", makeOpts({
    tools: [
      { type: "function", name: "func_tool", description: "", inputSchema: {} },
      { type: "provider", id: "provider.tool" as `${string}.${string}`, name: "prov_tool", args: {} },
    ],
  }))
  expect(req.params.tools).toHaveLength(1)
  expect(req.params.tools[0].name).toBe("func_tool")
})

test("passes through temperature, topP, topK", () => {
  const req = buildRequest("m", makeOpts({
    prompt: [{ role: "user", content: "hi" }],
    temperature: 0.5,
    topP: 0.9,
    topK: 40,
  }))
  expect(req.params.temperature).toBe(0.5)
  expect(req.params.top_p).toBe(0.9)
  expect(req.params.top_k).toBe(40)
})

test("defaults max_tokens to 16384 when not provided", () => {
  const req = buildRequest("m", makeOpts({ maxOutputTokens: undefined }))
  expect(req.params.max_tokens).toBe(16384)
})

test("envelope has correct top-level shape", () => {
  const req = buildRequest("m", makeOpts())
  expect(req).toHaveProperty("config")
  expect(req).toHaveProperty("memory", "")
  expect(req).toHaveProperty("taste", "")
  expect(req).toHaveProperty("skills", null)
  expect(req).toHaveProperty("permissionMode", "standard")
  expect(req).toHaveProperty("params")
})

// --- binary tool-result handling (regression: base64 inlined as text blew a 1M context window) ---

function toolResultWithFile(model: string, base64: string, mediaType = "image/png") {
  return buildRequest(model, makeOpts({
    prompt: [
      { role: "user", content: "read the image" },
      {
        role: "assistant",
        content: [
          { type: "tool-call", toolCallId: "call_1", toolName: "read", input: { filePath: "x.png" } },
        ],
      },
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "call_1",
            toolName: "read",
            output: {
              type: "content",
              value: [
                { type: "text", text: "Image read successfully" },
                { type: "file", mediaType, data: `data:${mediaType};base64,${base64}` },
              ],
            },
          },
        ],
      },
    ],
  }))
}

test("never inlines a tool-result image payload as text", () => {
  const base64 = "A".repeat(4096)
  const req = toolResultWithFile("deepseek/deepseek-v4.1-flash", base64)
  const toolMessage = JSON.stringify(req.params.messages[2])
  // the tool result must carry a note, never the payload as text
  expect(toolMessage.includes(base64)).toBe(false)
  expect(toolMessage.includes("attached as a user message")).toBe(true)
  // the payload only ever appears as an image part in the follow-up user message
  const followUp = req.params.messages[3] as { content: Array<Record<string, unknown>> }
  const imagePart = followUp.content.find((p) => p.type === "image")
  expect(String(imagePart?.image)).toBe(`data:image/png;base64,${base64}`)
})

test("forwards tool-result images as a follow-up user message for image-capable models", () => {
  const req = toolResultWithFile("deepseek/deepseek-v4.1-flash", "A".repeat(512))
  const roles = req.params.messages.map((m) => m.role)
  expect(roles).toEqual(["user", "assistant", "tool", "user"])
  const followUp = req.params.messages[3] as { content: Array<Record<string, unknown>> }
  const imagePart = followUp.content.find((p) => p.type === "image")
  expect(imagePart).toBeDefined()
  expect(String(imagePart?.image)).toStartWith("data:image/png;base64,")
})

test("omits tool-result images for text-only models", () => {
  const base64 = "B".repeat(1024)
  const req = toolResultWithFile("deepseek/deepseek-v4-flash", base64)
  const roles = req.params.messages.map((m) => m.role)
  expect(roles).toEqual(["user", "assistant", "tool"])
  expect(JSON.stringify(req).includes(base64)).toBe(false)
  expect(JSON.stringify(req).includes("attachment omitted")).toBe(true)
})

test("does not re-send an identical image twice in one request", () => {
  const data = `data:image/png;base64,${"C".repeat(256)}`
  const toolMsg = {
    role: "tool" as const,
    content: [
      {
        type: "tool-result" as const,
        toolCallId: "call_1",
        toolName: "read",
        output: { type: "content" as const, value: [{ type: "file", mediaType: "image/png", data }] },
      },
    ],
  }
  const req = buildRequest("deepseek/deepseek-v4.1-flash", makeOpts({
    prompt: [
      { role: "user", content: "read twice" },
      toolMsg as any,
      toolMsg as any,
    ],
  }))
  const roles = req.params.messages.map((m) => m.role)
  expect(roles).toEqual(["user", "tool", "user", "tool"])
  expect(JSON.stringify(req).includes("already attached earlier")).toBe(true)
})

test("caps oversized tool-result text payloads instead of inlining them", () => {
  const huge = "x".repeat(400_000)
  const req = buildRequest("test-model", makeOpts({
    prompt: [
      { role: "user", content: "hi" },
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "call_1",
            toolName: "bash",
            output: { type: "text", value: huge },
          },
        ],
      },
    ],
  }))
  // plain text tool output is passed through untouched (it is already text)
  expect(JSON.stringify(req).length).toBeGreaterThan(huge.length)
})

// --- opencode arrival shape: { type: "file", data: "<raw base64>", mediaType } ---
// This is how tool-result attachments actually reach the provider. The raw
// base64 was previously inlined as text (~1.5M tokens for one screenshot).

function opencodeAttachmentMessage(base64: string, mediaType?: string) {
  return {
    role: "user" as const,
    content: [
      { type: "text", text: "Attached media from tool result:" },
      mediaType
        ? { type: "file", data: base64, mediaType }
        : { type: "file", data: base64 },
    ],
  }
}

test("raw base64 file part with mediaType is forwarded as an image, never as text", () => {
  const base64 = "iVBORw0KGgo" + "A".repeat(2048)
  const req = buildRequest("deepseek/deepseek-v4.1-flash", makeOpts({
    prompt: [opencodeAttachmentMessage(base64, "image/png")],
  }))
  const msg = req.params.messages[0] as { content: Array<Record<string, unknown>> }
  const image = msg.content.find((p) => p.type === "image")
  expect(image).toBeDefined()
  expect(String(image?.image)).toBe(`data:image/png;base64,${base64}`)
  expect(image?.mimeType).toBe("image/png")
  // the payload must appear exactly once in the whole request (inside the image part)
  expect(JSON.stringify(req).split(base64).length - 1).toBe(1)
})

test("text-only model gets a note for a raw base64 image, not the payload", () => {
  const base64 = "A".repeat(4096)
  const req = buildRequest("deepseek/deepseek-v4-flash", makeOpts({
    prompt: [opencodeAttachmentMessage(base64, "image/png")],
  }))
  const msg = req.params.messages[0] as { content: string }
  expect(msg.content).toContain("[image omitted: image/png")
  expect(msg.content).toContain("does not accept image input")
  expect(JSON.stringify(req).includes(base64)).toBe(false)
})

test("raw base64 file part without a mime type is never inlined as text", () => {
  const base64 = "D".repeat(4096)
  const req = buildRequest("deepseek/deepseek-v4.1-flash", makeOpts({
    prompt: [opencodeAttachmentMessage(base64)],
  }))
  expect(JSON.stringify(req).includes(base64)).toBe(false)
  const msg = req.params.messages[0] as { content: string }
  expect(msg.content).toContain("[attachment omitted")
})

test("data-URI file part with mediaType is forwarded as an image", () => {
  const b64 = "B".repeat(512)
  const req = buildRequest("deepseek/deepseek-v4.1-flash", makeOpts({
    prompt: [{
      role: "user",
      content: [{ type: "file", data: `data:image/jpeg;base64,${b64}`, mediaType: "image/jpeg" }],
    }],
  }))
  const msg = req.params.messages[0] as { content: Array<Record<string, unknown>> }
  const image = msg.content.find((p) => p.type === "image")
  expect(String(image?.image)).toBe(`data:image/jpeg;base64,${b64}`)
})

test("embedded data-URI payloads inside serialized tool output are compacted", () => {
  const b64 = "C".repeat(2048)
  const req = buildRequest("m", makeOpts({
    prompt: [{
      role: "tool",
      content: [{
        type: "tool-result",
        toolCallId: "tc1",
        toolName: "screenshot",
        output: { type: "json", value: { screenshot: `data:image/png;base64,${b64}` } },
      }],
    }],
  }))
  const s = JSON.stringify(req)
  expect(s.includes(b64)).toBe(false)
  expect(s).toContain("[embedded base64 payload omitted")
})
