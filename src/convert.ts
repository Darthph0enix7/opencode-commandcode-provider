import type { LanguageModelV3CallOptions } from "@ai-sdk/provider"
import type {
  LanguageModelV3FunctionTool,
  LanguageModelV3Message,
  LanguageModelV3TextPart,
  LanguageModelV3ReasoningPart,
  LanguageModelV3ToolCallPart,
  LanguageModelV3ToolResultPart,
  LanguageModelV3ToolResultOutput,
} from "@ai-sdk/provider"

type CCMessage =
  | { role: "user"; content: string | unknown[] }
  | { role: "assistant"; content: CCAssistantContent[] }
  | { role: "tool"; content: CCToolResultContent[] }

type CCAssistantContent =
  | { type: "text"; text: string }
  | { type: "reasoning"; text: string }
  | { type: "tool-call"; toolCallId: string; toolName: string; input: unknown }

type CCToolResultContent = {
  type: "tool-result"
  toolCallId: string
  toolName: string
  output: { type: "text"; value: string } | { type: "error-text"; value: string }
}

type CCTool = {
  type: "function"
  name: string
  description?: string
  input_schema: unknown
}

interface CCRequestEnvelope {
  config: {
    workingDir: string
    date: string
    environment: string
    structure: unknown[]
    isGitRepo: boolean
    currentBranch: string
    mainBranch: string
    gitStatus: string
    recentCommits: unknown[]
  }
  memory: string
  taste: string
  skills: null
  permissionMode: string
  params: {
    model: string
    messages: CCMessage[]
    tools: CCTool[]
    system: string
    max_tokens: number
    stream: true
    temperature?: number
    top_p?: number
    top_k?: number
    reasoning_effort?: string
  }
}

function hasType(p: unknown, type: string): boolean {
  return typeof p === "object" && p !== null && (p as { type?: string }).type === type
}

function isTextPart(p: unknown): p is LanguageModelV3TextPart {
  return hasType(p, "text")
}

function isReasoningPart(p: unknown): p is LanguageModelV3ReasoningPart {
  return hasType(p, "reasoning")
}

function isToolCallPart(p: unknown): p is LanguageModelV3ToolCallPart {
  return hasType(p, "tool-call")
}

function isToolResultPart(p: unknown): p is LanguageModelV3ToolResultPart {
  return hasType(p, "tool-result")
}

function extractText(content: unknown): string {
  if (typeof content === "string") return content
  if (Array.isArray(content)) {
    const textParts = content.filter(isTextPart) as LanguageModelV3TextPart[]
    return textParts.map((p) => p.text).join("\n")
  }
  return ""
}

function toDataUriOrUrl(data: string | Uint8Array | URL, defaultMime: string = "image/png"): string {
  if (typeof data === "string") {
    if (data.startsWith("data:") || data.startsWith("http://") || data.startsWith("https://")) {
      return data
    }
    return `data:${defaultMime};base64,${data}`
  }
  if (data instanceof URL) {
    return data.toString()
  }
  if (data instanceof Uint8Array || (typeof Buffer !== "undefined" && Buffer.isBuffer(data))) {
    const b64 = Buffer.from(data).toString("base64")
    return `data:${defaultMime};base64,${b64}`
  }
  return String(data)
}

function convertUserContent(content: unknown): string | Array<{ type: string; [key: string]: unknown }> {
  if (typeof content === "string") return content
  if (!Array.isArray(content)) return ""

  const parts: Array<{ type: string; [key: string]: unknown }> = []
  let hasMultimodal = false

  for (const p of content) {
    if (!p || typeof p !== "object") continue
    const item = p as Record<string, unknown>

    if (item.type === "text" && typeof item.text === "string") {
      parts.push({ type: "text", text: item.text })
    } else if (item.type === "image" && item.image) {
      hasMultimodal = true
      const mimeType = typeof item.mimeType === "string" ? item.mimeType : "image/png"
      const imgUrl = toDataUriOrUrl(item.image as string | Uint8Array | URL, mimeType)
      parts.push({ type: "image", image: imgUrl, mimeType })
    } else if (item.type === "file" && item.data) {
      const mimeType = typeof item.mimeType === "string" ? item.mimeType : "application/octet-stream"
      if (mimeType.startsWith("image/")) {
        hasMultimodal = true
        const imgUrl = toDataUriOrUrl(item.data as string | Uint8Array | URL, mimeType)
        parts.push({ type: "image", image: imgUrl, mimeType })
      } else if (typeof item.data === "string" && !item.data.startsWith("data:")) {
        parts.push({ type: "text", text: item.data })
      }
    }
  }

  if (!hasMultimodal) {
    return parts.map((p) => (p.type === "text" ? String(p.text) : "")).join("\n")
  }
  return parts
}

function convertToolResultOutput(output: LanguageModelV3ToolResultOutput): CCToolResultContent["output"] {
  switch (output.type) {
    case "text":
      return { type: "text", value: output.value }
    case "error-text":
      return { type: "error-text", value: output.value }
    case "json":
      return { type: "text", value: JSON.stringify(output.value) }
    case "execution-denied":
      return { type: "error-text", value: output.reason ?? "Execution denied" }
    case "error-json":
      return { type: "error-text", value: JSON.stringify(output.value) }
    case "content":
      return { type: "text", value: output.value.map((v: Record<string, unknown>) => ("text" in v ? v.text : JSON.stringify(v))).join("\n") }
    default:
      return { type: "text", value: JSON.stringify(output) }
  }
}

function convertMessage(msg: LanguageModelV3Message): CCMessage | null {
  switch (msg.role) {
    case "user": {
      const content = convertUserContent(msg.content)
      return { role: "user", content: content as any }
    }
    case "assistant": {
      const parts: CCAssistantContent[] = []
      for (const part of msg.content) {
        if (isTextPart(part)) {
          parts.push({ type: "text", text: part.text })
        } else if (isReasoningPart(part)) {
          parts.push({ type: "reasoning", text: part.text })
        } else if (isToolCallPart(part)) {
          parts.push({
            type: "tool-call",
            toolCallId: part.toolCallId,
            toolName: part.toolName,
            input: part.input,
          })
        }
      }
      return { role: "assistant", content: parts }
    }
    case "tool": {
      const parts: CCToolResultContent[] = []
      for (const part of msg.content) {
        if (isToolResultPart(part)) {
          parts.push({
            type: "tool-result",
            toolCallId: part.toolCallId,
            toolName: part.toolName,
            output: convertToolResultOutput(part.output),
          })
        }
      }
      return { role: "tool", content: parts }
    }
    default:
      return null
  }
}

function convertTools(
  tools: Array<LanguageModelV3FunctionTool | { type: "provider"; id: `${string}.${string}`; name: string; args: Record<string, unknown> }> | undefined,
): CCTool[] {
  if (!tools) return []
  return tools
    .filter((t): t is LanguageModelV3FunctionTool => t.type === "function")
    .map((t) => ({
      type: "function" as const,
      name: t.name,
      description: t.description,
      input_schema: t.inputSchema,
    }))
}

function extractReasoningEffort(options: LanguageModelV3CallOptions): string | undefined {
  const opt = options as Record<string, unknown>
  if (typeof opt.reasoning_effort === "string") return opt.reasoning_effort
  if (typeof opt.reasoningEffort === "string") return opt.reasoningEffort
  if (typeof opt.effort === "string") return opt.effort
  if (typeof opt.variant === "string") return opt.variant

  if (opt.providerOptions && typeof opt.providerOptions === "object") {
    const po = opt.providerOptions as Record<string, unknown>
    for (const key of Object.keys(po)) {
      const val = po[key]
      if (val && typeof val === "object") {
        const sub = val as Record<string, unknown>
        if (typeof sub.reasoning_effort === "string") return sub.reasoning_effort
        if (typeof sub.reasoningEffort === "string") return sub.reasoningEffort
        if (typeof sub.effort === "string") return sub.effort
        if (typeof sub.variant === "string") return sub.variant
      }
    }
  }

  if (opt.options && typeof opt.options === "object") {
    const sub = opt.options as Record<string, unknown>
    if (typeof sub.reasoning_effort === "string") return sub.reasoning_effort
    if (typeof sub.reasoningEffort === "string") return sub.reasoningEffort
    if (typeof sub.effort === "string") return sub.effort
    if (typeof sub.variant === "string") return sub.variant
  }

  return undefined
}

export function buildRequest(
  modelId: string,
  options: LanguageModelV3CallOptions,
): CCRequestEnvelope {
  let systemPrompt = ""
  const messages: CCMessage[] = []

  for (const msg of options.prompt) {
    if (msg.role === "system") {
      systemPrompt += (systemPrompt ? "\n\n" : "") + msg.content
      continue
    }
    const converted = convertMessage(msg)
    if (converted) messages.push(converted)
  }

  const effort = extractReasoningEffort(options)

  const params: CCRequestEnvelope["params"] = {
    model: modelId,
    messages,
    tools: convertTools(options.tools),
    system: systemPrompt,
    max_tokens: options.maxOutputTokens ?? 16384,
    stream: true,
    ...(effort ? { reasoning_effort: effort } : {}),
  }

  if (options.temperature !== undefined) params.temperature = options.temperature
  if (options.topP !== undefined) params.top_p = options.topP
  if (options.topK !== undefined) params.top_k = options.topK

  return {
    config: {
      workingDir: process.cwd() ?? "/",
      date: new Date().toISOString().split("T")[0] ?? "",
      environment: `${process.platform}-${process.arch}`,
      // Stub: opencode does not expose project structure context
      structure: [],
      isGitRepo: false,
      currentBranch: "",
      mainBranch: "",
      gitStatus: "",
      recentCommits: [],
    },
    memory: "",
    // Stub: taste/memory/permissionMode are Command Code CLI features not exposed via provider API
    taste: "",
    skills: null,
    permissionMode: "standard",
    params,
  }
}
