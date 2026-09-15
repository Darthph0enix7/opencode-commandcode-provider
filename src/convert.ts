import type { LanguageModelV3CallOptions } from "@ai-sdk/provider"
import { readFileSync, writeFileSync } from "fs"
import { dirname, join } from "path"
import { fileURLToPath } from "url"
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

function convertUserContent(
  content: unknown,
  forwarder?: ImageForwarder,
): string | Array<{ type: string; [key: string]: unknown }> {
  if (typeof content === "string") return content
  if (!Array.isArray(content)) return ""

  const parts: Array<{ type: string; [key: string]: unknown }> = []
  let hasMultimodal = false

  for (const p of content) {
    if (!p || typeof p !== "object") continue
    const item = p as Record<string, unknown>

    if (item.type === "text" && typeof item.text === "string") {
      parts.push({ type: "text", text: item.text })
      continue
    }
    if (item.type === "file" || item.type === "image" || item.type === "media") {
      // `{ type: "file", data: "<raw base64>", mediaType: "image/png" }` is how
      // opencode hands tool-result attachments to providers. Emitting those as
      // text is what pushed ~1.5M tokens of base64 into a single request.
      if (convertBinaryPart(item, parts, forwarder ? forwarder.enabled : true)) {
        hasMultimodal = true
      }
      continue
    }
    // Unknown part carrying an obvious binary payload: never serialize it as text
    if (isBinaryLikePart(item)) {
      parts.push({ type: "text", text: describeBinaryPart(item) })
    }
  }

  if (!hasMultimodal) {
    return parts.map((p) => (p.type === "text" ? String(p.text) : "")).join("\n")
  }
  return parts
}

/** Extract the mime type from a `data:<mime>;base64,<payload>` URI. */
function dataUriMime(dataUri: string): string | null {
  if (!dataUri.startsWith("data:")) return null
  const semi = dataUri.indexOf(";")
  const comma = dataUri.indexOf(",")
  if (semi === -1 || comma === -1 || semi > comma) return null
  const mime = dataUri.slice(5, semi)
  return mime || null
}

/**
 * Media type of a binary part. opencode's serialiser emits `mediaType`; older
 * callers use `mimeType`; and when neither is declared, the payload's `data:`
 * header still tells us what it is.
 */
function detectMediaType(item: Record<string, unknown>): string {
  const declared = item.mediaType ?? item.mimeType ?? item.media_type
  if (typeof declared === "string" && declared.trim()) return declared.trim()
  const payload = item.data ?? item.url ?? item.image
  if (typeof payload === "string") {
    const mime = dataUriMime(payload)
    if (mime) return mime
  }
  return "application/octet-stream"
}

function describeOmittedImage(item: Record<string, unknown>, mimeType: string, reason: string): string {
  const bytes = byteLengthOf(item.data) || byteLengthOf(item.image) || byteLengthOf(item.url) || 0
  const size = bytes > 0 ? `, ${Math.max(1, Math.round(bytes / 1024))} KB` : ""
  return `[image omitted: ${mimeType}${size} — ${reason}]`
}

/**
 * Convert one binary content part (user attachment or tool-result media) into a
 * wire content part, appending to `parts`. Returns true when an actual image
 * part was emitted (i.e. the message now carries multimodal content).
 *
 * Images are emitted as `{ type: "image", image: "data:<mime>;base64,<b64>",
 * mimeType }` — byte-for-byte the shape the Command Code CLI itself sends.
 * Anything that cannot be transported becomes a short note; payloads are never
 * inlined into text.
 */
function convertBinaryPart(
  item: Record<string, unknown>,
  parts: Array<{ type: string; [key: string]: unknown }>,
  visionEnabled: boolean,
): boolean {
  const mimeType = detectMediaType(item)
  const payload = item.data ?? item.url ?? item.image

  if (mimeType.startsWith("image/")) {
    if (!visionEnabled) {
      parts.push({ type: "text", text: describeOmittedImage(item, mimeType, "this model does not accept image input") })
      return false
    }
    if (payload === undefined || payload === null) {
      parts.push({ type: "text", text: describeOmittedImage(item, mimeType, "no payload present") })
      return false
    }
    const imgUrl = toDataUriOrUrl(payload as string | Uint8Array | URL, mimeType)
    if (typeof imgUrl === "string" && imgUrl.startsWith("data:")) {
      parts.push({ type: "image", image: imgUrl, mimeType })
      return true
    }
    parts.push({
      type: "text",
      text: describeOmittedImage(item, mimeType, "the API accepts inline data URIs only, not remote URLs"),
    })
    return false
  }

  if (mimeType.startsWith("text/") && typeof payload === "string" && !payload.startsWith("data:")) {
    if (payload.length > MAX_INLINE_TOOL_TEXT_CHARS) {
      parts.push({
        type: "text",
        text: `[attachment omitted: ${mimeType}, ${Math.round(payload.length / 1024)} KB exceeds the inline limit]`,
      })
    } else {
      parts.push({ type: "text", text: payload })
    }
    return false
  }

  parts.push({ type: "text", text: describeBinaryPart(item) })
  return false
}

/**
 * Hard ceiling for any single inlined tool-result payload. Anything larger is
 * replaced by a size note instead of being inlined. Binary payloads (base64
 * images, PDFs, data URIs) are never inlined at all — see `describeBinaryPart`.
 */
const MAX_INLINE_TOOL_TEXT_CHARS = 200_000

/**
 * Image forwarding budget per request. Tool images cannot ride inside a
 * tool-result (Command Code accepts images only in user content), so they are
 * re-emitted as a synthetic user message right after the tool message. These
 * caps keep a pathological read (huge screenshot, repeated reads) bounded.
 */
const MAX_FORWARD_IMAGE_BYTES = 8 * 1024 * 1024
const MAX_FORWARD_IMAGES_TOTAL = 12
const MAX_FORWARD_IMAGE_BYTES_TOTAL = 16 * 1024 * 1024

interface ForwardedImage {
  mediaType: string
  dataUri: string
}

interface ImageForwarder {
  /** Whether the target model accepts image input at all. */
  enabled: boolean
  images: ForwardedImage[]
  seen: Set<string>
  bytes: number
}

function createImageForwarder(enabled: boolean): ImageForwarder {
  return { enabled, images: [], seen: new Set<string>(), bytes: 0 }
}

function imageFingerprint(dataUri: string): string {
  return `${dataUri.length}:${dataUri.slice(0, 64)}:${dataUri.slice(-64)}`
}

function kilobytes(dataUri: string): number {
  return Math.max(1, Math.round(dataUri.length / 1024))
}

/**
 * Decide what happens to one binary tool-result part.
 *
 * Image parts are handed to the forwarder (which re-emits them as a user
 * message for vision-capable models) and replaced in the tool result by a short
 * note. Everything else — and every image when the model cannot see, when the
 * payload is oversized, or when the budget is spent — becomes a size note so the
 * payload never reaches the prompt as text.
 */
function handleBinaryPart(item: Record<string, unknown>, forwarder?: ImageForwarder): string {
  const mediaType = detectMediaType(item)
  const raw = item.data ?? item.image ?? item.url
  if (!forwarder?.enabled || !mediaType.startsWith("image/")) {
    return describeBinaryPart(item)
  }
  const dataUri = toDataUriOrUrl(raw as string | Uint8Array | URL, mediaType)
  if (typeof dataUri !== "string" || !dataUri.startsWith("data:")) {
    return describeBinaryPart(item)
  }
  if (dataUri.length > MAX_FORWARD_IMAGE_BYTES) {
    return `[image omitted: ${mediaType}, ${kilobytes(dataUri)} KB exceeds the ${Math.round(MAX_FORWARD_IMAGE_BYTES / 1024 / 1024)} MB per-image forward limit]`
  }
  const fingerprint = imageFingerprint(dataUri)
  if (forwarder.seen.has(fingerprint)) {
    return `[identical ${mediaType} image already attached earlier in this conversation — not re-sent]`
  }
  if (
    forwarder.images.length >= MAX_FORWARD_IMAGES_TOTAL ||
    forwarder.bytes + dataUri.length > MAX_FORWARD_IMAGE_BYTES_TOTAL
  ) {
    return `[image omitted: per-request image forward budget reached — ${forwarder.images.length} image(s), ${Math.round(forwarder.bytes / 1024 / 1024)} MB already forwarded]`
  }
  forwarder.seen.add(fingerprint)
  forwarder.bytes += dataUri.length
  forwarder.images.push({ mediaType, dataUri })
  return `[${mediaType} image, ${kilobytes(dataUri)} KB — attached as a user message below]`
}

function byteLengthOf(data: unknown): number {
  if (typeof data === "string") return data.length
  if (data instanceof Uint8Array) return data.byteLength
  if (typeof Buffer !== "undefined" && Buffer.isBuffer(data)) return data.length
  return 0
}

/**
 * Describe a non-text tool-result part without serializing its payload.
 *
 * Tool results can carry binary attachments (opencode's `read` on an image
 * returns `{ type: "file", mediaType, data: "data:image/png;base64,..." }`).
 * Inlining those as text pushes megabytes of base64 into the prompt and blows
 * the model's context window, so they are summarised instead.
 */
function describeBinaryPart(item: Record<string, unknown>): string {
  const mediaType =
    typeof item.mediaType === "string"
      ? item.mediaType
      : typeof item.mimeType === "string"
        ? item.mimeType
        : "application/octet-stream"
  const bytes =
    byteLengthOf(item.data) || byteLengthOf(item.image) || byteLengthOf(item.url) || 0
  const size = bytes > 0 ? `, ${Math.max(1, Math.round(bytes / 1024))} KB` : ""
  return `[attachment omitted: ${mediaType}${size} — binary payloads are not inlined into text. Use a vision-capable model and send the image as a user attachment to inspect it.]`
}

function isBinaryLikePart(v: unknown): v is Record<string, unknown> {
  if (!v || typeof v !== "object") return false
  const item = v as Record<string, unknown>
  if (item.type === "file" || item.type === "image" || item.type === "media") return true
  if (typeof item.data === "string" && item.data.startsWith("data:")) return true
  if (typeof item.url === "string" && item.url.startsWith("data:")) return true
  return false
}

function inlineTextOrNote(v: unknown, forwarder?: ImageForwarder): string {
  if (typeof v === "string") return v
  if (!v || typeof v !== "object") return String(v ?? "")
  const item = v as Record<string, unknown>
  if (typeof item.text === "string") return item.text
  if (isBinaryLikePart(item)) return handleBinaryPart(item, forwarder)
  let serialized: string
  try {
    serialized = scrubEmbeddedPayloads(JSON.stringify(item) ?? "")
  } catch {
    return "[unserializable tool output omitted]"
  }
  if (serialized.length > MAX_INLINE_TOOL_TEXT_CHARS) {
    return `[tool output omitted: ${serialized.length} chars exceed the ${MAX_INLINE_TOOL_TEXT_CHARS} char inline limit]`
  }
  return serialized
}

/**
 * Safety net for serialised tool output: a `data:` URI buried inside an object
 * must never turn into thousands of tokens of base64 text. Structural handling
 * happens earlier (convertBinaryPart / handleBinaryPart); this catches the
 * leftovers (e.g. a payload nested inside an otherwise-ordinary JSON result).
 */
const EMBEDDED_DATA_URI_RE = /data:[a-z0-9][a-z0-9.+-]*\/[a-z0-9][a-z0-9.+-]*;base64,[A-Za-z0-9+/=\s]{512,}/gi

function scrubEmbeddedPayloads(text: string): string {
  if (!text.includes(";base64,")) return text
  return text.replace(
    EMBEDDED_DATA_URI_RE,
    (match) => `[embedded base64 payload omitted: ${Math.max(1, Math.round(match.length / 1024))} KB]`,
  )
}

function capSerialized(value: unknown, fallback: string): string {
  let serialized: string
  try {
    serialized = scrubEmbeddedPayloads(JSON.stringify(value) ?? "")
  } catch {
    return fallback
  }
  if (serialized.length > MAX_INLINE_TOOL_TEXT_CHARS) {
    return `[tool output omitted: ${serialized.length} chars exceed the ${MAX_INLINE_TOOL_TEXT_CHARS} char inline limit]`
  }
  return serialized
}

function convertToolResultOutput(
  output: LanguageModelV3ToolResultOutput,
  forwarder?: ImageForwarder,
): CCToolResultContent["output"] {
  switch (output.type) {
    case "text":
      return { type: "text", value: output.value }
    case "error-text":
      return { type: "error-text", value: output.value }
    case "json":
      return { type: "text", value: capSerialized(output.value, "[unserializable tool output omitted]") }
    case "execution-denied":
      return { type: "error-text", value: output.reason ?? "Execution denied" }
    case "error-json":
      return { type: "error-text", value: capSerialized(output.value, "[unserializable tool output omitted]") }
    case "content":
      return {
        type: "text",
        value: output.value.map((v: unknown) => inlineTextOrNote(v, forwarder)).join("\n"),
      }
    default:
      return { type: "text", value: capSerialized(output, "[unserializable tool output omitted]") }
  }
}

function convertMessage(msg: LanguageModelV3Message, forwarder?: ImageForwarder): CCMessage | null {
  switch (msg.role) {
    case "user": {
      const content = convertUserContent(msg.content, forwarder)
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
            output: convertToolResultOutput(part.output, forwarder),
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

/**
 * Lazily-loaded map of model id -> "accepts image input", built from the same
 * catalog this provider registers with opencode (models.cache.json wins, then
 * models.json). Unknown models are treated as image-capable: forwarding an image
 * can at worst be rejected by the upstream, never silently explode the prompt.
 */
let imageInputSupport: Map<string, boolean> | null = null

function loadImageInputSupport(): Map<string, boolean> {
  if (imageInputSupport) return imageInputSupport
  const support = new Map<string, boolean>()
  const packageRoot = join(dirname(fileURLToPath(import.meta.url)), "..")
  for (const file of ["models.cache.json", "models.json"]) {
    try {
      const parsed = JSON.parse(readFileSync(join(packageRoot, file), "utf-8")) as unknown
      const list = Array.isArray(parsed) ? parsed : Object.values((parsed ?? {}) as Record<string, unknown>)
      for (const raw of list) {
        if (!raw || typeof raw !== "object") continue
        const entry = raw as Record<string, unknown>
        const id = typeof entry.id === "string" ? entry.id : undefined
        if (!id) continue
        const modalities = (entry.modalities ?? {}) as Record<string, unknown>
        const inputs = Array.isArray(modalities.input) ? (modalities.input as unknown[]) : undefined
        const acceptsImages = inputs ? inputs.includes("image") : entry.attachment === true
        const key = id.toLowerCase()
        if (!support.has(key)) support.set(key, acceptsImages)
        const short = key.split("/").pop()
        if (short && !support.has(short)) support.set(short, acceptsImages)
      }
    } catch {
      // catalog missing/unreadable: fall through to the permissive default
    }
  }
  imageInputSupport = support
  return support
}

function modelAcceptsImageInput(modelId: string): boolean {
  const support = loadImageInputSupport()
  const key = modelId.toLowerCase()
  const hit = support.get(key) ?? support.get(key.split("/").pop() ?? key)
  return hit ?? true
}

export function buildRequest(
  modelId: string,
  options: LanguageModelV3CallOptions,
): CCRequestEnvelope {
  const rawDumpPath = process.env.COMMANDCODE_DEBUG_RAW
  if (rawDumpPath) {
    try {
      writeFileSync(rawDumpPath, JSON.stringify(options.prompt, null, 1), "utf-8")
    } catch {
      // diagnostics only
    }
  }
  let systemPrompt = ""
  const messages: CCMessage[] = []
  const forwarder = createImageForwarder(modelAcceptsImageInput(modelId))

  for (const msg of options.prompt) {
    if (msg.role === "system") {
      systemPrompt += (systemPrompt ? "\n\n" : "") + msg.content
      continue
    }
    const converted = convertMessage(msg, forwarder)
    if (converted) messages.push(converted)
    // Tool images cannot ride inside a tool-result, so they follow it as a
    // user message (Command Code accepts images only in user content).
    if (msg.role === "tool" && forwarder.images.length > 0) {
      const pending = forwarder.images.splice(0, forwarder.images.length)
      messages.push({
        role: "user",
        content: [
          { type: "text", text: "Images returned by the tool call(s) above:" },
          ...pending.map((image) => ({ type: "image", image: image.dataUri, mimeType: image.mediaType })),
        ],
      })
    }
  }

  const effort = extractReasoningEffort(options)

  // Opt-in wire inspection: COMMANDCODE_DEBUG_BODY=/path/to/body.json dumps the
  // exact request this provider is about to send (used to prove whether image
  // parts are forwarded as images or inlined as text).
  const debugBodyPath = process.env.COMMANDCODE_DEBUG_BODY
  if (debugBodyPath) {
    const dump = JSON.stringify(messages, null, 1)
    try {
      writeFileSync(debugBodyPath, dump, "utf-8")
    } catch {
      // diagnostics only — never fail the request over it
    }
  }

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
