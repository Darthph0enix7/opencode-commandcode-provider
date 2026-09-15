import { existsSync, readFileSync, realpathSync, renameSync, writeFileSync } from "fs"
import { dirname, join } from "path"
import { fileURLToPath } from "url"
import { resolveApiKey } from "./auth.js"

const __dirname = dirname(fileURLToPath(import.meta.url))
const PACKAGE_ROOT = join(__dirname, "..")
const BUNDLED_MODELS = join(PACKAGE_ROOT, "models.json")
const CACHE_MODELS = join(PACKAGE_ROOT, "models.cache.json")

export interface ModelEntry {
  id: string
  name: string
  tier: "premium" | "open-source"
  reasoning: boolean
  tool_call: boolean
  attachment?: boolean
  modalities?: {
    input: string[]
    output: string[]
  }
  cost: { input: number; output: number; cache_read?: number; cache_write?: number }
  limit: { context: number; output: number }
}

interface ApiModel {
  id: string
  name?: string
  context_length?: number
}

export interface SyncResult {
  models: ModelEntry[]
  source: string
}

export function loadCachedModels(file: string): ModelEntry[] | null {
  try {
    const parsed = JSON.parse(readFileSync(file, "utf-8"))
    if (Array.isArray(parsed) && parsed.length > 0) return parsed as ModelEntry[]
  } catch {
    // intentionally silent: unreadable or malformed cache is ignored
  }
  return null
}

function writeAtomic(file: string, data: string) {
  const tmp = `${file}.${process.pid}.tmp`
  writeFileSync(tmp, data)
  renameSync(tmp, file)
}

// Locate the model catalog bundled with the installed Command Code CLI.
// The catalog is the vendor-maintained list for the same API endpoint this
// provider talks to, so it is the most accurate metadata source available.
function findModelsMd(): string | null {
  const explicit = process.env.COMMANDCODE_MODELS_MD
  if (explicit && existsSync(explicit)) return explicit

  const dirs = (process.env.PATH ?? "").split(":").filter(Boolean)
  for (const dir of dirs) {
    for (const name of ["cmd", "command-code"]) {
      const bin = join(dir, name)
      if (!existsSync(bin)) continue
      let target = bin
      try {
        target = realpathSync(bin)
      } catch {
        // keep unresolved path
      }
      const candidates = [
        join(dirname(target), "bundled", "command-code-knowledge", "reference", "models.md"),
        join(dirname(target), "..", "dist", "bundled", "command-code-knowledge", "reference", "models.md"),
        join(dir, "..", "lib", "node_modules", "command-code", "dist", "bundled", "command-code-knowledge", "reference", "models.md"),
      ]
      for (const candidate of candidates) {
        if (existsSync(candidate)) return candidate
      }
    }
  }
  return null
}

function findCliMjs(): string | null {
  const dirs = (process.env.PATH ?? "").split(":").filter(Boolean)
  for (const dir of dirs) {
    for (const name of ["cmd", "command-code", "commandcode"]) {
      const bin = join(dir, name)
      if (!existsSync(bin)) continue
      let target = bin
      try {
        target = realpathSync(bin)
      } catch {}
      const candidates = [
        join(dirname(target), "..", "dist", "cli.mjs"),
        join(dir, "..", "lib", "node_modules", "command-code", "dist", "cli.mjs"),
        join(dirname(target), "cli.mjs"),
      ]
      for (const c of candidates) {
        if (existsSync(c)) return c
      }
    }
  }
  return null
}

function extractModalitiesMap(): Map<string, string[]> {
  const map = new Map<string, string[]>()
  try {
    const cliPath = findCliMjs()
    if (cliPath && existsSync(cliPath)) {
      const content = readFileSync(cliPath, "utf-8")
      const m = content.match(/(\$L=\{[^;]+);/)
      if (m) {
        const text = m[1].replace(/^\$L=/, "")
        const regex = /id:\s*"([^"]+)"[^{}]*?inputModalities:\s*(\[[^\]]*\])/g
        let match: RegExpExecArray | null
        while ((match = regex.exec(text)) !== null) {
          try {
            const rawArr = match[2].replace(/'/g, '"')
            map.set(match[1].toLowerCase(), JSON.parse(rawArr))
          } catch {}
        }
      }
    }
  } catch {}
  return map
}

function isVisionModelFallback(id: string): boolean {
  const lower = id.toLowerCase()
  return (
    lower.includes("v4.1") ||
    lower.includes("vision") ||
    lower.includes("27b") ||
    lower.includes("kimi-k2") ||
    lower.includes("minimax-m3") ||
    lower.includes("step-3.7") ||
    lower.includes("inkling") ||
    lower.includes("sonnet") ||
    lower.includes("opus") ||
    lower.includes("haiku") ||
    lower.includes("fable") ||
    lower.includes("gpt-5") ||
    lower.includes("gpt-6") ||
    lower.includes("gemini")
  )
}

function parseContext(raw: string): number | null {
  const match = /^([\d.]+)\s*([MK])$/i.exec(raw.trim())
  if (!match) return null
  const value = Number(match[1]) * (match[2].toUpperCase() === "M" ? 1_000_000 : 1_000)
  return Number.isFinite(value) && value > 0 ? Math.round(value) : null
}

function parseCost(raw: string): ModelEntry["cost"] | null {
  const match = /^\$([\d.]+)\s*\/\s*\$([\d.]+)(?:\s*·\s*cache\s*\$([\d.]+))?(?:\s*\(write\s*\$([\d.]+)\))?/i.exec(
    raw.trim(),
  )
  if (!match) return null
  const cost: ModelEntry["cost"] = { input: Number(match[1]), output: Number(match[2]) }
  if (match[3] !== undefined) cost.cache_read = Number(match[3])
  if (match[4] !== undefined) cost.cache_write = Number(match[4])
  return cost
}

export function parseModelsMd(text: string, hints: Map<string, ModelEntry>): ModelEntry[] {
  const entries: ModelEntry[] = []
  const modalitiesMap = extractModalitiesMap()
  let section = ""
  for (const line of text.split(/\r?\n/)) {
    const heading = /^##\s+(.+?)\s*$/.exec(line)
    if (heading) {
      section = heading[1]
      continue
    }
    if (!line.startsWith("| `")) continue

    const cells = line.split("|").slice(1, -1).map((cell) => cell.trim())
    if (cells.length < 5) continue

    const id = cells[0].replace(/`/g, "").trim()
    if (!id) continue
    const hint = hints.get(id.toLowerCase())
    const context = parseContext(cells[2]) ?? (hint?.limit.context || 200000)
    const efforts = cells[3]
    const cost = parseCost(cells[4]) ?? hint?.cost ?? { input: 0, output: 0 }
    const minPlan = cells[5] ? cells[5].trim() : ""
    const planBadge = minPlan ? ` [${minPlan.replace(" and above", "+")}]` : ""
    const displayName = `${cells[1]}${planBadge}`

    const cliMods = modalitiesMap.get(id.toLowerCase())
    const hasVision = cliMods ? cliMods.includes("image") : isVisionModelFallback(id)
    const inputMods = cliMods ?? (hasVision ? ["text", "image"] : ["text"])

    entries.push({
      id,
      name: displayName,
      tier: section.toLowerCase() === "open source" ? "open-source" : "premium",
      reasoning: efforts !== "" && efforts !== "—" && efforts !== "-",
      tool_call: true,
      attachment: inputMods.some((m) => m !== "text"),
      modalities: {
        input: inputMods,
        output: ["text"],
      },
      cost,
      limit: { context, output: hint?.limit.output ?? 65536 },
    })
  }
  return entries
}

async function fetchApiModels(): Promise<ApiModel[] | null> {
  const apiKey = resolveApiKey({})
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), 4000)
  try {
    const response = await fetch("https://api.commandcode.ai/provider/v1/models", {
      headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
      signal: controller.signal,
    })
    if (!response.ok) return null
    const payload = (await response.json()) as { data?: ApiModel[] }
    return Array.isArray(payload?.data) ? payload.data : null
  } catch {
    return null
  } finally {
    clearTimeout(timer)
  }
}

function mergeApiModels(entries: ModelEntry[], api: ApiModel[]): ModelEntry[] {
  const seen = new Map(entries.map((entry) => [entry.id.toLowerCase(), entry]))
  for (const model of api) {
    if (!model?.id) continue
    const key = model.id.toLowerCase()
    const existing = seen.get(key)
    if (existing) {
      if (model.context_length && (!existing.limit.context || existing.limit.context <= 0)) {
        existing.limit.context = model.context_length
      }
      continue
    }
    const entry: ModelEntry = {
      id: model.id,
      name: model.name?.trim() || model.id.split("/").pop() || model.id,
      tier: "open-source",
      reasoning: true,
      tool_call: true,
      cost: { input: 0, output: 0 },
      limit: { context: model.context_length && model.context_length > 0 ? model.context_length : 200000, output: 65536 },
    }
    entries.push(entry)
    seen.set(key, entry)
  }
  return entries
}

// Build the model list fresh on every opencode instance start:
//   1. CLI-bundled catalog (vendor-maintained, refreshed by `cmd update`)
//   2. Command Code's public model endpoint (picks up brand-new models)
//   3. last synced cache, then the bundled snapshot as an offline fallback.
export async function loadSyncedModels(): Promise<SyncResult> {
  const bundled = loadCachedModels(BUNDLED_MODELS)
  const cached = loadCachedModels(CACHE_MODELS)
  const hints = new Map<string, ModelEntry>()
  for (const entry of [...(cached ?? []), ...(bundled ?? [])]) hints.set(entry.id.toLowerCase(), entry)

  let docModels: ModelEntry[] | null = null
  try {
    const modelsMd = findModelsMd()
    if (modelsMd) {
      const parsed = parseModelsMd(readFileSync(modelsMd, "utf-8"), hints)
      if (parsed.length >= 5) docModels = parsed
    }
  } catch {
    // intentionally silent: fall through to the API or cache
  }

  const apiModels = await fetchApiModels()

  let models: ModelEntry[] | null = null
  let source = ""
  if (docModels?.length) {
    models = mergeApiModels(docModels, apiModels ?? [])
    source = apiModels?.length ? "cli-catalog+api" : "cli-catalog"
  } else if (apiModels?.length) {
    models = mergeApiModels([], apiModels)
    source = "api"
  }

  if (!models?.length) {
    models = cached ?? bundled ?? []
    source = cached ? "cache" : "bundled"
  }

  if (!models.length) throw new Error("Command Code model catalog unavailable (no CLI, API, cache, or bundled list)")

  if (source !== "cache" && source !== "bundled") {
    try {
      writeAtomic(CACHE_MODELS, JSON.stringify(models, null, 2) + "\n")
    } catch {
      // intentionally silent: cache is best-effort
    }
  }

  return { models, source }
}
