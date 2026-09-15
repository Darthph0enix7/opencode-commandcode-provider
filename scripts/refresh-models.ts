import { writeFileSync } from "fs"
import { dirname, join } from "path"
import { fileURLToPath } from "url"
import { loadSyncedModels } from "../src/model-sync.js"

const root = join(dirname(fileURLToPath(import.meta.url)), "..")

const { models, source } = await loadSyncedModels()
writeFileSync(join(root, "models.json"), JSON.stringify(models, null, 2) + "\n")

const premium = models.filter((model) => model.tier === "premium").length
console.log(`Wrote ${models.length} models to models.json (source: ${source}; premium: ${premium})`)
