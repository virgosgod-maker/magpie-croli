// Croli 网关适配层。运行时只使用已保存的网关 Key，不启动或读取官方客户端。
import { readFile } from "node:fs/promises"
import { homedir } from "node:os"
import { join } from "node:path"

import { rawCredentialOf } from "./credential.mjs"
import { CODES, OFFICIAL_UA, errorResponse, errorText, readBody } from "./http.mjs"
import { ensureStreamUsage, rewriteJsonBody, stripModelPrefix } from "./request-rewrite.mjs"
import { rewriteRefusal } from "./status.mjs"
import { foldOpenAI, openAIEvents } from "./openai-sse.mjs"
import { discoverDesktopProxy } from "./croli-desktop-proxy.mjs"

export const PROVIDER = "croli"
export const BASE = "https://ds.croli.net/v1"
export const RELAY_SEARCH = "https://api.croli.net/relay/api/relay/search"
export const CREDITS_BALANCE = "https://ds.croli.net/credits/v1/account/balance"

/**
 * 账号可用性判据：登录时只要求中转资源覆盖这几个。
 *
 * 其余规格只参与排序 —— 上游增减模型不会再把一次正常的登录误判成失败。
 *
 * ⚠ `grok-4.6` **不在这里**：2026-11-20 实测它既不在 `GET /v1/models`
 * 的返回里，直连也只会拿到 `503 No available channel for model grok-4.6
 * under group default`。把「当前完全不可服务」的模型放进核心判据，
 * 等于把登录的成败挂在一个随时会被上游删掉的名字上 —— 正是这条判据要消灭的脆性。
 * 它仍留在 `MODEL_SPECS` 里参与排序，活目录会按 `/v1/models` 自动把它过滤掉。
 *
 * 判据是「至少命中一个」，不是「全部命中」：2026-11-25 实测免费池里已经没有
 * `gpt-6-luna`（只剩 `gpt-5.6-luna`），要求全中会让正常账号被判成没有网关 Key。
 * 上游改名换型号是常态，这里只用来确认资源确实在服务模型。
 */
export const CORE_MODEL_IDS = Object.freeze([
  "DeepSeek-V4.1-Flash",
  "gpt-6-luna",
  "claude-opus-5.5",
])

// context / output / efforts 目前只作文档用途（代码只读 id 与 name）。
// 新增几项的上游窗口未公布，先按同族口径填写，实测后再订正。
export const MODEL_SPECS = Object.freeze([
  { id: "DeepSeek-V4.1-Flash", name: "DeepSeek V4.1 Flash", context: 1_048_576, output: 384_000, efforts: ["low", "medium", "high"] },
  { id: "gpt-6-luna", name: "GPT-6 Luna", context: 1_050_000, output: 128_000, efforts: ["low", "medium", "high"] },
  { id: "gpt-6.1-sol", name: "GPT-6.1 Sol", context: 1_050_000, output: 128_000, efforts: ["low", "medium", "high"] },
  { id: "gpt-5.6-sol", name: "GPT-5.6 Sol", context: 1_000_000, output: 128_000, efforts: ["low", "medium", "high"] },
  { id: "gpt-5.6-luna", name: "GPT-5.6 Luna", context: 1_000_000, output: 128_000, efforts: ["low", "medium", "high"] },
  { id: "claude-opus-5.5", name: "Claude Opus 5.5", context: 1_000_000, output: 128_000, efforts: ["low", "medium", "high"] },
  { id: "claude-sonnet-5", name: "Claude Sonnet 5", context: 1_000_000, output: 128_000, efforts: ["low", "medium", "high"] },
  { id: "grok-4.6", name: "Grok 4.6", context: 500_000, output: 450_000, efforts: ["low", "medium", "high"] },
])

function fail(message, status, code) {
  return Object.assign(new Error(String(message)), { status, code })
}

function cleanSecret(value) {
  const secret = String(value ?? "").trim()
  if (!secret || secret.length > 4096 || /[\u0000-\u001f\u007f]/.test(secret)) return ""
  return secret
}

export function gatewayKeyOf(auth) {
  return cleanSecret(rawCredentialOf(auth))
}

function assertKey(auth) {
  const key = gatewayKeyOf(auth)
  if (!key) throw fail("未配置 Croli 网关 API Key", 401, CODES.signInRequired)
  if (/^eyJ[A-Za-z0-9_-]+\./.test(key)) {
    throw fail("这里需要 csk_ 网关 Key，不是 Croli 登录 JWT；请用一次性导入或改填网关 Key", 401, CODES.signInRequired)
  }
  return key
}

function authHeaders(key, json = false) {
  return {
    "user-agent": OFFICIAL_UA,
    accept: "application/json",
    authorization: `Bearer ${key}`,
    ...(json ? { "content-type": "application/json" } : {}),
  }
}

async function fetchJson(url, { key, fetchImpl = fetch, timeoutMs = 30_000, method = "GET", body } = {}) {
  let response
  try {
    response = await fetchImpl(url, {
      method,
      headers: authHeaders(key, body !== undefined),
      ...(body !== undefined ? { body: typeof body === "string" ? body : JSON.stringify(body) } : {}),
      signal: AbortSignal.timeout(timeoutMs),
      redirect: "error",
    })
  } catch (error) {
    const timeout = error?.name === "TimeoutError" || error?.name === "AbortError"
    throw fail(`Croli 接口${timeout ? "超时" : "网络错误"}：${error?.message ?? error}`, timeout ? 504 : 503, timeout ? CODES.upstreamTimeout : CODES.upstreamUnavailable)
  }
  const data = await readBody(response)
  if (!response.ok) {
    const message = errorText(data) || `Croli 接口返回 HTTP ${response.status}`
    throw fail(message, response.status, data?.error?.code || data?.code || (response.status === 401 ? CODES.sessionExpired : CODES.upstreamError))
  }
  if (!data || typeof data !== "object" || Array.isArray(data) || data._text) {
    throw fail("Croli 接口没有返回有效 JSON", 502, CODES.upstreamError)
  }
  return data
}

export async function fetchLiveModels(auth, { fetchImpl = fetch } = {}) {
  const key = assertKey(auth)
  const data = await fetchJson(`${BASE}/models`, { key, fetchImpl })
  if (!Array.isArray(data.data)) throw fail("Croli /models 缺少 data 数组", 502, CODES.upstreamError)
  const present = new Set(data.data.map((row) => row?.id).filter(Boolean))
  return MODEL_SPECS.filter((spec) => present.has(spec.id)).map((spec) => ({ id: spec.id, name: spec.name }))
}

export async function exchangeRelayKey(jwt, { fetchImpl = fetch } = {}) {
  const token = cleanSecret(jwt)
  if (!/^eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/.test(token)) {
    throw fail("Croli auth.json 中没有有效 access_token", 401, CODES.sessionExpired)
  }
  const data = await fetchJson(RELAY_SEARCH, { key: token, fetchImpl })
  const resources = Array.isArray(data.resources) ? data.resources : []
  const candidates = resources.map((resource) => {
    const key = cleanSecret(resource?.sk ?? resource?.apiKey)
    const endpoint = String(resource?.url ?? resource?.endpoint ?? "").replace(/\/+$/, "")
    const models = Array.isArray(resource?.models) ? resource.models : []
    const matches = MODEL_SPECS.filter((spec) => models.includes(spec.id)).length
    const coreMatches = CORE_MODEL_IDS.filter((id) => models.includes(id)).length
    return { resource, key, endpoint, matches, coreMatches }
  }).filter((item) => item.key && item.endpoint === BASE && item.coreMatches >= 1)
    .sort((a, b) => b.matches - a.matches)
  if (!candidates.length) throw fail("免费资源中没有可用的 Croli 网关 Key（缺核心模型）", 404, CODES.modelNotFound)
  const selected = candidates[0]
  return { key: selected.key, endpoint: selected.endpoint, matchedModels: selected.matches }
}

function jwtPayload(token) {
  try {
    const part = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")
    return JSON.parse(Buffer.from(part, "base64").toString("utf8"))
  } catch {
    return {}
  }
}

export function defaultCroliAuthFiles(env = process.env) {
  if (env.CROLI_AUTH_FILE) return [env.CROLI_AUTH_FILE]
  const files = []
  if (env.APPDATA) files.push(join(env.APPDATA, "croli", "auth.json"))
  if (process.platform === "darwin") files.push(join(homedir(), "Library", "Application Support", "Croli", "auth.json"))
  files.push(join(homedir(), ".config", "croli", "auth.json"))
  return [...new Set(files)]
}

export async function synchronizeDesktopAuth(auth, { env = process.env } = {}) {
  if (auth?.metadata?.kind !== "croli-desktop-proxy") return auth
  for (const file of defaultCroliAuthFiles(env)) {
    try {
      const document = JSON.parse(await readFile(file, "utf8"))
      const payload = jwtPayload(document.access_token)
      if (!payload.sub || payload.sub !== (auth.metadata.userId || auth.uid) || !Number.isFinite(payload.exp)) continue
      return { ...auth, metadata: { ...auth.metadata, croliAccessToken: document.access_token, croliExpiresAt: new Date(payload.exp * 1000).toISOString() } }
    } catch {}
  }
  return auth
}

export async function importDesktopKey({ authFile, fetchImpl = fetch, env = process.env, desktopProxy = discoverDesktopProxy } = {}) {
  const files = authFile ? [authFile] : defaultCroliAuthFiles(env)
  let document
  let source
  let lastError
  for (const file of files) {
    try {
      document = JSON.parse(await readFile(file, "utf8"))
      source = file
      break
    } catch (error) {
      lastError = error
    }
  }
  if (!document) throw fail(`没有读到 Croli 登录态：${lastError?.message ?? "auth.json 不存在"}`, 401, CODES.signInRequired)
  const payload = jwtPayload(document.access_token)
  const uid = String(payload.sub || "")
  const proxy = uid && await desktopProxy({ type: "oauth", metadata: { userId: uid } }, { authFile: source })
  const exchanged = proxy ? { key: "croli-desktop-proxy", endpoint: BASE } : await exchangeRelayKey(document.access_token, { fetchImpl })
  const expiresAt = Number(document.expires_at) > 0 ? Number(document.expires_at) * 1000 : Number(payload.exp) > 0 ? Number(payload.exp) * 1000 : 0
  return {
    ...exchanged,
    uid,
    refreshToken: proxy ? "" : cleanSecret(document.refresh_token),
    expiresAt: proxy ? 0 : expiresAt,
    metadata: {
      kind: proxy ? "croli-desktop-proxy" : "croli-gateway-key",
      userId: uid,
      croliAccessToken: document.access_token,
      ...(expiresAt ? { croliExpiresAt: new Date(expiresAt).toISOString() } : {}),
    },
    source,
  }
}

export async function fetchCredits(auth, { fetchImpl = fetch, now = Date.now } = {}) {
  const metadata = auth?.metadata && typeof auth.metadata === "object" ? auth.metadata : {}
  const jwt = cleanSecret(metadata.croliAccessToken)
  if (!jwt) throw fail("API Key 模式不读取客户端额度；模型调用不受影响", 400, CODES.invalidRequest)
  const expiresAt = Date.parse(metadata.croliExpiresAt || "")
  if (Number.isFinite(expiresAt) && expiresAt <= now()) {
    throw fail("额度查询会话已过期；保存的 API Key 仍可继续调用", 401, CODES.sessionExpired)
  }
  const data = await fetchJson(CREDITS_BALANCE, { key: jwt, fetchImpl })
  const balance = Number(data.balance)
  if (!Number.isFinite(balance)) throw fail("额度接口缺少有效 balance", 502, CODES.upstreamError)
  const trial = data.trial && typeof data.trial === "object" ? data.trial : null
  return {
    balance,
    trial: trial?.active === true ? {
      dailyLimit: Number(trial.dailyLimitCredits),
      used: Number(trial.usedCredits),
      remaining: Number(trial.remainingCredits),
      nextResetAt: trial.nextResetAt,
      endsAt: trial.endsAt,
    } : null,
  }
}

function gatewayUrl(input) {
  const raw = typeof input === "string" ? input : input instanceof URL ? input.toString() : input?.url || ""
  const url = new URL(raw)
  const base = new URL(BASE)
  if (url.origin !== base.origin || !url.pathname.startsWith(`${base.pathname}/`)) {
    throw fail("拒绝向非 Croli 网关地址转发 API Key", 400, CODES.invalidRequest)
  }
  return url.toString()
}

/** 只改鉴权、模型前缀和 stream usage；messages/system/tools 保持原样。 */
export async function forwardRequest(input, init, auth, { fetchImpl = fetch, includeUsage = true, desktopProxy } = {}) {
  let key
  let url
  try {
    key = assertKey(auth)
    url = gatewayUrl(input)
  } catch (error) {
    return errorResponse(error.message, error.status || 400, error.code || CODES.invalidRequest)
  }

  const proxy = desktopProxy ? await desktopProxy(auth) : null
  if (!proxy && auth?.metadata?.kind === "croli-desktop-proxy") {
    return errorResponse("Croli 本机代理不可用，请打开同账号的免费 Codex 会话后重试", 503, CODES.upstreamUnavailable, { signIn: "kept" })
  }
  if (proxy) url = `${proxy.baseURL}${new URL(url).pathname.slice(new URL(BASE).pathname.length)}${new URL(url).search}`
  const request = input instanceof Request ? input : null
  const incoming = new Headers(init?.headers ?? request?.headers ?? {})
  const headers = { authorization: `Bearer ${proxy?.token || key}`, "user-agent": OFFICIAL_UA }
  for (const [name, value] of incoming) {
    const lower = name.toLowerCase()
    if (lower.startsWith("x-croli-")) continue
    if (["authorization", "x-api-key", "x-goog-api-key", "host", "content-length", "content-encoding", "user-agent"].includes(lower)) continue
    headers[name] = value
  }
  if (!Object.keys(headers).some((name) => name.toLowerCase() === "content-type")) headers["content-type"] = "application/json"

  let body = init?.body
  if (body === undefined && request) body = await request.clone().text()
  if (typeof body === "string") {
    body = rewriteJsonBody(body, [stripModelPrefix(PROVIDER), ...(includeUsage ? [ensureStreamUsage] : [])])
  }

  try {
    const response = await fetchImpl(url, {
      method: init?.method ?? request?.method ?? "POST",
      headers,
      body,
      signal: init?.signal ?? request?.signal ?? AbortSignal.timeout(300_000),
      redirect: "error",
    })
    if (proxy && response.status === 401 && (await response.clone().text()).trim() === "invalid proxy token") {
      return errorResponse("Croli 本机代理凭据已轮换，请保持免费 Codex 会话运行后重试", 503, CODES.upstreamUnavailable, { signIn: "kept" })
    }
    if (proxy && response.ok && response.headers.get("content-type")?.includes("text/event-stream")) {
      let nonStream = false
      try { nonStream = JSON.parse(body).stream === false && new URL(url).pathname.endsWith("/chat/completions") } catch {}
      if (nonStream) {
        let complete = false
        const events = (async function* () {
          for await (const event of openAIEvents(response, { signal: init?.signal })) {
            if (event.error) throw fail(errorText(event) || "Croli 流式响应包含错误", 503, CODES.upstreamUnavailable)
            if (event.choices?.some(choice => choice.finish_reason != null)) complete = true
            yield event
          }
        })()
        const completion = await foldOpenAI(events)
        if (!complete) return errorResponse("Croli 流式响应在完成前中断", 504, CODES.upstreamTimeout)
        const responseHeaders = new Headers(response.headers)
        responseHeaders.set("content-type", "application/json")
        responseHeaders.delete("content-length")
        responseHeaders.delete("content-encoding")
        return new Response(JSON.stringify(completion), { status: response.status, headers: responseHeaders })
      }
    }
    return rewriteRefusal(response)
  } catch (error) {
    const timeout = error?.name === "TimeoutError" || error?.name === "AbortError"
    return errorResponse(
      `Croli 网关${timeout ? "超时" : "网络错误"}：${error?.message ?? error}`,
      timeout ? 504 : 503,
      timeout ? CODES.upstreamTimeout : CODES.upstreamUnavailable,
    )
  }
}
