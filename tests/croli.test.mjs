import test from "node:test"
import assert from "node:assert/strict"
import { mkdtemp, writeFile, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"

import { CroliAuthPlugin } from "../index.mjs"
import { OFFICIAL_UA, CODES } from "../src/http.mjs"
import {
  BASE,
  RELAY_SEARCH,
  CREDITS_BALANCE,
  MODEL_SPECS,
  forwardRequest,
  fetchLiveModels,
  exchangeRelayKey,
  importDesktopKey,
  fetchCredits,
} from "../src/croli.mjs"

const auth = { type: "api", key: "csk_test" }
const json = (data, status = 200) => new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json" } })
const fakeJwt = (payload = {}) => [
  Buffer.from(JSON.stringify({ alg: "none" })).toString("base64url"),
  Buffer.from(JSON.stringify(payload)).toString("base64url"),
  "signature",
].join(".")

test("直通只改模型前缀与 stream usage，消息和工具 schema 原样保留", async () => {
  const originalMessages = [
    { role: "system", content: "CALLER SYSTEM PROMPT — KEEP BYTE FOR BYTE" },
    { role: "user", content: "只回复 OK" },
  ]
  const originalTools = [{ type: "function", function: { name: "strict_tool", parameters: { type: "object", additionalProperties: false } } }]
  const raw = JSON.stringify({ model: "croli/gpt-6-luna", messages: originalMessages, tools: originalTools, stream: true })
  let captured
  const upstream = new Response("data: [DONE]\n\n", { status: 200, headers: { "content-type": "text/event-stream" } })
  const result = await forwardRequest(`${BASE}/chat/completions`, { method: "POST", body: raw, headers: { "x-custom": "kept" } }, auth, {
    fetchImpl: async (url, init) => {
      captured = { url, init }
      return upstream
    },
  })
  assert.equal(result, upstream, "200/SSE 必须返回同一 Response，不能读流")
  assert.equal(captured.url, `${BASE}/chat/completions`)
  assert.equal(captured.init.headers.authorization, "Bearer csk_test")
  assert.equal(captured.init.headers["user-agent"], OFFICIAL_UA)
  assert.equal(captured.init.headers["x-custom"], "kept")
  assert.equal(captured.init.headers["x-croli-session"], undefined)
  const sent = JSON.parse(captured.init.body)
  assert.equal(sent.model, "gpt-6-luna")
  assert.deepEqual(sent.messages, originalMessages)
  assert.deepEqual(sent.tools, originalTools)
  assert.deepEqual(sent.stream_options, { include_usage: true })
  assert.equal(sent.system, undefined)
  assert.equal(sent.instructions, undefined)
  assert.equal(sent.prompt, undefined)
})

test("关闭 usage 或非流式时不添加 stream_options", async () => {
  for (const [includeUsage, stream] of [[false, true], [true, false]]) {
    let body
    await forwardRequest(`${BASE}/chat/completions`, { method: "POST", body: JSON.stringify({ model: "grok-4.6", messages: [], stream }) }, auth, {
      includeUsage,
      fetchImpl: async (_url, init) => {
        body = JSON.parse(init.body)
        return json({ choices: [] })
      },
    })
    assert.equal(body.stream_options, undefined)
  }
})

test("外部 URL 不会偷走 API Key", async () => {
  let called = false
  const response = await forwardRequest("https://attacker.invalid/v1/chat/completions", { body: "{}" }, auth, {
    fetchImpl: async () => { called = true; return json({}) },
  })
  assert.equal(called, false)
  assert.equal(response.status, 400)
  assert.equal((await response.json()).error.code, CODES.invalidRequest)
})

test("误贴 JWT 时返回清晰 401，不发给上游", async () => {
  let called = false
  const response = await forwardRequest(`${BASE}/chat/completions`, { body: "{}" }, { type: "api", key: fakeJwt({ sub: "user_test" }) }, {
    fetchImpl: async () => { called = true; return json({}) },
  })
  assert.equal(called, false)
  assert.equal(response.status, 401)
  assert.match((await response.json()).error.message, /不是 Croli 登录 JWT/)
})

test("模型级 400 翻成 404，响应原文保留", async () => {
  const response = await forwardRequest(`${BASE}/chat/completions`, { body: "{}" }, auth, {
    fetchImpl: async () => json({ error: { message: "Model not found" } }, 400),
  })
  assert.equal(response.status, 404)
  assert.equal(response.headers.get("x-magpie-vendor-status"), "400")
  assert.deepEqual(await response.json(), { error: { message: "Model not found" } })
})

test("网络错误与超时分流为 503 / 504", async () => {
  const timeout = await forwardRequest(`${BASE}/chat/completions`, { body: "{}" }, auth, {
    fetchImpl: async () => { throw new DOMException("timed out", "TimeoutError") },
  })
  assert.equal(timeout.status, 504)
  assert.equal((await timeout.json()).error.code, CODES.upstreamTimeout)
  const network = await forwardRequest(`${BASE}/chat/completions`, { body: "{}" }, auth, {
    fetchImpl: async () => { throw new TypeError("fetch failed") },
  })
  assert.equal(network.status, 503)
  assert.equal((await network.json()).error.code, CODES.upstreamUnavailable)
})

test("live 目录只保留四个目标模型，不把隐藏模型带进来", async () => {
  const rows = [...MODEL_SPECS.map((spec) => ({ id: spec.id })), { id: "claude-fable-5-2" }]
  const models = await fetchLiveModels(auth, {
    fetchImpl: async (url, init) => {
      assert.equal(url, `${BASE}/models`)
      assert.equal(init.headers.authorization, "Bearer csk_test")
      assert.equal(init.headers["user-agent"], OFFICIAL_UA)
      return json({ data: rows })
    },
  })
  assert.deepEqual(models.map((model) => model.id), MODEL_SPECS.map((spec) => spec.id))
})

test("live 缺少目标模型时不从静态表补回", async () => {
  const models = await fetchLiveModels(auth, { fetchImpl: async () => json({ data: [{ id: "grok-4.6" }] }) })
  assert.deepEqual(models.map((model) => model.id), ["grok-4.6"])
})

test("relay 选择覆盖核心模型的网关 Key", async () => {
  const result = await exchangeRelayKey(fakeJwt({ sub: "user_test" }), {
    fetchImpl: async (url) => {
      assert.equal(url, RELAY_SEARCH)
      return json({ resources: [
        { url: BASE, sk: "csk_one", models: ["grok-4.6"] },
        { url: BASE, sk: "csk_four", models: MODEL_SPECS.map((spec) => spec.id) },
      ] })
    },
  })
  assert.equal(result.key, "csk_four")
  assert.equal(result.matchedModels, MODEL_SPECS.length)
  await assert.rejects(() => exchangeRelayKey(fakeJwt(), {
    fetchImpl: async () => json({ resources: [{ url: "https://other.invalid/v1", sk: "csk_bad", models: ["grok-4.6"] }] }),
  }), /没有可用的 Croli 网关 Key/)
  await assert.rejects(() => exchangeRelayKey(fakeJwt(), {
    fetchImpl: async () => json({ resources: [{ url: BASE, sk: "csk_partial", models: ["grok-4.6"] }] }),
  }), /没有可用的 Croli 网关 Key/)
})

test("relay 核心判据是「至少命中一个」：上游删掉 gpt-6-luna 后仍能导入", async () => {
  const result = await exchangeRelayKey(fakeJwt({ sub: "user_test" }), {
    fetchImpl: async () => json({ resources: [
      { url: BASE, sk: "csk_no_core", models: ["grok-4.6", "gemini-4-argon"] },
      { url: BASE, sk: "csk_one_core", models: ["DeepSeek-V4.1-Flash", "gpt-6.1-sol", "claude-opus-5.5"] },
    ] }),
  })
  assert.equal(result.key, "csk_one_core")
})

test("一次性导入只依赖临时 auth 文件，返回独立 Key 与账号 metadata", async () => {
  const dir = await mkdtemp(join(tmpdir(), "magpie-croli-test-"))
  try {
    const file = join(dir, "auth.json")
    const token = fakeJwt({ sub: "user_test", exp: 2_000_000_000 })
    await writeFile(file, JSON.stringify({ access_token: token, expires_at: 2_000_000_000 }))
    const result = await importDesktopKey({ authFile: file, fetchImpl: async () => json({ resources: [{ url: BASE, sk: "csk_imported", models: MODEL_SPECS.map((spec) => spec.id) }] }) })
    assert.equal(result.key, "csk_imported")
    assert.equal(result.uid, "user_test")
    assert.equal(result.metadata.croliAccessToken, token)
    assert.equal(result.metadata.croliExpiresAt, new Date(2_000_000_000_000).toISOString())
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})

test("OAuth callback 完成一次性导入，之后保存的是网关 Key", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "magpie-croli-callback-"))
  const previous = process.env.CROLI_AUTH_FILE
  try {
    const file = join(dir, "auth.json")
    await writeFile(file, JSON.stringify({ access_token: fakeJwt({ sub: "user_callback", exp: 2_000_000_000 }), refresh_token: "refresh_callback", expires_at: 2_000_000_000 }))
    process.env.CROLI_AUTH_FILE = file
    t.mock.method(globalThis, "fetch", async () => json({ resources: [{ url: BASE, sk: "csk_callback", models: MODEL_SPECS.map((spec) => spec.id) }] }))
    const plugin = await CroliAuthPlugin()
    const flow = await plugin.auth.methods[0].authorize()
    const result = await flow.callback()
    assert.equal(result.type, "success")
    assert.equal(result.access, "csk_callback")
    assert.equal(result.expires, 2_000_000_000_000)
    assert.equal(result.refresh, "refresh_callback")
    assert.equal(result.uid, "user_callback")
  } finally {
    if (previous === undefined) delete process.env.CROLI_AUTH_FILE
    else process.env.CROLI_AUTH_FILE = previous
    await rm(dir, { recursive: true, force: true })
  }
})

test("真实额度形状映射为今日窗口和余额窗口", async (t) => {
  const token = fakeJwt({ sub: "user_usage" })
  const creditsResponse = () => json({ balance: 4000, trial: { active: true, dailyLimitCredits: 2000, usedCredits: 500, remainingCredits: 1500, nextResetAt: "2026-11-21T00:00:00Z", endsAt: "2026-11-27T00:00:00Z" } })
  t.mock.method(globalThis, "fetch", async (url) => {
    assert.equal(url, CREDITS_BALANCE)
    return creditsResponse()
  })
  const value = await fetchCredits({ metadata: { croliAccessToken: token, croliExpiresAt: "2099-01-01T00:00:00Z" } }, {
    now: () => 1_700_000_000_000,
  })
  assert.equal(value.balance, 4000)
  assert.equal(value.trial.used, 500)
  const plugin = await CroliAuthPlugin({ magpie: { option: () => false } })
  const usage = await plugin.auth.usage(async () => ({ type: "oauth", access: "csk_usage", metadata: { croliAccessToken: token, croliExpiresAt: "2099-01-01T00:00:00Z" } }))
  assert.equal(usage.windows.length, 2)
  assert.equal(usage.windows[0].used, 25)
  assert.equal(usage.windows[1].balance, 4000)
})

test("过期额度会话不导致 API Key 被标记失效", async () => {
  await assert.rejects(() => fetchCredits({ metadata: { croliAccessToken: fakeJwt(), croliExpiresAt: "2000-01-01T00:00:00Z" } }), /模型调用不受影响|仍可继续调用/)
  const plugin = await CroliAuthPlugin({ magpie: { option: () => false } })
  const usage = await plugin.auth.usage(async () => ({ type: "oauth", access: "csk_expired_usage", metadata: { croliAccessToken: fakeJwt(), croliExpiresAt: "2000-01-01T00:00:00Z" } }))
  assert.match(usage.error, /模型调用不受影响/)
  assert.equal(usage.signIn, undefined)
})

test("provider.models 仅把网关 401 标成 Key 失效", async (t) => {
  const plugin = await CroliAuthPlugin()
  t.mock.method(globalThis, "fetch", async () => json({ error: { message: "unauthorized" } }, 401))
  await assert.rejects(() => plugin.provider.models({}, { auth }), (error) => error.signIn === "expired")
})

test("积分读取失败时仍保留邀请窗口，不改写到 user 字段", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "croli-usage-invite-test-"))
  try {
    const plugin = await CroliAuthPlugin({}, { inviteCacheDir: dir })
    t.mock.method(globalThis, "fetch", async (url) => {
      if (url === CREDITS_BALANCE) return json({ code: "rate_limited", message: "too many" }, 429)
      if (String(url).includes("/v1/invite/code")) return json({ inviteCode: "Degraded123", inviterRewardCredits: 2500, inviteeRewardCredits: 3000 })
      if (String(url).includes("/v1/invite/list")) return json({ completedCount: 0, totalRewardCredits: 0 })
      throw new Error(`unexpected ${url}`)
    })
    const token = fakeJwt({ sub: "user_degraded" })
    const usage = await plugin.auth.usage(async () => ({ type: "oauth", access: "csk_degraded_usage", metadata: { userId: "user_degraded", croliAccessToken: token, croliExpiresAt: "2099-01-01T00:00:00Z" } }))
    assert.match(usage.plan, /积分暂不可用/)
    assert.equal(usage.error, undefined)
    assert.equal(usage.user, undefined)
    assert.equal(usage.windows.length, 1)
    assert.equal(usage.windows[0].display, "https://www.croli.net/invite/?code=Degraded123")
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})
