#!/usr/bin/env node

import { execFile } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

import { chromium } from 'playwright';

const execFileAsync = promisify(execFile);
const CHAT_INPUT_SELECTOR =
  '#chat-input [contenteditable="true"]:visible, ' +
  '#chat-input[contenteditable="true"]:visible';
const SEND_BUTTON_SELECTOR = '#send-message-button';
const RESPONSE_SELECTOR = '#response-content-container';
const STOP_BUTTON_SELECTOR = '#stop-response-button';
const VOICE_INPUT_BUTTON_SELECTOR = '#voice-input-button';
const CONTINUE_RESPONSE_BUTTON_SELECTOR = '#continue-response-button';
const DEFAULT_METRICS_URL = 'http://127.0.0.1:8000/metrics';
const DEFAULT_RESPONSE_TIMEOUT_MS = 900_000;
const PAGE_READY_TIMEOUT_MS = 90_000;

function usage() {
  return `Usage: node experiments/open_webui_live_demo.mjs \\
  --base-url URL \\
  --scenario FILE \\
  --raw-dir DIRECTORY \\
  --json FILE \\
  [--channel chrome]

Record a reproducible, real Open WebUI demo with Playwright.

Required arguments:
  --base-url URL       Authless Open WebUI origin, for example http://127.0.0.1:8081
  --scenario FILE      Scenario JSON produced by open_webui_demo_scenario.py
  --raw-dir DIRECTORY  Destination directory for unedited WebM recordings
  --json FILE          Destination capture artifact JSON

Optional arguments:
  --channel NAME       Playwright Chromium channel (default: chrome)
  --help               Show this help

The Open WebUI instance must be fresh/authless and have a default model selected.
The output JSON contains no cookies, browser storage, bearer keys, or prompt bodies.`;
}

function parseArguments(argv) {
  const options = {};
  const known = new Set(['base-url', 'scenario', 'raw-dir', 'json', 'channel']);

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--help' || argument === '-h') {
      options.help = true;
      continue;
    }
    if (!argument.startsWith('--')) {
      throw new Error(`unexpected positional argument: ${argument}`);
    }

    const equalsIndex = argument.indexOf('=');
    const key = argument.slice(2, equalsIndex === -1 ? undefined : equalsIndex);
    if (!known.has(key)) {
      throw new Error(`unknown argument --${key}; run with --help for usage`);
    }
    const value =
      equalsIndex === -1 ? argv[index + 1] : argument.slice(equalsIndex + 1);
    if (!value || (equalsIndex === -1 && value.startsWith('--'))) {
      throw new Error(`--${key} requires a value`);
    }
    if (equalsIndex === -1) {
      index += 1;
    }
    options[key] = value;
  }

  if (options.help) {
    return options;
  }
  for (const key of ['base-url', 'scenario', 'raw-dir', 'json']) {
    if (!options[key]) {
      throw new Error(`missing required argument --${key}; run with --help for usage`);
    }
  }

  const parsedBaseUrl = new URL(options['base-url']);
  if (!['http:', 'https:'].includes(parsedBaseUrl.protocol)) {
    throw new Error('--base-url must use http:// or https://');
  }
  if (parsedBaseUrl.username || parsedBaseUrl.password || parsedBaseUrl.search) {
    throw new Error('--base-url must not contain credentials or query parameters');
  }
  parsedBaseUrl.hash = '';

  const scenarioPath = resolve(options.scenario);
  const jsonPath = resolve(options.json);
  if (scenarioPath === jsonPath) {
    throw new Error('--json must not overwrite the scenario file');
  }

  return {
    baseUrl: parsedBaseUrl.href.replace(/\/$/, ''),
    scenarioPath,
    rawDir: resolve(options['raw-dir']),
    jsonPath,
    channel: options.channel || 'chrome',
  };
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function requireNonEmptyString(value, path) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${path} must be a non-empty string`);
  }
  return value;
}

function requirePositiveInteger(value, path) {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`${path} must be a positive integer`);
  }
  return value;
}

function requireHttpEndpoint(value, path) {
  const endpoint = requireNonEmptyString(value, path);
  let parsed;
  try {
    parsed = new URL(endpoint);
  } catch (error) {
    throw new Error(`${path} is not a valid URL: ${error.message}`);
  }
  if (!['http:', 'https:'].includes(parsed.protocol)) {
    throw new Error(`${path} must use http:// or https://`);
  }
  if (parsed.username || parsed.password) {
    throw new Error(`${path} must not contain credentials`);
  }
  return parsed.href;
}

function promptContent(prompt, path) {
  if (typeof prompt?.content === 'string' && prompt.content.trim() !== '') {
    return prompt.content;
  }
  if (Array.isArray(prompt?.messages)) {
    const lastUserMessage = [...prompt.messages]
      .reverse()
      .find((message) => message?.role === 'user' && typeof message?.content === 'string');
    if (lastUserMessage?.content.trim()) {
      return lastUserMessage.content;
    }
  }
  throw new Error(`${path}.content must be a non-empty string`);
}

function normalizedPrompt(prompt, path, { requireTokens = false } = {}) {
  if (!prompt || typeof prompt !== 'object' || Array.isArray(prompt)) {
    throw new Error(`${path} must be an object`);
  }
  const normalized = {
    promptId: requireNonEmptyString(prompt.prompt_id, `${path}.prompt_id`),
    label: requireNonEmptyString(prompt.label, `${path}.label`),
    content: promptContent(prompt, path),
    contentSha256:
      typeof prompt.content_sha256 === 'string'
        ? prompt.content_sha256
        : sha256(promptContent(prompt, path)),
    promptSha256: typeof prompt.prompt_sha256 === 'string' ? prompt.prompt_sha256 : null,
    renderedTokenCount: null,
  };
  if (requireTokens || prompt.rendered_token_count !== undefined) {
    normalized.renderedTokenCount = requirePositiveInteger(
      prompt.rendered_token_count,
      `${path}.rendered_token_count`,
    );
  }
  return normalized;
}

function validateScenario(rawScenario) {
  if (!rawScenario || typeof rawScenario !== 'object' || Array.isArray(rawScenario)) {
    throw new Error('scenario root must be a JSON object');
  }
  if (!Array.isArray(rawScenario.concurrent_prompts) || rawScenario.concurrent_prompts.length !== 4) {
    throw new Error('scenario.concurrent_prompts must contain exactly four prompts');
  }

  const longPrompt = normalizedPrompt(rawScenario.long_prompt, 'scenario.long_prompt', {
    requireTokens: true,
  });
  const needle = rawScenario.long_prompt?.needle;
  if (!needle || typeof needle !== 'object' || Array.isArray(needle)) {
    throw new Error('scenario.long_prompt.needle must be an object');
  }
  const exactNeedlePosition = needle.rendered_token_index;
  const approximateNeedlePosition = needle.rendered_token_index_approx;
  if (Number.isInteger(exactNeedlePosition) && exactNeedlePosition >= 0) {
    longPrompt.needlePosition = exactNeedlePosition;
    longPrompt.needlePositionExact = true;
  } else if (Number.isInteger(approximateNeedlePosition) && approximateNeedlePosition >= 0) {
    longPrompt.needlePosition = approximateNeedlePosition;
    longPrompt.needlePositionExact = false;
  } else {
    throw new Error(
      'scenario.long_prompt.needle must provide rendered_token_index or rendered_token_index_approx',
    );
  }
  longPrompt.needle = needle;

  const followUpObject =
    typeof rawScenario.follow_up === 'string'
      ? {
          prompt_id: 'common-follow-up',
          label: 'Common follow-up',
          content: rawScenario.follow_up,
        }
      : rawScenario.follow_up;

  const responseTimeoutMs =
    rawScenario.response_timeout_ms === undefined
      ? DEFAULT_RESPONSE_TIMEOUT_MS
      : requirePositiveInteger(rawScenario.response_timeout_ms, 'scenario.response_timeout_ms');

  return {
    kind: rawScenario.kind || null,
    schemaVersion: rawScenario.schema_version ?? null,
    declaredSha256: rawScenario.scenario_sha256 || null,
    longPrompt,
    concurrentPrompts: rawScenario.concurrent_prompts.map((prompt, index) =>
      normalizedPrompt(prompt, `scenario.concurrent_prompts[${index}]`),
    ),
    followUp: normalizedPrompt(followUpObject, 'scenario.follow_up'),
    providerMetricsUrl: requireHttpEndpoint(
      rawScenario.provider_metrics_url || DEFAULT_METRICS_URL,
      'scenario.provider_metrics_url',
    ),
    providerHealthUrl:
      rawScenario.provider_health_url === undefined
        ? null
        : requireHttpEndpoint(rawScenario.provider_health_url, 'scenario.provider_health_url'),
    responseTimeoutMs,
  };
}

async function loadScenario(path) {
  let bytes;
  try {
    bytes = await readFile(path);
  } catch (error) {
    throw new Error(`cannot read scenario ${path}: ${error.message}`);
  }
  let rawScenario;
  try {
    rawScenario = JSON.parse(bytes.toString('utf8'));
  } catch (error) {
    throw new Error(`invalid JSON in scenario ${path}: ${error.message}`);
  }
  return {
    scenario: validateScenario(rawScenario),
    fileSha256: sha256(bytes),
  };
}

function cleanError(error) {
  const message = error instanceof Error ? error.message : String(error);
  return message.replace(/https?:\/\/[^\s)]+/g, (match) => {
    try {
      const parsed = new URL(match);
      parsed.username = '';
      parsed.password = '';
      parsed.search = '';
      parsed.hash = '';
      return parsed.href;
    } catch {
      return '<redacted-url>';
    }
  });
}

function safeUrl(value) {
  const parsed = new URL(value);
  parsed.username = '';
  parsed.password = '';
  parsed.search = '';
  parsed.hash = '';
  return parsed.href;
}

function timestamp() {
  return new Date().toISOString();
}

function rounded(value, digits = 6) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return null;
  }
  return Number(value.toFixed(digits));
}

function offsetSeconds(originMs, eventMs) {
  return rounded((eventMs - originMs) / 1000);
}

function endpointFromBase(baseUrl, path) {
  return new URL(path.replace(/^\//, ''), `${baseUrl}/`).href;
}

async function fetchJson(url, timeoutMs = 5_000) {
  const response = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) });
  const body = await response.text();
  let data = null;
  try {
    data = body === '' ? null : JSON.parse(body);
  } catch {
    throw new Error(`HTTP ${response.status} returned non-JSON data from ${safeUrl(url)}`);
  }
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} from ${safeUrl(url)}`);
  }
  return { status: response.status, data };
}

function selectedHealth(data) {
  if (!data || typeof data !== 'object') {
    return null;
  }
  const fields = [
    'ok',
    'queue_depth',
    'pending_queue_depth',
    'running',
    'free_state_slots',
    'chat_sessions',
    'token_sessions',
    'timed_out_requests',
    'worker_alive',
  ];
  const selected = {};
  for (const field of fields) {
    if (data[field] !== undefined) {
      selected[field] = data[field];
    }
  }
  selected.has_last_error = data.last_error !== null && data.last_error !== undefined;
  return selected;
}

function selectedMetrics(data) {
  const fieldMap = {
    server: [
      'ready',
      'total_requests',
      'total_errors',
      'total_cancelled',
      'total_timed_out',
      'max_queue',
      'request_timeout_s',
      'chat_session_ttl_s',
      'max_chat_sessions',
      'pending_queue_depth',
      'tracked_requests',
      'chat_sessions',
      'worker_alive',
    ],
    engine: [
      'steps',
      'scheduled_tokens',
      'admitted_requests',
      'finished_requests',
      'error_count',
      'prefill_calls',
      'decode_batches',
      'decode_rows',
      'max_waiting',
      'max_running',
      'max_runnable_rows',
      'sessions_opened',
      'session_turns_completed',
      'session_reuse_hits',
      'session_reuse_misses',
      'prefix_tokens_reused',
      'full_reprefill_turns',
      'backpressure_events',
      'retraction_events',
      'queue_depth',
      'runnable_rows',
      'parked_sessions',
      'resident_sessions',
      'free_state_slots',
      'persistent_padded_decode',
      'persistent_padded_decode_steps',
      'persistent_padded_decode_cuda_graph',
      'use_native_gemma_forward',
      'native_gemma_attention_backend',
      'native_gemma_projection_backend',
      'native_gemma_weight_backend',
      'native_gemma_checkpoint_loader',
    ],
  };
  const selected = {};
  for (const [section, fields] of Object.entries(fieldMap)) {
    selected[section] = {};
    for (const field of fields) {
      if (data?.[section]?.[field] !== undefined) {
        selected[section][field] = data[section][field];
      }
    }
  }
  return selected;
}

function deriveHealthUrl(metricsUrl, explicitHealthUrl) {
  if (explicitHealthUrl) {
    return explicitHealthUrl;
  }
  const parsed = new URL(metricsUrl);
  parsed.pathname = parsed.pathname.replace(/\/metrics\/?$/, '/health');
  parsed.search = '';
  parsed.hash = '';
  return parsed.href;
}

async function providerProbe(scenario) {
  const metricsUrl = scenario.providerMetricsUrl;
  const healthUrl = deriveHealthUrl(metricsUrl, scenario.providerHealthUrl);
  const capturedAt = timestamp();
  const [healthResult, metricsResult] = await Promise.allSettled([
    fetchJson(healthUrl),
    fetchJson(metricsUrl),
  ]);

  return {
    captured_at: capturedAt,
    health:
      healthResult.status === 'fulfilled'
        ? {
            ok: true,
            endpoint: safeUrl(healthUrl),
            http_status: healthResult.value.status,
            values: selectedHealth(healthResult.value.data),
            error: null,
          }
        : {
            ok: false,
            endpoint: safeUrl(healthUrl),
            http_status: null,
            values: null,
            error: cleanError(healthResult.reason),
          },
    metrics:
      metricsResult.status === 'fulfilled'
        ? {
            ok: true,
            endpoint: safeUrl(metricsUrl),
            http_status: metricsResult.value.status,
            values: selectedMetrics(metricsResult.value.data),
            error: null,
          }
        : {
            ok: false,
            endpoint: safeUrl(metricsUrl),
            http_status: null,
            values: null,
            error: cleanError(metricsResult.reason),
          },
  };
}

function numericDelta(before, after) {
  if (!before || !after) {
    return null;
  }
  const result = {};
  for (const key of Object.keys(after)) {
    if (typeof after[key] === 'number' && typeof before[key] === 'number') {
      result[key] = rounded(after[key] - before[key]);
    } else if (
      after[key] &&
      before[key] &&
      typeof after[key] === 'object' &&
      typeof before[key] === 'object'
    ) {
      result[key] = numericDelta(before[key], after[key]);
    }
  }
  return result;
}

function providerDelta(beforeProbe, afterProbe) {
  return numericDelta(beforeProbe?.metrics?.values, afterProbe?.metrics?.values);
}

function reuseDelta(delta) {
  const engine = delta?.engine;
  if (!engine) {
    return null;
  }
  return {
    session_reuse_hits: engine.session_reuse_hits ?? null,
    session_reuse_misses: engine.session_reuse_misses ?? null,
    session_turns_completed: engine.session_turns_completed ?? null,
    full_reprefill_turns: engine.full_reprefill_turns ?? null,
    prefix_tokens_reused: engine.prefix_tokens_reused ?? null,
  };
}

function safeSessionRestarts(delta, completedTurns) {
  const server = delta?.server;
  const engine = delta?.engine;
  const reuseHits = engine?.session_reuse_hits;
  const sessionsOpened = engine?.sessions_opened;
  const cleanCounters = [
    server?.total_errors,
    server?.total_cancelled,
    server?.total_timed_out,
    engine?.error_count,
    engine?.session_reuse_misses,
  ];
  if (
    !Number.isFinite(reuseHits) ||
    !Number.isFinite(sessionsOpened) ||
    !cleanCounters.every((value) => value === 0) ||
    server?.total_requests !== completedTurns ||
    engine?.session_turns_completed !== completedTurns ||
    reuseHits + sessionsOpened !== completedTurns
  ) {
    return null;
  }
  return sessionsOpened;
}

async function queryGpus() {
  const { stdout } = await execFileAsync(
    'nvidia-smi',
    [
      '--query-gpu=index,name,memory.total,memory.used',
      '--format=csv,noheader,nounits',
    ],
    { timeout: 5_000, maxBuffer: 1024 * 1024 },
  );
  return stdout
    .trim()
    .split('\n')
    .filter(Boolean)
    .map((line) => {
      const parts = line.split(',').map((part) => part.trim());
      if (parts.length < 4) {
        throw new Error(`unexpected nvidia-smi row: ${line}`);
      }
      const [index, ...middle] = parts;
      const usedMiB = Number(middle.pop());
      const totalMiB = Number(middle.pop());
      const name = middle.join(', ');
      if (!Number.isFinite(totalMiB) || !Number.isFinite(usedMiB)) {
        throw new Error(`non-numeric memory value from nvidia-smi for GPU ${index}`);
      }
      return { index: Number(index), name, totalMiB, usedMiB };
    });
}

class GpuSampler {
  constructor(originMs, intervalMs = 250) {
    this.originMs = originMs;
    this.intervalMs = intervalMs;
    this.devices = new Map();
    this.errors = new Set();
    this.sampleCount = 0;
    this.timer = null;
    this.pending = null;
    this.firstSampleOffset = null;
    this.lastSampleOffset = null;
  }

  async sample() {
    if (this.pending) {
      return this.pending;
    }
    this.pending = (async () => {
      try {
        const sampleOffset = offsetSeconds(this.originMs, performance.now());
        const devices = await queryGpus();
        this.sampleCount += 1;
        this.firstSampleOffset ??= sampleOffset;
        this.lastSampleOffset = sampleOffset;
        for (const device of devices) {
          const previous = this.devices.get(device.index);
          this.devices.set(device.index, {
            index: device.index,
            name: device.name,
            total_mib: device.totalMiB,
            baseline_used_mib: previous?.baseline_used_mib ?? device.usedMiB,
            peak_used_mib: Math.max(previous?.peak_used_mib ?? device.usedMiB, device.usedMiB),
            last_used_mib: device.usedMiB,
          });
        }
      } catch (error) {
        this.errors.add(cleanError(error));
      } finally {
        this.pending = null;
      }
    })();
    return this.pending;
  }

  async start() {
    await this.sample();
    this.timer = setInterval(() => {
      void this.sample();
    }, this.intervalMs);
  }

  async stop() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    await this.sample();
    if (this.pending) {
      await this.pending;
    }
    return {
      sample_interval_ms: this.intervalMs,
      sample_count: this.sampleCount,
      first_sample_offset_s: this.firstSampleOffset,
      last_sample_offset_s: this.lastSampleOffset,
      devices: [...this.devices.values()].sort((left, right) => left.index - right.index),
      error: this.errors.size === 0 ? null : [...this.errors].join('; '),
    };
  }
}

function contextOptions(viewport, rawDir) {
  return {
    viewport,
    colorScheme: 'dark',
    recordVideo: {
      dir: rawDir,
      size: viewport,
    },
  };
}

async function installFreshUiState(context, webuiVersion) {
  await context.addInitScript((version) => {
    try {
      localStorage.clear();
      sessionStorage.clear();
      localStorage.setItem('theme', 'dark');
      if (version) {
        localStorage.setItem('version', version);
        localStorage.setItem(
          'settings',
          JSON.stringify({ version, showChangelog: false }),
        );
      }
    } catch {}
  }, webuiVersion);
}

async function dismissTransientUi(page) {
  await page.keyboard.press('Escape').catch(() => {});
  const names = [
    /^close$/i,
    /^dismiss$/i,
    /^got it$/i,
    /^okay,? let's go!?$/i,
    /^skip$/i,
    /^maybe later$/i,
  ];
  for (const name of names) {
    const buttons = page.getByRole('button', { name });
    const count = await buttons.count().catch(() => 0);
    for (let index = 0; index < count; index += 1) {
      const button = buttons.nth(index);
      if (await button.isVisible().catch(() => false)) {
        await button.click({ timeout: 2_000 }).catch(() => {});
      }
    }
  }
  const toastCloseButtons = page.locator(
    '[data-sonner-toast] button[aria-label="Close"], [data-sonner-toast] button[data-close-button]',
  );
  const toastCount = await toastCloseButtons.count().catch(() => 0);
  for (let index = 0; index < toastCount; index += 1) {
    await toastCloseButtons.nth(index).click({ timeout: 1_000 }).catch(() => {});
  }
}

async function waitForChatInput(page) {
  const input = page.locator(CHAT_INPUT_SELECTOR).first();
  try {
    await input.waitFor({ state: 'visible', timeout: PAGE_READY_TIMEOUT_MS });
  } catch (error) {
    const currentUrl = safeUrl(page.url());
    const authHint = /\/auth(?:\/|$)/.test(new URL(page.url()).pathname)
      ? ' The page redirected to authentication.'
      : '';
    throw new Error(
      `chat input did not become ready at ${currentUrl}.${authHint} ` +
        'Launch a dedicated fresh Open WebUI instance with WEBUI_AUTH=false and a default model.',
      { cause: error },
    );
  }
  return input;
}

async function waitForFollowUpReady(page) {
  const input = await waitForChatInput(page);
  const deadline = performance.now() + PAGE_READY_TIMEOUT_MS;
  while (performance.now() < deadline) {
    const [inputVisible, generationState] = await Promise.all([
      input.isVisible().catch(() => false),
      visibleGenerationState(page, 0, ''),
    ]);
    if (
      inputVisible &&
      !generationState.stopVisible &&
      (generationState.voiceVisible || generationState.continueVisible)
    ) {
      return;
    }
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 50));
  }
  throw new Error('chat input and completion controls did not become ready for follow-up');
}

async function preparePage(page, baseUrl) {
  await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: PAGE_READY_TIMEOUT_MS });
  await dismissTransientUi(page);
  await waitForChatInput(page);
  await page.waitForTimeout(750);
  await dismissTransientUi(page);
  const blockingModal = page.locator('.modal[aria-modal="true"]:visible');
  if ((await blockingModal.count()) > 0) {
    const changelogButton = page.getByRole('button', { name: /okay,? let's go!?/i });
    if ((await changelogButton.count()) > 0) {
      await changelogButton.last().click({ force: true, timeout: 2_000 });
    }
    await blockingModal.first().waitFor({ state: 'hidden', timeout: 5_000 });
  }
}

async function warmBrowser(browser, baseUrl, webuiVersion) {
  const context = await browser.newContext({
    viewport: { width: 1280, height: 720 },
    colorScheme: 'dark',
  });
  try {
    await installFreshUiState(context, webuiVersion);
    const page = await context.newPage();
    await preparePage(page, baseUrl);
  } finally {
    await context.close();
  }
}

async function createRecordedSession(browser, baseUrl, rawDir, viewport, webuiVersion) {
  const context = await browser.newContext(contextOptions(viewport, rawDir));
  try {
    await installFreshUiState(context, webuiVersion);
    const pageCreatedMonotonicMs = performance.now();
    const pageCreatedAt = timestamp();
    const page = await context.newPage();
    const video = page.video();
    await preparePage(page, baseUrl);
    return {
      context,
      page,
      video,
      pageCreatedMonotonicMs,
      pageCreatedAt,
    };
  } catch (error) {
    await context.close().catch(() => {});
    throw error;
  }
}

async function setFixedOverlay(page, title, lines) {
  await page.evaluate(
    ({ overlayTitle, overlayLines }) => {
      let overlay = document.getElementById('wkvm-capture-overlay');
      if (!overlay) {
        overlay = document.createElement('aside');
        overlay.id = 'wkvm-capture-overlay';
        document.body.appendChild(overlay);
      }
      Object.assign(overlay.style, {
        position: 'fixed',
        zIndex: '2147483646',
        top: '12px',
        right: '12px',
        maxWidth: 'min(520px, calc(100vw - 24px))',
        padding: '9px 12px',
        border: '1px solid rgba(96, 165, 250, 0.75)',
        borderRadius: '10px',
        background: 'rgba(8, 15, 29, 0.92)',
        boxShadow: '0 8px 28px rgba(0, 0, 0, 0.35)',
        color: '#e5eefc',
        font: '600 13px/1.35 ui-monospace, SFMono-Regular, Menlo, monospace',
        pointerEvents: 'none',
      });
      overlay.replaceChildren();
      const heading = document.createElement('div');
      heading.textContent = overlayTitle;
      heading.style.color = '#60a5fa';
      heading.style.fontWeight = '800';
      overlay.appendChild(heading);
      for (const line of overlayLines) {
        const row = document.createElement('div');
        row.textContent = line;
        overlay.appendChild(row);
      }
    },
    { overlayTitle: title, overlayLines: lines },
  );
}

async function setCountdown(page, text) {
  await page.evaluate((countdownText) => {
    let overlay = document.getElementById('wkvm-capture-countdown');
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.id = 'wkvm-capture-countdown';
      document.body.appendChild(overlay);
    }
    Object.assign(overlay.style, {
      position: 'fixed',
      zIndex: '2147483647',
      inset: '0',
      display: countdownText ? 'grid' : 'none',
      placeItems: 'center',
      background: 'rgba(3, 7, 18, 0.38)',
      color: '#ffffff',
      font: '900 96px/1 ui-sans-serif, system-ui, sans-serif',
      textShadow: '0 5px 25px rgba(0, 0, 0, 0.9)',
      pointerEvents: 'none',
    });
    overlay.textContent = countdownText;
  }, text);
}

async function synchronizedCountdown(sessions, label) {
  const startedMs = performance.now();
  for (const value of ['3', '2', '1']) {
    await Promise.all(sessions.map((session) => setCountdown(session.page, value)));
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 1_000));
  }
  await Promise.all(sessions.map((session) => setCountdown(session.page, 'GO')));
  await new Promise((resolveDelay) => setTimeout(resolveDelay, 250));
  await Promise.all(sessions.map((session) => setCountdown(session.page, '')));
  return {
    label,
    started_at_monotonic_ms: rounded(startedMs, 3),
    started_offsets_s: Object.fromEntries(
      sessions.map((session, index) => [
        String(index),
        offsetSeconds(session.pageCreatedMonotonicMs, startedMs),
      ]),
    ),
  };
}

async function prepareSubmission(page, content) {
  const input = await waitForChatInput(page);
  await input.fill(content, { timeout: 120_000 });
  const sendButton = page.locator(SEND_BUTTON_SELECTOR).first();
  await sendButton.waitFor({ state: 'visible', timeout: 30_000 });
  const enabledDeadline = performance.now() + 30_000;
  while (!(await sendButton.isEnabled().catch(() => false))) {
    if (performance.now() >= enabledDeadline) {
      throw new Error('send button remained disabled after filling the chat input');
    }
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 50));
  }
  const responses = page.locator(RESPONSE_SELECTOR);
  const priorResponseCount = await responses.count();
  return {
    sendButton,
    priorResponseCount,
    priorResponseText:
      priorResponseCount > 0
        ? (await responses.last().innerText().catch(() => '')).trim()
        : '',
  };
}

async function visibleGenerationState(page, priorResponseCount, priorResponseText) {
  return page.evaluate(
    ({
      responseSelector,
      stopSelector,
      voiceSelector,
      continueSelector,
      previousCount,
      previousText,
    }) => {
      const visible = (element) => {
        if (!element) return false;
        const style = window.getComputedStyle(element);
        const rect = element.getBoundingClientRect();
        return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
      };
      const responses = [...document.querySelectorAll(responseSelector)];
      const response = responses.at(-1) || null;
      const candidateText = (response?.innerText || response?.textContent || '').trim();
      const responseAdvanced =
        responses.length > previousCount || candidateText !== previousText;
      const text = responseAdvanced ? candidateText : '';
      const stopVisible = [...document.querySelectorAll(stopSelector)].some(visible);
      const voiceVisible = [...document.querySelectorAll(voiceSelector)].some(visible);
      const continueVisible = [...document.querySelectorAll(continueSelector)].some(visible);
      return {
        responseCount: responses.length,
        responseAdvanced,
        text,
        stopVisible,
        voiceVisible,
        continueVisible,
      };
    },
    {
      responseSelector: RESPONSE_SELECTOR,
      stopSelector: STOP_BUTTON_SELECTOR,
      voiceSelector: VOICE_INPUT_BUTTON_SELECTOR,
      continueSelector: CONTINUE_RESPONSE_BUTTON_SELECTOR,
      previousCount: priorResponseCount,
      previousText: priorResponseText,
    },
  );
}

async function submitAndMeasure(session, prepared, responseTimeoutMs) {
  const submittedAtMs = performance.now();
  await prepared.sendButton.click({ timeout: 30_000 });

  const deadline = submittedAtMs + responseTimeoutMs;
  let firstTokenAtMs = null;
  let completedAtMs = null;
  let finalText = '';
  let stableSinceMs = null;
  let stableText = null;

  while (performance.now() < deadline) {
    const state = await visibleGenerationState(
      session.page,
      prepared.priorResponseCount,
      prepared.priorResponseText,
    );
    const observedAtMs = performance.now();
    const generationEnded =
      !state.stopVisible && (state.voiceVisible || state.continueVisible);
    if (!generationEnded) {
      stableSinceMs = null;
      stableText = null;
    }
    if (firstTokenAtMs === null && state.responseAdvanced && state.text) {
      firstTokenAtMs = observedAtMs;
    }
    if (firstTokenAtMs !== null && state.text) {
      finalText = state.text;
      if (generationEnded) {
        if (stableText !== state.text) {
          stableText = state.text;
          stableSinceMs = observedAtMs;
        } else if (observedAtMs - stableSinceMs >= 1_000) {
          completedAtMs = observedAtMs;
          break;
        }
      }
    }
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 50));
  }

  if (firstTokenAtMs === null) {
    throw new Error(`no response text appeared within ${Math.round(responseTimeoutMs / 1000)} seconds`);
  }
  if (completedAtMs === null) {
    throw new Error(`response did not complete within ${Math.round(responseTimeoutMs / 1000)} seconds`);
  }

  const timing = {
    submitted_at_monotonic_ms: rounded(submittedAtMs, 3),
    first_token_at_monotonic_ms: rounded(firstTokenAtMs, 3),
    completed_at_monotonic_ms: rounded(completedAtMs, 3),
    submit_offset_s: offsetSeconds(session.pageCreatedMonotonicMs, submittedAtMs),
    first_token_offset_s: offsetSeconds(session.pageCreatedMonotonicMs, firstTokenAtMs),
    completion_offset_s: offsetSeconds(session.pageCreatedMonotonicMs, completedAtMs),
    ttft_s: rounded((firstTokenAtMs - submittedAtMs) / 1000),
    e2e_s: rounded((completedAtMs - submittedAtMs) / 1000),
  };
  return {
    timing,
    response_text: finalText,
    error: null,
  };
}

async function capturedTurn(session, content, responseTimeoutMs) {
  let submittedAtMs = null;
  try {
    const prepared = await prepareSubmission(session.page, content);
    submittedAtMs = performance.now();
    return await submitAndMeasure(session, prepared, responseTimeoutMs);
  } catch (error) {
    return {
      timing: {
        submitted_at_monotonic_ms: submittedAtMs === null ? null : rounded(submittedAtMs, 3),
        first_token_at_monotonic_ms: null,
        completed_at_monotonic_ms: null,
        submit_offset_s:
          submittedAtMs === null
            ? null
            : offsetSeconds(session.pageCreatedMonotonicMs, submittedAtMs),
        first_token_offset_s: null,
        completion_offset_s: null,
        ttft_s: null,
        e2e_s: null,
      },
      response_text: '',
      error: cleanError(error),
    };
  }
}

function chatIdentity(page) {
  const chatUrl = safeUrl(page.url());
  const match = new URL(page.url()).pathname.match(/\/c\/([^/?#]+)/);
  return {
    chat_url: chatUrl,
    chat_id: match ? decodeURIComponent(match[1]) : null,
  };
}

function promptArtifact(prompt) {
  return {
    prompt_id: prompt.promptId,
    label: prompt.label,
    content_sha256: prompt.contentSha256,
    prompt_sha256: prompt.promptSha256,
    rendered_token_count: prompt.renderedTokenCount,
  };
}

async function finalizeVideo(session, destinationPath) {
  if (!session.video) {
    throw new Error('Playwright did not attach a video recorder to the page');
  }
  const sourcePath = await session.video.path();
  await rename(sourcePath, destinationPath);
  return destinationPath;
}

async function captureLongPrompt(browser, options, scenario, webuiVersion, runPrefix) {
  const actOriginMs = performance.now();
  const gpuSampler = new GpuSampler(actOriginMs);
  const providerBefore = await providerProbe(scenario);
  await gpuSampler.start();
  let session = null;
  const result = {
    ...promptArtifact(scenario.longPrompt),
    needle_position: scenario.longPrompt.needlePosition,
    needle_position_exact: scenario.longPrompt.needlePositionExact,
    needle: scenario.longPrompt.needle,
    viewport: { width: 1280, height: 720 },
    video_path: null,
    page_created_at: null,
    page_created_monotonic_ms: null,
    timing: null,
    response_text: '',
    chat_url: null,
    chat_id: null,
    error: null,
    provider: null,
    gpu: null,
  };

  try {
    session = await createRecordedSession(
      browser,
      options.baseUrl,
      options.rawDir,
      result.viewport,
      webuiVersion,
    );
    result.page_created_at = session.pageCreatedAt;
    result.page_created_monotonic_ms = rounded(session.pageCreatedMonotonicMs, 3);
    const needleDescription = scenario.longPrompt.needlePositionExact
      ? `Needle starts at rendered token ${scenario.longPrompt.needlePosition.toLocaleString('en-US')} (${scenario.longPrompt.needle.index_base ?? 0}-based)`
      : `Needle near rendered token ${scenario.longPrompt.needlePosition.toLocaleString('en-US')} (scenario approximation)`;
    await setFixedOverlay(session.page, 'ACT 1 • LONG-CONTEXT RECALL', [
      `Rendered prompt: ${scenario.longPrompt.renderedTokenCount.toLocaleString('en-US')} exact tokenizer tokens`,
      needleDescription,
    ]);
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 1_000));
    const turn = await capturedTurn(
      session,
      scenario.longPrompt.content,
      scenario.responseTimeoutMs,
    );
    result.timing = turn.timing;
    result.response_text = turn.response_text;
    result.error = turn.error;
    Object.assign(result, chatIdentity(session.page));
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 1_000));
  } catch (error) {
    result.error = cleanError(error);
  } finally {
    if (session) {
      await session.context.close().catch((error) => {
        result.error ||= `context close failed: ${cleanError(error)}`;
      });
      try {
        result.video_path = await finalizeVideo(
          session,
          join(options.rawDir, `${runPrefix}-act1-long-context.webm`),
        );
      } catch (error) {
        result.error ||= `video finalization failed: ${cleanError(error)}`;
      }
    }
    result.gpu = await gpuSampler.stop();
    const providerAfter = await providerProbe(scenario);
    const delta = providerDelta(providerBefore, providerAfter);
    result.provider = {
      before: providerBefore,
      after: providerAfter,
      delta,
      session_reuse_delta: reuseDelta(delta),
    };
  }
  return result;
}

async function captureConcurrency(browser, options, scenario, webuiVersion, runPrefix) {
  const actOriginMs = performance.now();
  const gpuSampler = new GpuSampler(actOriginMs);
  const providerBefore = await providerProbe(scenario);
  await gpuSampler.start();
  const sessions = [];
  const result = {
    count: 4,
    viewport_per_session: { width: 960, height: 540 },
    synchronized_countdown_s: 3,
    countdowns: [],
    sessions: [],
    provider: null,
    gpu: null,
    error: null,
  };

  try {
    const creationResults = await Promise.allSettled(
      scenario.concurrentPrompts.map(() =>
        createRecordedSession(
          browser,
          options.baseUrl,
          options.rawDir,
          result.viewport_per_session,
          webuiVersion,
        ),
      ),
    );
    sessions.push(
      ...creationResults
        .filter((creationResult) => creationResult.status === 'fulfilled')
        .map((creationResult) => creationResult.value),
    );
    const creationErrors = creationResults
      .filter((creationResult) => creationResult.status === 'rejected')
      .map((creationResult) => cleanError(creationResult.reason));
    if (creationErrors.length > 0) {
      throw new Error(`failed to create ${creationErrors.length} recorded pages: ${creationErrors.join('; ')}`);
    }

    result.sessions = sessions.map((session, index) => ({
      ...promptArtifact(scenario.concurrentPrompts[index]),
      viewport: result.viewport_per_session,
      video_path: null,
      page_created_at: session.pageCreatedAt,
      page_created_monotonic_ms: rounded(session.pageCreatedMonotonicMs, 3),
      chat_url: null,
      chat_id: null,
      countdowns: [],
      first_turn: null,
      follow_up: {
        ...promptArtifact(scenario.followUp),
        timing: null,
        response_text: '',
        error: null,
      },
      error: null,
    }));

    await Promise.all(
      sessions.map((session, index) =>
        setFixedOverlay(session.page, `ACT 2 • ${scenario.concurrentPrompts[index].label}`, [
          `Independent chat ${index + 1} of 4`,
          'Four requests submit together',
        ]),
      ),
    );

    const firstPrepared = await Promise.all(
      sessions.map((session, index) =>
        prepareSubmission(session.page, scenario.concurrentPrompts[index].content).then(
          (value) => ({ status: 'fulfilled', value }),
          (reason) => ({ status: 'rejected', reason }),
        ),
      ),
    );
    const firstCountdown = await synchronizedCountdown(sessions, 'classic-prompts');
    result.countdowns.push(firstCountdown);
    result.sessions.forEach((sessionResult, index) => {
      sessionResult.countdowns.push({
        label: firstCountdown.label,
        started_offset_s: firstCountdown.started_offsets_s[String(index)],
      });
    });

    const firstTurns = await Promise.all(
      sessions.map(async (session, index) => {
        const prepared = firstPrepared[index];
        if (prepared.status === 'rejected') {
          return {
            timing: null,
            response_text: '',
            error: cleanError(prepared.reason),
          };
        }
        try {
          return await submitAndMeasure(session, prepared.value, scenario.responseTimeoutMs);
        } catch (error) {
          return {
            timing: null,
            response_text: '',
            error: cleanError(error),
          };
        }
      }),
    );
    firstTurns.forEach((turn, index) => {
      result.sessions[index].first_turn = turn;
      Object.assign(result.sessions[index], chatIdentity(sessions[index].page));
    });

    const providerAfterFirstTurn = await providerProbe(scenario);
    const eligibleSessions = sessions
      .map((session, index) => ({ session, index }))
      .filter(({ index }) => result.sessions[index].first_turn?.error === null);
    if (eligibleSessions.length !== 4) {
      for (const { index } of sessions.map((session, index) => ({ session, index }))) {
        if (result.sessions[index].first_turn?.error !== null) {
          result.sessions[index].follow_up.error = 'skipped because the first turn failed';
        }
      }
    }

    const followUpPrepared = await Promise.all(
      eligibleSessions.map(({ session }) =>
        waitForFollowUpReady(session.page)
          .then(() => prepareSubmission(session.page, scenario.followUp.content))
          .then(
            (value) => ({ status: 'fulfilled', value }),
            (reason) => ({ status: 'rejected', reason }),
          ),
      ),
    );
    if (eligibleSessions.length > 0) {
      await Promise.all(
        sessions.map((session, index) =>
          setFixedOverlay(session.page, `ACT 2 • ${scenario.concurrentPrompts[index].label}`, [
            `Independent chat ${index + 1} of 4`,
            'Common follow-up submits together',
          ]),
        ),
      );
      const followUpCountdown = await synchronizedCountdown(sessions, 'common-follow-up');
      result.countdowns.push(followUpCountdown);
      result.sessions.forEach((sessionResult, index) => {
        sessionResult.countdowns.push({
          label: followUpCountdown.label,
          started_offset_s: followUpCountdown.started_offsets_s[String(index)],
        });
      });

      const followUpTurns = await Promise.all(
        eligibleSessions.map(async ({ session }, eligibleIndex) => {
          const prepared = followUpPrepared[eligibleIndex];
          if (prepared.status === 'rejected') {
            return {
              timing: null,
              response_text: '',
              error: cleanError(prepared.reason),
            };
          }
          try {
            return await submitAndMeasure(session, prepared.value, scenario.responseTimeoutMs);
          } catch (error) {
            return {
              timing: null,
              response_text: '',
              error: cleanError(error),
            };
          }
        }),
      );
      followUpTurns.forEach((turn, eligibleIndex) => {
        const originalIndex = eligibleSessions[eligibleIndex].index;
        Object.assign(result.sessions[originalIndex].follow_up, turn);
        Object.assign(result.sessions[originalIndex], chatIdentity(sessions[originalIndex].page));
      });
    }

    const successfulFollowUps = result.sessions.filter(
      (sessionResult) => sessionResult.follow_up.error === null,
    ).length;
    const providerAfter = await providerProbe(scenario);
    const firstTurnDelta = providerDelta(providerBefore, providerAfterFirstTurn);
    const followUpDelta = providerDelta(providerAfterFirstTurn, providerAfter);
    const totalDelta = providerDelta(providerBefore, providerAfter);
    const followUpReuse = reuseDelta(followUpDelta);
    result.provider = {
      before: providerBefore,
      after_first_turn: providerAfterFirstTurn,
      after: providerAfter,
      first_turn_delta: firstTurnDelta,
      follow_up_delta: followUpDelta,
      delta: totalDelta,
      first_turn_session_reuse_delta: reuseDelta(firstTurnDelta),
      follow_up_session_reuse_delta: followUpReuse,
      session_reuse_delta: reuseDelta(totalDelta),
    };
    const reuseHits = followUpReuse?.session_reuse_hits;
    const sessionsOpened = followUpDelta?.engine?.sessions_opened;
    const safeRestarts = safeSessionRestarts(followUpDelta, successfulFollowUps);
    let reuseLine = 'Exact reuse telemetry unavailable';
    if (Number.isFinite(reuseHits) && safeRestarts !== null) {
      reuseLine = `${reuseHits}/4 exact reuse hits • ${safeRestarts} safe restarts`;
    } else if (Number.isFinite(reuseHits) && Number.isFinite(sessionsOpened)) {
      reuseLine = `${reuseHits}/4 exact reuse hits • ${sessionsOpened} session restarts`;
    }
    await Promise.all(
      sessions.map((session, index) =>
        setFixedOverlay(session.page, `ACT 2 • ${scenario.concurrentPrompts[index].label}`, [
          `${successfulFollowUps}/4 follow-ups completed`,
          reuseLine,
        ]),
      ),
    );
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 1_500));
  } catch (error) {
    result.error = cleanError(error);
  } finally {
    await Promise.allSettled(sessions.map((session) => session.context.close()));
    await Promise.all(
      sessions.map(async (session, index) => {
        try {
          const videoPath = await finalizeVideo(
            session,
            join(options.rawDir, `${runPrefix}-act2-${String(index + 1).padStart(2, '0')}.webm`),
          );
          if (result.sessions[index]) {
            result.sessions[index].video_path = videoPath;
          }
        } catch (error) {
          if (result.sessions[index]) {
            result.sessions[index].error = `video finalization failed: ${cleanError(error)}`;
          } else {
            result.error ||= `video finalization failed: ${cleanError(error)}`;
          }
        }
      }),
    );
    result.gpu = await gpuSampler.stop();
    if (!result.provider) {
      const providerAfter = await providerProbe(scenario);
      const delta = providerDelta(providerBefore, providerAfter);
      result.provider = {
        before: providerBefore,
        after_first_turn: null,
        after: providerAfter,
        first_turn_delta: null,
        follow_up_delta: null,
        delta,
        first_turn_session_reuse_delta: null,
        follow_up_session_reuse_delta: null,
        session_reuse_delta: reuseDelta(delta),
      };
    }
  }
  return result;
}

async function atomicWriteJson(path, value) {
  await mkdir(dirname(path), { recursive: true });
  const temporaryPath = join(dirname(path), `.${basename(path)}.${process.pid}.${randomUUID()}.tmp`);
  try {
    await writeFile(temporaryPath, `${JSON.stringify(value, null, 2)}\n`, {
      encoding: 'utf8',
      mode: 0o600,
      flag: 'wx',
    });
    await rename(temporaryPath, path);
  } catch (error) {
    await rm(temporaryPath, { force: true }).catch(() => {});
    throw error;
  }
}

async function gitHead() {
  try {
    const { stdout } = await execFileAsync('git', ['rev-parse', 'HEAD'], {
      timeout: 5_000,
      maxBuffer: 64 * 1024,
    });
    return stdout.trim() || null;
  } catch {
    return null;
  }
}

async function openWebuiVersion(baseUrl) {
  const versionEndpoint = endpointFromBase(baseUrl, '/api/version');
  const configEndpoint = endpointFromBase(baseUrl, '/api/config');
  const [versionResult, configResult] = await Promise.allSettled([
    fetchJson(versionEndpoint),
    fetchJson(configEndpoint),
  ]);
  return {
    ok: versionResult.status === 'fulfilled',
    endpoint: safeUrl(versionEndpoint),
    version:
      versionResult.status === 'fulfilled' ? versionResult.value.data?.version || null : null,
    auth_enabled:
      configResult.status === 'fulfilled'
        ? configResult.value.data?.features?.auth ?? null
        : null,
    error:
      versionResult.status === 'rejected' ? cleanError(versionResult.reason) : null,
    config_probe_error:
      configResult.status === 'rejected' ? cleanError(configResult.reason) : null,
  };
}

function summarize(artifact) {
  const turns = [];
  if (artifact.acts.long_prompt) {
    turns.push(artifact.acts.long_prompt);
  }
  for (const session of artifact.acts.concurrency?.sessions || []) {
    if (session.first_turn) turns.push(session.first_turn);
    if (session.follow_up) turns.push(session.follow_up);
  }
  const turnsSucceeded = turns.filter(
    (turn) => turn.error === null && typeof turn.response_text === 'string' && turn.response_text !== '',
  ).length;
  const turnsFailed = turns.length - turnsSucceeded;
  const captureErrors = [
    ...artifact.errors,
    artifact.acts.concurrency?.error,
    ...(artifact.acts.concurrency?.sessions || []).map((session) => session.error),
  ].filter(Boolean);
  const providerProbes = [
    artifact.acts.long_prompt?.provider?.before,
    artifact.acts.long_prompt?.provider?.after,
    artifact.acts.concurrency?.provider?.before,
    artifact.acts.concurrency?.provider?.after_first_turn,
    artifact.acts.concurrency?.provider?.after,
  ].filter(Boolean);
  const probeErrors = [
    artifact.provenance.open_webui?.error,
    artifact.provenance.open_webui?.config_probe_error,
    artifact.acts.long_prompt?.gpu?.error,
    artifact.acts.concurrency?.gpu?.error,
    ...providerProbes.flatMap((probe) => [probe.health?.error, probe.metrics?.error]),
  ].filter(Boolean);
  return {
    turns_attempted: turns.length,
    turns_succeeded: turnsSucceeded,
    turns_failed: turnsFailed,
    capture_errors: captureErrors.length,
    probe_errors: probeErrors.length,
    success:
      turns.length === 9 &&
      turnsFailed === 0 &&
      captureErrors.length === 0 &&
      probeErrors.length === 0,
  };
}

async function runCapture(options, loadedScenario) {
  await mkdir(options.rawDir, { recursive: true });
  const scriptPath = fileURLToPath(import.meta.url);
  const scriptBytes = await readFile(scriptPath);
  const webui = await openWebuiVersion(options.baseUrl);
  const runPrefix = `open-webui-${new Date().toISOString().replace(/[-:.]/g, '').replace('Z', 'z')}-${process.pid}`;
  const artifact = {
    kind: 'wkvm.open_webui.live_capture',
    schema_version: 1,
    captured_at: timestamp(),
    completed_at: null,
    base_url: safeUrl(options.baseUrl),
    scenario: {
      path: options.scenarioPath,
      sha256: loadedScenario.fileSha256,
      file_sha256: loadedScenario.fileSha256,
      declared_sha256: loadedScenario.scenario.declaredSha256,
      kind: loadedScenario.scenario.kind,
      schema_version: loadedScenario.scenario.schemaVersion,
    },
    provenance: {
      git_head: await gitHead(),
      script_sha256: sha256(scriptBytes),
      node_version: process.version,
      platform: process.platform,
      architecture: process.arch,
      browser: null,
      open_webui: webui,
    },
    recording: {
      raw_dir: options.rawDir,
      chromium_channel: options.channel,
      headless: true,
      videos_are_unedited: true,
      timing_scope: 'screen-observed Playwright monotonic time',
    },
    acts: {
      long_prompt: null,
      concurrency: null,
    },
    summary: null,
    errors: [],
  };

  let browser = null;
  try {
    if (webui.auth_enabled === true) {
      throw new Error(
        `Open WebUI at ${safeUrl(options.baseUrl)} has authentication enabled; ` +
          'launch a dedicated fresh instance with WEBUI_AUTH=false for credential-free capture',
      );
    }
    browser = await chromium.launch({ channel: options.channel, headless: true });
    artifact.provenance.browser = {
      engine: 'chromium',
      channel: options.channel,
      version: browser.version(),
    };
    await warmBrowser(browser, options.baseUrl, webui.version);
    artifact.acts.long_prompt = await captureLongPrompt(
      browser,
      options,
      loadedScenario.scenario,
      webui.version,
      runPrefix,
    );
    artifact.acts.concurrency = await captureConcurrency(
      browser,
      options,
      loadedScenario.scenario,
      webui.version,
      runPrefix,
    );
  } catch (error) {
    artifact.errors.push(cleanError(error));
  } finally {
    if (browser) {
      await browser.close().catch((error) => {
        artifact.errors.push(`browser close failed: ${cleanError(error)}`);
      });
    }
    artifact.completed_at = timestamp();
    artifact.summary = summarize(artifact);
    await atomicWriteJson(options.jsonPath, artifact);
  }

  return artifact;
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (options.help) {
    process.stdout.write(`${usage()}\n`);
    return;
  }
  const loadedScenario = await loadScenario(options.scenarioPath);
  const artifact = await runCapture(options, loadedScenario);
  process.stdout.write(`capture artifact: ${options.jsonPath}\n`);
  for (const videoPath of [
    artifact.acts.long_prompt?.video_path,
    ...(artifact.acts.concurrency?.sessions || []).map((session) => session.video_path),
  ].filter(Boolean)) {
    process.stdout.write(`raw video: ${videoPath}\n`);
  }
  if (!artifact.summary.success) {
    throw new Error(
      `capture finished with ${artifact.summary.turns_failed} failed turns and ` +
        `${artifact.summary.capture_errors} capture errors and ` +
        `${artifact.summary.probe_errors} probe errors; inspect ${options.jsonPath}`,
    );
  }
}

main().catch((error) => {
  process.stderr.write(`error: ${cleanError(error)}\n`);
  process.exitCode = 1;
});
