/**
 * Rewind response handler for Claude Code Proxy.
 *
 * Provides:
 *   - RewindMemoryStore  — in-memory KV store mirroring Python RewindStore semantics
 *   - createRewindHandler — detects rewind_retrieve tool calls in LLM responses,
 *                           executes retrieval, and constructs tool_result messages
 *
 * Part of claw-compactor Phase 5. License: MIT.
 */

import { createHash, randomBytes } from "node:crypto";

// ---------------------------------------------------------------------------
// RewindMemoryStore
// ---------------------------------------------------------------------------

/**
 * In-memory store for Rewind compressed/original pairs.
 * Mirrors the Python RewindStore interface: store(), retrieve(), search().
 *
 * Entries expire after ttlMs milliseconds (default 10 min).
 * Oldest entries are evicted when maxEntries is reached.
 */
export class RewindMemoryStore {
  /**
   * @param {number} maxEntries  maximum entries before LRU eviction (default 500)
   * @param {number} ttlMs       entry TTL in milliseconds (default 600_000)
   */
  constructor(maxEntries = 500, ttlMs = 600_000) {
    this._maxEntries = maxEntries;
    this._ttlMs = ttlMs;
    // Map<hashId, { original: string, compressed: string, storedAt: number }>
    this._entries = new Map();
  }

  /**
   * Store a compressed/original pair. Returns the 24-hex-character hash ID.
   *
   * The hash is derived from the compressed text so that identical compressions
   * produce the same hash (content-addressable), matching Python store behaviour.
   *
   * @param {string} original    full original text
   * @param {string} compressed  compressed text containing the marker
   * @returns {string} 24-character hex hash ID
   */
  store(original, compressed) {
    // Evict expired entries first
    this._evictExpired();

    // Content-addressable ID — first 24 hex chars of SHA-1 of compressed text
    const hashId = createHash("sha1")
      .update(compressed)
      .digest("hex")
      .slice(0, 24);

    // LRU eviction when at capacity (delete oldest inserted entry)
    if (this._entries.size >= this._maxEntries && !this._entries.has(hashId)) {
      const oldestKey = this._entries.keys().next().value;
      this._entries.delete(oldestKey);
    }

    this._entries.set(hashId, {
      original,
      compressed,
      storedAt: Date.now(),
    });

    return hashId;
  }

  /**
   * Retrieve the original text for a hash ID.
   * Returns null if not found or expired.
   *
   * @param {string} hashId
   * @returns {string|null}
   */
  retrieve(hashId) {
    const entry = this._entries.get(hashId);
    if (!entry) return null;
    if (Date.now() - entry.storedAt > this._ttlMs) {
      this._entries.delete(hashId);
      return null;
    }
    return entry.original;
  }

  /**
   * Retrieve the original text filtered to lines containing any of the keywords.
   * Falls back to full original when no keywords match.
   *
   * @param {string} hashId
   * @param {string[]} keywords
   * @returns {string|null}
   */
  search(hashId, keywords) {
    const original = this.retrieve(hashId);
    if (original === null) return null;
    if (!keywords || keywords.length === 0) return original;

    const lowerKws = keywords.map((k) => k.toLowerCase());
    const lines = original.split("\n");
    const matched = lines.filter((line) =>
      lowerKws.some((kw) => line.toLowerCase().includes(kw))
    );
    return matched.length > 0 ? matched.join("\n") : original;
  }

  /**
   * Number of live (non-expired) entries.
   * @returns {number}
   */
  get size() {
    this._evictExpired();
    return this._entries.size;
  }

  /** @private */
  _evictExpired() {
    const now = Date.now();
    for (const [key, entry] of this._entries) {
      if (now - entry.storedAt > this._ttlMs) {
        this._entries.delete(key);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Detection helpers
// ---------------------------------------------------------------------------

const TOOL_NAME = "rewind_retrieve";

/**
 * Detect a rewind_retrieve tool call in an OpenAI-format response object.
 *
 * Handles both:
 *   - OpenAI streaming delta / non-streaming: choices[].message.tool_calls[]
 *   - Anthropic direct: content[] blocks with type="tool_use"
 *
 * Returns the first rewind tool call found, or null.
 *
 * @param {object} response  parsed response JSON
 * @returns {object|null}  tool call object or null
 */
function detectToolCall(response) {
  // OpenAI format: choices[].message.tool_calls
  if (Array.isArray(response.choices)) {
    for (const choice of response.choices) {
      const msg = choice.message || choice.delta || {};
      if (Array.isArray(msg.tool_calls)) {
        for (const tc of msg.tool_calls) {
          if (tc?.function?.name === TOOL_NAME) return tc;
        }
      }
    }
  }

  // Anthropic format: content[] with type="tool_use"
  if (Array.isArray(response.content)) {
    for (const block of response.content) {
      if (block.type === "tool_use" && block.name === TOOL_NAME) return block;
    }
  }

  return null;
}

/**
 * Parse arguments from a tool call object.
 * Handles JSON strings (OpenAI) and plain objects (Anthropic input).
 *
 * @param {object} toolCall
 * @returns {{ hash_id: string, keywords?: string[] }}
 */
function parseToolCallArgs(toolCall) {
  // OpenAI: toolCall.function.arguments is a JSON string
  if (toolCall?.function?.arguments) {
    try {
      return JSON.parse(toolCall.function.arguments);
    } catch {
      return {};
    }
  }
  // Anthropic: toolCall.input is an object
  if (toolCall?.input && typeof toolCall.input === "object") {
    return toolCall.input;
  }
  return {};
}

// ---------------------------------------------------------------------------
// createRewindHandler
// ---------------------------------------------------------------------------

/**
 * Create a Rewind response handler bound to a RewindMemoryStore.
 *
 * @param {RewindMemoryStore} store
 * @returns {{ detectToolCall, handleRetrieval, createStreamHandler }}
 */
export function createRewindHandler(store) {
  /**
   * Execute retrieval for a detected tool call.
   * Returns an OpenAI-compatible tool_result message to be appended to the
   * conversation so the LLM can continue with the full content.
   *
   * @param {object} toolCall  tool call object from detectToolCall()
   * @returns {object}  { role: "tool", tool_call_id, content }
   */
  function handleRetrieval(toolCall) {
    const args = parseToolCallArgs(toolCall);
    const hashId = args.hash_id || "";
    const keywords = Array.isArray(args.keywords) ? args.keywords : [];

    let content;
    if (keywords.length > 0) {
      content = store.search(hashId, keywords);
    } else {
      content = store.retrieve(hashId);
    }

    const resultPayload =
      content !== null
        ? JSON.stringify({ status: "ok", content })
        : JSON.stringify({
            status: "not_found",
            message: `No content found for hash=${hashId}. It may have expired.`,
          });

    // OpenAI tool_result format
    return {
      role: "tool",
      tool_call_id: toolCall.id || `rewind-${hashId}`,
      content: resultPayload,
    };
  }

  /**
   * Create a streaming response handler.
   *
   * The returned object collects SSE chunks. When the stream ends,
   * call .finish() to get the buffered full response and any detected
   * tool call result that should be injected.
   *
   * Usage:
   *   const handler = createStreamHandler();
   *   for await (const chunk of responseStream) handler.push(chunk);
   *   const { toolResult, fullText } = handler.finish();
   *
   * @returns {{ push: function, finish: function }}
   */
  function createStreamHandler() {
    const chunks = [];

    /**
     * Push an SSE data line (string) or Buffer chunk into the buffer.
     * @param {string|Buffer} chunk
     */
    function push(chunk) {
      chunks.push(typeof chunk === "string" ? chunk : chunk.toString("utf8"));
    }

    /**
     * Finish buffering and detect tool calls in the accumulated response.
     *
     * @returns {{ toolResult: object|null, fullText: string, rawChunks: string[] }}
     */
    function finish() {
      const fullText = chunks.join("");

      // Parse SSE data lines into JSON objects and look for tool calls
      let toolResult = null;
      const dataLines = fullText
        .split("\n")
        .filter((l) => l.startsWith("data: ") && l !== "data: [DONE]");

      for (const line of dataLines) {
        try {
          const parsed = JSON.parse(line.slice(6));
          const tc = detectToolCall(parsed);
          if (tc) {
            toolResult = handleRetrieval(tc);
            break;
          }
        } catch {
          // Skip malformed lines
        }
      }

      // Also try parsing the whole text as a single JSON object (non-streaming)
      if (!toolResult) {
        try {
          const parsed = JSON.parse(fullText);
          const tc = detectToolCall(parsed);
          if (tc) toolResult = handleRetrieval(tc);
        } catch {
          // Not a single JSON blob — that's fine
        }
      }

      return { toolResult, fullText, rawChunks: [...chunks] };
    }

    return { push, finish };
  }

  return { detectToolCall, handleRetrieval, createStreamHandler };
}
