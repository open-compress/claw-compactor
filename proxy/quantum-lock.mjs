/**
 * Quantum Lock — KV Cache alignment for Claude Code Proxy.
 *
 * The Anthropic prompt cache keys on the first N tokens of system messages.
 * Dynamic content (dates, UUIDs, API keys, timestamps) anywhere near the
 * beginning of a system message destroys cache hits every request.
 *
 * QuantumLock stabilises system messages by:
 *   1. Extracting dynamic content (dates, UUIDs, JWTs, API keys, etc.)
 *   2. Moving extracted fragments to an appendix at the END of the message
 *   3. Keeping the stable prefix identical across requests → cache hits
 *
 * The "quantum" metaphor: dynamic values are collapsed into a deterministic
 * tail section so the wavefunction of the prefix stays locked (stable).
 *
 * Part of claw-compactor Phase 5. License: MIT.
 */

import { createHash } from "node:crypto";

// ---------------------------------------------------------------------------
// Dynamic content patterns
// ---------------------------------------------------------------------------

/**
 * Each pattern entry:
 *   name    — label used in the appendix comment
 *   regex   — global RegExp for matching
 *   replace — replacement string left in the prefix position
 */
const DYNAMIC_PATTERNS = [
  // ISO 8601 dates: 2026-03-17, 2025-12-01T10:30:00Z, 2024-01-01T00:00:00.000Z
  {
    name: "iso_date",
    regex:
      /\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?\b/g,
    replace: "<date>",
  },

  // Plain times: 10:30:00, 23:59:59
  {
    name: "time",
    regex: /\b\d{2}:\d{2}:\d{2}\b/g,
    replace: "<time>",
  },

  // JWTs: eyJ... (base64url header.payload.signature)
  {
    name: "jwt",
    regex: /\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b/g,
    replace: "<jwt>",
  },

  // API keys: sk-..., rk-... OR pk_live_..., pk_test_... (Stripe-style underscore separator)
  {
    name: "api_key",
    regex: /\b(?:(?:sk|rk)-[A-Za-z0-9_-]{16,}|(?:pk_live|pk_test)_[A-Za-z0-9_-]{16,})\b/g,
    replace: "<api_key>",
  },

  // UUIDs: 8-4-4-4-12 hex
  {
    name: "uuid",
    regex:
      /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/gi,
    replace: "<uuid>",
  },

  // Unix timestamps: 10-digit (seconds) or 13-digit (ms) standalone numbers
  {
    name: "unix_ts",
    regex: /\b(?:1[5-9]\d{8}|[2-9]\d{9}|\d{13})\b/g,
    replace: "<timestamp>",
  },

  // High-entropy hex strings: 32–64 hex chars (request IDs, trace IDs, session tokens)
  {
    name: "hex_id",
    regex: /\b[0-9a-f]{32,64}\b/gi,
    replace: "<id>",
  },
];

// ---------------------------------------------------------------------------
// extractDynamic
// ---------------------------------------------------------------------------

/**
 * Scan a system message for dynamic content fragments.
 *
 * Returns an array of matches, each containing:
 *   { name, original, replacement, index }
 *
 * Matches are de-duplicated by original value (same UUID appearing twice is
 * reported once with all indices).
 *
 * @param {string} systemMessage
 * @returns {Array<{ name: string, original: string, replacement: string, indices: number[] }>}
 */
export function extractDynamic(systemMessage) {
  if (typeof systemMessage !== "string") return [];

  const seen = new Map(); // original value -> entry

  for (const { name, regex, replace } of DYNAMIC_PATTERNS) {
    // Reset lastIndex since patterns are global and reused
    regex.lastIndex = 0;
    let match;
    while ((match = regex.exec(systemMessage)) !== null) {
      const key = match[0];
      if (seen.has(key)) {
        seen.get(key).indices.push(match.index);
      } else {
        seen.set(key, {
          name,
          original: key,
          replacement: replace,
          indices: [match.index],
        });
      }
    }
    regex.lastIndex = 0;
  }

  // Sort by first occurrence index so the appendix reflects document order
  return [...seen.values()].sort((a, b) => a.indices[0] - b.indices[0]);
}

// ---------------------------------------------------------------------------
// stabilize
// ---------------------------------------------------------------------------

/**
 * Stabilise a system message for KV cache alignment.
 *
 * Steps:
 *   1. Extract all dynamic fragments.
 *   2. Replace each occurrence in the text with its placeholder token.
 *   3. Append a clearly delimited "Dynamic context" section at the end
 *      listing the real values, so the model still has access to them.
 *
 * The stable prefix (everything before the appendix) is identical across
 * requests, maximising prompt-cache hit rates.
 *
 * Returns the original string unchanged if no dynamic content is found.
 *
 * @param {string} systemMessage
 * @returns {string}
 */
export function stabilize(systemMessage) {
  if (typeof systemMessage !== "string") return systemMessage;

  const fragments = extractDynamic(systemMessage);
  if (fragments.length === 0) return systemMessage;

  // Apply all replacements.
  // Process longest originals first to avoid partial substitution of substrings.
  const sorted = [...fragments].sort(
    (a, b) => b.original.length - a.original.length
  );

  let stabilized = systemMessage;
  for (const { original, replacement } of sorted) {
    // Escape special regex chars in the original value
    const escaped = original.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    stabilized = stabilized.replace(new RegExp(escaped, "g"), replacement);
  }

  // Build the appendix with actual dynamic values
  const appendixLines = ["", "---", "<!-- quantum-lock: dynamic context -->"];
  for (const { name, original } of fragments) {
    appendixLines.push(`${name}: ${original}`);
  }
  appendixLines.push("<!-- end quantum-lock -->");

  return stabilized + appendixLines.join("\n");
}

// ---------------------------------------------------------------------------
// getPrefixHash
// ---------------------------------------------------------------------------

/**
 * Compute a SHA-256 hash of the stable prefix of a system message.
 *
 * The "stable prefix" is the portion of the message before any dynamic
 * content appendix.  When this hash is identical across requests, it is
 * a strong signal that the prompt cache will hit.
 *
 * @param {string} systemMessage  raw (un-stabilised) system message
 * @returns {string}  hex digest (64 chars)
 */
export function getPrefixHash(systemMessage) {
  const stabilized = stabilize(systemMessage);

  // The stable prefix ends just before the quantum-lock appendix delimiter
  const appendixMarker = "\n---\n<!-- quantum-lock: dynamic context -->";
  const appendixIdx = stabilized.indexOf(appendixMarker);
  const prefix =
    appendixIdx === -1 ? stabilized : stabilized.slice(0, appendixIdx);

  return createHash("sha256").update(prefix, "utf8").digest("hex");
}

// ---------------------------------------------------------------------------
// createQuantumLock
// ---------------------------------------------------------------------------

/**
 * Create a QuantumLock instance.
 *
 * Returns an object with:
 *   extractDynamic(systemMessage) — list dynamic content fragments
 *   stabilize(systemMessage)      — move dynamic content to end, keep prefix stable
 *   getPrefixHash(systemMessage)  — hash of stable prefix for cache monitoring
 *
 * Also exposes:
 *   wrapRequest(body)             — stabilise the system message in a full
 *                                   chat completions request body (immutable)
 *
 * @returns {{ extractDynamic, stabilize, getPrefixHash, wrapRequest }}
 */
export function createQuantumLock() {
  /**
   * Stabilise the system message in a chat completions request body.
   * Returns a new body object (immutable — original is not mutated).
   *
   * @param {object} body  OpenAI-compatible request body
   * @returns {object}
   */
  function wrapRequest(body) {
    if (!body || !Array.isArray(body.messages)) return body;

    const messages = body.messages.map((msg) => {
      if (msg.role !== "system") return msg;

      if (typeof msg.content === "string") {
        const stable = stabilize(msg.content);
        return stable === msg.content ? msg : { ...msg, content: stable };
      }

      if (Array.isArray(msg.content)) {
        let changed = false;
        const newContent = msg.content.map((block) => {
          if (block.type !== "text") return block;
          const stable = stabilize(block.text);
          if (stable === block.text) return block;
          changed = true;
          return { ...block, text: stable };
        });
        return changed ? { ...msg, content: newContent } : msg;
      }

      return msg;
    });

    // Check whether anything actually changed
    const changed = messages.some((m, i) => m !== body.messages[i]);
    return changed ? { ...body, messages } : body;
  }

  return { extractDynamic, stabilize, getPrefixHash, wrapRequest };
}
