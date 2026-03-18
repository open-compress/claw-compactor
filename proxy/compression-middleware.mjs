/**
 * Compression middleware for Claude Code Proxy.
 *
 * Intercepts chat completion requests, runs all messages through the
 * claw-compactor Python pipeline (python3 -m scripts.pipeline --stdin --json),
 * scans compressed messages for Rewind markers, and injects the
 * rewind_retrieve tool definition when markers are present.
 *
 * Part of claw-compactor Phase 5. License: MIT.
 */

import { spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
// Root of claw-compactor repo (proxy/ is one level below root)
const REPO_ROOT = join(__dirname, "..");

// Rewind marker pattern — must match scripts/lib/rewind/marker.py
const MARKER_PATTERN =
  /\[(\d+) items? compressed to (\d+)\. Retrieve: hash=([a-f0-9]{24})\]/;

// OpenAI-format rewind_retrieve tool definition
const REWIND_TOOL_DEF = {
  type: "function",
  function: {
    name: "rewind_retrieve",
    description:
      "Retrieve the original uncompressed content for a compressed section. " +
      "Use this when you need more detail from a section marked with a retrieval hash.",
    parameters: {
      type: "object",
      properties: {
        hash_id: {
          type: "string",
          description: "The 24-character hash ID from the compression marker.",
        },
        keywords: {
          type: "array",
          items: { type: "string" },
          description: "Optional keywords to filter the retrieved content.",
        },
      },
      required: ["hash_id"],
    },
  },
};

// Anthropic-format rewind_retrieve tool definition
const REWIND_TOOL_DEF_ANTHROPIC = {
  name: "rewind_retrieve",
  description:
    "Retrieve the original uncompressed content for a compressed section. " +
    "Use this when you need more detail from a section marked with a retrieval hash.",
  input_schema: {
    type: "object",
    properties: {
      hash_id: {
        type: "string",
        description: "The 24-character hash ID from the compression marker.",
      },
      keywords: {
        type: "array",
        items: { type: "string" },
        description: "Optional keywords to filter the retrieved content.",
      },
    },
    required: ["hash_id"],
  },
};

/**
 * Extract all text content from a message object.
 * Handles both plain string content and content-block arrays (Anthropic format).
 *
 * @param {object} message
 * @returns {string}
 */
function extractMessageText(message) {
  const { content } = message;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((b) => b.type === "text")
      .map((b) => b.text)
      .join("\n");
  }
  return "";
}

/**
 * Return true if any message (or tool result content) contains a Rewind marker.
 *
 * @param {object[]} messages
 * @returns {boolean}
 */
function messagesHaveMarkers(messages) {
  for (const msg of messages) {
    const text = extractMessageText(msg);
    if (MARKER_PATTERN.test(text)) return true;

    // Tool role messages may carry result content arrays
    if (msg.role === "tool" && Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (typeof block.text === "string" && MARKER_PATTERN.test(block.text)) {
          return true;
        }
      }
    }
  }
  return false;
}

/**
 * Compress a single message's text content via the Python pipeline subprocess.
 * Spawns: python3 -m scripts.pipeline --stdin --json
 * stdin:  JSON { role, content }
 * stdout: JSON { role, content, compressed, markers }
 *
 * Returns the message unchanged on any error (fail-open).
 *
 * @param {object} message
 * @param {object} config
 * @returns {Promise<object>} compressed message
 */
function compressMessage(message, config) {
  return new Promise((resolve) => {
    const payload = JSON.stringify({
      role: message.role,
      content: extractMessageText(message),
    });

    const pythonBin = config.pythonBin || "python3";
    const args = ["-m", "scripts.pipeline", "--stdin", "--json"];
    if (config.tokenBudget) {
      args.push("--token-budget", String(config.tokenBudget));
    }

    const child = spawn(pythonBin, args, {
      cwd: REPO_ROOT,
      stdio: ["pipe", "pipe", "pipe"],
    });

    const stdoutChunks = [];
    const stderrChunks = [];

    child.stdout.on("data", (chunk) => stdoutChunks.push(chunk));
    child.stderr.on("data", (chunk) => stderrChunks.push(chunk));

    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      console.warn(
        `[CompressionMiddleware] Pipeline timeout for role=${message.role} — using original`
      );
      resolve(message);
    }, config.pipelineTimeoutMs || 10_000);

    child.on("close", (code) => {
      clearTimeout(timer);

      if (code !== 0) {
        const stderr = Buffer.concat(stderrChunks).toString().slice(0, 300);
        console.warn(
          `[CompressionMiddleware] Pipeline exited code=${code} role=${message.role}: ${stderr}`
        );
        resolve(message);
        return;
      }

      try {
        const raw = Buffer.concat(stdoutChunks).toString();
        const result = JSON.parse(raw);
        const compressedContent = result.content ?? extractMessageText(message);

        // Rebuild the message with the compressed content, preserving all other fields.
        // For string content, keep as string; for block arrays, replace text blocks.
        if (typeof message.content === "string") {
          resolve({ ...message, content: compressedContent });
        } else if (Array.isArray(message.content)) {
          // Replace text blocks with compressed text; preserve non-text blocks.
          let textReplaced = false;
          const newBlocks = message.content.map((block) => {
            if (block.type === "text" && !textReplaced) {
              textReplaced = true;
              return { ...block, text: compressedContent };
            }
            return block;
          });
          resolve({ ...message, content: newBlocks });
        } else {
          resolve({ ...message, content: compressedContent });
        }
      } catch (err) {
        console.warn(
          `[CompressionMiddleware] JSON parse error for role=${message.role}: ${err.message}`
        );
        resolve(message);
      }
    });

    child.on("error", (err) => {
      clearTimeout(timer);
      console.warn(
        `[CompressionMiddleware] spawn error: ${err.message} — using original`
      );
      resolve(message);
    });

    child.stdin.write(payload);
    child.stdin.end();
  });
}

/**
 * Determine whether a message role is eligible for compression.
 * Tool role messages are the primary compression targets (they accumulate fast).
 * User and assistant messages are also eligible.
 * System messages are skipped (handled by QuantumLock instead).
 *
 * @param {string} role
 * @param {object} config
 * @returns {boolean}
 */
function isCompressibleRole(role, config) {
  const { compressRoles = ["tool", "user", "assistant"] } = config;
  return compressRoles.includes(role);
}

/**
 * Inject the rewind_retrieve tool definition into a tools array.
 * Skips injection if it is already present. Immutable — returns new array.
 *
 * @param {object[]|undefined} tools  existing tools array
 * @param {string} provider  "anthropic" | "openai"
 * @returns {object[]}
 */
function injectRewindTool(tools, provider) {
  const existing = tools || [];
  const alreadyPresent = existing.some((t) => {
    const name = t.name || t?.function?.name;
    return name === "rewind_retrieve";
  });
  if (alreadyPresent) return existing;

  const toolDef =
    provider === "anthropic" ? REWIND_TOOL_DEF_ANTHROPIC : REWIND_TOOL_DEF;
  return [...existing, toolDef];
}

/**
 * Create the compression middleware.
 *
 * config options:
 *   pythonBin         {string}   python executable (default "python3")
 *   pipelineTimeoutMs {number}   per-message pipeline timeout ms (default 10000)
 *   tokenBudget       {number}   token budget hint passed to pipeline (optional)
 *   compressRoles     {string[]} roles to compress (default ["tool","user","assistant"])
 *   enabled           {boolean}  kill-switch (default true)
 *   provider          {string}   "anthropic" | "openai" (default "openai")
 *
 * The returned middleware function has the signature:
 *   compressRequest(requestBody) -> Promise<requestBody>
 *
 * It mutates nothing — it always returns a new request body object.
 *
 * @param {object} config
 * @returns {{ compressRequest: function }}
 */
export function createCompressionMiddleware(config = {}) {
  const cfg = {
    pythonBin: "python3",
    pipelineTimeoutMs: 10_000,
    compressRoles: ["tool", "user", "assistant"],
    enabled: true,
    provider: "openai",
    ...config,
  };

  /**
   * Compress all eligible messages in a request body.
   * Injects rewind_retrieve tool definition when compressed messages contain markers.
   *
   * @param {object} body  OpenAI-compatible chat completions request body
   * @returns {Promise<object>} new body with compressed messages (and possibly injected tool)
   */
  async function compressRequest(body) {
    if (!cfg.enabled) return body;

    const messages = body.messages;
    if (!Array.isArray(messages) || messages.length === 0) return body;

    // Compress eligible messages in parallel — order preserved via Promise.all
    const compressedMessages = await Promise.all(
      messages.map((msg) =>
        isCompressibleRole(msg.role, cfg)
          ? compressMessage(msg, cfg)
          : Promise.resolve(msg)
      )
    );

    // Scan compressed messages for Rewind markers
    const hasMarkers = messagesHaveMarkers(compressedMessages);

    // Inject rewind tool when markers exist
    const newTools = hasMarkers
      ? injectRewindTool(body.tools, cfg.provider)
      : body.tools;

    return {
      ...body,
      messages: compressedMessages,
      ...(newTools !== undefined ? { tools: newTools } : {}),
    };
  }

  return { compressRequest };
}
