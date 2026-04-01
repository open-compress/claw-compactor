"""ContentStripper — strip images and documents before summarization.

Inspired by Claude Code's pre-summarization content stripping: before
sending conversation history to the summarizer, large binary content
(base64 images, embedded documents) is replaced with lightweight
placeholders. This prevents the summarizer from wasting tokens on
non-textual content.

Usage::

    from claw_compactor.fusion.content_stripper import strip_images_and_docs

    cleaned_messages, stats = strip_images_and_docs(messages)

Part of claw-compactor v8. License: MIT.
"""
from __future__ import annotations

import re
from typing import Any

from claw_compactor.tokens import estimate_tokens


# Patterns for detecting embedded content.
_BASE64_DATA_URI_RE = re.compile(
    r'data:(?P<mime>[a-zA-Z0-9/+.-]+);base64,(?P<b64>[A-Za-z0-9+/=\n]{100,})',
)
_MARKDOWN_IMAGE_RE = re.compile(
    r'!\[(?P<alt>[^\]]*)\]\((?P<url>[^)]+)\)',
)
_HTML_IMG_RE = re.compile(
    r'<img\s[^>]*src=["\'](?P<url>[^"\']+)["\'][^>]*/?>',
    re.IGNORECASE,
)
_DOCUMENT_BLOCK_RE = re.compile(
    r'```(?:pdf|doc|docx|xlsx|csv)\s*\n.*?```',
    re.DOTALL,
)

# Placeholder templates.
_IMAGE_PLACEHOLDER = "[image: {mime}, ~{size_kb}KB]"
_MARKDOWN_IMAGE_PLACEHOLDER = "[image: {alt}]"
_DOCUMENT_PLACEHOLDER = "[embedded document removed, ~{size_kb}KB]"


def strip_images_and_docs(
    messages: list[dict[str, Any]],
    strip_base64: bool = True,
    strip_markdown_images: bool = True,
    strip_html_images: bool = True,
    strip_document_blocks: bool = True,
    min_base64_length: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Strip images and documents from messages, replacing with placeholders.

    Parameters
    ----------
    messages:
        OpenAI-format message list.
    strip_base64:
        Replace base64 data URIs with placeholders.
    strip_markdown_images:
        Replace markdown image syntax with alt-text placeholders.
    strip_html_images:
        Replace HTML img tags with placeholders.
    strip_document_blocks:
        Replace embedded document code blocks with placeholders.
    min_base64_length:
        Minimum base64 string length to trigger stripping.

    Returns
    -------
    (cleaned_messages, stats)
    """
    stats: dict[str, Any] = {
        "images_stripped": 0,
        "documents_stripped": 0,
        "tokens_saved": 0,
        "multipart_images_stripped": 0,
    }

    result_messages: list[dict[str, Any]] = []

    for msg in messages:
        content = msg.get("content")

        # Handle multipart content (OpenAI list format).
        if isinstance(content, list):
            new_parts, part_stats = _strip_multipart(content)
            result_messages.append({**msg, "content": new_parts})
            stats["multipart_images_stripped"] += part_stats["images_stripped"]
            stats["tokens_saved"] += part_stats["tokens_saved"]
            continue

        if not isinstance(content, str) or not content:
            result_messages.append(msg)
            continue

        original_tokens = estimate_tokens(content)
        cleaned = content

        # Strip base64 data URIs.
        if strip_base64:
            def _replace_base64(match: re.Match) -> str:
                mime = match.group("mime")
                b64 = match.group("b64")
                size_kb = round(len(b64) * 3 / 4 / 1024, 1)
                stats["images_stripped"] += 1
                return _IMAGE_PLACEHOLDER.format(mime=mime, size_kb=size_kb)

            cleaned = _BASE64_DATA_URI_RE.sub(_replace_base64, cleaned)

        # Strip markdown images.
        if strip_markdown_images:
            def _replace_md_image(match: re.Match) -> str:
                alt = match.group("alt") or "unnamed"
                stats["images_stripped"] += 1
                return _MARKDOWN_IMAGE_PLACEHOLDER.format(alt=alt)

            cleaned = _MARKDOWN_IMAGE_RE.sub(_replace_md_image, cleaned)

        # Strip HTML images.
        if strip_html_images:
            def _replace_html_image(match: re.Match) -> str:
                stats["images_stripped"] += 1
                return "[image removed]"

            cleaned = _HTML_IMG_RE.sub(_replace_html_image, cleaned)

        # Strip document blocks.
        if strip_document_blocks:
            def _replace_doc(match: re.Match) -> str:
                size_kb = round(len(match.group(0)) / 1024, 1)
                stats["documents_stripped"] += 1
                return _DOCUMENT_PLACEHOLDER.format(size_kb=size_kb)

            cleaned = _DOCUMENT_BLOCK_RE.sub(_replace_doc, cleaned)

        new_tokens = estimate_tokens(cleaned)
        stats["tokens_saved"] += original_tokens - new_tokens

        result_messages.append({**msg, "content": cleaned})

    return result_messages, stats


def _strip_multipart(
    parts: list[Any],
) -> tuple[list[Any], dict[str, int]]:
    """Strip image parts from multipart content."""
    stats = {"images_stripped": 0, "tokens_saved": 0}
    new_parts: list[Any] = []

    for part in parts:
        if not isinstance(part, dict):
            new_parts.append(part)
            continue

        part_type = part.get("type", "")

        if part_type == "image_url":
            # Replace image_url with a text placeholder.
            url = part.get("image_url", {}).get("url", "")
            if url.startswith("data:"):
                mime_match = re.match(r'data:([^;]+)', url)
                mime = mime_match.group(1) if mime_match else "unknown"
                size_kb = round(len(url) * 3 / 4 / 1024, 1)
                placeholder = _IMAGE_PLACEHOLDER.format(mime=mime, size_kb=size_kb)
            else:
                placeholder = f"[image: {url[:80]}]"

            new_parts.append({"type": "text", "text": placeholder})
            stats["images_stripped"] += 1
            stats["tokens_saved"] += max(0, estimate_tokens(url) - estimate_tokens(placeholder))

        elif part_type == "image":
            # Another image format variant.
            new_parts.append({"type": "text", "text": "[image removed]"})
            stats["images_stripped"] += 1

        else:
            new_parts.append(part)

    return new_parts, stats
