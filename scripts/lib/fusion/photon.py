"""Photon — image optimiser FusionStage for the claw-compactor pipeline.

Detects base64-encoded images embedded in message content (OpenAI, Anthropic,
and Google GenAI multi-modal formats), applies size-based resizing / quality
reduction via Pillow when available, converts PNG screenshots to JPEG, and
sets OpenAI ``detail: "low"`` to cap vision-token cost.

order = 8  (runs early; images bloat context most aggressively)

Part of claw-compactor. License: MIT.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import math
import re
from typing import Any

from lib.fusion.base import FusionContext, FusionResult, FusionStage
from lib.tokens import estimate_tokens

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Pillow import
# ---------------------------------------------------------------------------
try:
    from PIL import Image as _PILImage  # type: ignore[import]
    PILLOW_AVAILABLE = True
except ImportError:
    _PILImage = None  # type: ignore[assignment]
    PILLOW_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Size thresholds (decoded bytes)
_THRESHOLD_1MB = 1 * 1024 * 1024   # 1 MB — resize to 512 px wide, quality=85
_THRESHOLD_2MB = 2 * 1024 * 1024   # 2 MB — resize to 384 px wide, quality=75

# Resize targets: (max_width, jpeg_quality)
_RESIZE_1MB = (512, 85)
_RESIZE_2MB = (384, 75)

# PNG → JPEG conversion quality
_PNG_JPEG_QUALITY = 85

# Regex: match a full data-URI base64 payload
_DATA_URI_RE = re.compile(
    r"data:image/(?P<fmt>[a-zA-Z0-9+.\-]+);base64,(?P<b64>[A-Za-z0-9+/=\s]+)"
)

# OpenAI "detail" field values
_DETAIL_LOW = "low"
_DETAIL_HIGH = "high"
_DETAIL_AUTO = "auto"


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_image_tokens(width: int, height: int) -> int:
    """Estimate vision tokens for an image using OpenAI tile formula.

    Formula: (width/512) * (height/512) * 85 + 170
    Rounded up to nearest integer.
    """
    tiles_w = math.ceil(width / 512)
    tiles_h = math.ceil(height / 512)
    return int(math.ceil(tiles_w * tiles_h * 85 + 170))


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _decode_b64(b64_str: str) -> bytes:
    """Decode a base64 string (strips whitespace first)."""
    cleaned = b64_str.strip().replace("\n", "").replace(" ", "")
    return base64.b64decode(cleaned)


def _encode_b64(data: bytes) -> str:
    """Encode bytes to a base64 string (no newlines)."""
    return base64.b64encode(data).decode("ascii")


def _image_size_bytes(b64_str: str) -> int:
    """Return decoded byte size of a base64 image payload."""
    try:
        return len(_decode_b64(b64_str))
    except Exception:
        return 0


def _resize_and_encode(
    raw: bytes,
    max_width: int,
    jpeg_quality: int,
    source_fmt: str,
) -> tuple[bytes, str]:
    """Resize *raw* image bytes to *max_width* and return (new_bytes, mime_type).

    The output format is always JPEG.  *source_fmt* is used only for logging.
    Requires Pillow.
    """
    img = _PILImage.open(io.BytesIO(raw))
    orig_w, orig_h = img.size

    if orig_w > max_width:
        ratio = max_width / orig_w
        new_h = max(1, int(orig_h * ratio))
        img = img.resize((max_width, new_h), _PILImage.LANCZOS)

    # Convert to RGB for JPEG (removes alpha channel if present)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    return buf.getvalue(), "jpeg"


def _png_to_jpeg(raw: bytes, quality: int = _PNG_JPEG_QUALITY) -> tuple[bytes, str]:
    """Convert PNG bytes to JPEG.  Requires Pillow."""
    img = _PILImage.open(io.BytesIO(raw))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), "jpeg"


# ---------------------------------------------------------------------------
# Per-image optimisation (returns updated data-URI or original on failure)
# ---------------------------------------------------------------------------

def _optimise_image_data_uri(
    fmt: str,
    b64_payload: str,
) -> tuple[str, str, int, int]:
    """Optimise a single image represented as ``(fmt, b64_payload)``.

    Returns ``(new_fmt, new_b64, original_bytes, new_bytes)``.

    Without Pillow, only records sizes; no transformation is applied.
    """
    raw = _decode_b64(b64_payload)
    original_bytes = len(raw)
    fmt_lower = fmt.lower().replace("image/", "")

    if not PILLOW_AVAILABLE:
        # Cannot resize without Pillow; return unchanged
        return fmt_lower, b64_payload, original_bytes, original_bytes

    try:
        if original_bytes >= _THRESHOLD_2MB:
            new_raw, new_fmt = _resize_and_encode(raw, _RESIZE_2MB[0], _RESIZE_2MB[1], fmt_lower)
        elif original_bytes >= _THRESHOLD_1MB:
            new_raw, new_fmt = _resize_and_encode(raw, _RESIZE_1MB[0], _RESIZE_1MB[1], fmt_lower)
        elif fmt_lower == "png":
            new_raw, new_fmt = _png_to_jpeg(raw)
        else:
            # Nothing to do
            return fmt_lower, b64_payload, original_bytes, original_bytes

        new_b64 = _encode_b64(new_raw)
        return new_fmt, new_b64, original_bytes, len(new_raw)
    except Exception as exc:
        logger.warning("Photon: image optimisation failed (%s); keeping original.", exc)
        return fmt_lower, b64_payload, original_bytes, original_bytes


# ---------------------------------------------------------------------------
# Content traversal helpers
# ---------------------------------------------------------------------------

def _process_openai_content(content: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], int, int]:
    """Walk an OpenAI message content list and optimise image_url blocks.

    Returns ``(new_content, markers, saved_bytes, total_original_bytes)``.
    """
    new_content: list[dict[str, Any]] = []
    markers: list[str] = []
    saved = 0
    original_total = 0

    for block in content:
        if not isinstance(block, dict) or block.get("type") != "image_url":
            new_content.append(block)
            continue

        image_url = block.get("image_url", {})
        if not isinstance(image_url, dict):
            new_content.append(block)
            continue

        url = image_url.get("url", "")
        detail = image_url.get("detail", _DETAIL_AUTO)

        # Always set detail:low for token savings
        new_detail = _DETAIL_LOW
        updated_url_obj: dict[str, Any] = {**image_url, "detail": new_detail}

        m = _DATA_URI_RE.match(url) if isinstance(url, str) else None
        if m:
            fmt = m.group("fmt")
            b64 = m.group("b64")
            new_fmt, new_b64, orig_b, new_b = _optimise_image_data_uri(fmt, b64)
            original_total += orig_b
            saved += orig_b - new_b
            new_data_uri = f"data:image/{new_fmt};base64,{new_b64}"
            updated_url_obj = {**updated_url_obj, "url": new_data_uri}
            markers.append(
                f"photon:openai_image orig={orig_b}B new={new_b}B fmt={fmt}->{new_fmt}"
            )
        else:
            # External URL — only set detail:low
            if detail != _DETAIL_LOW:
                markers.append(f"photon:openai_detail_low url={url[:60]}")

        new_block: dict[str, Any] = {**block, "image_url": updated_url_obj}
        new_content.append(new_block)

    return new_content, markers, saved, original_total


def _process_anthropic_content(content: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], int, int]:
    """Walk an Anthropic message content list and optimise image blocks.

    Anthropic format::

        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "<b64>"}}

    Returns ``(new_content, markers, saved_bytes, total_original_bytes)``.
    """
    new_content: list[dict[str, Any]] = []
    markers: list[str] = []
    saved = 0
    original_total = 0

    for block in content:
        if not isinstance(block, dict) or block.get("type") != "image":
            new_content.append(block)
            continue

        source = block.get("source", {})
        if not isinstance(source, dict) or source.get("type") != "base64":
            new_content.append(block)
            continue

        media_type = source.get("media_type", "image/jpeg")
        b64_data = source.get("data", "")
        fmt = media_type.replace("image/", "")

        new_fmt, new_b64, orig_b, new_b = _optimise_image_data_uri(fmt, b64_data)
        original_total += orig_b
        saved += orig_b - new_b

        new_source: dict[str, Any] = {
            **source,
            "media_type": f"image/{new_fmt}",
            "data": new_b64,
        }
        new_block: dict[str, Any] = {**block, "source": new_source}
        new_content.append(new_block)
        markers.append(
            f"photon:anthropic_image orig={orig_b}B new={new_b}B fmt={fmt}->{new_fmt}"
        )

    return new_content, markers, saved, original_total


def _process_google_content(content: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], int, int]:
    """Walk a Google GenAI ``parts`` list and optimise inlineData image parts.

    Google format::

        {"inlineData": {"mimeType": "image/png", "data": "<b64>"}}

    Returns ``(new_content, markers, saved_bytes, total_original_bytes)``.
    """
    new_content: list[dict[str, Any]] = []
    markers: list[str] = []
    saved = 0
    original_total = 0

    for part in content:
        if not isinstance(part, dict) or "inlineData" not in part:
            new_content.append(part)
            continue

        inline = part["inlineData"]
        if not isinstance(inline, dict):
            new_content.append(part)
            continue

        mime = inline.get("mimeType", "image/jpeg")
        b64_data = inline.get("data", "")
        fmt = mime.replace("image/", "")

        new_fmt, new_b64, orig_b, new_b = _optimise_image_data_uri(fmt, b64_data)
        original_total += orig_b
        saved += orig_b - new_b

        new_inline: dict[str, Any] = {
            **inline,
            "mimeType": f"image/{new_fmt}",
            "data": new_b64,
        }
        new_part: dict[str, Any] = {**part, "inlineData": new_inline}
        new_content.append(new_part)
        markers.append(
            f"photon:google_image orig={orig_b}B new={new_b}B fmt={fmt}->{new_fmt}"
        )

    return new_content, markers, saved, original_total


# ---------------------------------------------------------------------------
# Data-URI scanning in plain text / JSON strings
# ---------------------------------------------------------------------------

def _scan_and_replace_data_uris(text: str) -> tuple[str, list[str], int, int]:
    """Find all data-URI image payloads inside *text* and optimise them.

    Handles plain-text content that embeds images as data URIs (e.g. raw JSON
    serialised into the content string).

    Returns ``(new_text, markers, saved_bytes, original_bytes)``.
    """
    markers: list[str] = []
    saved = 0
    original_total = 0

    def replacer(m: re.Match) -> str:
        nonlocal saved, original_total
        fmt = m.group("fmt")
        b64 = m.group("b64")
        new_fmt, new_b64, orig_b, new_b = _optimise_image_data_uri(fmt, b64)
        original_total += orig_b
        saved += orig_b - new_b
        markers.append(
            f"photon:inline_image orig={orig_b}B new={new_b}B fmt={fmt}->{new_fmt}"
        )
        return f"data:image/{new_fmt};base64,{new_b64}"

    new_text = _DATA_URI_RE.sub(replacer, text)
    return new_text, markers, saved, original_total


# ---------------------------------------------------------------------------
# PhotonStage
# ---------------------------------------------------------------------------

class PhotonStage(FusionStage):
    """Image optimiser fusion stage.

    Detects base64 images in message content in OpenAI, Anthropic, and Google
    GenAI multi-modal formats.  Applies size-based resizing (Pillow required),
    PNG-to-JPEG conversion, and OpenAI ``detail:low`` token caps.

    Without Pillow installed, only the OpenAI ``detail:low`` optimisation is
    applied; all other paths degrade gracefully (images are passed through
    unchanged, markers still emitted for accounting).
    """

    name = "photon"
    order = 8

    def should_apply(self, ctx: FusionContext) -> bool:
        content = ctx.content.strip()
        # Apply if the content looks like it may contain images
        if "base64" in content:
            return True
        if '"image_url"' in content or '"image"' in content or "inlineData" in content:
            return True
        # Check for data: URI scheme
        if "data:image/" in content:
            return True
        return False

    def apply(self, ctx: FusionContext) -> FusionResult:  # noqa: C901
        content = ctx.content
        original_tokens = estimate_tokens(content)
        all_markers: list[str] = []
        all_warnings: list[str] = []
        total_saved = 0
        total_original = 0

        # ------------------------------------------------------------------
        # Attempt to parse content as JSON (multi-modal message list)
        # ------------------------------------------------------------------
        parsed: Any = None
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            parsed = None

        if parsed is not None:
            # Could be a list of content blocks or a single object
            if isinstance(parsed, list):
                # Try to figure out the format from the first image block found
                new_parsed, markers, sv, orig = _dispatch_list(parsed)
                all_markers.extend(markers)
                total_saved += sv
                total_original += orig
                if markers:
                    try:
                        content = json.dumps(new_parsed, ensure_ascii=False)
                    except Exception as exc:
                        all_warnings.append(f"photon: JSON re-serialisation failed: {exc}")

            elif isinstance(parsed, dict):
                # Might be a message object with a "content" key
                inner = parsed.get("content")
                if isinstance(inner, list):
                    new_inner, markers, sv, orig = _dispatch_list(inner)
                    all_markers.extend(markers)
                    total_saved += sv
                    total_original += orig
                    if markers:
                        try:
                            new_parsed = {**parsed, "content": new_inner}
                            content = json.dumps(new_parsed, ensure_ascii=False)
                        except Exception as exc:
                            all_warnings.append(
                                f"photon: JSON re-serialisation failed: {exc}"
                            )

        # ------------------------------------------------------------------
        # Scan plain-text content for inline data URIs
        # ------------------------------------------------------------------
        if "data:image/" in content:
            new_content, markers, sv, orig = _scan_and_replace_data_uris(content)
            if markers:
                content = new_content
                all_markers.extend(markers)
                total_saved += sv
                total_original += orig

        if not all_markers:
            all_warnings.append("photon: no images detected or optimised")

        if not PILLOW_AVAILABLE:
            all_warnings.append(
                "photon: Pillow not installed; only detail:low applied. "
                "Install Pillow for full image resizing support."
            )

        compressed_tokens = estimate_tokens(content)

        return FusionResult(
            content=content,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            markers=all_markers,
            warnings=all_warnings,
        )


# ---------------------------------------------------------------------------
# Format dispatch helper
# ---------------------------------------------------------------------------

def _dispatch_list(
    blocks: list[Any],
) -> tuple[list[Any], list[str], int, int]:
    """Detect format and dispatch to the right processor.

    Tries OpenAI → Anthropic → Google in order. Processes whichever format is
    detected; if none match, returns the list unchanged.
    """
    # OpenAI: blocks have type == "image_url"
    if any(
        isinstance(b, dict) and b.get("type") == "image_url" for b in blocks
    ):
        return _process_openai_content(blocks)

    # Anthropic: blocks have type == "image" with source.type == "base64"
    if any(
        isinstance(b, dict) and b.get("type") == "image"
        and isinstance(b.get("source"), dict)
        for b in blocks
    ):
        return _process_anthropic_content(blocks)

    # Google: blocks (parts) have "inlineData" key
    if any(isinstance(b, dict) and "inlineData" in b for b in blocks):
        return _process_google_content(blocks)

    return blocks, [], 0, 0
