"""Run-Length Encoding for structured data patterns.

Detects and compresses structured repetitive patterns:
- IP address families → common prefix extraction
- File paths → $WS/ shorthand
- Enumeration lists → compact format
- Repeated section headers

Part of claw-compactor. License: MIT.
"""

import re
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default workspace path to shorten
DEFAULT_WS_PATHS = [
    "/home/user/workspace",
]

# IP pattern
_IP_RE = re.compile(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b')


def compress_paths(text: str, workspace_paths: Optional[List[str]] = None) -> str:
    """Replace long workspace paths with $WS shorthand."""
    if not text:
        return ""
    paths = workspace_paths or DEFAULT_WS_PATHS
    result = text
    for ws in sorted(paths, key=len, reverse=True):
        result = result.replace(ws, "$WS")
    return result


def decompress_paths(text: str, workspace_path: str) -> str:
    """Expand $WS back to the full workspace path."""
    if not text:
        return ""
    return text.replace("$WS", workspace_path)


def compress_ip_families(text: str, min_occurrences: int = 2) -> Tuple[str, Dict[str, str]]:
    """Group IPs by common prefix and compress families.

    Returns (compressed_text, prefix_map) where prefix_map maps
    $IPn labels to the common prefix.
    Only compresses families with min_occurrences+ IPs sharing a 3-octet prefix.
    """
    if not text:
        return "", {}

    ips = _IP_RE.findall(text)
    if not ips:
        return text, {}

    # Group by first 3 octets
    families: Dict[str, List[str]] = {}
    for ip in ips:
        parts = ip.split('.')
        prefix = '.'.join(parts[:3]) + '.'
        families.setdefault(prefix, []).append(ip)

    # Only compress families with min_occurrences+ members
    prefix_map: Dict[str, str] = {}
    result = text
    idx = 0
    for prefix, members in sorted(families.items(), key=lambda x: -len(x[1])):
        if len(members) < min_occurrences:
            continue
        label = f"$IP{idx}" if idx > 0 else "$IP"
        prefix_map[label] = prefix
        for ip in set(members):
            parts = ip.split('.')
            suffix = parts[3]
            result = result.replace(ip, f"{label}.{suffix}")
        idx += 1

    return result, prefix_map


def decompress_ip_families(text: str, prefix_map: Dict[str, str]) -> str:
    """Expand compressed IP references back to full IPs."""
    if not text or not prefix_map:
        return text
    result = text
    for label, prefix in prefix_map.items():
        result = re.sub(
            re.escape(label) + r'\.(\d{1,3})',
            lambda m, p=prefix: p + m.group(1),
            result,
        )
    return result


def compress_enumerations(text: str) -> str:
    """Compress comma-separated lists of ALL-CAPS short codes.

    Only compresses lists with 4+ items that are all uppercase short tokens.
    E.g. "BTC, ETH, SOL, BNB, DOGE" → "[BTC,ETH,SOL,BNB,DOGE]"
    """
    if not text:
        return ""

    # Match comma-separated uppercase tokens
    pattern = re.compile(r'((?:[A-Z][A-Z0-9]{1,6})(?:\s*,\s*(?:[A-Z][A-Z0-9]{1,6})){3,})')

    def _compact(m: re.Match) -> str:
        items = [s.strip() for s in m.group(0).split(',')]
        return '[' + ','.join(items) + ']'

    return pattern.sub(_compact, text)


def compress_repeated_headers(text: str) -> str:
    """Compress repeated identical section headers.

    When the same header text appears multiple times, keep only the first
    and merge contents.
    """
    if not text:
        return ""
    lines = text.split('\n')
    seen_headers: Dict[str, int] = {}
    result: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if this is a header
        if line.startswith('#'):
            header_text = line.lstrip('#').strip()
            if header_text in seen_headers:
                # Skip this header, but keep its body content
                i += 1
                while i < len(lines) and not lines[i].startswith('#'):
                    if lines[i].strip():
                        result.append(lines[i])
                    i += 1
                continue
            else:
                seen_headers[header_text] = len(result)
        result.append(line)
        i += 1
    return '\n'.join(result)


def compress(text: str, workspace_paths: Optional[List[str]] = None) -> str:
    """Apply all RLE-style compressions to *text*."""
    if not text:
        return ""
    result = compress_paths(text, workspace_paths)
    result, _ = compress_ip_families(result)
    result = compress_enumerations(result)
    return result


def decompress(text: str, workspace_path: str, ip_prefix_map: Optional[Dict[str, str]] = None) -> str:
    """Reverse all RLE-style compressions."""
    if not text:
        return ""
    result = decompress_paths(text, workspace_path)
    if ip_prefix_map:
        result = decompress_ip_families(result, ip_prefix_map)
    return result
