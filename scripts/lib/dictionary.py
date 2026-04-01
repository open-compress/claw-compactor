"""Dictionary-based compression using auto-learned codebooks.

Scans workspace memory files, learns high-frequency n-grams, builds a
codebook mapping long phrases to short `$XX` codes, and applies/reverses
substitutions for lossless compression.

Part of claw-compactor. License: MIT.
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Code format: $AA .. $ZZ (676 slots), then $AAA.. if needed
_CODE_RE = re.compile(r'\$[A-Z]{2,3}')

# Min occurrences for a phrase to be codebook-worthy
MIN_FREQ = 3
# Min raw length to be worth replacing (shorter than this → no savings)
MIN_PHRASE_LEN = 6
# Max codebook entries
MAX_CODEBOOK = 200

# IP address pattern
_IP_RE = re.compile(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b')
# Absolute path pattern (Unix)
_PATH_RE = re.compile(r'(/[A-Za-z0-9_.~-]+){3,}')


def _generate_codes(n: int) -> List[str]:
    """Generate *n* unique short codes: $AA..$ZZ, then $AAA.. if needed."""
    codes: List[str] = []
    # 2-letter codes: $AA .. $ZZ (676)
    for i in range(26):
        for j in range(26):
            codes.append('$' + chr(65 + i) + chr(65 + j))
            if len(codes) >= n:
                return codes
    # 3-letter codes if needed
    for i in range(26):
        for j in range(26):
            for k in range(26):
                codes.append('$' + chr(65 + i) + chr(65 + j) + chr(65 + k))
                if len(codes) >= n:
                    return codes
    return codes


def _tokenize_ngrams(text: str, min_n: int = 2, max_n: int = 5) -> Counter:
    """Extract word n-grams from *text*, filtering by minimum length."""
    counter: Counter = Counter()
    if not text:
        return counter
    words = text.split()
    for n in range(min_n, max_n + 1):
        for i in range(len(words) - n + 1):
            gram = ' '.join(words[i:i + n])
            if len(gram) >= MIN_PHRASE_LEN:
                counter[gram] += 1
    return counter


def _extract_ip_prefixes(texts: List[str]) -> Dict[str, int]:
    """Find frequently occurring IP prefixes (3-octet) across *texts*."""
    counter: Counter = Counter()
    for text in texts:
        for ip in _IP_RE.findall(text):
            parts = ip.split('.')
            prefix = '.'.join(parts[:3]) + '.'
            counter[prefix] += 1
    return {prefix: count for prefix, count in counter.items() if count >= 2}


def _extract_path_prefixes(texts: List[str]) -> Dict[str, int]:
    """Find frequently occurring path prefixes (directory components) across *texts*."""
    all_paths: List[str] = []
    for text in texts:
        for m in _PATH_RE.finditer(text):
            all_paths.append(m.group())
    
    if len(all_paths) < 2:
        return {}
    
    # Extract directory prefixes at various depths
    counter: Counter = Counter()
    for path in all_paths:
        parts = path.split('/')
        # Generate prefixes of increasing length (at least 3 components)
        for depth in range(3, len(parts)):
            prefix = '/'.join(parts[:depth])
            counter[prefix] += 1
    
    return {prefix: count for prefix, count in counter.items() if count >= 2}


def build_codebook(
    texts: List[str],
    min_freq: int = MIN_FREQ,
    max_entries: int = MAX_CODEBOOK,
) -> Dict[str, str]:
    """Build a codebook from a list of text documents.

    Scans for high-frequency n-grams, IPs, and paths. Returns a dict
    mapping short codes ($XX) to the phrases they replace.
    """
    if not texts:
        return {}

    # Gather candidates: n-grams + IPs + paths
    combined = Counter()
    for text in texts:
        combined.update(_tokenize_ngrams(text))

    # Add IPs and paths
    ip_freqs = _extract_ip_prefixes(texts)
    for ip, count in ip_freqs.items():
        if len(ip) >= MIN_PHRASE_LEN:
            combined[ip] = max(combined.get(ip, 0), count)

    path_freqs = _extract_path_prefixes(texts)
    for path, count in path_freqs.items():
        if len(path) >= MIN_PHRASE_LEN:
            combined[path] = max(combined.get(path, 0), count)

    # Filter by min_freq and sort by savings potential (freq * len)
    candidates = [
        (phrase, count)
        for phrase, count in combined.items()
        if count >= min_freq and len(phrase) >= MIN_PHRASE_LEN
    ]
    candidates.sort(key=lambda x: x[1] * len(x[0]), reverse=True)

    # Take top entries, avoiding overlapping phrases
    codes = _generate_codes(min(len(candidates), max_entries))
    codebook: Dict[str, str] = {}
    used_phrases: Set[str] = set()

    for (phrase, _count), code in zip(candidates, codes):
        # Skip if this phrase is a substring of an already-selected phrase
        skip = False
        for existing in used_phrases:
            if phrase in existing or existing in phrase:
                skip = True
                break
        if skip:
            continue
        codebook[code] = phrase
        used_phrases.add(phrase)
        if len(codebook) >= max_entries:
            break

    return codebook


def _normalize_codebook(codebook: Dict[str, str]) -> Dict[str, str]:
    """Normalize codebook to {code: phrase} format.
    
    Accepts either {code: phrase} or {phrase: code} format.
    Detects format by checking if keys start with '$'.
    """
    if not codebook:
        return {}
    # Check first key to determine format
    first_key = next(iter(codebook))
    if first_key.startswith('$'):
        return codebook  # Already {code: phrase}
    else:
        # {phrase: code} -> {code: phrase}
        return {code: phrase for phrase, code in codebook.items()}


_DOLLAR_ESCAPE = "\x00DLR\x00"  # sentinel for literal '$' in source text


def compress_text(text: str, codebook: Dict[str, str]) -> str:
    """Apply codebook substitutions to *text*. Lossless.
    
    Accepts codebook in either {code: phrase} or {phrase: code} format.
    Pre-existing '$' characters are escaped so they survive roundtrip.
    """
    if not text or not codebook:
        return text
    normalized = _normalize_codebook(codebook)
    # Escape pre-existing '$' to avoid collisions with codes
    result = text.replace("$", _DOLLAR_ESCAPE)
    # Sort by phrase length descending to avoid partial matches
    for code, phrase in sorted(normalized.items(), key=lambda x: -len(x[1])):
        escaped_phrase = phrase.replace("$", _DOLLAR_ESCAPE)
        result = result.replace(escaped_phrase, code)
    # Restore pre-existing '$' that were not part of any codebook phrase
    result = result.replace(_DOLLAR_ESCAPE, "$")
    return result


def decompress_text(text: str, codebook: Dict[str, str]) -> str:
    """Reverse codebook substitutions. Lossless.
    
    Accepts codebook in either {code: phrase} or {phrase: code} format.
    """
    if not text or not codebook:
        return text
    normalized = _normalize_codebook(codebook)
    result = text
    # Sort by code length descending to handle $AAA before $AA
    for code, phrase in sorted(normalized.items(), key=lambda x: -len(x[0])):
        result = result.replace(code, phrase)
    # Unescape literal '$' characters
    result = result.replace(_DOLLAR_ESCAPE, "$")
    return result


def save_codebook(codebook: Dict[str, str], path: Path) -> None:
    """Save codebook to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": 1, "entries": codebook}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_codebook(path: Path) -> Dict[str, str]:
    """Load codebook from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Codebook not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "entries" not in data:
        raise ValueError(f"Invalid codebook format: {path}")
    return data["entries"]


def compression_stats(
    texts_or_original, codebook_or_compressed=None, codebook=None
) -> Dict[str, object]:
    """Calculate compression statistics.
    
    Can be called as:
      compression_stats(texts_dict, codebook) — where texts_dict maps filenames to content
      compression_stats(original_str, compressed_str, codebook)
    """
    if codebook is not None:
        # 3-arg form: (original, compressed, codebook)
        original = texts_or_original
        compressed = codebook_or_compressed
        orig_len = len(original)
        comp_len = len(compressed)
    elif isinstance(texts_or_original, dict) and isinstance(codebook_or_compressed, dict):
        # 2-arg form: (texts_dict, codebook)
        codebook = codebook_or_compressed
        all_text = '\n'.join(texts_or_original.values())
        original = all_text
        compressed = compress_text(all_text, codebook)
        orig_len = len(original)
        comp_len = len(compressed)
    else:
        return {"original_chars": 0, "compressed_chars": 0, "gross_reduction_pct": 0.0,
                "codebook_entries": 0, "codes_used": 0}

    reduction = ((orig_len - comp_len) / orig_len * 100) if orig_len else 0.0

    # Count how many codes are actually used in the compressed text
    normalized = _normalize_codebook(codebook)
    codes_used = sum(1 for code in normalized if code in compressed)

    # Net reduction accounts for codebook overhead
    codebook_overhead = sum(len(k) + len(v) + 2 for k, v in normalized.items())  # code: phrase + separator
    net_saved = orig_len - comp_len - codebook_overhead
    net_reduction = (net_saved / orig_len * 100) if orig_len else 0.0

    return {
        "original_chars": orig_len,
        "compressed_chars": comp_len,
        "gross_reduction_pct": round(reduction, 2),
        "net_reduction_pct": round(net_reduction, 2),
        "codebook_entries": len(codebook),
        "codes_used": codes_used,
    }
