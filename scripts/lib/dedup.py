"""Deduplication engine using shingle hashing.

Uses n-gram (shingle) fingerprinting for efficient near-duplicate detection
without O(n^2) pairwise comparison. Groups entries by section, then compares
shingle sets using Jaccard similarity.

Part of claw-compactor. License: MIT.
"""

import hashlib
import logging
from typing import List, Dict, Any, Set, Tuple

logger = logging.getLogger(__name__)

# Configuration
SHINGLE_SIZE = 3  # n-gram size (words)
SIMILARITY_THRESHOLD = 0.6  # Jaccard similarity threshold for "duplicate"


def _shingles(text: str, k: int = SHINGLE_SIZE) -> Set[int]:
    """Generate a set of k-word shingle hashes from *text*.

    Each shingle is a hash of *k* consecutive words.
    Returns a set of integer hashes.
    """
    words = text.split()
    if not words:
        return set()
    if len(words) < k:
        return {hash(' '.join(words))}
    result: Set[int] = set()
    for i in range(len(words) - k + 1):
        shingle = ' '.join(words[i:i + k])
        result.add(hash(shingle))
    return result


def jaccard(a: Set[int], b: Set[int]) -> float:
    """Compute Jaccard similarity between two shingle sets.

    Returns 1.0 for identical sets, 0.0 for disjoint.
    If both are empty, returns 1.0.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a) + len(b) - intersection
    return intersection / union if union else 0.0


def find_duplicates(
    entries: List[str],
    threshold: float = SIMILARITY_THRESHOLD,
    k: int = SHINGLE_SIZE,
) -> List[Dict[str, Any]]:
    """Find near-duplicate groups among *entries*.

    Returns a list of dicts, each with:
        - indices: list of indices that are near-duplicates
        - similarity: average Jaccard similarity within the group

    Uses O(n^2) pairwise comparison with shingle hashing.
    """
    if len(entries) < 2:
        return []

    shingle_sets = [_shingles(e, k) for e in entries]
    used: Set[int] = set()
    groups: List[Dict[str, Any]] = []

    for i in range(len(entries)):
        if i in used:
            continue
        group_indices = [i]
        total_sim = 0.0
        count = 0
        for j in range(i + 1, len(entries)):
            if j in used:
                continue
            sim = jaccard(shingle_sets[i], shingle_sets[j])
            if sim >= threshold:
                group_indices.append(j)
                used.add(j)
                total_sim += sim
                count += 1
        if len(group_indices) > 1:
            avg_sim = total_sim / count if count else threshold
            groups.append({
                "indices": group_indices,
                "similarity": round(avg_sim, 4),
            })
            used.update(group_indices)

    return groups


def merge_duplicates(
    entries: List[str],
    groups: List[Dict[str, Any]],
) -> List[str]:
    """Merge duplicate groups, keeping the longest entry in each group.

    Entries not in any group are passed through unchanged.
    """
    if not groups:
        return list(entries)

    removed: Set[int] = set()
    for g in groups:
        indices = g["indices"]
        # Keep the longest
        best = max(indices, key=lambda idx: len(entries[idx]))
        for idx in indices:
            if idx != best:
                removed.add(idx)

    return [e for i, e in enumerate(entries) if i not in removed]
