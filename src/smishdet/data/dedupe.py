"""Near-duplicate clustering (MinHash + LSH banding, verified by Jaccard).

Clusters double as `template_group` ids so that near-identical campaign messages
never straddle the train/test boundary.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict

from smishdet.data.normalize import dedupe_key

_MASK = (1 << 61) - 1


def _shingles(key: str, k: int = 4) -> set[str]:
    if len(key) <= k:
        return {key}
    return {key[i : i + k] for i in range(len(key) - k + 1)}


def _h(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def cluster_near_duplicates(
    texts: list[str],
    threshold: float = 0.85,
    num_perm: int = 64,
    bands: int = 16,
    seed: int = 42,
) -> list[int]:
    """Return a cluster id per text. Texts with Jaccard >= threshold share a cluster."""
    rng = random.Random(seed)
    params = [(rng.randrange(1, _MASK), rng.randrange(0, _MASK)) for _ in range(num_perm)]
    rows = num_perm // bands

    sets = [_shingles(dedupe_key(t)) for t in texts]
    parent = list(range(len(texts)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for i, sh in enumerate(sets):
        hashed = [_h(s) for s in sh]
        sig = [min((a * x + b) & _MASK for x in hashed) for a, b in params]
        for band in range(bands):
            buckets[(band, tuple(sig[band * rows : (band + 1) * rows]))].append(i)

    for members in buckets.values():
        first = members[0]
        for other in members[1:]:
            if find(first) != find(other) and _jaccard(sets[first], sets[other]) >= threshold:
                parent[find(other)] = find(first)

    return [find(i) for i in range(len(texts))]
