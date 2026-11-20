"""Group-aware train/val/test split: every group lands wholly in one split."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence


def group_split(
    groups: Sequence[object],
    ratios: dict[str, float] | None = None,
    seed: int = 42,
) -> list[str]:
    ratios = ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
    members: dict[object, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)

    keys = list(members)
    random.Random(seed).shuffle(keys)

    total = len(groups)
    names = list(ratios)
    targets = {n: ratios[n] * total for n in names}
    counts = dict.fromkeys(names, 0)
    out = [""] * total
    for g in keys:
        # assign to the split furthest below its target (greedy, deterministic)
        name = max(names, key=lambda n: targets[n] - counts[n])
        for i in members[g]:
            out[i] = name
        counts[name] += len(members[g])
    return out
