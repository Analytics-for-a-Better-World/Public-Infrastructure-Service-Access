"""Sequential coverage frontier DP with safe upper bounds and optional beam.

No whole-instance exactness claim after any beam truncation or interruption.
All objective arithmetic is integral. Facility/demand masks use Python integers.
"""

from bisect import insort
from random import Random
from time import perf_counter


class BitWeights:
    def __init__(self, weights):
        self.weights = tuple(weights)
        self.cache = {0: 0}

    def __call__(self, mask):
        found = self.cache.get(mask)
        if found is not None:
            return found
        value = 0
        bits = mask
        while bits:
            low = bits & -bits
            value += self.weights[low.bit_length() - 1]
            bits ^= low
        if len(self.cache) < 20000:
            self.cache[mask] = value
        return value


class ByteWeights:
    """Exact int64 lookup only when the sum of all nonnegative weights fits."""

    def __init__(self, weights):
        import numpy as np

        if sum(weights) > (1 << 63) - 1:
            raise ValueError("byte oracle integer range")
        self.np = np
        self.size = (len(weights) + 7) // 8
        padded = np.zeros(self.size * 8, dtype=np.int64)
        padded[: len(weights)] = weights
        self.table = np.zeros((self.size, 256), dtype=np.int64)
        for bit in range(8):
            block = 1 << bit
            self.table[:, block : 2 * block] = self.table[:, :block] + padded[bit::8, None]
        self.rows = np.arange(self.size)
        self.cache = {0: 0}

    def __call__(self, mask):
        found = self.cache.get(mask)
        if found is not None:
            return found
        codes = self.np.frombuffer(mask.to_bytes(self.size, "little"), dtype=self.np.uint8)
        value = int(self.table[self.rows, codes].sum(dtype=self.np.int64))
        if len(self.cache) < 20000:
            self.cache[mask] = value
        return value


def solve_beam_dp(
    facilities,
    weights,
    p,
    incumbent,
    deadline,
    *,
    beam=256,
    completion_interval=32,
    weight_oracle=None,
    ranking="legacy",
    diversity="count",
    bounds="both",
    completion="incumbent",
    completion_limit=0,
    preserve_prefix=True,
    tie_break="deterministic",
    seed=0,
    rank_weights=(1, 1, 0),
):
    """Return protected feasible best; beam=None runs untruncated exact DP.

    Facilities are coverage masks IN processing order; incumbent indices refer
    to that order. State key = (selected count, covered unfinished demands).
    Value = (finalized coverage score, full chosen facility traceback mask).
    Bounds concern whole solutions, not disconnected bucket contributions.
    """
    started = perf_counter()
    n = len(facilities)
    if type(p) is not int or not 0 <= p <= n:
        raise ValueError("invalid cardinality")
    if beam is not None and (type(beam) is not int or beam < 1):
        raise ValueError("invalid beam")
    if type(completion_interval) is not int or completion_interval < 1:
        raise ValueError("invalid completion interval")
    if ranking not in ("legacy", "guide", "upper", "coverage", "mixed"):
        raise ValueError("invalid ranking")
    if diversity not in ("count", "global") or bounds not in ("both", "coverage", "budget", "none"):
        raise ValueError("invalid diversity or bounds")
    if completion not in ("incumbent", "static", "random") or tie_break not in (
        "deterministic",
        "random",
    ):
        raise ValueError("invalid completion or tie break")
    if (
        type(completion_limit) is not int
        or completion_limit < 0
        or type(preserve_prefix) is not bool
        or type(seed) is not int
    ):
        raise ValueError("invalid limit, prefix flag or seed")
    if (
        len(rank_weights) != 3
        or any(type(x) is not int or x < 0 for x in rank_weights)
        or not any(rank_weights)
    ):
        raise ValueError("rank weights must be three nonnegative integers, not all zero")
    tie_rng = Random(seed ^ 0x6A09E667)
    if any(type(w) is not int or w < 0 for w in weights):
        raise ValueError("nonnegative Python integer weights required")
    if any(type(mask) is not int or mask < 0 or mask >> len(weights) for mask in facilities):
        raise ValueError("invalid coverage mask")
    incumbent = list(incumbent)
    if (
        len(incumbent) != p
        or len(set(incumbent)) != p
        or any(type(j) is not int or not 0 <= j < n for j in incumbent)
    ):
        raise ValueError("invalid incumbent")
    weight = (
        (
            ByteWeights(weights)
            if len(weights) > 64 and sum(weights) < (1 << 63)
            else BitWeights(weights)
        )
        if weight_oracle is None
        else weight_oracle
    )
    best_mask = sum(1 << j for j in incumbent)
    union = 0
    for j in incumbent:
        union |= facilities[j]
    best = weight(union)
    stats = dict(
        completed=False,
        truncated=False,
        bound_pruned=0,
        infeasible_pruned=0,
        merged=0,
        beam_dropped=0,
        expanded=0,
        layers=0,
        improvements=[],
        reason="deadline",
    )
    stats["controls"] = dict(
        beam=beam,
        completion_interval=completion_interval,
        ranking=ranking,
        diversity=diversity,
        bounds=bounds,
        completion=completion,
        completion_limit=completion_limit,
        preserve_prefix=preserve_prefix,
        tie_break=tie_break,
        seed=seed,
        rank_weights=list(rank_weights),
    )
    stats["completion_evaluations"] = 0

    def finish(reason):
        stats.update(reason=reason, seconds=perf_counter() - started)
        return dict(
            objective=best, selected=[j for j in range(n) if best_mask >> j & 1], stats=stats
        )

    if perf_counter() >= deadline:
        return finish("deadline")
    suffix = [0] * (n + 1)
    suffix_inc = [0] * (n + 1)
    inc_set = set(incumbent)
    # Static per-facility totals overestimate any future marginal contribution.
    # Their r-largest suffix sums give a safe, budget-aware upper bound.
    top_sums = [None] * (n + 1)
    top_sums[n] = (0,)
    largest = []
    totals = []
    for mask in facilities:
        if perf_counter() >= deadline:
            return finish("deadline")
        totals.append(weight(mask))
    for j in range(n - 1, -1, -1):
        if perf_counter() >= deadline:
            return finish("deadline")
        suffix[j] = suffix[j + 1] | facilities[j]
        suffix_inc[j] = suffix_inc[j + 1] | (facilities[j] if j in inc_set else 0)
        insort(largest, totals[j])
        if len(largest) > p:
            largest.pop(0)
        sums = [0]
        for value in reversed(largest):
            sums.append(sums[-1] + value)
        top_sums[j] = tuple(sums)
    total = weight(suffix[0])
    # Precompute candidate completion priority. This constructs feasible lower
    # bounds only; no completion estimate is ever used as an upper bound.
    if completion == "incumbent":
        priority = sorted(range(n), key=lambda j: (j not in inc_set, -totals[j], j))
    elif completion == "static":
        priority = sorted(range(n), key=lambda j: (-totals[j], j))
    else:
        priority = list(range(n))
        Random(seed ^ 0xBB67AE85).shuffle(priority)

    def upper_bound(value, live, layer, r):
        coverage_bound = value + total - finalized_total
        if bounds == "none":
            return total
        if bounds == "coverage":
            return coverage_bound
        budget_bound = value + weight(live) + top_sums[layer][r]
        return budget_bound if bounds == "budget" else min(coverage_bound, budget_bound)

    def rank_row(key, value, layer):
        count, live = key
        guide = value[0] + weight(live | suffix_inc[layer])
        covered_score = value[0] + weight(live)
        if ranking in ("legacy", "guide"):
            score = guide
        elif ranking == "upper":
            score = value[2]
        elif ranking == "coverage":
            score = covered_score
        else:
            score = (
                rank_weights[0] * guide
                + rank_weights[1] * value[2]
                + rank_weights[2] * covered_score
            )
        tie = tie_rng.getrandbits(64) if tie_break == "random" else -live
        return (score, value[2], tie, key, value)

    states = {(0, 0): (0, 0)}
    finalized_total = 0
    inc_live = inc_score = inc_count = inc_path = 0
    stats["incumbent_prefix_restored"] = 0
    for j, facility in enumerate(facilities):
        if perf_counter() >= deadline:
            return finish("deadline")
        future = suffix[j + 1]
        last = facility & ~future
        finalized_total += weight(last)
        remaining = n - j - 1
        if j in inc_set:
            inc_live |= facility
            inc_count += 1
            inc_path |= 1 << j
        inc_score += weight(inc_live & last)
        inc_live &= future
        candidates = {}
        for (count, covered), (score, chosen) in states.items():
            for take in (0, 1):
                stats["expanded"] += 1
                if stats["expanded"] % 128 == 0 and perf_counter() >= deadline:
                    return finish("deadline")
                c = count + take
                r = p - c
                if r < 0 or r > remaining:
                    stats["infeasible_pruned"] += 1
                    continue
                cov = covered | (facility if take else 0)
                value = score + weight(cov & last)
                live = cov & future
                path = chosen | ((1 << j) if take else 0)
                upper = upper_bound(value, live, j + 1, r)
                if bounds != "none" and upper <= best:
                    stats["bound_pruned"] += 1
                    continue
                key = (c, live)
                old = candidates.get(key)
                if old is not None:
                    stats["merged"] += 1
                if old is None or value > old[0]:
                    candidates[key] = (value, path, upper)
        # Keep the incumbent's partial path available for changes arising late
        # in the facility order. A better state with the same key dominates it.
        inc_key = (inc_count, inc_live)
        inc_r = p - inc_count
        inc_upper = upper_bound(inc_score, inc_live, j + 1, inc_r)
        if (
            preserve_prefix
            and (bounds == "none" or inc_upper > best)
            and (inc_key not in candidates or candidates[inc_key][0] < inc_score)
        ):
            candidates[inc_key] = (inc_score, inc_path, inc_upper)
            stats["incumbent_prefix_restored"] += 1
        stats["layers"] = j + 1
        if not candidates:
            stats["completed"] = True
            return finish("frontier_exhausted")
        # Complete partial paths using a common deterministic future ordering.
        # The selected count is EXACT even when different from the incumbent's.
        if (j + 1) % completion_interval == 0 or j == n - 1:
            completions = [(0, 0)]
            cmask = ccov = 0
            for k in priority:
                if k > j:
                    cmask |= 1 << k
                    ccov |= facilities[k]
                    completions.append((cmask, ccov))
                    if len(completions) > p:
                        break
            completion_items = list(candidates.items())
            if completion_limit and len(completion_items) > completion_limit:
                completion_items.sort(
                    key=lambda item: (
                        item[1][0] + weight(item[0][1] | suffix_inc[j + 1]),
                        item[1][2],
                    ),
                    reverse=True,
                )
                completion_items = completion_items[:completion_limit]
            # Always evaluate final feasible states, even with a completion cap.
            if j == n - 1:
                completion_items = list(candidates.items())
            for (count, live), (score, chosen, _upper) in completion_items:
                if perf_counter() >= deadline:
                    return finish("deadline")
                cmask, ccov = completions[p - count]
                value = score + weight(live | ccov)
                stats["completion_evaluations"] += 1
                if value > best:
                    best = value
                    best_mask = chosen | cmask
                    stats["improvements"].append(
                        dict(
                            layer=j + 1,
                            objective=best,
                            elapsed=perf_counter() - started,
                            selected=[k for k in range(n) if best_mask >> k & 1],
                        )
                    )
        candidates = {
            key: value for key, value in candidates.items() if bounds == "none" or value[2] > best
        }
        if beam is not None and len(candidates) > beam:
            # Avoid collapsing onto one partial cardinality. Within each count,
            # compare actual finalized coverage plus future-incumbent coverage.
            groups = {}
            for key, value in candidates.items():
                count, live = key
                groups.setdefault(count if diversity == "count" else 0, []).append(
                    rank_row(key, value, j + 1)
                )
            for group in groups.values():
                group.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
            kept = {}
            depth = 0
            while len(kept) < beam:
                layer = [group[depth] for group in groups.values() if len(group) > depth]
                if not layer:
                    break
                layer.sort(
                    key=lambda row: (
                        (
                            (-row[1], -row[0], row[3])
                            if tie_break == "deterministic"
                            else (-row[1], -row[0], -row[2], row[3])
                        )
                        if ranking == "legacy"
                        else (-row[0], -row[1], -row[2], row[3])
                    )
                )
                for _, _, _, key, value in layer[: beam - len(kept)]:
                    kept[key] = value
                depth += 1
            stats["truncated"] = True
            if preserve_prefix and inc_key in candidates and inc_key not in kept:
                kept.pop(next(reversed(kept)))
                kept[inc_key] = candidates[inc_key]
            stats["beam_dropped"] += len(candidates) - len(kept)
            candidates = kept
        states = {key: value[:2] for key, value in candidates.items()}
    stats["completed"] = True
    return finish("complete")
