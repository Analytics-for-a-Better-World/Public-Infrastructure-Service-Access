"""Conditional ruin/recreate using own elites, coverage candidates and beam DP."""

from collections import OrderedDict
from functools import lru_cache
from math import isfinite
from time import perf_counter

from .beam_dp import BitWeights, ByteWeights, solve_beam_dp


class RepairMemo:
    """Bounded LRU, bound to immutable snapshots of the model and ordering."""

    def __init__(self, capacity=512):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("invalid cache capacity")
        self.capacity = capacity
        self.signature = None
        self.exact = OrderedDict()
        self.hits = dict(prepared=0, exact=0, invalidations=0)

        # One decorator-backed cache per workspace. Deadline is an execution
        # guard, not a mathematical input. Expiry raises and is never cached.
        @lru_cache(maxsize=capacity)
        def prepared(key):
            incumbent, free, elite_union, extra_cap, fraction, greedy_steps = key
            facilities, weights, order_index = self.signature
            pool, count = _prepare_pool(
                facilities,
                self.weight,
                list(incumbent),
                list(free),
                [elite_union] if elite_union else [],
                self.fixed_mask,
                self.deadline,
                extra_cap,
                fraction,
                greedy_steps,
                order_index,
            )
            if pool is None:
                raise PreparationExpired()
            return tuple(pool), count

        self.prepared = prepared

    def bind(self, facilities, weights, order_index):
        signature = (
            tuple(facilities),
            tuple(weights),
            None if order_index is None else tuple(order_index),
        )
        if self.signature != signature:
            if self.signature is not None:
                self.hits["invalidations"] += 1
            self.prepared.cache_clear()
            self.exact.clear()
            self.signature = signature

    def prepare(self, key, weight, fixed_mask, deadline):
        self.weight = weight
        self.fixed_mask = fixed_mask
        self.deadline = deadline
        before = self.prepared.cache_info().hits
        try:
            value = self.prepared(key)
        except PreparationExpired:
            return None, False
        hit = self.prepared.cache_info().hits > before
        self.hits["prepared"] += int(hit)
        return value, hit

    def get(self, kind, key):
        table = getattr(self, kind)
        if key not in table:
            return None
        table.move_to_end(key)
        self.hits[kind] += 1
        return table[key]

    def put(self, kind, key, value):
        table = getattr(self, kind)
        table[key] = value
        table.move_to_end(key)
        while len(table) > self.capacity:
            table.popitem(last=False)


class PreparationExpired(Exception):
    pass


def repair_region(
    facilities,
    weights,
    incumbent,
    elites,
    free,
    deadline,
    *,
    extra_cap=48,
    elite_fraction=0.5,
    greedy_steps=8,
    beam=64,
    seed=0,
    order_index=None,
    weight_oracle=None,
    memo=None,
    cache_mode="off",
):
    n = len(facilities)
    if cache_mode not in ("off", "prepare", "exact", "both"):
        raise ValueError("invalid cache mode")
    if cache_mode != "off" and not isinstance(memo, RepairMemo):
        raise ValueError("enabled cache requires RepairMemo")
    incumbent = list(incumbent)
    free = list(free)
    if any(type(w) is not int or w < 0 for w in weights) or any(
        type(mask) is not int or mask < 0 or mask >> len(weights) for mask in facilities
    ):
        raise ValueError("invalid weights or coverage masks")
    if any(type(j) is not int for j in free):
        raise ValueError("invalid free site")
    if beam is not None and (type(beam) is not int or beam < 1):
        raise ValueError("invalid beam")
    if type(seed) is not int:
        raise ValueError("invalid seed")
    if (
        len(set(incumbent)) != len(incumbent)
        or len(set(free)) != len(free)
        or not set(free) <= set(incumbent)
    ):
        raise ValueError("invalid incumbent/free region")
    if any(type(j) is not int or not 0 <= j < n for j in incumbent):
        raise ValueError("invalid site")
    if (
        type(extra_cap) is not int
        or extra_cap < 0
        or type(greedy_steps) is not int
        or greedy_steps < 0
        or type(elite_fraction) not in (int, float)
        or not isfinite(elite_fraction)
        or not 0 <= elite_fraction <= 1
    ):
        raise ValueError("invalid pool controls")
    if any(
        len(set(elite)) != len(elite)
        or len(elite) != len(incumbent)
        or any(type(j) is not int or not 0 <= j < n for j in elite)
        for elite in elites
    ):
        raise ValueError("invalid elite")
    weight = weight_oracle or (
        ByteWeights(weights)
        if len(weights) > 64 and sum(weights) < (1 << 63)
        else BitWeights(weights)
    )

    def union(ids):
        result = 0
        for j in ids:
            result |= facilities[j]
        return result

    parent = weight(union(incumbent))
    fixed = sorted(set(incumbent) - set(free))
    fixed_mask = union(fixed)
    stats = dict(reason="deadline", pool_sites=[], elite_candidates=0)

    def fallback():
        return dict(objective=parent, selected=incumbent.copy(), stats=stats)

    if not free or perf_counter() >= deadline:
        return fallback()
    elite_union = tuple(sorted(set().union(*(set(elite) for elite in elites)))) if elites else ()
    key = (
        tuple(sorted(incumbent)),
        tuple(sorted(free)) if order_index is not None else tuple(free),
        elite_union,
    )
    key += (extra_cap, elite_fraction, greedy_steps)
    if cache_mode != "off":
        memo.bind(facilities, weights, order_index)
    proof = memo.get("exact", key) if cache_mode in ("exact", "both") else None
    if proof is not None:
        proof_pool, proof_fixed, candidate_count, proof_seed, proof_parent = proof
        if proof_parent != parent:
            raise RuntimeError("stale exact proof")
        stats.update(
            reason="exact_no_improvement_cache",
            exact_cached=True,
            pool_sites=list(proof_pool),
            free_sites=free,
            fixed_sites=list(proof_fixed),
            elite_candidates=candidate_count,
            cached_proof_seed=proof_seed,
            cached_proof_parent=proof_parent,
        )
        return fallback()
    prepared, hit = (
        memo.prepare(key, weight, fixed_mask, deadline)
        if cache_mode in ("prepare", "both")
        else (None, False)
    )
    if prepared is not None:
        pool, candidate_count = prepared
        pool = list(pool)
        stats["elite_candidates"] = candidate_count
        stats["prepared_cache_hit"] = hit
    elif cache_mode in ("prepare", "both"):
        return fallback()
    else:
        pool = None
    if pool is None:
        pool, candidate_count = _prepare_pool(
            facilities,
            weight,
            incumbent,
            free,
            elites,
            fixed_mask,
            deadline,
            extra_cap,
            elite_fraction,
            greedy_steps,
            order_index,
        )
        if pool is None:
            return fallback()
        stats["elite_candidates"] = candidate_count
        stats["prepared_cache_hit"] = False
    lookup = {j: index for index, j in enumerate(pool)}
    stats.update(pool_sites=pool, free_sites=free, fixed_sites=fixed)
    result = solve_beam_dp(
        [facilities[j] & ~fixed_mask for j in pool],
        weights,
        len(free),
        [lookup[j] for j in free],
        deadline,
        beam=beam,
        completion_interval=8,
        ranking="guide",
        diversity="count",
        tie_break="random",
        seed=seed,
    )
    chosen = fixed + [pool[j] for j in result["selected"]]
    actual = weight(union(chosen))
    expected = weight(fixed_mask) + result["objective"]
    if (
        len(chosen) != len(set(chosen))
        or len(chosen) != len(incumbent)
        or actual != expected
        or actual < parent
    ):
        raise RuntimeError("conditional reconstruction failed")
    stats.update(reason=result["stats"]["reason"], dp=result["stats"], raw_objective=actual)
    # Only a completed UNTRUNCATED no-gain calculation closes this exact model.
    # Improvements are never memoized: equal optimal ties may guide later search.
    if (
        cache_mode in ("exact", "both")
        and actual == parent
        and result["stats"]["completed"]
        and not result["stats"]["truncated"]
    ):
        memo.put("exact", key, (tuple(pool), tuple(fixed), stats["elite_candidates"], seed, parent))
    return dict(objective=actual, selected=chosen, stats=stats)


def _prepare_pool(
    facilities,
    weight,
    incumbent,
    free,
    elites,
    fixed_mask,
    deadline,
    extra_cap,
    elite_fraction,
    greedy_steps,
    order_index,
):
    n = len(facilities)
    closed = set(range(n)) - set(incumbent)
    residual = {}
    ranked = []
    for index, j in enumerate(sorted(closed)):
        if index % 128 == 0 and perf_counter() >= deadline:
            return None, 0
        mask = facilities[j] & ~fixed_mask
        gain = weight(mask)
        if gain:
            residual[j] = mask
            ranked.append((gain, j))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    elite_sites = set().union(*(set(elite) for elite in elites)) if elites else set()
    elite_ranked = [j for gain, j in ranked if j in elite_sites]
    extras = elite_ranked[: int(extra_cap * elite_fraction)]
    candidate_count = len(extras)
    # Diverse gap proposals affect only candidate construction. DP weights and
    # all accepted objectives remain original, with fixed coverage removed once.
    proposals = [j for gain, j in ranked[: max(extra_cap * 4, greedy_steps)]]
    remaining = 0
    for mask in residual.values():
        remaining |= mask
    for j in extras:
        remaining &= ~residual[j]
    for _ in range(min(greedy_steps, extra_cap - len(extras))):
        if perf_counter() >= deadline:
            return None, 0
        available = [j for j in proposals if j not in extras]
        if not available:
            break
        chosen = max(available, key=lambda j: (weight(residual[j] & remaining), -j))
        if not weight(residual[chosen] & remaining):
            break
        extras.append(chosen)
        remaining &= ~residual[chosen]
    for _gain, j in ranked:
        if len(extras) >= extra_cap:
            break
        if j not in extras:
            extras.append(j)
    pool = free + extras
    if order_index is not None:
        pool.sort(key=lambda j: (order_index[j], j))
    return pool, candidate_count
