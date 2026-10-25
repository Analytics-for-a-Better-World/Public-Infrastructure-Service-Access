"""Feasible heuristic menus for large parent-union coverage components."""

from time import perf_counter

import numpy as np


def component_swap_deltas(matrix, numeric_weights, coverage, bin_exact):
    exclusive = numeric_weights * (coverage == 1)
    gains = (matrix @ (numeric_weights * (coverage == 0))).astype(np.int64)
    losses = (matrix @ exclusive).astype(np.int64)
    overlap = (matrix * exclusive) @ matrix.T
    if bin_exact:
        overlap = overlap.astype(np.int64)
    return gains[None, :] - losses[:, None] + overlap


def extend_large_menu(
    ins, sites, residual, budget, base_menu, parent_counts, bin_exact, deadline, refine_steps=12
):
    by_count = {cost: (cost, int(value), int(mask)) for cost, value, mask in base_menu}
    stats = dict(skipped=False, refinement_moves=0, constructed=0)

    def result():
        return list(by_count.values()), stats

    if perf_counter() >= deadline:
        return result()
    k = len(sites)
    used = np.zeros(ins.n_demand, dtype=bool)
    rows = []
    for j in sites:
        row = ins.demand_of(j)
        row = row[residual[row]]
        rows.append(row)
        used[row] = True
    demands = np.flatnonzero(used)
    n = len(demands)
    estimated_bytes = 17 * k * n + 24 * k * k + 40 * n + 32 * k + 5 * ins.n_demand
    if k * n > 2_000_000 or estimated_bytes > 64 * 1024 * 1024:
        stats["skipped"] = True
        return result()
    positions = np.full(ins.n_demand, -1, dtype=np.int32)
    positions[demands] = np.arange(n)
    binary = np.zeros((k, n), dtype=bool)
    for i, row in enumerate(rows):
        binary[i, positions[row]] = True
    dtype = np.float64 if bin_exact else np.int64
    matrix = binary.astype(dtype)
    weights = ins.weights[demands]
    numeric_weights = weights.astype(dtype)

    def record(mask, value):
        count = mask.bit_count()
        if count > budget:
            return
        selected = [i for i in range(k) if mask & (1 << i)]
        union = binary[selected].any(axis=0) if selected else np.zeros(n, dtype=bool)
        actual = int(weights[union].sum())
        if actual != value:
            raise AssertionError("Large-menu construction differs from true union objective")
        if count not in by_count or value > by_count[count][1]:
            by_count[count] = (count, int(value), int(mask))

    forward = {}
    reverse = {}
    selected = np.zeros(k, dtype=bool)
    covered = np.zeros(n, dtype=bool)
    mask = 0
    value = 0
    record(mask, value)
    forward[0] = (mask, value)
    for count in range(1, min(k, budget) + 1):
        if perf_counter() >= deadline:
            return result()
        gains = (matrix @ (numeric_weights * (~covered))).astype(np.int64)
        active = np.flatnonzero(~selected)
        chosen = int(active[np.argmax(gains[active])])
        if perf_counter() >= deadline:
            return result()
        selected[chosen] = True
        covered |= binary[chosen]
        mask |= 1 << chosen
        value += int(gains[chosen])
        record(mask, value)
        forward[count] = (mask, value)
        stats["constructed"] += 1
    selected[:] = True
    coverage = binary.sum(axis=0, dtype=np.int32)
    mask = (1 << k) - 1
    value = int(weights.sum())
    record(mask, value)
    if k <= budget:
        reverse[k] = (mask, value)
    for count in range(k - 1, -1, -1):
        if perf_counter() >= deadline:
            return result()
        losses = (matrix @ (numeric_weights * (coverage == 1))).astype(np.int64)
        active = np.flatnonzero(selected)
        chosen = int(active[np.argmin(losses[active])])
        if perf_counter() >= deadline:
            return result()
        selected[chosen] = False
        coverage -= binary[chosen]
        mask &= ~(1 << chosen)
        value -= int(losses[chosen])
        record(mask, value)
        if count <= budget:
            reverse[count] = (mask, value)
        stats["constructed"] += 1
    seen = set()
    for count in sorted(set(parent_counts)):
        for construction in [forward, reverse]:
            if count not in construction:
                continue
            mask, value = construction[count]
            if mask in seen:
                continue
            seen.add(mask)
            selected = np.asarray([bool(mask & (1 << i)) for i in range(k)])
            coverage = binary[selected].sum(axis=0, dtype=np.int32)
            for _ in range(refine_steps):
                if perf_counter() >= deadline:
                    return result()
                outs = np.flatnonzero(selected)
                entries = np.flatnonzero(~selected)
                if not len(outs) or not len(entries):
                    break
                delta = component_swap_deltas(matrix, numeric_weights, coverage, bin_exact)
                cross = delta[np.ix_(outs, entries)]
                i, j = divmod(int(np.argmax(cross)), len(entries))
                gain = int(cross[i, j])
                if gain <= 0:
                    break
                if perf_counter() >= deadline:
                    return result()
                out, entry = int(outs[i]), int(entries[j])
                selected[out] = False
                selected[entry] = True
                coverage -= binary[out]
                coverage += binary[entry]
                mask = (mask & ~(1 << out)) | (1 << entry)
                value += gain
                record(mask, value)
                stats["refinement_moves"] += 1
    return result()
