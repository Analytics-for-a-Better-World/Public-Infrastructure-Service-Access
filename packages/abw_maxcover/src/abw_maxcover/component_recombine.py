"""Experimental exact small-component parent-union recombination; no external solver."""

from functools import lru_cache
from time import perf_counter

import numpy as np

from .large_component_menus import extend_large_menu


@lru_cache(maxsize=16)
def subset_counts(k):
    values = np.arange(1 << k, dtype=np.uint32)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(values)
    return np.fromiter((int(v).bit_count() for v in values), dtype=np.uint8, count=len(values))


@lru_cache(maxsize=16)
def subset_groups(k):
    counts = subset_counts(k)
    return tuple(np.flatnonzero(counts == count) for count in range(k + 1))


def exact_component_menu(ins, sites, residual, budget):
    k = len(sites)
    size = 1 << k
    patterns = np.zeros(ins.n_demand, dtype=np.int32)
    for bit, j in enumerate(sites):
        row = ins.demand_of(j)
        row = row[residual[row]]
        patterns[row] |= 1 << bit
    support = np.flatnonzero(patterns)
    frequency = np.zeros(size, dtype=np.int64)
    np.add.at(frequency, patterns[support], ins.weights[support])
    zeta = frequency.copy()
    for bit in range(k):
        stride = 1 << bit
        view = zeta.reshape(-1, 2 * stride)
        view[:, stride:] += view[:, :stride]
    total = int(frequency.sum())
    values = total - zeta[(size - 1) ^ np.arange(size)]
    groups = subset_groups(k)
    menu = []
    for count in range(min(k, budget) + 1):
        masks = groups[count]
        chosen = int(masks[np.argmax(values[masks])])
        menu.append((count, int(values[chosen]), chosen))
    return menu


def component_recombine(start, guide, deadline, exact_cap=12, greedy_large=False):
    """Return a strict-better completed offspring or None and diagnostic counts.

    Large components have a restricted parent/empty/all menu. Only completed
    searches have the better-parent feasibility bound; timeout returns None.
    """
    ins = start.ins
    if guide.ins is not ins or guide.p != start.p:
        raise ValueError("Parents must share instance and budget")
    if type(exact_cap) is not int or not 1 <= exact_cap <= 16:
        raise ValueError("Exact component cap must be within 1..16")
    stats = dict(
        components=0,
        exact_components=0,
        subsets=0,
        large_components=0,
        completed=False,
        expanded_groups=0,
        greedy_groups=0,
        greedy_refinement_moves=0,
        greedy_memory_skips=0,
    )
    if perf_counter() >= deadline:
        return None, stats
    a = set(map(int, start.order[: start.p]))
    b = set(map(int, guide.order[: guide.p]))
    common = sorted(a & b)
    different = sorted(a ^ b)
    budget = start.p - len(common)
    if not different:
        return None, stats
    covered = np.zeros(ins.n_demand, dtype=bool)
    for j in common:
        covered[ins.demand_of(j)] = True
    common_value = sum(map(int, ins.weights[covered]))
    residual = ~covered & (ins.weights > 0)
    parent = list(range(len(different)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = np.full(ins.n_demand, -1, dtype=np.int32)
    for index, j in enumerate(different):
        if perf_counter() >= deadline:
            return None, stats
        row = ins.demand_of(j)
        for demand in row[residual[row]]:
            previous = int(owner[demand])
            if previous < 0:
                owner[demand] = index
            else:
                x, y = root(index), root(previous)
                if x != y:
                    parent[x] = y
    groups = {}
    for i, j in enumerate(different):
        groups.setdefault(root(i), []).append(j)
    menus = []
    components = list(groups.values())
    stats["components"] = len(components)
    for sites in components:
        if perf_counter() >= deadline:
            return None, stats
        if len(sites) <= exact_cap:
            menu = exact_component_menu(ins, sites, residual, budget)
            stats["exact_components"] += 1
            stats["subsets"] += 1 << len(sites)
            stats["expanded_groups"] += int(len(sites) > 12)
        else:
            stats["large_components"] += 1
            masks = [0, (1 << len(sites)) - 1]
            masks += [sum(1 << i for i, j in enumerate(sites) if j in p) for p in [a, b]]
            by_count = {}
            for mask in masks:
                count = mask.bit_count()
                if count > budget:
                    continue
                union = np.zeros(ins.n_demand, dtype=bool)
                for i, j in enumerate(sites):
                    if mask & (1 << i):
                        union[ins.demand_of(j)] = True
                value = sum(map(int, ins.weights[union & residual]))
                if count not in by_count or value > by_count[count][1]:
                    by_count[count] = (count, value, mask)
            menu = list(by_count.values())
            if greedy_large:
                counts = [sum(j in parent for j in sites) for parent in [a, b]]
                menu, extra = extend_large_menu(
                    ins, sites, residual, budget, menu, counts, start.bin_exact, deadline
                )
                stats["greedy_groups"] += 1
                stats["greedy_refinement_moves"] += extra["refinement_moves"]
                stats["greedy_memory_skips"] += int(extra["skipped"])
        menus.append(menu)
    dp = [None] * (budget + 1)
    dp[0] = 0
    history = []
    for menu in menus:
        if perf_counter() >= deadline:
            return None, stats
        nxt = [None] * (budget + 1)
        choices = [None] * (budget + 1)
        for used, value in enumerate(dp):
            if value is None:
                continue
            for cost, gain, mask in menu:
                total = used + cost
                if total <= budget and (nxt[total] is None or value + gain > nxt[total]):
                    nxt[total] = value + gain
                    choices[total] = (used, mask)
        dp = nxt
        history.append(choices)
    if dp[budget] is None:
        raise AssertionError("Parent-feasible recombination lost budget feasibility")
    objective = common_value + dp[budget]
    if objective < max(start.f, guide.f):
        raise AssertionError("Completed recombination worse than parent")
    stats["completed"] = True
    stats["menu_objective"] = objective
    if objective <= max(start.f, guide.f) or perf_counter() >= deadline:
        return None, stats
    selected = set(common)
    used = budget
    for sites, choices in zip(reversed(components), reversed(history), strict=True):
        used, mask = choices[used]
        selected.update(j for i, j in enumerate(sites) if mask & (1 << i))
    if len(selected) != start.p:
        raise AssertionError("Wrong offspring budget")
    child = start.copy()
    for out, entry in zip(sorted(a - selected), sorted(selected - a), strict=True):
        if perf_counter() >= deadline:
            return None, stats
        child.exchange(out, entry)
    if child.f != objective:
        raise AssertionError("Independent state objective differs from component DP")
    if perf_counter() >= deadline:
        return None, stats
    return child, stats
