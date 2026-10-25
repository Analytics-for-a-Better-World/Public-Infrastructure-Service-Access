"""Deterministic gap candidates and exact fixed-pool neighborhood, solver-free."""

from time import perf_counter

import numpy as np

from .exclusion_menu import decode_selected_sites, treewidth_exclusion_menu
from .swap_matrix import SwapMatrix


def exclusion_neighborhood(start, deadline, extra_cap=24):
    stats = dict(completed=False, reason=None, extra_count=0, width=0, operations=0)
    ins = start.ins
    p = start.p
    if p + extra_cap > 160:
        stats["reason"] = "vertex_cap"
        return None, stats
    if perf_counter() >= deadline:
        stats["reason"] = "deadline"
        return None, stats
    if p * ins.n_facilities > 2_000_000 or len(ins.ji_indices) > 2_000_000:
        stats["reason"] = "matrix_cap"
        return None, stats
    selected = sorted(map(int, start.order[:p]))
    closed = np.flatnonzero(~start.selected)
    matrix = SwapMatrix(start)
    masks = {}

    def bits(j):
        if j not in masks:
            value = 0
            for demand in ins.demand_of(j):
                if ins.weights[demand] > 0:
                    value |= 1 << int(demand)
            masks[j] = value
        return masks[j]

    selected_masks = [bits(j) for j in selected]
    proposals = []

    def eligible(j):
        return not any(not bits(j) & ~mask for mask in selected_masks)

    for pos, out in enumerate(map(int, start.order[:p])):
        if perf_counter() >= deadline:
            stats["reason"] = "deadline"
            return None, stats
        delta = start.gains + matrix.bonus[pos] - matrix.loss[pos]
        ranked = closed[np.argsort(-delta[closed], kind="stable")]
        for index, j in enumerate(map(int, ranked)):
            if index % 256 == 0 and perf_counter() >= deadline:
                stats["reason"] = "deadline"
                return None, stats
            if eligible(j):
                proposals.append((int(delta[j]), int(matrix.loss[pos]), out, ranked))
                break
    extras = []
    present = set()
    for _gain, _loss, _out, ranked in sorted(proposals, key=lambda row: (-row[0], -row[1], row[2])):
        if len(extras) >= extra_cap:
            break
        for index, j in enumerate(map(int, ranked)):
            if index % 256 == 0 and perf_counter() >= deadline:
                stats["reason"] = "deadline"
                return None, stats
            if j not in present and eligible(j):
                extras.append(j)
                present.add(j)
                break
    # Prune only dominated extras. All p incumbent sites remain available.
    extras = [
        j
        for j in extras
        if not any(
            h != j and not bits(j) & ~bits(h) and (bits(j) != bits(h) or h < j) for h in extras
        )
    ]
    stats["pool_rule"] = "near_neutral_per_outgoing_original_delta"
    stats["extra_count"] = len(extras)
    stats["pool_sites"] = selected + extras
    if not extras:
        stats["reason"] = "no_positive_extra"
        return None, stats
    sites = selected + extras
    k = len(sites) - p
    menu, details = treewidth_exclusion_menu(ins, sites, np.ones(ins.n_demand, bool), k, deadline)
    stats.update(details)
    stats["exact_completed"] = details["completed"]
    if menu is None:
        return None, stats
    _, objective, excluded = next(entry for entry in menu if entry[0] == k)
    chosen = decode_selected_sites(sites, excluded)
    if not (len(chosen) == p and len(set(chosen)) == p):
        raise RuntimeError("neighborhood budget")
    coverage = np.zeros(ins.n_demand, bool)
    for j in chosen:
        coverage[ins.demand_of(j)] = True
    actual = sum(map(int, ins.weights[coverage]))
    if not (actual == objective and objective >= start.f):
        raise RuntimeError("neighborhood original objective")
    stats["menu_objective"] = objective
    if objective <= start.f:
        return None, stats
    child = start.copy()
    target = set(chosen)
    for out, entry in zip(
        sorted(set(selected) - target), sorted(target - set(selected)), strict=True
    ):
        if perf_counter() >= deadline:
            stats.update(completed=False, reason="child_deadline")
            return None, stats
        child.exchange(out, entry)
    if not (child.f == objective):
        raise RuntimeError("neighborhood state objective")
    if perf_counter() >= deadline:
        stats.update(completed=False, reason="child_deadline")
        return None, stats
    return child, stats
