"""CEM-inspired feasible rollouts + resumable, bounded DP-state search.

Integer coverage bounds are independent of learned probabilities. Deferred and
memory-dropped states remain represented in the returned upper bound. This is
an experimental hybrid, not a verbatim implementation of a published algorithm.
Live in-memory sessions retain search state across calls; no disk checkpoint. Queue probability
hints are captured at insertion; updated models guide new nodes and rollouts.
"""

import heapq
from bisect import insort
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite, log
from random import Random
from time import perf_counter

from .beam_dp import BitWeights, ByteWeights


class EliteModel:
    def __init__(
        self, n, p, seeds, score, *, enabled=True, size=16, models=2, alpha=0.3, epsilon=0.005
    ):
        self.n = n
        self.p = p
        self.enabled = enabled
        self.size = size
        self.alpha = alpha
        self.epsilon = epsilon
        self.models = models
        self.archive = {mask: score(mask) for mask in seeds}
        self.updates = 0
        self.last_fit = None
        self.fit_skips = 0
        self.q = [[max(epsilon, min(1 - epsilon, p / n if n else 0.5))] * n for _ in range(models)]
        if enabled:
            self.fit()

    def observe(self, mask, value):
        self.archive[mask] = value
        ranked = sorted(self.archive, key=lambda m: (-self.archive[m], m))
        # Distance two deduplicates fixed-cardinality masks; separate farthest
        # clusters below provide the additional model diversity.
        kept = []
        for mask in ranked:
            if all((mask ^ other).bit_count() >= 2 for other in kept):
                kept.append(mask)
            if len(kept) == self.size:
                break
        self.archive = {mask: self.archive[mask] for mask in kept}

    def fit(self):
        if not self.enabled or not self.archive:
            return
        signature = tuple(sorted(self.archive.items()))
        if signature == self.last_fit:
            self.fit_skips += 1
            return
        self.last_fit = signature
        ranked = sorted(self.archive, key=lambda m: (-self.archive[m], m))
        centers = [ranked[0]]
        while len(centers) < min(self.models, len(ranked)):
            centers.append(
                max(
                    (m for m in ranked if m not in centers),
                    key=lambda m: (min((m ^ c).bit_count() for c in centers), self.archive[m], -m),
                )
            )
        groups = [[] for _ in centers]
        for mask in ranked:
            index = min(range(len(centers)), key=lambda k: ((mask ^ centers[k]).bit_count(), k))
            groups[index].append(mask)
        for k in range(self.models):
            group = groups[k % len(groups)]
            for j in range(self.n):
                frequency = sum(mask >> j & 1 for mask in group) / len(group)
                self.q[k][j] = max(
                    self.epsilon,
                    min(1 - self.epsilon, (1 - self.alpha) * self.q[k][j] + self.alpha * frequency),
                )
        self.updates += 1

    def preference(self, mask, depth):
        if not depth:
            return 0.0
        return max(
            sum(log(q[j] if mask >> j & 1 else 1 - q[j]) for j in range(depth)) / depth
            for q in self.q
        )

    def sample(self, depth, r, rng, component):
        # Weighted sampling without replacement: exact remaining cardinality.
        # Frequency updates are CEM-inspired, not a conditional-Bernoulli MLE.
        q = self.q[component % self.models]
        keys = [
            (-log(max(rng.random(), 1e-300)) / (q[j] / (1 - q[j])), j) for j in range(depth, self.n)
        ]
        return [j for _, j in heapq.nsmallest(r, keys)]


@dataclass(frozen=True)
class Node:
    depth: int
    count: int
    live: int
    value: int
    chosen: int
    upper: int


def _search(
    facilities,
    weights,
    p,
    incumbent,
    deadline,
    *,
    elites=(),
    beam=16,
    agenda_limit=4096,
    dominance_limit=100000,
    max_expansions=100000,
    use_learning=True,
    elite_size=16,
    models=2,
    alpha=0.3,
    epsilon=0.005,
    rollout_interval=1,
    rollout_nodes=2,
    rollouts=2,
    polish_moves=1,
    seed=0,
    weight_oracle=None,
):
    start = perf_counter()
    n = len(facilities)
    incumbent = list(incumbent)
    if type(p) is not int or not 0 <= p <= n:
        raise ValueError("invalid budget")
    for value, minimum in [
        (beam, 1),
        (dominance_limit, 1),
        (max_expansions, 0),
        (elite_size, 1),
        (models, 1),
        (rollout_interval, 1),
        (rollout_nodes, 0),
        (rollouts, 0),
        (polish_moves, 0),
    ]:
        if type(value) is not int or value < minimum:
            raise ValueError("invalid search control")
    if agenda_limit is not None and (type(agenda_limit) is not int or agenda_limit < 1):
        raise ValueError("invalid agenda limit")
    if type(seed) is not int or type(use_learning) is not bool:
        raise ValueError("invalid seed/learning flag")
    if (
        any(type(x) not in (int, float) or not isfinite(x) for x in [alpha, epsilon])
        or not 0 < alpha <= 1
        or not 0 < epsilon < 0.5
    ):
        raise ValueError("invalid learning controls")
    if any(type(w) is not int or w < 0 for w in weights) or any(
        type(m) is not int or m < 0 or m >> len(weights) for m in facilities
    ):
        raise ValueError("invalid coverage model")
    seeds = [incumbent] + [list(sites) for sites in elites]
    if any(
        len(s) != p or len(set(s)) != p or any(type(j) is not int or not 0 <= j < n for j in s)
        for s in seeds
    ):
        raise ValueError("invalid feasible seed")
    weight = weight_oracle or (
        ByteWeights(weights)
        if len(weights) > 64 and sum(weights) < 1 << 63
        else BitWeights(weights)
    )

    def union(mask):
        cov = 0
        while mask:
            low = mask & -mask
            cov |= facilities[low.bit_length() - 1]
            mask ^= low
        return cov

    def score(mask):
        return weight(union(mask))

    seed_masks = [sum(1 << j for j in sites) for sites in seeds]
    best_mask = max(seed_masks, key=lambda mask: (score(mask), -mask))
    best = score(best_mask)
    initial = score(seed_masks[0])
    stats = dict(
        expanded=0,
        merged=0,
        bound_pruned=0,
        deferred=0,
        memory_dropped=0,
        rollouts=0,
        polish_calls=0,
        events=[],
        batches=0,
    )
    heap = []
    serial = 0
    active_bound = 0
    lost_bound = 0
    preparing = True
    safe_total = sum(weights)

    def finish(reason):
        upper = max([best, lost_bound, active_bound] + [row[-1].upper for row in heap])
        if preparing:
            upper = max(upper, safe_total)
        stats.update(
            reason=reason,
            seconds=perf_counter() - start,
            agenda_states=len(heap),
            lost_upper=lost_bound,
            active_upper=active_bound,
            cem_updates=0 if preparing else model.updates,
        )
        stats["fit_skips"] = 0 if preparing else model.fit_skips
        archive = (
            []
            if preparing
            else [
                dict(objective=model.archive[mask], selected=[j for j in range(n) if mask >> j & 1])
                for mask in sorted(model.archive, key=lambda m: (-model.archive[m], m))
            ]
        )
        return deepcopy(
            dict(
                objective=best,
                selected=[j for j in range(n) if best_mask >> j & 1],
                upper_bound=upper,
                gap=upper - best,
                certified=upper == best,
                stats=stats,
                archive=archive,
            )
        )

    while perf_counter() >= deadline:
        deadline, max_expansions = yield finish("deadline_preparation")
    suffix = [0] * (n + 1)
    suffix_inc = [0] * (n + 1)
    top = [None] * (n + 1)
    top[n] = (0,)
    totals = [weight(mask) for mask in facilities]
    largest = []
    for j in range(n - 1, -1, -1):
        while perf_counter() >= deadline:
            deadline, max_expansions = yield finish("deadline_preparation")
        suffix[j] = suffix[j + 1] | facilities[j]
        suffix_inc[j] = suffix_inc[j + 1] | (facilities[j] if seed_masks[0] >> j & 1 else 0)
        insort(largest, totals[j])
        if len(largest) > p:
            largest.pop(0)
        sums = [0]
        for value in reversed(largest):
            sums.append(sums[-1] + value)
        top[j] = tuple(sums)
    total = weight(suffix[0])
    finalized = [0] * (n + 1)
    for j in range(n):
        finalized[j + 1] = finalized[j] + weight(facilities[j] & ~suffix[j + 1])
    model = EliteModel(
        n,
        p,
        seed_masks,
        score,
        enabled=use_learning,
        size=elite_size,
        models=models,
        alpha=alpha,
        epsilon=epsilon,
    )
    preparing = False
    rng = Random(seed)
    seen = {}

    def bound(depth, count, live, value):
        return min(value + total - finalized[depth], value + weight(live) + top[depth][p - count])

    def enqueue(node):
        nonlocal serial
        if node.upper <= best:
            stats["bound_pruned"] += 1
            return
        key = (node.depth, node.count, node.live)
        if seen.get(key, -1) >= node.value:
            stats["merged"] += 1
            return
        if len(seen) >= dominance_limit:
            seen.clear()
        seen[key] = node.value
        guide = node.value + weight(node.live | suffix_inc[node.depth])
        # Learned preferences are scheduling hints only; NEVER pruning bounds.
        heapq.heappush(
            heap, (-node.upper, -guide, -model.preference(node.chosen, node.depth), serial, node)
        )
        serial += 1

    def accept(mask, phase):
        nonlocal best, best_mask
        if mask.bit_count() != p or mask >> n:
            raise RuntimeError("completion budget")
        value = score(mask)
        model.observe(mask, value)
        if value > best:
            best = value
            best_mask = mask
            stats["events"].append(
                dict(
                    objective=value,
                    phase=phase,
                    elapsed=perf_counter() - start,
                    selected=[j for j in range(n) if mask >> j & 1],
                )
            )
        return value

    def polish(mask):
        for _ in range(polish_moves):
            value = score(mask)
            changed = False
            selected = [j for j in range(n) if mask >> j & 1]
            closed = [j for j in range(n) if not mask >> j & 1]
            for out in selected:
                for entry in closed:
                    if perf_counter() >= deadline:
                        return mask
                    trial = mask ^ (1 << out) ^ (1 << entry)
                    if score(trial) > value:
                        mask = trial
                        changed = True
                        break
                if changed:
                    break
            if not changed:
                break
        return mask

    if best > initial:
        stats["events"].append(
            dict(
                objective=best,
                phase="initial_elite",
                elapsed=perf_counter() - start,
                selected=[j for j in range(n) if best_mask >> j & 1],
            )
        )
    enqueue(Node(0, 0, 0, 0, 0, bound(0, 0, 0, 0)))
    while heap:
        while perf_counter() >= deadline:
            deadline, max_expansions = yield finish("deadline")
        batch = []
        for _ in range(min(beam, len(heap))):
            batch.append(heapq.heappop(heap)[-1])
        active_bound = max(node.upper for node in batch)
        stats["batches"] += 1
        for pos, node in enumerate(batch):
            while perf_counter() >= deadline:
                deadline, max_expansions = yield finish("deadline_active")
            while stats["expanded"] >= max_expansions:
                deadline, max_expansions = yield finish("expansion_limit")
            if node.upper <= best:
                stats["bound_pruned"] += 1
                continue
            if node.value < seen.get((node.depth, node.count, node.live), node.value):
                stats["merged"] += 1
                continue
            if node.depth == n:
                accept(node.chosen, "leaf")
                continue
            if pos < rollout_nodes and stats["batches"] % rollout_interval == 0:
                remaining = p - node.count
                # Common incumbent-based feasible completion, followed by
                # learned (or uniform, in ablation) randomized completions.
                future = sorted(
                    range(node.depth, n),
                    key=lambda j: (not (seed_masks[0] >> j & 1), -totals[j], j),
                )
                choices = [future[:remaining]] + [
                    model.sample(node.depth, remaining, rng, k) for k in range(rollouts)
                ]
                for choices_remaining in choices:
                    while perf_counter() >= deadline:
                        deadline, max_expansions = yield finish("deadline_rollout")
                    mask = node.chosen | sum(1 << j for j in choices_remaining)
                    accept(mask, "completion")
                    stats["rollouts"] += 1
                    if polish_moves and perf_counter() < deadline:
                        stats["polish_calls"] += 1
                        accept(polish(mask), "polished_completion")
            j = node.depth
            stats["expanded"] += 1
            last = facilities[j] & ~suffix[j + 1]
            for take in (0, 1):
                count = node.count + take
                if count > p or p - count > n - j - 1:
                    continue
                cov = node.live | (facilities[j] if take else 0)
                value = node.value + weight(cov & last)
                live = cov & suffix[j + 1]
                child = Node(
                    j + 1,
                    count,
                    live,
                    value,
                    node.chosen | ((1 << j) if take else 0),
                    bound(j + 1, count, live, value),
                )
                enqueue(child)
        model.fit()
        active_bound = 0
        stats["deferred"] += max(0, len(heap) - beam)
        if agenda_limit is not None and len(heap) > agenda_limit:
            keep = heapq.nsmallest(agenda_limit, heap)
            kept_serial = {row[-2] for row in keep}
            dropped = [row[-1] for row in heap if row[-2] not in kept_serial]
            lost_bound = max([lost_bound] + [node.upper for node in dropped])
            stats["memory_dropped"] += len(dropped)
            heap = keep
            heapq.heapify(heap)
    yield finish("exhausted")


class BeamSession:
    """One immutable regional model, resumed in memory (not thread safe).

    max_expansions is a per-call allowance. A pause retains even an unfinished
    batch, unlike restarting from its incumbent. Returned records are snapshots.
    """

    def __init__(self, facilities, weights, p, incumbent, **controls):
        self.args = (tuple(facilities), tuple(weights), p, tuple(incumbent))
        self.controls = dict(controls)
        if "elites" in self.controls:
            self.controls["elites"] = tuple(tuple(x) for x in self.controls["elites"])
        self.generator = None
        self.last = None
        self.calls = 0
        self.active_seconds = 0.0

    def run(self, deadline, *, max_expansions=100000):
        if type(deadline) not in (int, float) or not isfinite(deadline):
            raise ValueError("finite monotonic deadline required")
        if type(max_expansions) is not int or max_expansions < 0:
            raise ValueError("invalid expansion allowance")
        start = perf_counter()
        before = 0 if self.last is None else self.last["stats"]["expanded"]
        if self.last is not None and self.last["stats"]["reason"] == "exhausted":
            result = deepcopy(self.last)
        elif self.generator is None:
            self.generator = _search(
                *self.args, deadline, max_expansions=max_expansions, **self.controls
            )
            result = next(self.generator)
        else:
            result = self.generator.send((deadline, before + max_expansions))
        self.calls += 1
        elapsed = perf_counter() - start
        self.active_seconds += elapsed
        result["session"] = dict(
            calls=self.calls,
            expanded_this_call=result["stats"]["expanded"] - before,
            call_seconds=elapsed,
            active_seconds=self.active_seconds,
        )
        self.last = deepcopy(result)
        return result


def solve_cem_beam(
    facilities, weights, p, incumbent, deadline, *, max_expansions=100000, **controls
):
    return BeamSession(facilities, weights, p, incumbent, **controls).run(
        deadline, max_expansions=max_expansions
    )
