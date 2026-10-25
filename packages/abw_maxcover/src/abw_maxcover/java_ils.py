"""Source-directed AILS-PR replication; no JVM or solver required for search.

Java-compatible RNG streams and permutation ordering support trajectory checks.
Cooperative deadlines and protected public incumbents are explicit wrapper changes.
"""

import hashlib
import struct
from dataclasses import dataclass
from functools import lru_cache
from math import acos, ceil, cos, pi, sin
from time import perf_counter

import numpy as np

from .component_recombine import component_recombine
from .faithful_local import local_descent as faithful_local_descent
from .results import HeuristicResult, MaxCoverResult


@lru_cache(maxsize=8192)
def _jump_transform(steps):
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("Nonnegative integer steps required")
    mask = (1 << 48) - 1
    aa, cc, a, c = 1, 0, 0x5DEECE66D, 11
    while steps:
        if steps & 1:
            aa, cc = aa * a & mask, (cc * a + c) & mask
        a, c = a * a & mask, (a + 1) * c & mask
        steps >>= 1
    return aa, cc


class JavaRandom:
    def __init__(self, seed):
        self.seed = (int(seed) ^ 0x5DEECE66D) & ((1 << 48) - 1)

    def next(self, bits):
        self.seed = (self.seed * 0x5DEECE66D + 11) & ((1 << 48) - 1)
        return self.seed >> (48 - bits)

    def integer(self, bound):
        if bound <= 0:
            raise ValueError("Positive bound required")
        if bound & (bound - 1) == 0:
            return (bound * self.next(31)) >> 31
        while True:
            bits = self.next(31)
            value = bits % bound
            if bits - value + (bound - 1) < (1 << 31):
                return value

    def advance(self, steps):
        a, c = _jump_transform(steps)
        self.seed = (a * self.seed + c) & ((1 << 48) - 1)

    def boolean(self):
        return self.next(1) != 0


@dataclass
class JavaILSConfig:
    seed: int = 1
    time_limit_seconds: float = 60.0
    iterations: int | None = None
    fake_clock: bool = False
    guide_tournament: int = 1
    neutral_steps: int = 0
    dynamic_relink: bool = True
    component_relink: bool = True
    exact_component_cap: int = 12
    large_component_greedy: bool = False
    anchored_component: bool = False
    global_record_feedback: bool = False
    faithful_local: bool = False
    initial_starts: int = 1
    seed_capacity: int = 40


def elite_index(rng, size, tournament):
    """Uniform (1) or minimum rank of a uniformly sampled rank tuple."""
    value = rng.integer(size**tournament)
    best = size - 1
    for _ in range(tournament):
        best = min(best, value % size)
        value //= size
    return best


class Mean:
    def __init__(self):
        self.count = 0
        self.value = 0.0

    def add(self, x):
        self.count += 1
        self.value = (
            (self.value * (self.count - 1) + x) / self.count
            if self.count < 30
            else self.value * (1 - 1 / 30) + x * (1 / 30)
        )


class Feedback:
    def __init__(self, p):
        self.omega = float(min(10, p))
        self.p = p
        self.mean = Mean()

    def observe(self, distance):
        self.mean.add(distance)
        if self.mean.count % 30 == 0:
            self.omega = (
                min(self.p, max(1.0, self.omega / self.mean.value * 10))
                if self.mean.value
                else float(self.p)
            )


class Acceptance:
    def __init__(self, now, limit):
        self.mean = Mean()
        self.ceiling = float(2**63 - 1)
        self.next_ceiling = float(2**63 - 1)
        self.eta = 1.0
        self.alpha = 1.0
        self.now = now
        self.limit = limit
        self.started = None

    def accepts(self, f):
        if self.started is None:
            self.started = self.now()
        n = self.mean.count + 1
        if n % 30 == 0:
            self.ceiling, self.next_ceiling = self.next_ceiling, 0.0
        self.next_ceiling = max(self.next_ceiling, f)
        self.ceiling = max(self.ceiling, f)
        self.mean.add(f)
        if n % 30 == 0:
            fraction = ((self.now() - self.started) / 1000) / self.limit
            self.alpha = 0.1 ** (fraction / n)
        self.eta = max(0.1, self.eta * self.alpha)
        threshold = int(self.mean.value + (1 - self.eta) * (self.ceiling - self.mean.value))
        return f > threshold


class State:
    def __init__(self, ins, p):
        self.ins = ins
        self.p = p
        self.order = np.arange(ins.n_facilities, dtype=np.int32)
        self.positions = self.order.copy()
        self.selected = np.zeros(ins.n_facilities, dtype=bool)
        self.coverage = np.zeros(ins.n_demand, dtype=np.int32)
        self.gains = np.zeros(ins.n_facilities, dtype=np.int64)
        self.f = 0
        self.bin_exact = (
            all(int(w) >= 0 for w in ins.weights) and sum(map(int, ins.weights)) <= 2**53
        )

    def copy(self):
        x = State.__new__(State)
        x.ins = self.ins
        x.p = self.p
        x.f = self.f
        x.bin_exact = self.bin_exact
        for name in ("order", "positions", "selected", "coverage", "gains"):
            setattr(x, name, getattr(self, name).copy())
        return x

    def swap_positions(self, a, b):
        ja, jb = int(self.order[a]), int(self.order[b])
        self.order[a], self.order[b] = jb, ja
        self.positions[ja], self.positions[jb] = b, a

    def adjust(self, demands, sign):
        if not len(demands):
            return
        self.f += sign * int(self.ins.weights[demands].sum())
        # NumPy performs only the arithmetic; this remains a Python algorithm.
        starts = self.ins.ij_indptr[demands]
        counts = self.ins.ij_indptr[demands + 1] - starts
        ends = np.cumsum(counts, dtype=np.intp)
        bases = ends - counts
        offsets = np.repeat(starts - bases, counts) + np.arange(int(ends[-1]))
        facilities = self.ins.ij_indices[offsets]
        values = np.repeat(self.ins.weights[demands], counts)
        if self.bin_exact:
            delta = np.bincount(facilities, weights=values, minlength=self.ins.n_facilities).astype(
                np.int64
            )
            delta[self.selected] = 0
            self.gains -= sign * delta
        else:
            mask = ~self.selected[facilities]
            np.add.at(self.gains, facilities[mask], -sign * values[mask])

    def add(self, j):
        self.selected[j] = True
        self.gains[j] = 0
        row = self.ins.demand_of(j)
        changed = row[self.coverage[row] == 0]
        self.adjust(changed, 1)
        self.coverage[row] += 1

    def remove(self, j):
        self.selected[j] = False
        row = self.ins.demand_of(j)
        changed = row[self.coverage[row] == 1]
        self.adjust(changed, -1)
        self.coverage[row] -= 1

    def exchange(self, out, entry):
        a, b = int(self.positions[out]), int(self.positions[entry])
        self.remove(out)
        self.add(entry)
        self.swap_positions(a, b)

    def result(self):
        return HeuristicResult(
            list(map(int, self.order[: self.p])),
            self.f,
            self.coverage.copy(),
            [self.f],
            [0.0],
            0.0,
        )


def local_descent(s, shuffle_rng, tie_rng, deadline):
    """Exact complete one-swap descent, with a new uniform best-tie rule."""
    for _ in range(s.p):
        if perf_counter() >= deadline:
            return False
        s.swap_positions(shuffle_rng.integer(s.p), shuffle_rng.integer(s.p))
    cache = {} if s.p * s.ins.n_facilities <= 2_000_000 else None
    zero = np.zeros(s.ins.n_facilities, dtype=np.int64)
    improved = True
    base_max = int(s.gains.max())
    while improved:
        improved = False
        for position in range(s.p):
            if perf_counter() >= deadline:
                return False
            out = int(s.order[position])
            row = s.ins.demand_of(out)
            exclusive = row[s.coverage[row] == 1]
            cached = None if cache is None else cache.get(out)
            if cached is not None and np.array_equal(exclusive, cached[0]):
                bonus, active, loss = cached[1:]
            else:
                loss = int(s.ins.weights[exclusive].sum())
                bonus = zero
                if len(exclusive):
                    bonus = np.zeros(s.ins.n_facilities, dtype=np.int64)
                    starts = s.ins.ij_indptr[exclusive]
                    counts = s.ins.ij_indptr[exclusive + 1] - starts
                    bases = np.cumsum(counts, dtype=np.intp) - counts
                    offsets = np.repeat(starts - bases, counts) + np.arange(int(counts.sum()))
                    if s.bin_exact:
                        bonus = np.bincount(
                            s.ins.ij_indices[offsets],
                            weights=np.repeat(s.ins.weights[exclusive], counts),
                            minlength=s.ins.n_facilities,
                        ).astype(np.int64)
                    else:
                        np.add.at(
                            bonus,
                            s.ins.ij_indices[offsets],
                            np.repeat(s.ins.weights[exclusive], counts),
                        )
                active = np.flatnonzero(bonus)
                if cache is not None:
                    cache[out] = (exclusive, bonus, active, loss)
            candidates = active[~s.selected[active]]
            maximum = base_max
            if len(candidates):
                maximum = max(maximum, int((s.gains[candidates] + bonus[candidates]).max()))
            if maximum <= loss:
                continue
            # The global base gain covers every non-overlapping replacement.
            # Only accepted moves need a full tie extraction, in sequential ID order.
            options = np.flatnonzero(~s.selected & (s.gains + bonus == maximum))
            entry = int(options[tie_rng.integer(len(options))])
            s.exchange(out, entry)
            if cache is not None:
                cache.pop(out, None)
            base_max = int(s.gains.max())
            improved = True
    return True


def neutral_step(s, rng, forbidden, deadline):
    """Try at most eight outgoing facilities and sixteen incoming ties per outgoing."""
    positions = list(range(s.p))
    for k in range(min(8, s.p)):
        if perf_counter() >= deadline:
            return False
        idx = k + rng.integer(s.p - k)
        positions[k], positions[idx] = positions[idx], positions[k]
        out = int(s.order[positions[k]])
        row = s.ins.demand_of(out)
        exclusive = row[s.coverage[row] == 1]
        loss = int(s.ins.weights[exclusive].sum())
        bonus = np.zeros(s.ins.n_facilities, dtype=np.int64)
        if len(exclusive):
            starts = s.ins.ij_indptr[exclusive]
            counts = s.ins.ij_indptr[exclusive + 1] - starts
            bases = np.cumsum(counts, dtype=np.intp) - counts
            offsets = np.repeat(starts - bases, counts) + np.arange(int(counts.sum()))
            values = np.repeat(s.ins.weights[exclusive], counts)
            ids = s.ins.ij_indices[offsets]
            if s.bin_exact:
                bonus = np.bincount(ids, weights=values, minlength=s.ins.n_facilities).astype(
                    np.int64
                )
            else:
                np.add.at(bonus, ids, values)
        candidates = np.flatnonzero(~s.selected & (s.gains + bonus == loss))
        for q in range(min(16, len(candidates))):
            if perf_counter() >= deadline:
                return False
            idx = q + rng.integer(len(candidates) - q)
            candidates[q], candidates[idx] = candidates[idx], candidates[q]
            entry = int(candidates[q])
            if entry in forbidden:
                continue
            # Identical rows are deliberately excluded; equal covered unions are allowed.
            if np.array_equal(np.sort(row), np.sort(s.ins.demand_of(entry))):
                continue
            old_f = s.f
            s.exchange(out, entry)
            if s.f != old_f:
                raise AssertionError("Neutral step changed exact objective")
            forbidden.add(out)
            return True
    return False


def geographic_distance(coords, a, b):
    if a == b:
        return 0.0
    x, y = coords[a]
    xx, yy = coords[b]
    rad = pi / 180
    v = cos(x * rad) * cos(xx * rad) * cos((yy - y) * rad) + sin(x * rad) * sin(xx * rad)
    angle = acos(v) if -1 <= v <= 1 else float("nan")
    # Java casts angle to float before float multiplication by earth radius.
    return float(np.float32(np.float32(angle) * np.float32(6378100)))


def perturb(s, kind, rng, feedback, neighbors, coords, deadline):
    omega = feedback.omega
    heuristic = rng.integer(2)
    concentric = None
    if kind == 1:
        center = int(s.order[rng.integer(s.p)])
        distances = []
        for j in map(int, s.order[: s.p]):
            if perf_counter() >= deadline:
                return False
            distances.append((geographic_distance(coords, j, center), j))
        distances.sort(key=lambda item: item[0])
        concentric = [j for _, j in distances]
    for index in range(int(ceil(omega))):
        if perf_counter() >= deadline:
            return False
        out = int(s.order[rng.integer(s.p)]) if kind == 0 else concentric[index]
        position = int(s.positions[out])
        s.remove(out)
        if heuristic == 0:
            entry = int(s.order[s.p + rng.integer(len(s.order) - s.p)])
        else:
            row = neighbors[out]
            entry = None
            for _ in range(len(row)):
                j = int(row[rng.integer(len(row))])
                if not s.selected[j]:
                    entry = j
                    break
            if entry is None:
                # Sparse Java adaptation's deterministic fallback.
                entry = next((int(j) for j in row if not s.selected[j]), out)
        s.add(entry)
        s.swap_positions(position, int(s.positions[entry]))
    return True


def insert_elite(pool, s, target=10, capacity=40):
    if len(pool) >= capacity and s.f < pool[-1].f:
        return
    close = [
        i for i, x in enumerate(pool) if len(set(s.order[: s.p]) - set(x.order[: x.p])) < target
    ]
    if any(pool[i].f > s.f for i in close):
        return
    if len(close) == 1:
        pool[close[0]] = s.copy()
    elif close:
        i = 0
        while i < len(pool):
            if len(set(s.order[: s.p]) - set(pool[i].order[: s.p])) < target:
                pool[i] = pool[-1]
                pool.pop()
            else:
                i += 1
        pool.append(s.copy())
    elif len(pool) < capacity:
        pool.append(s.copy())
    else:
        pool[-1] = s.copy()
    pool.sort(key=lambda x: -x.f)


def static_matched_relink(start, guide, rng, deadline):
    s = start.copy()
    exits = [int(j) for j in s.order[: s.p] if not guide.selected[j]]
    entries = [int(j) for j in guide.order[: guide.p] if not s.selected[j]]
    for _ in range(len(exits)):
        a, b = rng.integer(len(exits)), rng.integer(len(exits))
        exits[a], exits[b] = exits[b], exits[a]
    pairs = []
    for out in exits:
        if perf_counter() >= deadline:
            return None
        row = s.ins.demand_of(out)
        exclusive = row[s.coverage[row] == 1]
        bonus = np.zeros(s.ins.n_facilities, dtype=np.int64)
        if len(exclusive):
            starts = s.ins.ij_indptr[exclusive]
            counts = s.ins.ij_indptr[exclusive + 1] - starts
            bases = np.cumsum(counts, dtype=np.intp) - counts
            offsets = np.repeat(starts - bases, counts) + np.arange(int(counts.sum()))
            if s.bin_exact:
                bonus = np.bincount(
                    s.ins.ij_indices[offsets],
                    weights=np.repeat(s.ins.weights[exclusive], counts),
                    minlength=s.ins.n_facilities,
                ).astype(np.int64)
            else:
                np.add.at(
                    bonus, s.ins.ij_indices[offsets], np.repeat(s.ins.weights[exclusive], counts)
                )
        values = s.gains[entries] + bonus[entries]
        idx = int(np.argmax(values))
        entry = entries[idx]
        entries[idx] = entries[-1]
        entries.pop()
        pairs.append((out, entry))
    best = None
    if perf_counter() >= deadline:
        return best
    differences = []
    for out, entry in pairs:
        if perf_counter() >= deadline:
            return best
        differences.append(
            np.setdiff1d(s.ins.demand_of(out), s.ins.demand_of(entry), assume_unique=True)
        )
    lengths = np.asarray([len(row) for row in differences], dtype=np.intp)
    starts = np.cumsum(lengths, dtype=np.intp) - lengths
    nonempty = lengths > 0
    flat = np.concatenate(differences) if nonempty.any() else np.empty(0, dtype=np.int32)
    losses = np.zeros(len(pairs), dtype=np.int64)
    entry_ids = np.asarray([entry for _, entry in pairs], dtype=np.int32)
    active = list(range(len(pairs)))
    while active:
        if perf_counter() >= deadline:
            break
        if nonempty.any():
            losses[nonempty] = np.add.reduceat(
                np.where(s.coverage[flat] == 1, s.ins.weights[flat], 0),
                starts[nonempty],
            )
        indices = np.asarray(active, dtype=np.intp)
        scores = s.gains[entry_ids[indices]] - losses[indices]
        index = int(np.argmax(scores))
        out, entry = pairs[active[index]]
        s.exchange(out, entry)
        active[index] = active[-1]
        active.pop()
        if s.f > start.f and s.f > guide.f and (best is None or s.f > best.f):
            best = s.copy()
    return best


def dynamic_pair_scores(s, exits, entries):
    """Exact current exchange deltas over all allowed outgoing/incoming pairs."""
    owner = np.full(s.ins.n_demand, -1, dtype=np.int32)
    loss = np.zeros(len(exits), dtype=np.int64)
    for i, out in enumerate(exits):
        row = s.ins.demand_of(out)
        exclusive = row[s.coverage[row] == 1]
        owner[exclusive] = i
        loss[i] = int(s.ins.weights[exclusive].sum())
    rows = [s.ins.demand_of(entry) for entry in entries]
    counts = np.asarray([len(row) for row in rows], dtype=np.intp)
    demands = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
    columns = np.repeat(np.arange(len(entries)), counts)
    owners = owner[demands]
    keep = owners >= 0
    ids = owners[keep].astype(np.intp) * len(entries) + columns[keep]
    weights = s.ins.weights[demands[keep]]
    size = len(exits) * len(entries)
    if s.bin_exact:
        bonus = np.bincount(ids, weights=weights, minlength=size).astype(np.int64)
    else:
        bonus = np.zeros(size, dtype=np.int64)
        np.add.at(bonus, ids, weights)
    return s.gains[entries][None, :] - loss[:, None] + bonus.reshape(len(exits), len(entries))


class DynamicPathScores:
    """Frozen path incidence; current coverage/gains and active order stay dynamic."""

    def __init__(self, s, exits, entries):
        self.exits = np.asarray(exits, dtype=np.int32)
        self.entries = np.asarray(entries, dtype=np.int32)
        self.k = len(exits)
        out_rows = [s.ins.demand_of(j) for j in exits]
        in_rows = [s.ins.demand_of(j) for j in entries]
        self.out_demands = np.concatenate(out_rows) if out_rows else np.empty(0, dtype=np.int32)
        self.in_demands = np.concatenate(in_rows) if in_rows else np.empty(0, dtype=np.int32)
        self.out_ids = np.repeat(np.arange(self.k), [len(row) for row in out_rows])
        self.in_ids = np.repeat(np.arange(self.k), [len(row) for row in in_rows])

    def scores(self, s, active_out, active_in):
        owner = np.full(s.ins.n_demand, -1, dtype=np.int32)
        exclusive = (s.coverage[self.out_demands] == 1) & s.selected[self.exits[self.out_ids]]
        demands = self.out_demands[exclusive]
        rows = self.out_ids[exclusive]
        owner[demands] = rows
        if s.bin_exact:
            loss = np.bincount(rows, weights=s.ins.weights[demands], minlength=self.k).astype(
                np.int64
            )
        else:
            loss = np.zeros(self.k, dtype=np.int64)
            np.add.at(loss, rows, s.ins.weights[demands])
        owners = owner[self.in_demands]
        keep = owners >= 0
        ids = owners[keep].astype(np.intp) * self.k + self.in_ids[keep]
        weights = s.ins.weights[self.in_demands[keep]]
        if s.bin_exact:
            bonus = np.bincount(ids, weights=weights, minlength=self.k * self.k).astype(np.int64)
        else:
            bonus = np.zeros(self.k * self.k, dtype=np.int64)
            np.add.at(bonus, ids, weights)
        all_scores = s.gains[self.entries][None, :] - loss[:, None] + bonus.reshape(self.k, self.k)
        return all_scores[np.ix_(active_out, active_in)]


def matched_relink(start, guide, rng, deadline):
    if perf_counter() >= deadline:
        return None
    s = start.copy()
    exits = [int(j) for j in s.order[: s.p] if not guide.selected[j]]
    entries = [int(j) for j in guide.order[: guide.p] if not s.selected[j]]
    for _ in range(len(exits)):
        a, b = rng.integer(len(exits)), rng.integer(len(exits))
        exits[a], exits[b] = exits[b], exits[a]
    if not exits:
        return None
    data = DynamicPathScores(s, exits, entries)
    active_out = list(range(len(exits)))
    active_in = list(range(len(entries)))
    best = None
    while active_out:
        if perf_counter() >= deadline:
            break
        scores = data.scores(s, active_out, active_in)
        if perf_counter() >= deadline:
            break
        i, j = divmod(int(np.argmax(scores)), len(active_in))
        out, entry = exits[active_out[i]], entries[active_in[j]]
        predicted = int(scores[i, j])
        old_f = s.f
        s.exchange(out, entry)
        if s.f - old_f != predicted:
            raise AssertionError("Dynamic path exchange delta mismatch")
        active_out[i] = active_out[-1]
        active_out.pop()
        active_in[j] = active_in[-1]
        active_in.pop()
        if s.f > start.f and s.f > guide.f and (best is None or s.f > best.f):
            best = s.copy()
    return best


def state_digest(states, pool):
    md = hashlib.sha256()

    def number(x):
        md.update(struct.pack(">q", int(x)))

    def one(s):
        number(s.f)
        number(len(s.order))
        number(s.p)
        for i, j in enumerate(s.order):
            number(j)
            number(s.gains[i])
        for value in s.coverage:
            number(value)

    for s in states:
        one(s)
    number(len(pool))
    for s in pool:
        one(s)
    return md.hexdigest()


def run_java_ils(
    instance,
    budget,
    *,
    config=None,
    facility_coordinates=None,
    facility_neighbors=None,
    incumbent_callback=None,
    trace_callback=None,
):
    started = perf_counter()
    config = config or JavaILSConfig()
    if type(config.seed_capacity) is not int or not 1 <= config.seed_capacity <= 40:
        raise ValueError("seed_capacity must be integer within1..40")
    if type(config.initial_starts) is not int or not 1 <= config.initial_starts <= 40:
        raise ValueError("initial_starts must be integer within1..40")
    if type(config.faithful_local) is not bool:
        raise ValueError("faithful_local must be bool")
    if type(config.global_record_feedback) is not bool:
        raise ValueError("global_record_feedback must be bool")
    if type(config.anchored_component) is not bool:
        raise ValueError("Anchored component recombination must be boolean")
    if type(config.large_component_greedy) is not bool:
        raise ValueError("Large-component greedy menus must be boolean")
    if type(config.exact_component_cap) is not int or not 1 <= config.exact_component_cap <= 16:
        raise ValueError("Exact component cap must be an integer within1..16")
    if type(config.component_relink) is not bool:
        raise ValueError("Component recombination must be boolean")
    if type(config.dynamic_relink) is not bool:
        raise ValueError("Dynamic relink must be boolean")
    relinker = matched_relink if config.dynamic_relink else static_matched_relink
    if type(config.neutral_steps) is not int or config.neutral_steps < 0:
        raise ValueError("Neutral steps must be a nonnegative integer")
    if not np.isfinite(config.time_limit_seconds) or config.time_limit_seconds <= 0:
        raise ValueError("Positive finite time required")
    if type(budget) not in (int, np.int32, np.int64) or not 1 <= budget < instance.n_facilities:
        raise ValueError("Require 1<=budget<candidates")
    if type(config.guide_tournament) is not int or not 1 <= config.guide_tournament <= 4:
        raise ValueError("Guide tournament must be an integer from1to4")
    if config.iterations is not None and (
        type(config.iterations) is not int or config.iterations < 0
    ):
        raise ValueError("Nonnegative integer iterations required")
    if np.any(instance.weights < 0) or sum(map(int, instance.weights)) > np.iinfo(np.int64).max:
        raise ValueError("Nonnegative int64-safe weights required")
    coords = np.asarray(facility_coordinates, dtype=float)
    if coords.shape != (instance.n_facilities, 2) or not np.all(np.isfinite(coords)):
        raise ValueError("Finite coordinates required")
    if facility_neighbors is None or len(facility_neighbors) != instance.n_facilities:
        raise ValueError("Ordered radius neighborhoods required")
    for j, row in enumerate(facility_neighbors):
        row = np.asarray(row)
        if (
            row.ndim != 1
            or row.dtype.kind not in "iu"
            or not len(row)
            or row[0] < 0
            or row[-1] >= instance.n_facilities
            or np.any(row[1:] <= row[:-1])
            or j not in row
        ):
            raise ValueError("Invalid radius neighborhood")
    deadline = started + config.time_limit_seconds
    fake = [0]

    def now():
        if config.fake_clock:
            value = fake[0]
            fake[0] += 1
            return value
        return int(perf_counter() * 1000)

    streams = {i: JavaRandom(config.seed + 1000003 * i) for i in (0, 1, 2, 4, 5, 6, 8, 9, 11, 13)}
    now()  # Match Java search-start clock tick in deterministic parity mode.
    descent = faithful_local_descent if config.faithful_local else local_descent
    current = State(instance, budget)
    reference = State(instance, budget)
    public_best = State(instance, budget)
    public_best.selected[:budget] = True
    public_best.coverage = np.bincount(
        instance.ji_indices[: instance.ji_indptr[budget]], minlength=instance.n_demand
    ).astype(np.int32)
    public_best.f = int(instance.weights[public_best.coverage > 0].sum())
    # The fallback is also a reusable search state, including exact closed gains.
    fallback_nonempty = np.diff(instance.ji_indptr) > 0
    fallback_weights = np.where(public_best.coverage > 0, 0, instance.weights)
    public_best.gains[fallback_nonempty] = np.add.reduceat(
        fallback_weights[instance.ji_indices], instance.ji_indptr[:-1][fallback_nonempty]
    )
    public_best.gains[public_best.selected] = 0
    events = 0

    def publish(s, phase):
        nonlocal public_best, events
        if events and s.f <= public_best.f:
            return
        public_best = s.copy()
        events += 1
        if incumbent_callback is not None:
            incumbent_callback(
                dict(
                    objective=s.f,
                    sites=list(map(int, s.order[:budget])),
                    phase=phase,
                    elapsed_seconds=perf_counter() - started,
                )
            )

    publish(public_best, "deadline_fallback")
    nonempty = np.diff(instance.ji_indptr) > 0
    reference.gains[nonempty] = np.add.reduceat(
        instance.weights[instance.ji_indices], instance.ji_indptr[:-1][nonempty]
    )
    for i in range(budget):
        if perf_counter() >= deadline:
            result = MaxCoverResult(
                budget,
                "java_style_ails_pr",
                public_best.f,
                list(map(int, public_best.order[:budget])),
                coverage=public_best.coverage.copy(),
            )
            result.total_seconds = result.solve_seconds = perf_counter() - started
            result.metadata = {
                "iterations": 0,
                "stopping_reason": "constructor_deadline",
                "experimental": True,
                "protected_wrapper_incumbents": True,
            }
            return result
        reference.swap_positions(i, i + streams[8].integer(instance.n_facilities - i))
        reference.add(int(reference.order[i]))
    publish(reference, "random_constructor")
    initial_complete = descent(reference, streams[2], streams[4], deadline)
    internal_best = reference.copy()
    best_f = 0
    publish(reference, "initial_descent")
    pool = []
    insert_elite(pool, reference, capacity=config.seed_capacity)
    feedback = [Feedback(budget), Feedback(budget)]
    acceptance = Acceptance(now, config.time_limit_seconds)
    seed_started = perf_counter()
    seed_records = [
        dict(
            index=0,
            objective=reference.f,
            sites=list(map(int, reference.order[:budget])),
            descent_complete=bool(initial_complete),
            rng_owners=[8, 2, 4],
        )
    ]
    seed_best_index = 0
    seed_partial = 0
    if config.initial_starts > 1:
        if public_best.f > reference.f:
            reference = public_best.copy()
            seed_best_index = -1
            insert_elite(pool, reference, capacity=config.seed_capacity)
        seed_gains = np.zeros(instance.n_facilities, dtype=np.int64)
        seed_gains[nonempty] = np.add.reduceat(
            instance.weights[instance.ji_indices], instance.ji_indptr[:-1][nonempty]
        )
        for start_index in range(1, config.initial_starts):
            if perf_counter() >= deadline:
                break
            trial = State(instance, budget)
            trial.gains[:] = seed_gains
            owner = 20 + 3 * start_index
            construction_rng = JavaRandom(config.seed + 1000003 * owner)
            shuffle_rng = JavaRandom(config.seed + 1000003 * (owner + 1))
            tie_rng = JavaRandom(config.seed + 1000003 * (owner + 2))
            partial = False
            for position in range(budget):
                if perf_counter() >= deadline:
                    partial = True
                    break
                trial.swap_positions(
                    position, position + construction_rng.integer(instance.n_facilities - position)
                )
                trial.add(int(trial.order[position]))
            if partial:
                seed_partial += 1
                break
            publish(trial, "extra_constructor")
            complete = descent(trial, shuffle_rng, tie_rng, deadline)
            publish(trial, "extra_initial_descent")
            insert_elite(pool, trial, capacity=config.seed_capacity)
            seed_records.append(
                dict(
                    index=start_index,
                    objective=trial.f,
                    sites=list(map(int, trial.order[:budget])),
                    descent_complete=bool(complete),
                    rng_owners=[owner, owner + 1, owner + 2],
                )
            )
            if trial.f > reference.f:
                reference = trial.copy()
                seed_best_index = start_index
        internal_best = reference.copy()
    seed_seconds = perf_counter() - seed_started
    seeded_pool_size = len(pool)
    seeded_reference_objective = reference.f
    seeded_distances = [
        len(set(item.order[:budget]) - set(reference.order[:budget])) for item in pool
    ]
    neutral_rng = JavaRandom(config.seed + 19000057)
    neutral_moves = neutral_improvements = neutral_attempts = 0
    iterations = 0
    completed_descents = 0
    pr_improvements = 0
    record_feedback_current = record_feedback_relink = 0
    component_attempts = component_children = component_incumbents = 0
    component_exact_groups = component_large_groups = component_subsets = (
        component_expanded_groups
    ) = 0
    component_greedy_groups = component_greedy_refinement_moves = component_greedy_memory_skips = 0
    anchor_key = None
    anchor_tried = set()
    anchor_attempts = anchor_children = anchor_incumbents = anchor_cache_skips = (
        anchor_completed
    ) = 0
    anchor_distance_sum = anchor_exact_groups = anchor_large_groups = 0
    if trace_callback:
        trace_callback(0, state_digest((current, reference, internal_best), pool))

    def analyze(s, phase):
        nonlocal best_f, internal_best
        if s.f > best_f:
            best_f = s.f
            internal_best = s.copy()
            now()
        publish(s, phase)

    while (
        iterations < config.iterations
        if config.iterations is not None
        else perf_counter() < deadline
    ):
        if perf_counter() >= deadline:
            break
        iterations += 1
        if config.neutral_steps and iterations % 25 == 0:
            trial = public_best.copy()
            forbidden = set()
            neutral_attempts += 1
            for _ in range(config.neutral_steps):
                if not neutral_step(trial, neutral_rng, forbidden, deadline):
                    break
                neutral_moves += 1
                descent(trial, streams[1], streams[4], deadline)
                if trial.f > public_best.f:
                    neutral_improvements += 1
                    analyze(trial, "neutral_walk")
                    insert_elite(pool, trial)
                    reference = trial.copy()
        current = reference.copy()
        kind = streams[0].integer(2)
        perturb(
            current,
            kind,
            streams[11 if kind == 0 else 13],
            feedback[kind],
            facility_neighbors,
            coords,
            deadline,
        )
        done = descent(current, streams[1], streams[4], deadline)
        completed_descents += int(done)
        distance = len(set(current.order[:budget]) - set(reference.order[:budget]))
        current_is_record = current.f > public_best.f
        analyze(current, "iterated")
        feedback[kind].observe(distance)
        if distance >= 10 or (distance < 10 and current.f < reference.f):
            insert_elite(pool, current)
        if acceptance.accepts(current.f):
            reference = current.copy()
        if config.global_record_feedback and current_is_record and current.f > reference.f:
            reference = current.copy()
            record_feedback_current += 1
        if perf_counter() < deadline:
            choice = streams[0].integer(3)

            def pick():
                return pool[elite_index(streams[9], len(pool), config.guide_tournament)]

            if choice == 0:
                a, b = pick(), current
            elif choice == 1:
                a, b = current, pick()
            else:
                a, b = pick(), pick()
            linked = None
            relink_phase = "path_relink"
            if config.component_relink and iterations % 10 == 0:
                component_attempts += 1
                linked, diagnostics = component_recombine(
                    a,
                    b,
                    deadline,
                    exact_cap=config.exact_component_cap,
                    greedy_large=config.large_component_greedy and iterations % 50 == 0,
                )
                component_exact_groups += diagnostics["exact_components"]
                component_large_groups += diagnostics["large_components"]
                component_subsets += diagnostics["subsets"]
                component_expanded_groups += diagnostics["expanded_groups"]
                component_greedy_groups += diagnostics["greedy_groups"]
                component_greedy_refinement_moves += diagnostics["greedy_refinement_moves"]
                component_greedy_memory_skips += diagnostics["greedy_memory_skips"]
                if linked is not None:
                    component_children += 1
                    relink_phase = "component_recombine"
            if config.anchored_component and iterations % 50 == 0 and perf_counter() < deadline:
                key = tuple(sorted(map(int, public_best.order[:budget])))
                if key != anchor_key:
                    anchor_key = key
                    anchor_tried.clear()
                anchor_set = set(key)
                candidates = []
                for guide in pool:
                    guide_set = set(map(int, guide.order[:budget]))
                    guide_key = tuple(sorted(guide_set))
                    distance = len(guide_set - anchor_set)
                    if guide_key in anchor_tried:
                        anchor_cache_skips += 1
                        continue
                    if distance >= 2:
                        candidates.append((distance, -guide.f, guide_key, guide))
                if candidates:
                    distance, _, guide_key, guide = min(candidates, key=lambda item: item[:3])
                    anchor_attempts += 1
                    anchor_distance_sum += distance
                    anchor_start = public_best
                    extra, details = component_recombine(
                        anchor_start,
                        guide,
                        deadline,
                        exact_cap=config.exact_component_cap,
                        greedy_large=False,
                    )
                    anchor_exact_groups += details["exact_components"]
                    anchor_large_groups += details["large_components"]
                    anchor_completed += int(details["completed"])
                    if (
                        extra is None
                        and details["completed"]
                        and details["menu_objective"] <= max(anchor_start.f, guide.f)
                    ):
                        if len(anchor_tried) >= 4096:
                            anchor_tried.clear()
                        anchor_tried.add(guide_key)
                    if extra is not None:
                        anchor_children += 1
                        if linked is None or extra.f > linked.f:
                            linked = extra
                            relink_phase = "anchor_component"
            if linked is None:
                linked = relinker(a, b, streams[5], deadline)
            if linked is not None:
                raw_global = linked.f > public_best.f
                if relink_phase == "component_recombine" and raw_global:
                    component_incumbents += 1
                if relink_phase == "anchor_component" and raw_global:
                    anchor_incumbents += 1
                publish(linked, "raw_" + relink_phase)
                descent(linked, streams[6], streams[4], deadline)
                # Java PR best state owns stream6, retained across calls.
                insert_elite(pool, linked)
                if (
                    relink_phase == "component_recombine"
                    and not raw_global
                    and linked.f > public_best.f
                ):
                    component_incumbents += 1
                if (
                    relink_phase == "anchor_component"
                    and not raw_global
                    and linked.f > public_best.f
                ):
                    anchor_incumbents += 1
                linked_is_record = raw_global or linked.f > public_best.f
                analyze(linked, relink_phase)
                if config.global_record_feedback and linked_is_record and linked.f > reference.f:
                    reference = linked.copy()
                    record_feedback_relink += 1
                pr_improvements += 1
        if trace_callback:
            trace_callback(iterations, state_digest((current, reference, internal_best), pool))
    result = MaxCoverResult(
        budget,
        "java_style_ails_pr",
        public_best.f,
        list(map(int, public_best.order[:budget])),
        coverage=public_best.coverage.copy(),
    )
    result.total_seconds = result.solve_seconds = perf_counter() - started
    result.metadata = {
        "iterations": iterations,
        "completed_iterated_descents": completed_descents,
        "pr_improvements": pr_improvements,
        "component_attempts": component_attempts,
        "component_children": component_children,
        "component_incumbents": component_incumbents,
        "component_exact_groups": component_exact_groups,
        "component_large_groups": component_large_groups,
        "component_subsets": component_subsets,
        "component_expanded_groups": component_expanded_groups,
        "exact_component_cap": config.exact_component_cap,
        "large_component_greedy": config.large_component_greedy,
        "anchored_component": config.anchored_component,
        "initial_starts_requested": config.initial_starts,
        "seed_capacity": config.seed_capacity,
        "initial_start_records": seed_records,
        "initial_best_index": seed_best_index,
        "initial_partial_constructors_discarded": seed_partial,
        "extra_seeding_seconds": seed_seconds,
        "seeded_pool_size": seeded_pool_size,
        "seeded_reference_objective": seeded_reference_objective,
        "seeded_reference_distances": seeded_distances,
        "faithful_local": config.faithful_local,
        "fallback_gains_initialized": True,
        "global_record_feedback": config.global_record_feedback,
        "record_feedback_current": record_feedback_current,
        "record_feedback_relink": record_feedback_relink,
        "anchor_attempts": anchor_attempts,
        "anchor_children": anchor_children,
        "anchor_incumbents": anchor_incumbents,
        "anchor_completed": anchor_completed,
        "anchor_cached_guides_skipped": anchor_cache_skips,
        "anchor_distance_sum": anchor_distance_sum,
        "anchor_exact_groups": anchor_exact_groups,
        "anchor_large_groups": anchor_large_groups,
        "component_greedy_groups": component_greedy_groups,
        "component_greedy_refinement_moves": component_greedy_refinement_moves,
        "component_greedy_memory_skips": component_greedy_memory_skips,
        "neutral_attempts": neutral_attempts,
        "neutral_moves": neutral_moves,
        "neutral_improvements": neutral_improvements,
        "internal_best_objective": internal_best.f,
        "omega": [x.omega for x in feedback],
        "elite_size": len(pool),
        "experimental": True,
        "protected_wrapper_incumbents": True,
        "guide_tournament": config.guide_tournament,
    }
    return result


def load_java_sparse(path):
    """Read Java's ordered bipartite snapshot without reinterpreting neighborhoods."""
    from pathlib import Path

    data = memoryview(Path(path).read_bytes())
    offset = 0

    def values(code, count=1):
        nonlocal offset
        result = struct.unpack_from(">" + code * count, data, offset)
        offset += struct.calcsize(">" + code * count)
        return result

    magic, n, m, r = values("i", 4)
    if magic != 0x53504D31 or n < 1 or m < 1:
        raise ValueError("Wrong sparse magic")
    coords = np.asarray(values("d", 2 * n)).reshape(n, 2)
    weights = np.asarray(values("q", m), dtype=np.int64)

    def row(limit):
        count = values("i")[0]
        if not 0 <= count <= limit:
            raise ValueError("Invalid sparse degree")
        result = np.asarray(values("i", count), dtype=np.int32)
        if len(result) and (
            result[0] < 0 or result[-1] >= limit or np.any(result[1:] <= result[:-1])
        ):
            raise ValueError("Invalid sparse row")
        return result

    rows = [row(m).tolist() for _ in range(n)]
    neighborhoods = []
    for _ in range(2):
        group = []
        for j in range(n):
            neighborhood = row(n)
            if j not in neighborhood:
                raise ValueError("Neighborhood must contain itself")
            group.append(neighborhood)
        neighborhoods.append(group)
    if offset != len(data):
        raise ValueError("Trailing sparse bytes")
    from .instance import build_instance_from_facility_map

    return (
        build_instance_from_facility_map(dict(enumerate(rows)), weights),
        coords,
        neighborhoods[0],
    )
