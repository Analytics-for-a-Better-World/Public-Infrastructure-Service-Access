"""Verbatim source-directed Java local tie/LCG kernel; outer search is hybrid."""

from time import perf_counter

import numpy as np


def local_descent(s, shuffle_rng, tie_rng, deadline):
    for _ in range(s.p):
        if perf_counter() >= deadline:
            return False
        s.swap_positions(shuffle_rng.integer(s.p), shuffle_rng.integer(s.p))
    cache = {} if s.p * s.ins.n_facilities <= 2_000_000 else None
    zero_bonus = np.zeros(s.ins.n_facilities, dtype=np.int64)
    baseline = None
    improved = True
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
                bonus, loss, active = cached[1], cached[2], cached[3]
            else:
                loss = int(s.ins.weights[exclusive].sum())
                bonus = zero_bonus
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
                    cache[out] = (exclusive, bonus, loss, active)
            tail = s.order[s.p :]
            if not len(tail):
                continue
            if baseline is None:
                base = s.gains[tail]
                base_events = np.flatnonzero(base == np.maximum.accumulate(np.maximum(base, 0)))
                baseline = base, base_events
            base, base_events = baseline
            positions = s.positions[active]
            overlap = positions[positions >= s.p] - s.p
            frontier = np.union1d(base_events, overlap)
            values = base[frontier] + bonus[tail[frontier]]
            maxima = np.maximum.accumulate(np.maximum(values, 0))
            mask = values == maxima
            events, event_values = frontier[mask], values[mask]
            best_gain = int(event_values[-1]) if len(event_values) else 0
            index = 0
            if best_gain <= loss:
                ties = (
                    int(np.count_nonzero(event_values == np.r_[0, event_values[:-1]]))
                    if len(event_values)
                    else 0
                )
                tie_rng.advance(ties)
            else:
                best_gain = 0
                for offset, raw_value in zip(events, event_values, strict=True):
                    if perf_counter() >= deadline:
                        return False
                    value = int(raw_value)
                    if value > best_gain or (value == best_gain and tie_rng.boolean()):
                        best_gain, index = value, s.p + int(offset)
            if best_gain > loss:
                entry = int(s.order[index])
                s.remove(out)
                s.add(entry)
                s.swap_positions(position, index)
                if cache is not None:
                    cache.pop(out, None)
                baseline = None
                improved = True
    return True
