"""Transient complete-one-swap bonus matrix with exact integer updates."""

import numpy as np


def reverse_edges(ins, demands):
    starts = ins.ij_indptr[demands]
    counts = ins.ij_indptr[demands + 1] - starts
    ends = np.cumsum(counts, dtype=np.intp)
    offsets = np.repeat(starts - (ends - counts), counts) + np.arange(
        int(ends[-1]) if len(ends) else 0
    )
    return ins.ij_indices[offsets], counts


class SwapMatrix:
    def __init__(self, s):
        ins = s.ins
        self.owner = np.full(ins.n_demand, -1, dtype=np.int32)
        for pos, j in enumerate(s.order[: s.p]):
            row = ins.demand_of(int(j))
            exclusive = row[s.coverage[row] == 1]
            self.owner[exclusive] = pos
        demands = np.flatnonzero(self.owner >= 0)
        facilities, counts = reverse_edges(ins, demands)
        slots = np.repeat(self.owner[demands], counts)
        values = np.repeat(ins.weights[demands], counts)
        if s.bin_exact:
            self.bonus = (
                np.bincount(
                    slots * ins.n_facilities + facilities,
                    weights=values,
                    minlength=s.p * ins.n_facilities,
                )
                .astype(np.int64)
                .reshape(s.p, ins.n_facilities)
            )
            self.loss = np.bincount(
                self.owner[demands], weights=ins.weights[demands], minlength=s.p
            ).astype(np.int64)
        else:
            self.bonus = np.zeros((s.p, ins.n_facilities), dtype=np.int64)
            self.loss = np.zeros(s.p, dtype=np.int64)
            np.add.at(self.bonus, (slots, facilities), values)
            np.add.at(self.loss, self.owner[demands], ins.weights[demands])

    def exchange(self, s, out, entry):
        ins = s.ins
        affected = np.union1d(ins.demand_of(out), ins.demand_of(entry))
        old = self.owner[affected].copy()
        s.exchange(out, entry)
        new = np.full(len(affected), -1, dtype=np.int32)
        unique = np.flatnonzero(s.coverage[affected] == 1)
        facilities, counts = reverse_edges(ins, affected[unique])
        selected = s.selected[facilities]
        indexes = np.repeat(unique, counts)[selected]
        new[indexes] = s.positions[facilities[selected]]
        changed = old != new
        demands = affected[changed]
        old = old[changed]
        fresh = new[changed]
        weights = ins.weights[demands]
        prior = old >= 0
        after = fresh >= 0
        np.add.at(self.loss, old[prior], -weights[prior])
        np.add.at(self.loss, fresh[after], weights[after])
        facilities, counts = reverse_edges(ins, demands)
        old_slots = np.repeat(old, counts)
        new_slots = np.repeat(fresh, counts)
        values = np.repeat(weights, counts)
        prior = old_slots >= 0
        after = new_slots >= 0
        np.add.at(self.bonus, (old_slots[prior], facilities[prior]), -values[prior])
        np.add.at(self.bonus, (new_slots[after], facilities[after]), values[after])
        self.owner[affected] = new
