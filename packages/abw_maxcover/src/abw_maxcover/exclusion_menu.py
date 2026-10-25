"""Exact fixed-pool coverage: DP masks/counts encode EXCLUDED sites. Not a whole-instance solver."""

from time import perf_counter


class Incomplete(Exception):
    pass


def decode_selected_sites(sites, excluded):
    if type(excluded) is not int or excluded < 0 or excluded >> len(sites):
        raise ValueError("invalid exclusion mask")
    return [j for bit, j in enumerate(sites) if not excluded >> bit & 1]


def solve_exclusion_menu(
    n, clauses, deadline, *, budget=None, width_cap=8, entry_cap=200_000, operation_cap=2_000_000
):
    if type(n) is not int or not 0 <= n <= 160:
        raise ValueError("n must be integer within0..160")
    if budget is not None and (type(budget) is not int or budget < 0):
        raise ValueError("budget must be nonnegative integer")
    for value in [width_cap, entry_cap, operation_cap]:
        if type(value) is not int or value < 0:
            raise ValueError("caps must be nonnegative integers")
    budget = n if budget is None else min(n, budget)
    stats = dict(completed=False, reason=None, width=0, estimated_entries=0, operations=0)

    def clock():
        if perf_counter() >= deadline:
            raise Incomplete("deadline")

    def tick():
        stats["operations"] += 1
        if stats["operations"] > operation_cap:
            raise Incomplete("operation_cap")
        if stats["operations"] % 256 == 0:
            clock()

    def combine(left, right):
        merged = {}
        for ca, (va, ma) in left.items():
            for cb, (vb, mb) in right.items():
                tick()
                cost = ca + cb
                if cost > budget:
                    continue
                if not (not ma & mb):
                    raise RuntimeError("eliminated variable counted twice")
                value = va + vb
                if cost not in merged or value > merged[cost][0]:
                    merged[cost] = (value, ma | mb)
        return merged

    try:
        clock()
        weights = {}
        adj = [0] * n
        for mask, weight in clauses:
            clock()
            if (
                type(mask) is not int
                or mask < 0
                or mask >> n
                or type(weight) is not int
                or weight < 0
            ):
                raise ValueError("invalid OR clause")
            if mask and weight:
                weights[mask] = weights.get(mask, 0) + weight
        for mask in weights:
            clock()
            for v in range(n):
                if mask >> v & 1:
                    adj[v] |= mask & ~(1 << v)
        active = (1 << n) - 1
        bags = []
        while active:
            clock()
            choices = []
            for v in range(n):
                if not active >> v & 1:
                    continue
                clock()
                neighbors = adj[v] & active
                vertices = [u for u in range(n) if neighbors >> u & 1]
                missing = sum((neighbors & ~adj[u] & ~(1 << u)).bit_count() for u in vertices) // 2
                choices.append((missing, len(vertices), v, neighbors, vertices))
            _, width, v, neighbors, vertices = min(choices)
            stats["width"] = max(stats["width"], width)
            if width > width_cap:
                raise Incomplete("width_cap")
            bags.append((v, tuple(vertices)))
            for u in vertices:
                adj[u] |= neighbors & ~(1 << u)
            active &= ~(1 << v)
        rank = {v: i for i, (v, _) in enumerate(bags)}
        local = {v: [] for v in rank}
        buckets = {v: [] for v in rank}
        sizes = {v: [] for v in rank}
        for mask, weight in weights.items():
            v = min((v for v in rank if mask >> v & 1), key=rank.get)
            local[v].append((mask, weight))
        for v, sep in bags:
            count = 1 + sum(sizes[v])
            stats["estimated_entries"] += (1 << len(sep)) * (min(count, budget) + 1)
            if stats["estimated_entries"] > entry_cap:
                raise Incomplete("entry_cap")
            if sep:
                sizes[min(sep, key=rank.get)].append(count)
        roots = []
        for v, sep in bags:
            clock()
            bag = (v,) + sep
            positions = {u: i for i, u in enumerate(bag)}
            bag_mask = sum(1 << u for u in bag)
            if not (all((not mask & ~bag_mask for mask, weight in local[v]))):
                raise RuntimeError("clause escaped elimination bag")
            local_masks = [
                (sum(1 << positions[u] for u in bag if mask >> u & 1), weight)
                for mask, weight in local[v]
            ]
            incoming = buckets[v]
            buckets[v] = []
            if not (all((set(scope).issubset(positions) for scope, table in incoming))):
                raise RuntimeError("message escaped elimination bag")
            projections = [tuple(positions[u] for u in scope) for scope, table in incoming]
            table = [{} for _ in range(1 << len(sep))]
            for assignment in range(1 << len(bag)):
                clock()
                chosen = assignment & 1
                value = sum(weight for mask, weight in local_masks if (mask & assignment) != mask)
                current = {chosen: (value, (1 << v) if chosen else 0)} if chosen <= budget else {}
                for (_, child_table), projection in zip(incoming, projections, strict=True):
                    key = sum(
                        ((assignment >> position) & 1) << bit
                        for bit, position in enumerate(projection)
                    )
                    current = combine(current, child_table[key])
                target = table[assignment >> 1]
                for cost, pair in current.items():
                    if cost not in target or pair[0] > target[cost][0]:
                        target[cost] = pair
            if sep:
                buckets[min(sep, key=rank.get)].append((sep, table))
            else:
                roots.append(table[0])
        final = {0: (0, 0)}
        for root in roots:
            final = combine(final, root)
        clock()
        for cost, (_value, mask) in final.items():
            if not (mask.bit_count() == cost):
                raise RuntimeError("cardinality traceback mismatch")
        stats.update(completed=True, reason="complete")
        return [(cost, value, mask) for cost, (value, mask) in sorted(final.items())], stats
    except Incomplete as error:
        stats["reason"] = str(error)
        return None, stats


def treewidth_exclusion_menu(ins, sites, residual, budget, deadline, **kwargs):
    patterns = {}
    for bit, j in enumerate(sites):
        if perf_counter() >= deadline:
            return None, dict(completed=False, reason="deadline")
        for index, demand in enumerate(ins.demand_of(j)):
            if index % 256 == 0 and perf_counter() >= deadline:
                return None, dict(completed=False, reason="deadline")
            if residual[demand]:
                patterns[int(demand)] = patterns.get(int(demand), 0) | (1 << bit)
    clauses = [(mask, int(ins.weights[demand])) for demand, mask in patterns.items()]
    return solve_exclusion_menu(len(sites), clauses, deadline, budget=budget, **kwargs)
