import itertools
import random
from time import perf_counter

import pytest

from abw_maxcover.cem_beam import BeamSession, EliteModel, solve_cem_beam
from abw_maxcover.regional_repair import RepairMemo, repair_region


def score(facilities, weights, selected):
    cov = 0
    for j in selected:
        cov |= facilities[j]
    return sum(w for d, w in enumerate(weights) if cov >> d & 1)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("learning", [False, True])
def test_resumed_bounds_against_independent_enumeration(seed, learning):
    rng = random.Random(seed)
    n = 7
    p = 3
    f = [rng.getrandbits(8) for _ in range(n)]
    w = [rng.randrange(10) for _ in range(8)]
    w[0] += 2**54
    optimum = max(score(f, w, ids) for ids in itertools.combinations(range(n), p))
    session = BeamSession(
        f,
        w,
        p,
        list(range(p)),
        beam=2,
        agenda_limit=None,
        use_learning=learning,
        rollouts=1,
        polish_moves=1,
    )
    old = score(f, w, range(p))
    for _call in range(200):
        result = session.run(perf_counter() + 10, max_expansions=1)
        assert old <= result["objective"] <= optimum <= result["upper_bound"]
        assert score(f, w, result["selected"]) == result["objective"]
        old = result["objective"]
        if result["stats"]["reason"] == "exhausted":
            break
    else:
        pytest.fail("resumed enumeration did not finish")
    assert result["certified"] and old == optimum
    result["selected"].clear()
    assert session.run(perf_counter() + 1)["selected"]


@pytest.mark.parametrize("seed", range(12))
def test_memory_cap_keeps_valid_lost_bound(seed):
    rng = random.Random(seed)
    f = [rng.getrandbits(10) for _ in range(9)]
    w = list(range(10))
    p = 4
    optimum = max(score(f, w, ids) for ids in itertools.combinations(range(9), p))
    result = solve_cem_beam(
        f, w, p, list(range(p)), perf_counter() + 5, beam=1, agenda_limit=1, rollout_nodes=0
    )
    assert result["objective"] <= optimum <= result["upper_bound"]
    assert not result["certified"] or result["objective"] == optimum


def test_active_batch_is_retained_at_zero_expansions():
    s = BeamSession([1, 2], [1, 5], 1, [0], beam=1, agenda_limit=None, rollout_nodes=0)
    first = s.run(perf_counter() + 5, max_expansions=0)
    assert first["objective"] == 1 and first["upper_bound"] >= 5
    final = s.run(perf_counter() + 5, max_expansions=20)
    assert final["objective"] == final["upper_bound"] == 5


def test_archive_change_controls_fitting():
    model = EliteModel(4, 2, [3], lambda m: m.bit_count())
    old = [q.copy() for q in model.q]
    updates = model.updates
    for _ in range(10):
        model.fit()
    assert model.updates == updates and model.fit_skips == 10 and model.q == old
    model.observe(12, 2)
    model.fit()
    assert model.updates == updates + 1
    model.observe(12, 2)
    model.fit()
    assert model.updates == updates + 1


@pytest.mark.parametrize("deadline", [True, float("nan"), float("inf")])
def test_session_rejects_invalid_deadline(deadline):
    with pytest.raises(ValueError):
        BeamSession([1], [1], 1, [0]).run(deadline)


def test_regional_memo_expiry_invalidation_and_immutable_results():
    f = [1, 2, 4, 3, 6]
    w = [1, 2, 3]
    memo = RepairMemo(2)
    kwargs = dict(
        facilities=f,
        weights=w,
        incumbent=[0, 1],
        elites=[[1, 2]],
        free=[0],
        extra_cap=2,
        beam=None,
        memo=memo,
        cache_mode="both",
    )
    expired = repair_region(deadline=0, **kwargs)
    assert expired["objective"] == 3 and memo.prepared.cache_info().currsize == 0
    first = repair_region(deadline=perf_counter() + 5, **kwargs)
    first["selected"].clear()
    second = repair_region(deadline=perf_counter() + 5, **kwargs)
    assert score(f, w, second["selected"]) == second["objective"] >= 3
    assert memo.hits["prepared"] >= 1
    previous = memo.hits["invalidations"]
    repair_region(deadline=perf_counter() + 5, **dict(kwargs, weights=[1, 2, 30]))
    assert memo.hits["invalidations"] == previous + 1
