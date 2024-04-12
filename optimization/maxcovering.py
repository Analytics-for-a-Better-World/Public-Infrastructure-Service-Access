"""

Optimizing weighted maximal covering problems, J. Gromicho 2023.

We consider the optimization problem introduced in the paper below:

@Article{Church1974,
author="Church, Richard
and ReVelle, Charles",
title="The maximal covering location problem",
journal="Papers of the Regional Science Association",
year="1974",
month="Dec",
day="01",
volume="32",
number="1",
pages="101--118",
issn="1435-5957",
doi="10.1007/BF01942293",
url="https://doi.org/10.1007/BF01942293"
}

This module includes mathematical optimization solvers and heuristics.

Although this is now a very well-known problem and there are several
open source implementations
(example https://github.com/cyang-kth/maximum-coverage-location)
we implemented here the research conducted during the following
master theses:

An Optimization Tool for Facility Location in Developing Countries
Case study for Timor-Leste by Joyce Antonissen, U. Tilburg, 2020

Solving Large Maximum Covering Location Problems with a GRASP Heuristic
Case-study for stroke facility allocation in Vietnam
by Fleur Theulen, U. Tilburg, 2022

The mathematical optimization models are related to the work of Joyce, but
while she used a double index formulation for the budgeted version of the
problem, we use a much more scalable single index formulation.

This is implemented directly in gurobipy to allow very fast model building and
optimization with the gurobi solver and also in pyomo to allow optimization
with any solver supported by pyomo, including many open source solvers.

The greedy and LocalSearch heuristics are faster and more scalable
re-implementations of those by Fleur.

Also includes an heuristic from Church and ReVelle that is not part of Fleur's
thesis.

Is still lacks a number of heuristics from Fleur:
 - 'adaptive' version of LocalSearch that tries to avoid hopeless attempts.
 - GRASP
 - path relinking

Besides functions documented with docstring, this model also exports the
following:

verbose_gurobi_code : dict to description of the termination codes of gurobi
"""

from time import perf_counter as pc
import copy
import numpy as np
import gurobipy as gb
import pyomo.environ as pyo

# own modules
import optdata as od

verbose_gurobi_code = {gb.GRB.LOADED: 'loaded',
                       gb.GRB.OPTIMAL: 'optimal',
                       gb.GRB.INFEASIBLE: 'infeasible',
                       gb.GRB.INF_OR_UNBD: 'infeasible or unbounded',
                       gb.GRB.UNBOUNDED: 'unbounded',
                       gb.GRB.CUTOFF: 'cutoff',
                       gb.GRB.ITERATION_LIMIT: 'iteration limit',
                       gb.GRB.NODE_LIMIT: 'node limit',
                       gb.GRB.TIME_LIMIT: 'time limit',
                       gb.GRB.SOLUTION_LIMIT: 'solution limit',
                       gb.GRB.INTERRUPTED: 'interrupted',
                       gb.GRB.NUMERIC: 'numeric',
                       gb.GRB.SUBOPTIMAL: 'suboptimal',
                       gb.GRB.INPROGRESS: 'in progress',
                       gb.GRB.USER_OBJ_LIMIT: 'user objective limit'}


# Optimizing with Pyomo, see https://mobook.github.io/MO-book/intro.html
def GetPyomoSolver(solverName: str,
                   timeLimit: float = None,
                   mipGap: float = None) -> pyo.SolverFactory:
    """
    This function returns a Pyomo solver object based on the solver name.
    Sets the time limit and mip gap if provided.

    Parameters
    ----------
    solverName : str
        Name of the solver to be used.
    timeLimit : float, optional
        Time limit for the solver (default None) in seconds.
    mipGap : float, optional
        MIP gap for the solver. The default (None) uses the default specific
        to each solver.

    Returns
    -------
    pyo.SolverFactory
        Pyomo solver object.
    """
    solver = pyo.SolverFactory(solverName)
    if timeLimit:
        if 'cplex' in solverName:
            solver.options['timelimit'] = timeLimit
        elif 'cbc' in solverName:
            solver.options['sec'] = np.ceil(timeLimit)
        elif 'gurobi' in solverName:
            solver.options['TimeLimit'] = timeLimit
    if mipGap:
        if 'cplex' in solverName:
            solver.options['mipgap'] = mipGap
        elif 'cbc' in solverName:
            solver.options['allowableGap'] = mipGap
        elif 'gurobi' in solverName:
            solver.options['MipGap'] = mipGap
    return solver

def OptimizeWithPyomo(w: list, I: list,  # noqa: E741
                      J: list, IJ: dict, budget_list: list,
                      parsimonious: bool = True, maxTimeInSeconds: int = 5*60,
                      mipGap: float = 1e-8, trace: bool = False,
                      already_open: list = [], solver: str = 'cbc',
                      progress: callable = lambda iterable: iterable) \
                          -> dict[int, dict[str, any]]:
    """
    Instantiates and solves the weighted maximal covering problem with the
    solver specified for the data provided.

    Parameters
    ----------
    w : list
        List of sizes of each household.
    I : list
        List of indices of households.
    J : list
        List of indices if (potential) locations.
    IJ : dict
        Dictionary of households to the locations within reach.
    budget_list : list
        List of budgets to optimize for.
        These limit the number of facilities to select, in addition to those
        (if any) listed in already_open.
    parsimonious : bool, optional
        Whether to optimize minimize the number of open facilities needed to
        reach the optimal coverage (default is True).
    maxTimeInSeconds : int, optional
        Maximum time in seconds to run the optimization (default is 5*60).
    mipGap : float, optional
        MIP gap to use for optimization (default is 1e-8).
    trace : bool, optional
        Whether to trace the optimization (default is False).
    already_open : list, optional
        List of facilities that are already open (default is []).
    solver : str, optional
        Solver to use for optimization (default is 'cbc').
    progress : callable, optional
        Callable (function) to use for progress tracking (default is the
        identity).

    Returns
    -------
    result : dict[int, dict[str, any]]
        dict of dicts containing the optimization results at the outer level
        one entry per budget in budget_list and the inner level 'value',
        'solution','modeling','solving', 'termination','upper'
    """

    # ensure that all facilities already open are given to a variable
    J = sorted(set(J) | set(already_open))

    # ensure that only reachable customers are considered
    I = sorted(set(I) & set(IJ.keys()))  # noqa: E741
    
    result = dict()

    start = pc()

    M = pyo.ConcreteModel('max_coverage')

    M.I = pyo.Set(initialize=I)  # noqa: E741
    M.J = pyo.Set(initialize=J)

    M.budget = pyo.Param(mutable=True, default=0)

    M.X = pyo.Var(M.J, domain=pyo.Binary)
    M.Y = pyo.Var(M.I, domain=pyo.Binary)

    for j in already_open:
        M.X[j].fix(1)

    @M.Expression()
    def nof_open_facilities(M):
        return pyo.quicksum(M.X[j] for j in M.J)

    @M.Expression()
    def weighted_coverage(M):
        return pyo.quicksum(w[i]*M.Y[i] for i in M.I)

    coef_x = -1/(max(budget_list)+1) if parsimonious else 0

    @M.Objective(sense=pyo.maximize)
    def coverage(M):
        return M.weighted_coverage + M.nof_open_facilities * coef_x

    @M.Constraint(M.I)
    def serve_if_open(M, i):
        return M.Y[i] <= pyo.quicksum(M.X[j] for j in IJ[i])

    @M.Constraint()
    def in_the_budget(M):
        return M.nof_open_facilities <= M.budget

    solver = GetPyomoSolver(solver, maxTimeInSeconds, mipGap)

    for p in progress(budget_list):
        M.budget = p + len(already_open)
        modeling = pc()-start
        start = pc()
        solver_result = solver.solve(M, tee=trace)
        result[p] = dict(
            modeling=modeling,
            solving=pc()-start,
            value=0+M.weighted_coverage(),
            solution=[j for j,x in M.X.items() if x() > .5],
            termination=solver_result.solver.termination_condition,
            upper=0+solver_result.problem.upper_bound)
        start = pc()

    return result


def OptimizeWithGurobipy(w: list, I: list,  # noqa: E741
                         J: list, IJ: dict,
                         budget_list: list, parsimonious: bool = True,
                         maxTimeInSeconds: int = 5*60, mipGap: float = 1e-8,
                         trace: bool = False, already_open: list = [],
                         progress: callable = lambda iterable: iterable) \
                             -> dict[int, dict[str, any]]:
    """
    Instantiates the weighted maximal covering problem using gurobipy for the
    data provided and solves it with gurobi .

    Parameters
    ----------
    w : list
        List of sizes of each household.
    I : list
        List of indices of households.
    J : list
        List of indices if (potential) locations.
    IJ : dict
        Dictionary of households to the locations within reach.
    budget_list : list
        List of budgets to optimize for. These limit the number of facilities
        to select, in addition to those (if any) listed in already_open.
    parsimonious : bool, optional
        Whether to optimize minimize the number of open facilities needed to
        reach the optimal coverage (default is True).
    maxTimeInSeconds (float, optional):
        Max solve time. Defaults to 5*60.
        See https://www.gurobi.com/documentation/9.5/refman/timelimit.html
    mipGap ([type], optional):
        Max MIP gap. Defaults to 1e-8.
        See https://www.gurobi.com/documentation/9.5/refman/mipgap2.html
    trace (bool, optional):
        Show solve log. Defaults to False.
        See https://www.gurobi.com/documentation/9.5/refman/outputflag.html
    already_open : list, optional
        List of facilities that are already open (default is []).
    progress : callable, optional
        Callable (function) to use for progress tracking (default is the
        identity).

    Returns
    -------
    result : dict[int, dict[str, any]]
        dict of dicts containing the optimization results at the outer level
        one entry per budget in budget_list and the inner level 'value',
        'solution','modeling','solving','termination','upper'
    """

    # ensure that all facilities already open are given to a variable
    J = sorted(set(J) | set(already_open))

    # ensure that only reachable customers are considered
    I = sorted(set(I) & set(IJ.keys()))  # noqa: E741

    result = dict()
    start = pc()

    M = gb.Model('max_coverage')
    M.ModelSense = gb.GRB.MAXIMIZE

    M.Params.OutputFlag = trace
    M.Params.MIPGap = mipGap
    M.Params.TimeLimit = maxTimeInSeconds

    if parsimonious:
        X = M.addVars(J, obj=-1/(max(budget_list)+1), vtype=gb.GRB.BINARY)
    else:
        X = M.addVars(J, vtype=gb.GRB.BINARY)
    Y = M.addVars(I, obj=w[I], vtype=gb.GRB.BINARY)

    for j in already_open:
        X[j].lb = X[j].ub = 1

    M.addConstrs((Y[i] <= (gb.quicksum(X[j] for j in IJ[i]))) for i in I)
    budget = M.addLConstr(X.sum() >= 0)

    for p in progress(budget_list):
        M.remove(budget)
        budget = M.addLConstr(X.sum() <= p + len(already_open))
        modeling = pc()-start
        start = pc()
        M.optimize()
        result[p] = dict(modeling=modeling,
                         solving=pc()-start,
                         value=0+M.objVal,
                         solution=[j for j in J if X[j].x > .5],
                         termination=verbose_gurobi_code[M.status],
                         upper=0+M.ObjBound)
        start = pc()

    return result


# Heuristics
def Greedy(w: np.ndarray, IJ: dict, JI: dict, nof_facilities: np.uint,
           budget_list: list, progress: callable = lambda iterable: iterable) \
               -> dict[int, dict[str, any]]:
    """
    Fleur's Greedy algorithm for the weighted budgeted maximal covering
    facility location problem. Note that this is the GA (Greedy Addition)
    algorithm described by Church and ReVelle

    Parameters
    ----------
    w : np.ndarray
        Weight matrix.
    IJ : dict
        Dictionary of households to facilities.
    JI : dict
        Dictionary of facilities to households.
    nof_facilities : np.uint
        Number of facilities.
    budget_list : list
        List of budgets.
    progress : callable, optional
        Callable (function) to use for progress tracking (default is the
        identity).
    Returns
    -------
    result : dict[int, dict[str, any]]
        dict of dicts containing the 'value', 'solution', 'increments',
        'solving' and 'coverage' for each budget.
    """

    result = dict()
    start = pc()
    greedy_selected, greedy_added = [], []

    nof_households = len(w)
    coverage = np.zeros(nof_households, dtype=np.uint16)
    greedy_val = -np.ones(nof_facilities, dtype=int)

    may_change = np.array(list(JI.keys()))
    i = prev = -1
    for p in progress(sorted(budget_list)):
        for i in range(prev+1, min(p, nof_facilities)):
            greedy_val[may_change] = [w[JI[j][coverage[JI[j]] == 0]].sum()
                                      for j in may_change]
            select = np.argmax(greedy_val)
            if greedy_val[select] <= 0:
                break

            coverage[JI[select]] += 1
            greedy_selected.append(select)
            greedy_added.append(greedy_val[select])

            # Greedy only changes if coverage overlap with selected facility
            may_change = np.unique(np.concatenate([IJ[i] for i in JI[select]]))
        prev = i

        result[p] = dict(solving=pc()-start,
                         value=sum(greedy_added),
                         solution=greedy_selected.copy(),
                         increments=greedy_added.copy(),
                         coverage=coverage.copy())
        start = pc()

    return result


def LocalSearch(solution: list, coverage: np.ndarray, objective: int, J: list,
                JI: dict, household: list) \
                    -> tuple[list, list, int, list, list, float]:
    """
    This function performs a local search algorithm to optimize coverage by
    attempting to swap one open for one closed facility for as long as that
    increases coverage.

    Parameters
    ----------
    solution : list
        The initial solution to be optimized.
    coverage : np.ndarray
        The population coverage of the initial solution.
    objective : int
        The objective value of the initial solution.
    J : list
        The list of all possible facilities.
    JI : dict
        Dictionary of facilities to households.
    household : list
        The headcount of the households.
    Returns
    -------
    sol : list
        The optimized solution.
    obj : int
        The objective value of the optimized solution.
    cov : list
        The coverage of the optimized solution.
    objectives : list
        The list of objective values of the improved solutions found.
    times : list
        The list of times taken to reach each objective value.
    final_time : float
        The total time taken to reach the optimized solution.
    """

    sol = copy.copy(solution)
    cov = copy.copy(coverage)
    obj = objective
    times, objectives = [0], [obj]

    candidates = np.setdiff1d(J, sol, assume_unique=True)

    start = pc()
    while True:
        modified = False
        for i in range(len(sol)):
            rsi = JI[sol[i]]
            hh_out = household[rsi[cov[rsi] == 1]].sum()
            cov[rsi] -= 1
            for j in range(len(candidates)):
                rcj = JI[candidates[j]]
                hh_in = household[rcj[cov[rcj] == 0]].sum()
                if hh_in > hh_out:
                    cov[rcj] += 1
                    obj += hh_in - hh_out
                    hh_out = hh_in
                    sol[i], candidates[j] = candidates[j], sol[i]
                    modified = True
                    rsi = JI[sol[i]]
                    cov[rsi] -= 1
                    times.append(pc() - start)
                    objectives.append(obj)
            cov[rsi] += 1
        if not modified:
            break

    final_time = pc() - start
    times.append(final_time), objectives.append(obj)

    return sol, obj, cov, objectives, times, final_time


def GreedyLS(w: np.ndarray,
             IJ: dict,
             JI: dict,
             nof_facilities: np.uint,
             budget_list: list,
             progress: callable = lambda iterable: iterable
             ) -> dict[int, dict[str, any]]:

    """
    A first implementation of the GAS (Greedy Addition with Substitution)
    algorithm described by Church and ReVelle.
    Note that the performance of this algorithm may still be improved.

    Parameters
    ----------
    w : np.ndarray
        Weight matrix.
    IJ : dict
        Dictionary of households to facilities.
    JI : dict
        Dictionary of facilities to households.
    nof_facilities : np.uint
        Number of facilities.
    budget_list : list
        List of budgets.
    progress : callable, optional
        Callable (function) to use for progress tracking (default is the
        identity).
    Returns
    -------
    result : dict[int, dict[str, any]]
        dict of dicts containing the 'value', 'solution', 'increments',
        'solving' and 'coverage' for each budget.
    """

    def GetCoverage(facilities, nof_homes, JI):
        # 'u1' is the smallest unsigned datatype for int: takes 1 byte
        coverage = np.zeros(nof_homes, dtype=np.dtype('u1'))
        for j in facilities:
            coverage[JI[j]] += 1
        return coverage

    def GetSolutionValue(solution, household, JI):
        if solution:
            return household[od.all_in([JI[s] for s in solution])].sum()
        return 0

    result = dict()
    start = pc()
    solution, greedy_added = [], []

    nof_households = len(w)
    coverage = np.zeros(nof_households, dtype=np.uint16)
    greedy_val = -np.ones(nof_facilities, dtype=int)

    J = list(JI.keys())

    i = prev = -1
    for p in progress(sorted(budget_list)):
        may_change = np.array(J)
        for i in range(prev+1, min(p, nof_facilities)):
            greedy_val[may_change] = [w[JI[j][coverage[JI[j]] == 0]].sum()
                                      for j in may_change]
            select = np.argmax(greedy_val)
            if greedy_val[select] <= 0:
                break

            coverage[JI[select]] += 1
            solution.append(select)
            greedy_added.append(greedy_val[select])

            # Greedy only changes if coverage overlap with selected facility
            may_change = np.unique(np.concatenate([IJ[i] for i in JI[select]]))
        prev = i

        solution, objective, coverage, *_ = \
            LocalSearch(solution, coverage,
                        GetSolutionValue(solution, w, JI), J, JI, w)

        result[p] = dict(solving=pc()-start, value=objective,
                         solution=solution, coverage=coverage)
        start = pc()

    return result
