"""
Weighted Maximal Covering Location Problem
J. Gromicho, 2026

Implementation of exact and heuristic methods for the Weighted
Maximal Covering Location Problem (MCLP).

Original problem:

Church, R. and ReVelle, C. (1974).
The maximal covering location problem.
Papers of the Regional Science Association, 32(1), 101-118.
doi: 10.1007/BF01942293

Related theses:

Joyce Antonissen (2020)
An Optimization Tool for Facility Location in Developing Countries
Case study for Timor Leste
https://github.com/Analytics-for-a-Better-World/Public-Infrastructure-Service-Access/blob/main/publications/Joyce_Optimisation_Model.pdf

Fleur Theulen (2022)
Solving Large Maximum Covering Location Problems with a GRASP Heuristic
Case study for stroke facility allocation in Vietnam
https://drive.google.com/file/d/14jijFt_QJPSOwHG05rgv847Tg1qHErQD/view

Contents
--------

Exact models:
- gurobipy implementation for fast construction and solution with Gurobi
- Pyomo implementation for solver agnostic optimization

Heuristics:
- Greedy
- LocalSearch
- Church and ReVelle inspired heuristic

Not yet implemented:
- Adaptive LocalSearch
- GRASP
- Path relinking

Exports:
- gurobicode: dict mapping Gurobi termination codes to descriptions
"""

from time import perf_counter as pc
import numpy as np
import pandas as pd
import gurobipy as gb
import pyomo.environ as pyo


gurobicode = {
    gb.GRB.LOADED: 'loaded',
    gb.GRB.OPTIMAL: 'optimal',
    gb.GRB.INFEASIBLE: 'infeasible',
    gb.GRB.INF_OR_UNBD: 'inf_or_unbd',
    gb.GRB.UNBOUNDED: 'unbounded',
    gb.GRB.CUTOFF: 'cutoff',
    gb.GRB.ITERATION_LIMIT: 'iteration_limit',
    gb.GRB.NODE_LIMIT: 'node_limit',
    gb.GRB.TIME_LIMIT: 'time_limit',
    gb.GRB.SOLUTION_LIMIT: 'solution_limit',
    gb.GRB.INTERRUPTED: 'interrupted',
    gb.GRB.NUMERIC: 'numeric',
    gb.GRB.SUBOPTIMAL: 'suboptimal',
    gb.GRB.INPROGRESS: 'inprogress',
    gb.GRB.USER_OBJ_LIMIT: 'user_obj_limit',
}


def all_in(list_of_lists: list[list]) -> np.ndarray:
    """
    Return the sorted unique values appearing in a list of lists.

    Parameters
    ----------
    list_of_lists : list[list]
        Nested list containing values to be flattened and deduplicated.

    Returns
    -------
    np.ndarray
        Sorted array of unique values. Returns an empty array when the input
        is empty or all inner lists are empty.
    """
    non_empty = [np.asarray(values) for values in list_of_lists if len(values)]
    if not non_empty:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(non_empty))


def CreateIndexMapping(
    all_facs: dict,
    household: list,
    covered: set = set(),
) -> tuple[np.array, np.array, dict, dict]:
    """
    Build index mappings between households and reachable facilities.

    Parameters
    ----------
    all_facs : dict
        Dictionary mapping each facility index to the household indices in its
        catchment area.
    household : list
        Household data. Only the length is used here.
    covered : set, optional
        Set of household indices that are already covered and should therefore
        be excluded from the returned mappings.

    Returns
    -------
    tuple[np.array, np.array, dict, dict]
        A tuple ``(I, J, IJ, JI)`` where:

        - ``I`` is the array of uncovered households that can still be served
        - ``J`` is the array of facilities that can serve at least one
          uncovered household
        - ``IJ`` maps each uncovered household to the facilities that can
          cover it
        - ``JI`` maps each facility to the uncovered households it can cover
    """
    n_households = len(household)
    covered_array = np.asarray(sorted(covered), dtype=int)
    not_covered = np.setdiff1d(
        np.arange(n_households, dtype=int),
        covered_array,
        assume_unique=True,
    )

    j_to_i: dict[int, np.ndarray] = {}
    for j, households_in_range in all_facs.items():
        remaining = np.setdiff1d(
            np.asarray(households_in_range),
            covered_array,
            assume_unique=True,
        )
        if remaining.size > 0:
            j_to_i[j] = remaining

    i_to_j: dict[int, list[int]] = {int(i): [] for i in not_covered}
    for j, households_in_range in j_to_i.items():
        for i in households_in_range:
            if int(i) in i_to_j:
                i_to_j[int(i)].append(int(j))

    ij = {
        i: np.unique(np.asarray(reachable_facs, dtype=int))
        for i, reachable_facs in i_to_j.items()
        if reachable_facs
    }

    if ij:
        I = np.unique(np.fromiter(ij.keys(), dtype=int))
        J = np.unique(np.concatenate(list(ij.values())))
    else:
        I = np.array([], dtype=int)
        J = np.array([], dtype=int)

    return I, J, ij, j_to_i


def CheckSanityIndexMapping(
    I: list,
    J: list,
    IJ: dict,
    JI: dict,
    w: list,
) -> bool:
    """
    Check consistency of household and facility index mappings.

    Parameters
    ----------
    I : list
        Household indices.
    J : list
        Facility indices.
    IJ : dict
        Mapping from each household to facilities that can cover it.
    JI : dict
        Mapping from each facility to households in its catchment area.
    w : list
        Household weights.

    Returns
    -------
    bool
        ``True`` if all checks pass. Raises ``AssertionError`` otherwise.
    """
    I_set = set(I)
    J_set = set(J)

    ij_values = all_in(list(IJ.values()))
    ji_values = all_in(list(JI.values()))

    assert I_set.issuperset(set(IJ.keys())), 'household out of bounds'
    assert J_set.issuperset(set(JI.keys())), 'potential facility out of bounds'
    assert J_set.issuperset(set(ij_values)), 'potential facility out of bounds'
    assert I_set.issuperset(set(ji_values)), 'household out of bounds'
    assert set(ji_values).issubset(set(range(len(w)))), 'household weight out of bounds'

    return True


def GetPyomoSolver(
    solverName: str,
    timeLimit: float = None,
    mipGap: float = None,
) -> pyo.SolverFactory:
    """
    Create and configure a Pyomo solver instance.

    Parameters
    ----------
    solverName : str
        Name of the solver to be used.
    timeLimit : float, optional
        Solve time limit in seconds.
    mipGap : float, optional
        Relative MIP gap.

    Returns
    -------
    pyo.SolverFactory
        Configured Pyomo solver object.
    """
    solver_path = r'D:\joaquimg\Dropbox\Python\solvers\new cbc master\bin'

    if solverName == 'cbc':
        solver = pyo.SolverFactory(solverName, executable=solver_path + r'\cbc.exe')
        solver.options['threads'] = 8
    elif solverName == 'cplex':
        solver = pyo.SolverFactory('cplex_direct')
    elif solverName == 'gurobi':
        solver = pyo.SolverFactory('gurobi_direct')
    elif solverName == 'glpk':
        solver = pyo.SolverFactory(solverName, executable=solver_path + r'\glpsol.exe')
    else:
        solver = pyo.SolverFactory(solverName)

    if timeLimit is not None:
        if solverName == 'cplex':
            solver.options['timelimit'] = timeLimit
        elif solverName == 'cbc':
            solver.options['sec'] = int(np.ceil(timeLimit))
        elif solverName == 'gurobi':
            solver.options['TimeLimit'] = timeLimit

    if mipGap is not None:
        if solverName == 'cplex':
            solver.options['mipgap'] = mipGap
        elif solverName == 'cbc':
            solver.options['allowableGap'] = mipGap
        elif solverName == 'gurobi':
            solver.options['MipGap'] = mipGap

    return solver


def PyomoOptimize(
    w: list,
    I: list,
    J: list,
    IJ: dict,
    budget_list: list,
    parsimonious: bool = True,
    maxTimeInSeconds: int = 5 * 60,
    mipGap: float = 1e-8,
    trace: bool = False,
    already_open: list = [],
    solver: str = 'cbc',
    progress: callable = lambda iterable: iterable,
) -> pd.DataFrame:
    """
    Solve the weighted maximal covering problem with Pyomo.

    Parameters
    ----------
    w : list
        Household weights.
    I : list
        Household indices.
    J : list
        Facility indices.
    IJ : dict
        Mapping from each household to facilities that can cover it.
    budget_list : list
        Budgets to evaluate.
    parsimonious : bool, optional
        If ``True``, break ties by preferring fewer open facilities.
    maxTimeInSeconds : int, optional
        Maximum solve time per optimization run.
    mipGap : float, optional
        Relative MIP gap passed to the solver.
    trace : bool, optional
        If ``True``, display the solver log.
    already_open : list, optional
        Facilities forced open.
    solver : str, optional
        Solver name.
    progress : callable, optional
        Wrapper for progress reporting, for example ``tqdm``.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by budget with columns:
        ``value``, ``solution``, ``modeling``, ``solving``,
        ``termination``, and ``upper``.
    """
    result = pd.DataFrame(
        index=budget_list,
        columns=['value', 'solution', 'modeling', 'solving', 'termination', 'upper'],
    )

    w_array = np.asarray(w)
    budget_max = max(budget_list) if budget_list else 0

    start = pc()

    M = pyo.ConcreteModel('max_coverage')

    M.I = pyo.Set(initialize=I)
    M.J = pyo.Set(initialize=J)

    M.budget = pyo.Param(mutable=True, default=0)

    M.X = pyo.Var(M.J, domain=pyo.Binary)
    M.Y = pyo.Var(M.I, domain=pyo.Binary)

    for j in already_open:
        if j in J:
            M.X[j].fix(1)

    @M.Expression()
    def nof_open_facilities(M) -> pyo.Expression:
        return pyo.quicksum(M.X[j] for j in M.J)

    @M.Expression()
    def weighted_coverage(M) -> pyo.Expression:
        return pyo.quicksum(float(w_array[i]) * M.Y[i] for i in M.I)

    coef_x = -1 / (budget_max + 1) if parsimonious else 0.0

    @M.Objective(sense=pyo.maximize)
    def coverage(M) -> pyo.Objective:
        return M.weighted_coverage + coef_x * M.nof_open_facilities

    @M.Constraint(M.I)
    def serve_if_open(M, i) -> pyo.Constraint:
        return M.Y[i] <= pyo.quicksum(M.X[j] for j in IJ[i])

    @M.Constraint()
    def in_the_budget(M) -> pyo.Constraint:
        return M.nof_open_facilities <= M.budget

    solver_instance = GetPyomoSolver(solver, maxTimeInSeconds, mipGap)

    for p in progress(budget_list):
        M.budget.set_value(p)

        result.at[p, 'modeling'] = pc() - start
        start = pc()

        solver_result = solver_instance.solve(M, tee=trace)

        result.at[p, 'solving'] = pc() - start
        result.at[p, 'value'] = int(np.ceil(pyo.value(M.weighted_coverage) - np.finfo(np.float16).eps))
        result.at[p, 'solution'] = [j for j in J if pyo.value(M.X[j]) >= 0.5]
        result.at[p, 'termination'] = solver_result.solver.termination_condition

        lower_bound = getattr(solver_result.problem, 'lower_bound', None)
        upper_bound = getattr(solver_result.problem, 'upper_bound', None)
        bounds = []
        if lower_bound is not None:
            bounds.append(abs(int(np.round(lower_bound + np.finfo(np.float16).eps))))
        if upper_bound is not None:
            bounds.append(abs(int(np.round(upper_bound + np.finfo(np.float16).eps))))
        result.at[p, 'upper'] = max(bounds) if bounds else None

        start = pc()

    return result


def make_pyomo_optimizer_using(this_solver: str) -> callable:
    """
    Create a closure that solves instances with a fixed Pyomo solver.

    Parameters
    ----------
    this_solver : str
        Solver name.

    Returns
    -------
    callable
        Function with the same call pattern as ``PyomoOptimize``, except that
        the solver is fixed to ``this_solver``.
    """
    def optimizer(
        w,
        I,
        J,
        IJ,
        budget_list,
        parsimonious=True,
        maxTimeInSeconds=5 * 60,
        mipGap=1e-8,
        trace=False,
        already_open=[],
    ):
        """
        Solve a weighted maximal covering instance with a predefined solver.

        Parameters
        ----------
        w : list
            Household weights.
        I : list
            Household indices.
        J : list
            Facility indices.
        IJ : dict
            Mapping from each household to facilities that can cover it.
        budget_list : list
            Budgets to evaluate.
        parsimonious : bool, optional
            Whether to break ties in favour of fewer open facilities.
        maxTimeInSeconds : int, optional
            Time limit per solve.
        mipGap : float, optional
            Relative MIP gap.
        trace : bool, optional
            Whether to show solver output.
        already_open : list, optional
            Facilities fixed open.

        Returns
        -------
        pd.DataFrame
            Optimization results by budget.
        """
        return PyomoOptimize(
            w,
            I,
            J,
            IJ,
            budget_list,
            parsimonious,
            maxTimeInSeconds,
            mipGap,
            trace,
            already_open,
            this_solver,
        )

    return optimizer


def OptimizeWithGurobipy(
    w: list,
    I: list,
    J: list,
    IJ: dict,
    budget_list: list,
    parsimonious: bool = True,
    maxTimeInSeconds: int = 5 * 60,
    mipGap: float = 1e-8,
    trace: bool = False,
    already_open: list = [],
    progress: callable = lambda iterable: iterable,
) -> pd.DataFrame:
    """
    Solve the weighted maximal covering problem with gurobipy.

    Parameters
    ----------
    w : list
        Household weights.
    I : list
        Household indices.
    J : list
        Facility indices.
    IJ : dict
        Mapping from each household to facilities that can cover it.
    budget_list : list
        Budgets to evaluate.
    parsimonious : bool, optional
        If ``True``, break ties by preferring fewer open facilities.
    maxTimeInSeconds : int, optional
        Maximum solve time per optimization run.
    mipGap : float, optional
        Relative MIP gap.
    trace : bool, optional
        If ``True``, display the Gurobi log.
    already_open : list, optional
        Facilities forced open.
    progress : callable, optional
        Wrapper for progress reporting, for example ``tqdm``.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by budget with columns:
        ``value``, ``solution``, ``modeling``, ``solving``,
        ``termination``, and ``upper``.
    """
    result = pd.DataFrame(
        index=budget_list,
        columns=['value', 'solution', 'modeling', 'solving', 'termination', 'upper'],
    )

    w_array = np.asarray(w)
    budget_max = max(budget_list) if budget_list else 0

    start = pc()

    M = gb.Model('max_coverage')
    M.ModelSense = gb.GRB.MAXIMIZE
    M.Params.OutputFlag = int(trace)
    M.Params.MIPGap = mipGap
    M.Params.TimeLimit = maxTimeInSeconds

    x_obj = -1 / (budget_max + 1) if parsimonious else 0.0
    X = M.addVars(J, obj=x_obj, vtype=gb.GRB.BINARY, name='X')
    Y = M.addVars(I, obj={i: float(w_array[i]) for i in I}, vtype=gb.GRB.BINARY, name='Y')

    for j in set(already_open).intersection(set(J)):
        X[j].lb = 1.0
        X[j].ub = 1.0

    M.addConstrs(
        (Y[i] <= gb.quicksum(X[j] for j in IJ[i]) for i in I),
        name='serve_if_open',
    )

    budget = M.addConstr(X.sum() >= 0, name='budget')

    for p in progress(budget_list):
        M.remove(budget)
        budget = M.addConstr(X.sum() <= p, name=f'budget_{p}')
        M.update()

        result.at[p, 'modeling'] = pc() - start
        start = pc()

        M.optimize()

        result.at[p, 'solving'] = pc() - start
        result.at[p, 'termination'] = gurobicode.get(M.status, f'unknown_status_{M.status}')

        if M.SolCount > 0:
            weighted_coverage_value = sum(float(w_array[i]) for i in I if Y[i].X >= 0.5)
            result.at[p, 'value'] = int(np.ceil(weighted_coverage_value))
            result.at[p, 'solution'] = [j for j in J if X[j].X >= 0.5]
        else:
            result.at[p, 'value'] = None
            result.at[p, 'solution'] = None

        result.at[p, 'upper'] = int(np.floor(M.ObjBound)) if np.isfinite(M.ObjBound) else None

        start = pc()

    return result