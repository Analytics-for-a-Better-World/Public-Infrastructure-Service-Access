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
import pandas as pd
import gurobipy as gb
import pyomo.environ as pyo

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


# helper functions
def all_in(list_of_lists: list[list]) -> np.ndarray:
    """
    Returns a numpy array of unique elements from a list of lists

    Parameters:
        list_of_lists (list[list]): A list of lists

    Returns:
        numpy.ndarray: A numpy array of unique elements
    """
    return np.unique(np.concatenate(list_of_lists))


def CreateIndexMapping(all_facs: dict, household: list, covered: set = set()) \
        -> tuple[np.array, np.array, dict, dict]:
    """
    CreateIndexMapping creates a mapping between the indices of the households
    and the facilities.

    Parameters:
    all_facs (dict): A dictionary of facilities and their associated indices.
    household (list): A list of households.
    covered (set): A set of indices that are already covered.

    Returns:
    I (np.array): An array of indices of households.
    J (np.array): An array of indices of facilities.
    IJ (dict): A dictionary of households to the facilities that they reach.
    JI (dict): A dictionary of facilities to the households in catchment area.
    """
    not_covered = np.setdiff1d(np.arange(len(household)),
                               covered,
                               assume_unique=True)
    JI = {j: np.setdiff1d(i, covered, assume_unique=True)
          for j, i in all_facs.items()}
    JI = {j: i for j, i in JI.items() if len(i)}
    IJ = {i: [] for i in not_covered}
    for j, I in JI.items():
        for i in I:
            if i in IJ.keys():
                IJ[i].append(j)
    IJ = {i: np.unique(j) for i, j in IJ.items() if len(j)}
    I = np.unique(list(IJ.keys()))  # noqa: E741
    J = np.unique(np.concatenate(list(IJ.values())))
    return I, J, IJ, JI


def CheckIndexMapping(I: list,  # noqa: E741
                      J: list, IJ: dict, JI: dict, w: list) -> bool:
    """
    Checks if the index mapping is valid.

    Parameters:
    I (list): List of households
    J (list): List of potential facilities
    IJ (dict): Mapping from households to potential facilities
    JI (dict): Mapping from potential facilities to households
    w (list): List of weights

    Returns:
    bool: True if the index mapping is valid, fires an assertion otherwise
    """
    assert set(I) == set(range(len(w))), 'unknown household'
    assert set(I).issuperset(set(IJ.keys())), 'unknown household'
    assert set(J).issuperset(set(JI.keys())), 'unknown facility'
    assert set(J).issuperset(all_in(list(IJ.values()))), 'unknown facility'
    assert set(I).issuperset(all_in(list(JI.values()))), 'unknown household'
    assert set(all_in(list(JI.values()))).issubset(set(range(len(w)))),\
        'unknown household'
    return True


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
    # change if needed!!!
    solver_path = r'D:\joaquimg\Dropbox\Python\solvers\new cbc master\bin'
    if solverName == 'cbc':
        solver = pyo.SolverFactory(solverName,
                                   executable=solver_path+r'\cbc.exe')
        solver.options['threads'] = 8
    elif solverName == 'cplex':
        solver = pyo.SolverFactory('cplex_direct')
    elif solverName == 'gurobi':
        solver = pyo.SolverFactory('gurobi_direct')
    elif solverName == 'glpk':
        solver = pyo.SolverFactory(solverName,
                                   executable=solver_path+r'\glpsol.exe')
    else:
        solver = pyo.SolverFactory(solverName)
    if timeLimit:
        if solverName == 'cplex':
            solver.options['timelimit'] = timeLimit
        elif solverName == 'cbc':
            solver.options['sec'] = np.ceil(timeLimit)
        elif solverName == 'gurobi':
            solver.options['TimeLimit'] = timeLimit
    if mipGap:
        if solverName == 'cplex':
            solver.options['mipgap'] = mipGap
        elif solverName == 'cbc':
            solver.options['allowableGap'] = mipGap
        elif solverName == 'gurobi':
            solver.options['MipGap'] = mipGap
    return solver


def OptimizeWithPyomo(w: list, I: list,  # noqa: E741
                      J: list, IJ: dict, budget_list: list,
                      parsimonious: bool = True, maxTimeInSeconds: int = 5*60,
                      mipGap: float = 1e-8, trace: bool = False,
                      already_open: list = [], solver: str = 'cbc',
                      progress: callable = lambda iterable: iterable) \
                          -> pd.DataFrame:
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
    result : pandas.DataFrame
        DataFrame containing the optimization results with one row per budget
        in budget_list and columns 'value','solution','modeling','solving',
        'termination','upper'
    """

    # ensure that all facilities already open are given to a variable
    J = list(set(J) | set(already_open))

    result = pd.DataFrame(index=budget_list,
                          columns=['value', 'solution', 'modeling',
                                   'solving', 'termination', 'upper'])

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
        result.at[p, 'modeling'] = pc()-start
        start = pc()
        solve_result = solver.solve(M, tee=trace)
        result.at[p, 'solving'] = pc()-start
        result.at[p, 'value'] = M.weighted_coverage()
        result.at[p, 'solution'] = [j for j in J if pyo.value(M.X[j]) >= .5]
        result.at[p, 'termination'] = solve_result.solver.termination_condition
        result.at[p, 'upper'] = solve_result.problem.upper_bound
        start = pc()

    return result


# a simple closure
def make_pyomo_optimizer_using(this_solver: str) -> callable:
    """
    A simple closure to a function instantiating the specified this_solver
    able to solve an instance to be defined.

    Parameters
    ----------
        this_solver (str): the name of the solver to use.
    """
    def optimizer(w, I,  # noqa: E741
                  J, IJ, budget_list, parsimonious=True,
                  maxTimeInSeconds=5*60, mipGap=1e-8, trace=False,
                  already_open=[]):
        return OptimizeWithPyomo(w, I, J, IJ, budget_list, parsimonious,
                                 maxTimeInSeconds, mipGap, trace, already_open,
                                 this_solver)
    return optimizer


def OptimizeWithGurobipy(w: list, I: list,  # noqa: E741
                         J: list, IJ: dict,
                         budget_list: list, parsimonious: bool = True,
                         maxTimeInSeconds: int = 5*60, mipGap: float = 1e-8,
                         trace: bool = False, already_open: list = [],
                         progress: callable = lambda iterable: iterable) \
                             -> pd.DataFrame:
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
    result : pandas.DataFrame
        DataFrame containing the optimization results with one row per budget
        in budget_list and columns 'value','solution','modeling','solving',
        'termination','upper'
    """

    # ensure that all facilities already open are given to a variable
    J = list(set(J) | set(already_open))

    result = pd.DataFrame(index=budget_list, columns=['value',
                                                      'solution',
                                                      'modeling',
                                                      'solving',
                                                      'termination',
                                                      'upper'])

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
        result.at[p, 'modeling'] = pc()-start
        start = pc()
        M.optimize()
        result.at[p, 'solving'] = pc()-start
        result.at[p, 'value'] = M.objVal
        result.at[p, 'solution'] = [j for j in J if X[j].x >= .5]
        result.at[p, 'termination'] = verbose_gurobi_code[M.status]
        result.at[p, 'upper'] = M.ObjBound
        start = pc()

    return result


# Heuristics
def Greedy(w: np.ndarray, IJ: dict, JI: dict, nof_facilities: np.uint,
           budget_list: list, progress: callable = lambda iterable: iterable) \
               -> pd.DataFrame:
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
    result : pd.DataFrame
        DataFrame containing the value, solution, increments, solving and
        coverage for each budget.
    """

    result = pd.DataFrame(index=sorted(budget_list), columns=['value',
                                                              'solution',
                                                              'increments',
                                                              'solving',
                                                              'coverage'])

    start = pc()
    greedy_selected, greedy_added = [], []

    nof_households = len(w)
    coverage = np.zeros(nof_households, dtype=np.uint16)
    greedy_val = -np.ones(nof_facilities, dtype=int)

    may_change = np.array(list(JI.keys()))
    i = prev = -1
    for p in progress(result.index):
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

        result.at[p, 'solving'] = pc()-start
        result.at[p, 'value'] = sum(greedy_added)
        result.at[p, 'solution'] = greedy_selected.copy()
        result.at[p, 'increments'] = greedy_added.copy()
        result.at[p, 'coverage'] = coverage.copy()
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


def IncrementalSolutions(existing_facs: pd.DataFrame,
                         potential_facs: pd.DataFrame,
                         population: np.array,
                         max_number_additional_facilities: int,
                         optimize: callable = OptimizeWithGurobipy,
                         parsimonious: bool = True,
                         maxTimeInSeconds: int = 60,
                         mipGap: float = 1e-15) -> pd.DataFrame:
    """
    This function finds the incremental solutions for the given existing
    facilities, potential facilities, population, and maximum number of
    additional facilities.
    It uses the optimize function to solve the optimization problem and returns
    the incremental solutions in a DataFrame.

    Parameters
    ----------
    existing_facs : pd.DataFrame
        DataFrame containing existing facilities.
    potential_facs : pd.DataFrame
        DataFrame containing potential facilities.
    population : np.array
        Array containing population.
    max_number_additional_facilities : int
        Maximum number of additional facilities.
    optimize : callable
        Optimization function to be used, defaults to OptimizeWithGurobipy.
    parsimonious : bool
        Boolean value to indicate if the solution to be found by optimize
        should have the minimum number of facilities that achieve the maximal
        coverage, defaults to True.
    maxTimeInSeconds : int
        Maximum time in seconds for optimization in optimize, defaults to 60.
    mipGap : float
        maximum MIP gap for optimization in optimize, defaults to 1e-15

    Returns
    -------
    greedy : pd.DataFrame
        DataFrame containing the incremental solutions, facility per facility
        that recreate that optimal one for max_number_additional_facilities
        adding one facility a the time
    """

    covered = all_in(existing_facs.pop_with_access)
    household = population.astype(np.uint).values
    percent_covered = household[covered].sum()/household.sum()
    covered_set = set(covered)

    # Solve optimally once: for  the highest budget
    I, J, IJ, JI = CreateIndexMapping(potential_facs.pop_with_access,
                                      household,
                                      covered=covered)
    optimization = optimize(household, I, J, IJ,
                            [max_number_additional_facilities],
                            parsimonious=parsimonious,
                            maxTimeInSeconds=maxTimeInSeconds, mipGap=mipGap)
    optimization['nof'] = [len(s) for s in optimization.solution]

    coverage = (optimization.value/household.sum()+percent_covered).to_frame()
    coverage['served'] = [np.unique(list(covered_set.union(
        all_in(potential_facs.loc[s].pop_with_access.values))))
                          for s in optimization.solution.values]
    coverage['validation'] = [household[s].sum()/household.sum()
                              for s in coverage.served.values]

    # Open the optimal solution in greedy steps
    best = optimization.loc[optimization.index[-1]].solution
    served = coverage.loc[coverage.index[-1]].served

    bestJI = {j: JI[j] for j in best}
    bestIJ = {i: [] for i in served}
    for j, ii in bestJI.items():
        for i in ii:
            bestIJ[i].append(j)
    bestIJ = {i: np.unique(jj) for i, jj in bestIJ.items() if len(jj)}
    greedy = Greedy(household, bestIJ, bestJI,
                    len(potential_facs)+len(existing_facs),
                    np.arange(0, max_number_additional_facilities, 1))

    # Complement the data
    greedy['served'] = [covered] + \
        [np.unique(list(covered_set.union(set(
            all_in(potential_facs.loc[s].pop_with_access.values)))))
         for s in greedy.solution[1:].values]
    greedy['coverage'] = [household[s].sum()/household.sum()
                          for s in greedy.served.values]

    return greedy


def GreedyLS(w: np.ndarray, IJ: dict, JI: dict, nof_facilities: np.uint,
             budget_list: list,
             progress: callable = lambda iterable: iterable) -> pd.DataFrame:

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
    result : pd.DataFrame
        DataFrame containing the value, solution, increments, solving and
        coverage for each budget.
    """

    def GetCoverage(facilities, nof_homes, JI):
        # 'u1' is the smallest unsigned datatype for int: takes 1 byte
        coverage = np.zeros(nof_homes, dtype=np.dtype('u1'))
        for j in facilities:
            coverage[JI[j]] += 1
        return coverage

    def GetSolutionValue(solution, household, JI):
        if solution:
            return household[all_in([JI[s] for s in solution])].sum()
        return 0

    result = pd.DataFrame(index=sorted(budget_list), columns=['value',
                                                              'solution',
                                                              'solving',
                                                              'coverage'])

    start = pc()
    solution, greedy_added = [], []

    nof_households = len(w)
    coverage = np.zeros(nof_households, dtype=np.uint16)
    greedy_val = -np.ones(nof_facilities, dtype=int)

    J = list(JI.keys())

    prev = -1
    for p in progress(result.index):
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

        result.at[p, 'solving'] = pc()-start
        result.at[p, 'value'] = objective
        result.at[p, 'solution'] = solution
        result.at[p, 'coverage'] = coverage
        start = pc()

    return result
