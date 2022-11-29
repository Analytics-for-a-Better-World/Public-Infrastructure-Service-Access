from time import perf_counter as pc
import numpy as np
import pandas as pd
import gurobipy as gb
import pyomo.environ as pyo


def GetPyomoSolver(solverName, timeLimit=None, mipGap=None):
    if solverName == 'cbc':
        solver = pyo.SolverFactory(solverName, executable=r'D:\joaquimg\Dropbox\Python\solvers\cbc master\bin\cbc.exe')
        solver.options['threads'] = 8
    elif solverName == 'cplex':
        solver = pyo.SolverFactory('cplex_direct')
    elif solverName == 'gurobi':
        solver = pyo.SolverFactory('gurobi_direct')
    elif solverName == 'glpk':
        solver = pyo.SolverFactory(solverName,
                                   executable=r'D:\joaquimg\Dropbox\Python\solvers\cbc master\bin\glpsol.exe')
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


def OpenOptimize(w, I, J, IJ, budget_list, parsimonious=True, maxTimeInSeconds=5 * 60, mipGap=1e-8, trace=False,
                 solver='cbc'):
    """Solves the weighted maximum coverage problem with the solver specified, see https://en.wikipedia.org/wiki/Maximum_coverage_problem

    Args:
        w (array): w[i] is the weight of i in I
        I (array): indices to be served
        J (array): indices of potential services
        IJ (dictionary of arrays): per i in I the list of j in J that are accessible from i
        budget_list (list of integer): list of the maximum number of services to open
        maxTimeInSeconds (float, optional): Max solve time. Defaults to 5*60. 
        mipGap ([type], optional): Max MIP gap. Defaults to 1e-8. 
        trace (bool, optional): Show solve log. Defaults to False. 
        solver (string): the solver to use
        
    Returns:
        dataframe: one row per budget in budget_list and columns 'value','solution','modeling','solving','termination','upper'
    """

    result = pd.DataFrame(index=budget_list,
                          columns=['value', 'solution', 'modeling', 'solving', 'termination', 'upper'])

    start = pc()

    M = pyo.ConcreteModel('max_coverage')

    M.I = pyo.Set(initialize=I)
    M.J = pyo.Set(initialize=J)

    M.budget = pyo.Param(mutable=True, default=0)

    M.X = pyo.Var(M.J, domain=pyo.Binary)
    M.Y = pyo.Var(M.I, domain=pyo.Binary)

    M.nof_open_facilities = pyo.Expression(expr=pyo.quicksum(M.X[j] for j in M.J))
    M.weighted_coverage = pyo.Expression(expr=pyo.quicksum(w[i] * M.Y[i] for i in M.I))

    coef_x = -1 / (max(budget_list) + 1) if parsimonious else 0

    @M.Objective(sense=pyo.maximize)
    def coverage(M):
        return M.weighted_coverage + coef_x * M.nof_open_facilities

    @M.Constraint(M.I)
    def serve_if_open(M, i):
        return M.Y[i] <= pyo.quicksum(M.X[j] for j in IJ[i])

    @M.Constraint()
    def in_the_budget(M):
        return M.nof_open_facilities <= M.budget

    solver = GetPyomoSolver(solver, maxTimeInSeconds, mipGap)

    for p in budget_list:
        M.budget = p
        result.at[p, 'modeling'] = pc() - start
        start = pc()
        solver_result = solver.solve(M, tee=trace)
        result.at[p, 'solving'] = pc() - start
        result.at[p, 'value'] = int(np.ceil(M.weighted_coverage() - np.finfo(np.float16).eps))
        result.at[p, 'solution'] = [j for j in J if pyo.value(M.X[j]) >= .5]
        result.at[p, 'termination'] = solver_result.solver.termination_condition
        result.at[p, 'upper'] = max([abs(int(np.round(solver_result.problem.lower_bound + np.finfo(np.float16).eps))),
                                     abs(int(np.round(solver_result.problem.upper_bound + np.finfo(np.float16).eps)))])
        start = pc()

    return result


# a simple closure
def make_optimizer_using(this_solver):
    def optimizer(w, I, J, IJ, budget_list, parsimonious=True, maxTimeInSeconds=5 * 60, mipGap=1e-8, trace=False):
        return OpenOptimize(w, I, J, IJ, budget_list, parsimonious, maxTimeInSeconds, mipGap, trace, this_solver)

    return optimizer


gurobicode = {gb.GRB.LOADED: 'loaded',
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
              gb.GRB.USER_OBJ_LIMIT: 'user_obj_limit'}


def Optimize(w, I, J, IJ, budget_list, parsimonious=True, maxTimeInSeconds=5 * 60, mipGap=1e-8, trace=False):
    """Solves the weighted maximum coverage problem with gurobi, see https://en.wikipedia.org/wiki/Maximum_coverage_problem

    Args:
        w (array): w[i] is the weight of i in I
        I (array): indices to be served
        J (array): indices of potential services
        IJ (dictionary of arrays): per i in I the list of j in J that may access a service in i
        budget_list (list of integer): list of the maximum number of services to open
        maxTimeInSeconds (float, optional): Max solve time. Defaults to 5*60. See https://www.gurobi.com/documentation/9.5/refman/timelimit.html 
        mipGap ([type], optional): Max MIP gap. Defaults to 1e-8. See https://www.gurobi.com/documentation/9.5/refman/mipgap2.html
        trace (bool, optional): Show solve log. Defaults to False. See https://www.gurobi.com/documentation/9.5/refman/outputflag.html

    Returns:
        dataframe: one row per budget in budget_list and columns 'value','solution','modeling','solving','termination','upper'
    """

    result = pd.DataFrame(index=budget_list,
                          columns=['value', 'solution', 'modeling', 'solving', 'termination', 'upper'])

    start = pc()

    M = gb.Model('max_coverage')
    M.ModelSense = gb.GRB.MAXIMIZE

    M.Params.OutputFlag = trace
    M.Params.MIPGap = mipGap
    M.Params.TimeLimit = maxTimeInSeconds

    if parsimonious:
        X = M.addVars(J, obj=-1 / (max(budget_list) + 1), vtype=gb.GRB.BINARY)
    else:
        X = M.addVars(J, vtype=gb.GRB.BINARY)
    Y = M.addVars(I, obj=w[I], vtype=gb.GRB.BINARY)

    M.addConstrs((Y[i] <= (gb.quicksum(X[j] for j in IJ[i]))) for i in I)
    budget = M.addLConstr(X.sum() >= 0)

    for p in budget_list:
        M.remove(budget)
        budget = M.addLConstr(X.sum() <= p)
        result.at[p, 'modeling'] = pc() - start
        start = pc()
        M.optimize()
        result.at[p, 'solving'] = pc() - start
        result.at[p, 'value'] = int(np.ceil(M.objVal))
        result.at[p, 'solution'] = [j for j in J if X[j].x >= .5]
        result.at[p, 'termination'] = gurobicode[M.status]
        result.at[p, 'upper'] = int(np.floor(M.ObjBound))
        start = pc()

    return result


def Greedy(w, IJ, JI, budget_list):
    budget_list = sorted(budget_list)
    result = pd.DataFrame(index=budget_list, columns=['value', 'solution', 'increments', 'solving', 'coverage'])

    start = pc()
    greedy_selected, greedy_added = [], []
    coverage = np.zeros(len(w), dtype=np.uint16)
    greedy_val = -np.ones(len(w), dtype=int)

    J = list(JI.keys())
    may_change = np.array(J)
    prev = -1
    for p in budget_list:
        for i in range(prev + 1, min(p, len(J))):
            greedy_val[may_change] = [w[JI[j][coverage[JI[j]] == 0]].sum() for j in may_change]
            select = np.argmax(greedy_val)
            if greedy_val[select] == 0:
                break

            coverage[JI[select]] += 1
            greedy_selected.append(select)
            greedy_added.append(greedy_val[select])

            # Greedy only changes if coverage overlap with selected facility
            may_change = np.unique(np.concatenate([IJ[i] for i in JI[select]]))
        prev = i

        result.at[p, 'solving'] = pc() - start
        result.at[p, 'value'] = sum(greedy_added)
        result.at[p, 'solution'] = greedy_selected.copy()
        result.at[p, 'increments'] = greedy_added.copy()
        result.at[p, 'coverage'] = coverage.copy()
        start = pc()

    return result


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    import re
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def Solve(household, current, potential, accessibility, budgets, optimize=Optimize, type='ID'):
    values = pd.DataFrame()
    solutions = pd.DataFrame()

    columns = [c for c in current[accessibility].columns if c.startswith(type)]
    columns.sort(key=natural_keys, reverse=True)
    for column in columns:
        covered = np.unique(np.concatenate(current[accessibility][column])).astype(np.uint)
        percent_covered = household[covered].sum() / household.sum()

        # First solve optimally for the largest budget
        aux = potential[accessibility][['Cluster_ID', column]].set_index('Cluster_ID', drop=True)
        JI = {j: np.setdiff1d(i, covered, assume_unique=True) for j, i in aux[column].to_dict().items()}
        JI = {j: i for j, i in JI.items() if len(i)}
        IJ = {i: [] for i in np.setdiff1d(np.arange(len(household)), covered, assume_unique=True)}
        for j, I in JI.items():
            for i in I:
                if i in IJ.keys():
                    IJ[i].append(j)
        IJ = {i: np.unique(j) for i, j in IJ.items() if len(j)}
        I = np.unique(list(IJ.keys()))
        J = np.unique(np.concatenate(list(IJ.values())))
        optimization = optimize(household, I, J, IJ, [max(budgets)], parsimonious=True, maxTimeInSeconds=5,
                                mipGap=1e-15)
        optimization['nof'] = [len(s) for s in optimization.solution]
        coverage = (optimization.value / household.sum() + percent_covered).to_frame()
        coverage['served'] = [
            np.unique(list(set(np.concatenate(list(aux.loc[s][column]))).union(covered))).astype(np.uint) for s in
            optimization.solution.values]
        coverage['validation'] = [household[s].sum() / household.sum() for s in coverage.served.values]

        # Open the optimal solution in greedy steps
        best = optimization.loc[optimization.index[-1]].solution
        served = coverage.loc[coverage.index[-1]].served

        bestJI = {j: i for j, i in JI.items() if j in best}
        bestIJ = {i: [] for i in served}
        for j, I in bestJI.items():
            for i in I:
                bestIJ[i].append(j)
        bestIJ = {i: np.unique(j) for i, j in bestIJ.items() if len(j)}
        greedy = Greedy(household, bestIJ, bestJI, budgets)
        greedy['served'] = [
            np.unique(list(set(np.concatenate(list(aux.loc[s][column]))).union(covered))).astype(np.uint) for s in
            greedy.solution.values]
        greedy['coverage'] = [household[s].sum() / household.sum() for s in greedy.served.values]

        case = '_'.join(column.split('_')[1:])
        values[case] = greedy['coverage']
        solutions[case] = greedy['solution']

    return values, solutions


class Tree(dict):  # auto-vivification
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value


def CurrentValues(current, household, accessibilities):
    result = Tree()
    population = household.sum()
    for a in accessibilities:
        columns = [c for c in current[a].columns if c.startswith('ID')]
        columns.sort(key=natural_keys, reverse=True)
        for c in columns:
            covered = np.unique(np.concatenate(current[a][c])).astype(np.uint)
            result[a][c.partition('_')[-1]] = household[covered].sum() / population
    return result


def GoBackInTime(df_tests_lab, current, potential, accessibilities, to_date='05/01'):
    back_to_basics = df_tests_lab[['ShortDate', 'Laboratory', 'Province Name']].sort_values(
        ['ShortDate', 'Province Name', 'Laboratory']).drop_duplicates(subset='Laboratory', keep='first').reset_index(
        drop=True)
    the_first_ones = set(back_to_basics[back_to_basics.ShortDate <= to_date].Laboratory.values)

    new_current = dict()
    new_potential = dict()
    for a in accessibilities:
        new_current[a] = current[a][current[a].L_NAME.isin(the_first_ones)]
        to_move_to_potential = current[a][~current[a].L_NAME.isin(the_first_ones)].rename(
            columns={'Hosp_ID': 'Cluster_ID', 'L_NAME': 'Name'})
        new_potential[a] = pd.concat((to_move_to_potential, potential[a]))

    return new_current, new_potential


def ComputeCoverageFromSolutions(result, current, potential, household, accessibilities):
    coverage = dict()
    for accessibility in accessibilities:
        coverage[accessibility] = pd.DataFrame(index=result[accessibility][1].index)
        for col in result[accessibility][1].columns:
            column = 'ID_' + col
            covered = np.unique(np.concatenate(current[accessibility][column].values)).astype(np.uint)
            aux = potential[accessibility][['Cluster_ID', column]].set_index('Cluster_ID', drop=True)
            served = [np.unique(list(set(np.concatenate(list(aux.loc[s][column]))).union(covered))).astype(np.uint) for
                      s in result[accessibility][1][col].values]
            coverage[accessibility][col] = [household[s].sum() / household.sum() for s in served]
    return coverage
