import numpy as np
import gurobipy as gp
from pyscipopt import Model
from enum import Enum
from utils import gurobi_env
THREADS = 4
gurobi_env.setParam('Threads', THREADS)

class SolverStatus(Enum):
    OPTIMAL = 2  # success
    INFEASIBLE = 3  # fail
    UNBOUNDED = 5  # fail
    TIME_LIMIT = 9  # success
    SOLUTION_LIMIT = 10  # success
    INTERRUPTED = 11  # fail
    OBJ_LIMIT = 15  # success
    GAP_LIMIT = 18  # success
    SIZE_LIMIT = 19  # fail
    UNDEFINED = 20  # undefined


__gurobi_status_dict__ = {
    gp.GRB.OPTIMAL: SolverStatus.OPTIMAL,
    gp.GRB.TIME_LIMIT: SolverStatus.TIME_LIMIT,
    gp.GRB.SOLUTION_LIMIT: SolverStatus.SOLUTION_LIMIT,
    gp.GRB.INTERRUPTED: SolverStatus.INTERRUPTED,
    gp.GRB.INFEASIBLE: SolverStatus.INFEASIBLE,
    gp.GRB.UNBOUNDED: SolverStatus.UNBOUNDED,
    gp.GRB.INF_OR_UNBD: SolverStatus.UNBOUNDED,
}

__scip_status_dict__ = {
    "optimal": SolverStatus.OPTIMAL,
    "timelimit": SolverStatus.TIME_LIMIT,
    "sollimit": SolverStatus.SOLUTION_LIMIT,
    "gaplimit": SolverStatus.GAP_LIMIT,
    "infeasible": SolverStatus.INFEASIBLE,
    "unbounded": SolverStatus.UNBOUNDED,
    "inforunbd": SolverStatus.UNBOUNDED,
    "userinterrupt": SolverStatus.INTERRUPTED,
    "unknown": SolverStatus.UNDEFINED
}



def gurobi_subproblem_solve(
        var_feat: np.ndarray,  # Variable types and bounds
        constr_feat: np.ndarray,  # Constraint types and right-hand side
        obj_feat: np.ndarray,  # Objective function types
        edge_index: np.ndarray,  # Index of hyperedges corresponding to edges
        edge_attr: np.ndarray,  # Attributes of hyperedges corresponding to edge_features
        cur_sol: np.ndarray,  # Current solution
        time_limit: float,  # Time limit for optimization
        neighborhood: np.ndarray = None,  # Neighborhood (0-1 vector)
        solution_limit: int = None,  # solution limit for solve
        gap_limit: float = None,  # gap limit for solve
        obj_limit: float = None,  # obj limit for solve
        problem_name: str = None,  # Name of the problem file
        logger=None
):
    """
    Solve the optimization problem using Gurobi.

    Returns:
        cur_sol (np.ndarray): Current solution
        cur_obj (float): Objective value of current solution
        status (SolverStatus): Status of the solver
    """

    model = gp.Model(env=gurobi_env)

    # Add variables
    num_variables = var_feat.shape[0]
    var_types = []
    var_lb = []
    var_ub = []
    for feature in var_feat:
        if np.array_equal(feature[:3], [1, 0, 0]):
            var_types.append(gp.GRB.CONTINUOUS)
        elif np.array_equal(feature[:3], [0, 1, 0]):
            var_types.append(gp.GRB.BINARY)
        else:
            var_types.append(gp.GRB.INTEGER)

        # Add bounds
        lb = -gp.GRB.INFINITY if feature[5] == 1 else feature[3]
        ub = gp.GRB.INFINITY if feature[6] == 1 else feature[4]
        var_lb.append(lb)
        var_ub.append(ub)

    if neighborhood is None:
        neighborhood = np.ones(num_variables)
    variables = []
    for i in range(num_variables):
        if neighborhood[i] == 0:
            variables.append(cur_sol[i])
        elif neighborhood[i] == 1:
            v = model.addVar(vtype=var_types[i], lb=var_lb[i], ub=var_ub[i], name=str(i))
            v.setAttr('Start', cur_sol[i])
            variables.append(v)
    model.update()

    # Add constraints
    num_constraints = constr_feat.shape[0]
    constr_list = []
    for j in range(len(edge_index)):
        i = edge_index[j][0] - num_variables - 2
        if 0 <= i < num_constraints:
            if len(constr_list) <= i:
                constr_list.extend([gp.QuadExpr()
                                   for _ in range(i - len(constr_list) + 1)])

            quad_expr = constr_list[i]
            if edge_index[j][2] == num_variables:
                quad_expr.add(variables[edge_index[j][1]], edge_attr[j])
            elif edge_index[j][2] == num_variables + 1:
                quad_expr.add(variables[edge_index[j][1]] *
                              variables[edge_index[j][1]], edge_attr[j])
            else:
                quad_expr.add(variables[edge_index[j][1]] *
                              variables[edge_index[j][2]], edge_attr[j])

    for i, quad_expr in enumerate(constr_list):
        sense = constr_feat[i][9:12]
        rhs = constr_feat[i][12]
        if (sense == [1, 0, 0]).all():
            model.addQConstr(quad_expr, gp.GRB.LESS_EQUAL, rhs)
        elif (sense == [0, 1, 0]).all():
            model.addQConstr(quad_expr, gp.GRB.GREATER_EQUAL, rhs)
        else:
            model.addQConstr(quad_expr, gp.GRB.EQUAL, rhs)
    model.update()

    # Set objective
    obj_expr = gp.QuadExpr()
    for j in range(len(edge_index)):
        if edge_index[j][0] == num_constraints + num_variables + 2:
            if edge_index[j][2] == num_variables:
                obj_expr.add(variables[edge_index[j][1]], edge_attr[j])
            elif edge_index[j][2] == num_variables + 1:
                obj_expr.add(variables[edge_index[j][1]] *
                             variables[edge_index[j][1]], edge_attr[j])
            else:
                obj_expr.add(variables[edge_index[j][1]] *
                             variables[edge_index[j][2]], edge_attr[j])
    obj_sense = gp.GRB.MINIMIZE if obj_feat[0][13] == 1 else gp.GRB.MAXIMIZE
    model.setObjective(obj_expr, obj_sense)

    # Optimize
    model.setParam('NonConvex', 2)
    model.setParam('Seed', 42)
    model.setParam("Threads", THREADS)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    if solution_limit is not None:
        model.setParam('SolutionLimit', solution_limit)
    if gap_limit is not None:
        model.setParam('MIPGap', gap_limit)
    if obj_limit is not None:
        model.setParam('BestObjStop', obj_limit)
    model.update()

    model.optimize()
    status = SolverStatus(__gurobi_status_dict__[model.status])
    # return current solution
    try:
        cur_obj = model.ObjVal
        for i in range(num_variables):
            if neighborhood[i] == 1:
                cur_sol[i] = model.getVarByName(str(i)).X
    except:
        cur_obj = -1
    if logger is not None:
        logger.info(
            f"Subproblem: {problem_name}, obj: {cur_obj}, status: {status}, gap:{model.MIPGap}, time: {model.Runtime}")

    return (cur_sol, cur_obj, status)


def scip_subproblem_solve(
        var_feat: np.ndarray,  # Variable types and bounds
        constr_feat: np.ndarray,  # Constraint types and right-hand side
        obj_feat: np.ndarray,  # Objective function types
        edge_index: np.ndarray,  # Index of hyperedges corresponding to edges
        edge_attr: np.ndarray,  # Attributes of hyperedges corresponding to edge_features
        cur_sol: np.ndarray,  # Current solution
        time_limit: float,  # Time limit for optimization
        neighborhood: np.ndarray = None,  # Neighborhood (0-1 vector)
        solution_limit: int = None,  # solution limit for solve
        gap_limit: float = None,  # gap limit for solve
        problem_name: str = None,  # Name of the problem file
        logger=None
):
    """
    Solve the optimization problem using SCIP.

    Returns:
        cur_sol (np.ndarray): Current solution
        cur_obj (float): Objective value of current solution
        status (int): Status of the optimization
    """

    model = Model("SCIPModel")

    model.setRealParam('limits/time', time_limit)
    if solution_limit is not None:
        model.setIntParam('limits/solutions', solution_limit)
    if gap_limit is not None:
        model.setRealParam('limits/gap', gap_limit)

    model.hideOutput()

    num_variables = var_feat.shape[0]
    variables = []
    for i in range(num_variables):
        if np.array_equal(var_feat[i][:3], [0, 1, 0]):
            vtype = "B"
        elif np.array_equal(var_feat[i][:3], [0, 0, 1]):
            vtype = "I"
        else:
            vtype = "C"
        lb = -float('inf')if var_feat[i][5] == 1 else var_feat[i][3]
        ub = float('inf') if var_feat[i][6] == 1 else var_feat[i][4]
        var = model.addVar(str(i), vtype=vtype, lb=lb, ub=ub)
        variables.append(var)
    
    initial_sol = model.createSol()
    for i, var in enumerate(variables):
        model.setSolVal(initial_sol, var, cur_sol[i])

    for i in range(num_variables):
        if neighborhood[i] == 0:
            model.fixVar(variables[i], cur_sol[i])

    num_constraints = constr_feat.shape[0]
    constr_list = [0 for _ in range(num_constraints)]
    for j in range(len(edge_index)):
        i = edge_index[j][0] - num_variables - 2
        if 0 <= i < num_constraints:
            if edge_index[j][2] == num_variables:
                constr_list[i] += variables[edge_index[j][1]] * edge_attr[j]
            elif edge_index[j][2] == num_variables + 1:
                constr_list[i] += variables[edge_index[j][1]] * \
                    variables[edge_index[j][1]] * edge_attr[j]
            else:
                constr_list[i] += variables[edge_index[j][1]] * \
                    variables[edge_index[j][2]] * edge_attr[j]

    for i in range(num_constraints):
        sense = constr_feat[i][9:12]
        rhs = constr_feat[i][12]
        if np.array_equal(sense, [1, 0, 0]):
            model.addCons(constr_list[i] <= rhs)
        elif np.array_equal(sense, [0, 1, 0]):
            model.addCons(constr_list[i] >= rhs)
        else:
            model.addCons(constr_list[i] == rhs)

    obj = model.addVar("obj", vtype="C", lb=-float('inf'), ub=float('inf'))
    obj_expr = 0
    for j in range(len(edge_index)):
        if edge_index[j][0] == num_constraints + num_variables + 2:
            if edge_index[j][2] == num_variables:
                obj_expr += variables[edge_index[j][1]] * edge_attr[j]
            elif edge_index[j][2] == num_variables + 1:
                obj_expr += variables[edge_index[j][1]] * \
                    variables[edge_index[j][1]] * edge_attr[j]
            else:
                obj_expr += variables[edge_index[j][1]] * \
                    variables[edge_index[j][2]] * edge_attr[j]
    model.addCons(obj == obj_expr)
    model.setObjective(obj, "minimize" if obj_feat[0][13] == 1 else "maximize")

    model.optimize()
    status = SolverStatus(__scip_status_dict__[model.getStatus()])
    try:
        cur_obj = model.getObjVal()
        for i in range(num_variables):
            cur_sol[i] = model.getVal(variables[i])
    except:
        cur_obj = -1
    if logger is not None:
        logger.info(
            f"Subproblem: {problem_name}, obj: {cur_obj}, status: {status}, time: {model.getSolvingTime()}")

    return (cur_sol, cur_obj, status)


def get_solver(solver_name):
    if solver_name == "gurobi":
        return gurobi_subproblem_solve
    elif solver_name == "scip":
        return scip_subproblem_solve
    else:
        raise NotImplementedError
