# %%
import numpy as np
import multiprocessing
import time
import random
import logging
from enum import Enum
from SubSolver import get_solver, SolverStatus
from pprint import pprint, pformat

CROSS_TIME_LIMIT = 30
SEARCH_TIME_LIMIT = 60
INITIAL_TIME_LIMIT = 30

INFINITY = 1e100


class VType(Enum):
    CONTINUOUS = 0
    BINARY = 1
    INTEGER = 2


class CType(Enum):
    LESS_EQUAL = 0
    GREATER_EQUAL = 1
    EQUAL = 2


class Sense(Enum):
    MINIMIZE = 0
    MAXIMIZE = 1


class Problem(object):

    def __init__(self,
                 solver: str,
                 var_feat: np.ndarray,  # Variable types and bounds
                 constr_feat: np.ndarray,  # Constraint types and right-hand side
                 obj_feat: np.ndarray,  # Objective function types
                 edge_attr: np.ndarray,  # Attributes of hyperedges corresponding to edge_features
                 edge_index: np.ndarray,  # Index of hyperedges corresponding to edges
                 cur_val: np.ndarray = None,
                 cur_obj: float = None):

        # set solver
        self.subproblem_solve = get_solver(solver)
        # set variables
        var_types, var_lb, var_ub = [], [], []
        for feature in var_feat:
            if np.array_equal(feature[:3], [1, 0, 0]):
                var_types.append(VType.CONTINUOUS)
            elif np.array_equal(feature[:3], [0, 1, 0]):
                var_types.append(VType.BINARY)
                lb, ub = 0, 1
            else:
                var_types.append(VType.INTEGER)

            lb = -INFINITY if feature[5] == 1 else feature[3]
            ub = INFINITY if feature[6] == 1 else feature[4]
            var_lb.append(lb)
            var_ub.append(ub)

        self.var_feat = var_feat
        self.var_types = np.array(var_types)
        self.var_lb = np.array(var_lb)
        self.var_ub = np.array(var_ub)
        self.n_var = len(var_feat)

        # set constraints
        self.constr_feat = constr_feat
        self.n_constr = len(constr_feat)

        # set edges
        self.edge_index = edge_index
        self.edge_attr = edge_attr

        # set objective sense
        self.obj_feat = obj_feat
        self.obj_sense = Sense.MINIMIZE if obj_feat[0][13] == 1 else Sense.MAXIMIZE

        # set current model status
        self.is_feasible = False
        self.is_unbounded = False
        self.cur_val = cur_val if cur_val is not None else np.zeros(self.n_var)
        self.cur_obj = cur_obj if cur_obj is not None else INFINITY

    def set_variables(self, indices: np.ndarray, values: np.ndarray):
        """
        Set the values of variables to the given values.
        """
        for i, val in zip(indices, values):
            self.cur_val[int(i)] = np.round(val)

    def get_data(self):
        """
        Get the data of the problem.
        """
        return self.var_feat, self.constr_feat, self.obj_feat, self.edge_index, self.edge_attr, self.cur_val


class RepairPolicy(object):
    "Only for binary problem"

    def __init__(self, p: Problem):
        self.problem = p

    def get_repair(self, cur_sol, neiborhood):
        raise NotImplementedError(
            "RepairPolicy has to be implemented by subclasses.")


class QuickRepairPolicy(RepairPolicy):

    def get_repair(
        self,
        cur_sol: np.ndarray,
        neighborhood: np.ndarray,  # Neighborhood (0-1 vector)
        max_size: int
    ) -> np.ndarray:
        """
        Check and repair the constraints that are doomed to be infeasible.
        """
        var_feat, constr_feat, _, edge_index, edge_attr, _ = self.problem.get_data()
        num_variables = var_feat.shape[0]
        for feature in var_feat:
            # ensure that the problem is binary
            if not np.array_equal(feature[:3], [0, 1, 0]):
                pprint("The problem is too complex to be repaired")
                return

        var_lb = np.zeros(num_variables)  # NOTE: only for binary variables
        var_ub = np.ones(num_variables)  # NOTE: only for binary variables
        if neighborhood is None:
            neighborhood = np.zeros(num_variables)
        size = np.count_nonzero(neighborhood)
        mask = (neighborhood == 0)
        var_lb[mask] = cur_sol[mask]
        var_ub[mask] = cur_sol[mask]

        num_constraints = constr_feat.shape[0]
        constr_list = []
        for j in range(len(edge_index)):
            if edge_index[j][0] == num_variables + 2:
                index = j
                break
        for i in range(num_constraints):
            sense = constr_feat[i][9:12]
            rhs = constr_feat[i][12]
            constr_list.append([])
            if (sense == [1, 0, 0]).all():  # <=
                constr_lb = 0
                # find edge_index related to constraint i
                for j in range(index, len(edge_index)):
                    if edge_index[j][0] == num_variables + i + 2:
                        if edge_index[j][2] == num_variables or edge_index[j][2] == num_variables + 1:
                            constr_lb += edge_attr[j]*var_lb[edge_index[j][1]
                                                             ] if edge_attr[j] > 0 else edge_attr[j]*var_ub[edge_index[j][1]]
                            constr_list[i].append(edge_index[j][1])
                        else:
                            quad_lb = max(
                                0, var_lb[edge_index[j][1]]+var_lb[edge_index[j][2]]-1)
                            quad_ub = min(
                                var_ub[edge_index[j][1]], var_ub[edge_index[j][2]])
                            constr_lb += edge_attr[j] * \
                                quad_lb if edge_attr[j] > 0 else edge_attr[j]*quad_ub
                            constr_list[i].append(edge_index[j][1])
                            constr_list[i].append(edge_index[j][2])
                    else:
                        index = j
                        break
                if constr_lb > rhs:
                    for k in constr_list[i]:
                        if (neighborhood[k] == 0 and size < max_size):
                            neighborhood[k] = 1
                            var_lb[k] = 0
                            var_ub[k] = 1
                            size += 1
                        elif size >= max_size:
                            break

            elif (sense == [0, 1, 0]).all():   # >=
                constr_ub = 0
                # find edge_index related to constraint i
                for j in range(index, len(edge_index)):
                    if edge_index[j][0] == num_variables + i + 2:
                        if edge_index[j][2] == num_variables or edge_index[j][2] == num_variables + 1:
                            constr_ub += edge_attr[j]*var_ub[edge_index[j][1]
                                                             ] if edge_attr[j] > 0 else edge_attr[j]*var_lb[edge_index[j][1]]
                            constr_list[i].append(edge_index[j][1])
                        else:
                            quad_lb = max(
                                0, var_lb[edge_index[j][1]]+var_lb[edge_index[j][2]]-1)
                            quad_ub = min(
                                var_ub[edge_index[j][1]], var_ub[edge_index[j][2]])
                            constr_ub += edge_attr[j] * \
                                quad_ub if edge_attr[j] > 0 else edge_attr[j]*quad_lb
                            constr_list[i].append(edge_index[j][1])
                            constr_list[i].append(edge_index[j][2])
                    else:
                        index = j
                        break
                if constr_ub < rhs:
                    for k in constr_list[i]:
                        if (neighborhood[k] == 0 and size < max_size):
                            neighborhood[k] = 1
                            var_lb[k] = 0
                            var_ub[k] = 1
                            size += 1
                        elif size >= max_size:
                            break
            else:
                constr_lb = 0
                constr_ub = 0
                # find edge_index related to constraint i
                for j in range(index, len(edge_index)):
                    if edge_index[j][0] == num_variables + i + 2:
                        if edge_index[j][2] == num_variables or edge_index[j][2] == num_variables + 1:
                            constr_lb += edge_attr[j]*var_lb[edge_index[j][1]
                                                             ] if edge_attr[j] > 0 else edge_attr[j]*var_ub[edge_index[j][1]]
                            constr_ub += edge_attr[j]*var_ub[edge_index[j][1]
                                                             ] if edge_attr[j] > 0 else edge_attr[j]*var_lb[edge_index[j][1]]
                            constr_list[i].append(edge_index[j][1])
                        else:
                            quad_lb = max(
                                0, var_lb[edge_index[j][1]]+var_lb[edge_index[j][2]]-1)
                            quad_ub = min(
                                var_ub[edge_index[j][1]], var_ub[edge_index[j][2]])
                            constr_lb += edge_attr[j] * \
                                quad_lb if edge_attr[j] > 0 else edge_attr[j]*quad_ub
                            constr_ub += edge_attr[j] * \
                                quad_ub if edge_attr[j] > 0 else edge_attr[j]*quad_lb
                            constr_list[i].append(edge_index[j][1])
                            constr_list[i].append(edge_index[j][2])
                    else:
                        index = j
                        break
                if constr_ub < rhs or constr_lb > rhs:
                    for k in constr_list[i]:
                        if (neighborhood[k] == 0 and size < max_size):
                            neighborhood[k] = 1
                            var_lb[k] = 0
                            var_ub[k] = 1
                            size += 1
                        elif size >= max_size:
                            break

        return neighborhood


class CautiousRepairPolicy(RepairPolicy):

    def get_repair(
        self,
        cur_sol: np.ndarray,
        neighborhood: np.ndarray,  # Neighborhood (0-1 vector)
        max_size: int
    ) -> np.ndarray:
        """
        Check and repair the constraints that are doomed to be infeasible.
        """
        var_feat, constr_feat, _, edge_index, edge_attr, _ = self.problem.get_data()
        num_variables = var_feat.shape[0]
        for feature in var_feat:
            if not np.array_equal(feature[:3], [0, 1, 0]):
                pprint("The problem is too complex to be repaired")
                return

        var_lb = np.zeros(num_variables)
        var_ub = np.ones(num_variables)
        if neighborhood is None:
            neighborhood = np.zeros(num_variables)
        size = np.count_nonzero(neighborhood)
        mask = (neighborhood == 0)
        var_lb[mask] = cur_sol[mask]
        var_ub[mask] = cur_sol[mask]

        num_constraints = constr_feat.shape[0]
        constr_list = []
        for j in range(len(edge_index)):
            if edge_index[j][0] == num_variables + 2:
                index = j
                break
        for i in range(num_constraints):
            sense = constr_feat[i][9:12]
            rhs = constr_feat[i][12]
            constr_list.append([])
            if (sense == [1, 0, 0]).all():  # <=
                constr_lb = 0
                # find edge_index related to constraint i
                for j in range(index, len(edge_index)):
                    if edge_index[j][0] == num_variables + i + 2:
                        if edge_index[j][2] == num_variables or edge_index[j][2] == num_variables + 1:
                            expr_lb = edge_attr[j]*var_lb[edge_index[j][1]
                                                          ] if edge_attr[j] > 0 else edge_attr[j]*var_ub[edge_index[j][1]]
                            constr_lb += expr_lb
                            constr_list[i].append(
                                [0, edge_index[j][1], edge_index[j][1], edge_attr[j], expr_lb])
                        else:
                            quad_lb = max(
                                0, var_lb[edge_index[j][1]]+var_lb[edge_index[j][2]]-1)
                            quad_ub = min(
                                var_ub[edge_index[j][1]], var_ub[edge_index[j][2]])
                            expr_lb = edge_attr[j] * \
                                quad_lb if edge_attr[j] > 0 else edge_attr[j]*quad_ub
                            constr_lb += edge_attr[j] * \
                                quad_lb if edge_attr[j] > 0 else edge_attr[j]*quad_ub
                            constr_list[i].append(
                                [1, edge_index[j][1], edge_index[j][2], edge_attr[j], expr_lb])
                    else:
                        index = j
                        break
                for k, var_list in enumerate(constr_list[i]):
                    if constr_lb > rhs:
                        if var_list[0] == 0 and neighborhood[var_list[1]] == 0 and size < max_size:
                            neighborhood[var_list[1]] = 1
                            size += 1
                            constr_lb = constr_lb - var_list[4]
                            var_lb[var_list[1]] = 0
                            var_ub[var_list[1]] = 1
                            if var_list[3] < 0:
                                constr_lb += var_list[3]
                        elif var_list[0] == 1 and neighborhood[var_list[1]] * neighborhood[var_list[2]] == 0 and size < max_size:
                            if neighborhood[var_list[1]] == 0:
                                size += 1
                            if neighborhood[var_list[2]] == 0:
                                size += 1
                            neighborhood[var_list[1]] = 1
                            neighborhood[var_list[2]] = 1
                            constr_lb = constr_lb - var_list[4]
                            var_lb[var_list[1]] = 0
                            var_ub[var_list[1]] = 1
                            var_lb[var_list[2]] = 0
                            var_ub[var_list[2]] = 1
                            if var_list[3] < 0:
                                constr_lb += var_list[3]
                    else:
                        break

            elif (sense == [0, 1, 0]).all():  # >=
                constr_ub = 0
                for j in range(index, len(edge_index)):
                    if edge_index[j][0] == num_variables + i + 2:
                        if edge_index[j][2] == num_variables or edge_index[j][2] == num_variables + 1:
                            expr_ub = edge_attr[j]*var_ub[edge_index[j][1]
                                                          ] if edge_attr[j] > 0 else edge_attr[j]*var_lb[edge_index[j][1]]
                            constr_ub += expr_ub
                            constr_list[i].append(
                                [0, edge_index[j][1], edge_index[j][1], edge_attr[j], expr_ub])
                        else:
                            quad_lb = max(
                                0, var_lb[edge_index[j][1]]+var_lb[edge_index[j][2]]-1)
                            quad_ub = min(
                                var_ub[edge_index[j][1]], var_ub[edge_index[j][2]])
                            expr_ub = edge_attr[j] * \
                                quad_ub if edge_attr[j] > 0 else edge_attr[j]*quad_lb
                            constr_ub += expr_ub
                            constr_list[i].append(
                                [0, edge_index[j][1], edge_index[j][2], edge_attr[j], expr_ub])
                    else:
                        index = j
                        break
                for k, var_list in enumerate(constr_list[i]):
                    if constr_ub < rhs:
                        if var_list[0] == 0 and neighborhood[var_list[1]] == 0 and size < max_size:
                            size += 1
                            neighborhood[var_list[1]] = 1
                            constr_ub = constr_ub - var_list[4]
                            var_lb[var_list[1]] = 0
                            var_ub[var_list[1]] = 1
                            if var_list[3] > 0:
                                constr_ub += var_list[3]
                        elif var_list[0] == 1 and neighborhood[var_list[1]] * neighborhood[var_list[2]] == 0 and size < max_size:
                            if neighborhood[var_list[1]] == 0:
                                size += 1
                            if neighborhood[var_list[2]] == 0:
                                size += 1
                            neighborhood[var_list[1]] = 1
                            neighborhood[var_list[2]] = 1
                            constr_ub = constr_ub - var_list[4]
                            var_lb[var_list[1]] = 0
                            var_ub[var_list[1]] = 1
                            var_lb[var_list[2]] = 0
                            var_ub[var_list[2]] = 1
                            if var_list[3] > 0:
                                constr_ub += var_list[3]
                    else:
                        break
            else:
                constr_lb = 0
                constr_ub = 0
                for j in range(index, len(edge_index)):
                    if edge_index[j][0] == num_variables + i + 2:
                        if edge_index[j][2] == num_variables or edge_index[j][2] == num_variables + 1:
                            expr_lb = edge_attr[j]*var_lb[edge_index[j][1]
                                                          ] if edge_attr[j] > 0 else edge_attr[j]*var_ub[edge_index[j][1]]
                            expr_ub = edge_attr[j]*var_ub[edge_index[j][1]
                                                          ] if edge_attr[j] > 0 else edge_attr[j]*var_lb[edge_index[j][1]]
                            constr_lb += expr_lb
                            constr_ub += expr_ub
                            constr_list[i].append(
                                [0, edge_index[j][1], edge_index[j][1], edge_attr[j], expr_lb, expr_ub])
                        else:
                            quad_lb = max(
                                0, var_lb[edge_index[j][1]]+var_lb[edge_index[j][2]]-1)
                            quad_ub = min(
                                var_ub[edge_index[j][1]], var_ub[edge_index[j][2]])
                            expr_lb = edge_attr[j] * \
                                quad_lb if edge_attr[j] > 0 else edge_attr[j]*quad_ub
                            expr_ub = edge_attr[j] * \
                                quad_ub if edge_attr[j] > 0 else edge_attr[j]*quad_lb
                            constr_lb += expr_lb
                            constr_ub += expr_ub
                            constr_list[i].append(
                                [0, edge_index[j][1], edge_index[j][2], edge_attr[j], expr_lb, expr_ub])
                    else:
                        index = j
                        break
                for k, var_list in enumerate(constr_list[i]):
                    if constr_ub < rhs or constr_lb > rhs:
                        if var_list[0] == 0 and neighborhood[var_list[1]] == 0 and size < max_size:
                            size += 1
                            neighborhood[var_list[1]] = 1
                            constr_lb = constr_lb - var_list[4]
                            constr_ub = constr_ub - var_list[5]
                            var_lb[var_list[1]] = 0
                            var_ub[var_list[1]] = 1
                            if var_list[3] > 0:
                                constr_ub += var_list[3]
                            else:
                                constr_lb += var_list[3]
                        elif var_list[0] == 1 and neighborhood[var_list[1]] * neighborhood[var_list[2]] == 0 and size < max_size:
                            if neighborhood[var_list[1]] == 0:
                                size += 1
                            if neighborhood[var_list[2]] == 0:
                                size += 1
                            neighborhood[var_list[1]] = 1
                            neighborhood[var_list[2]] = 1
                            constr_lb = constr_lb - var_list[4]
                            constr_ub = constr_ub - var_list[5]
                            var_lb[var_list[1]] = 0
                            var_ub[var_list[1]] = 1
                            var_lb[var_list[2]] = 0
                            var_ub[var_list[2]] = 1
                            if var_list[3] > 0:
                                constr_ub += var_list[3]
                            else:
                                constr_lb += var_list[3]
                    else:
                        break

        return neighborhood


class InitialSolutionPolicy(object):
    def __init__(self, p: Problem):
        self.problem = p

    def get_feasible_solution(cur_val):
        raise NotImplementedError(
            "InitialPolicy has to be implemented by subclasses.")


class VariableRelaxationPolicy(InitialSolutionPolicy):

    def __init__(self, p: Problem):
        super().__init__(p)

    def get_feasible_solution(self,
                              logger,
                              # the index,predicted values and predicted loss of each variable
                              val_and_logit: np.ndarray,
                              repair_policy: RepairPolicy,  # repair policy
                              alpha=0.1,  # the threshold of the percentage of variables to be optimized
                              alpha_step=0.05,  # the step of alpha
                              alpha_ub=0.25,  # the upper bound of alpha
                              max_size=1000):
        """
        Get a feasible solution from the given values by relaxing the variables.
        """
        start_time = time.time()
        sorted_indices = val_and_logit[:, 2].argsort()
        val_and_logit = val_and_logit[sorted_indices]

        # set the variables to the predicted values
        indices = val_and_logit[:, 0].reshape(-1)
        values = val_and_logit[:, 1].reshape(-1)
        self.problem.set_variables(indices, values)

        var_feat, constr_feat, obj_feat, edge_index, edge_attr, cur_sol = self.problem.get_data()

        while (self.problem.is_feasible == False) and (alpha <= alpha_ub):
            num_to_optimize = int(alpha * len(val_and_logit))
            one_indices = indices[-num_to_optimize:].astype(int)
            neighborhood = np.zeros(self.problem.n_var, dtype=int)
            neighborhood[one_indices] = 1
            cur_sol = self.problem.cur_val
            neighborhood = repair_policy.get_repair(
                cur_sol, neighborhood, max_size)

            alpha = np.nonzero(neighborhood)[
                0].shape[0] / neighborhood.shape[0]
            cur_sol, obj, status = self.problem.subproblem_solve(var_feat, constr_feat, obj_feat, edge_index, edge_attr,
                                                                 cur_sol, time_limit=INITIAL_TIME_LIMIT, neighborhood=neighborhood, solution_limit=1)
            if status == SolverStatus.OPTIMAL or status == SolverStatus.SOLUTION_LIMIT:
                self.problem.is_feasible = True
                self.problem.cur_obj = obj
                self.problem.cur_val = cur_sol
                logger.info(f"Feasible solution found alpha:{alpha} n_var:{np.nonzero(neighborhood)[0].shape[0]}")
                logger.info(f"Time used: {time.time() - start_time :.2f} seconds")
                break

            else:
                alpha += alpha_step
                if alpha >= alpha_ub:
                    logger.info(f'Failed to find initial feasible solution')
                    break
        return cur_sol, obj


class NeighborhoodPolicy(object):
    """
    Neighborhood search policy

    Attributes:
        size: number of total variables
        losses: loss of each variable, i.e., the likelihood of confidence
        rate: the rate of variables allowed to be optimized
        predictX: the predicted value of each variable
        curX: the current value of each variable
    """

    def __init__(self, p: Problem) -> None:
        self.problem = p

    def get_neighborhood(self, block_size):
        raise NotImplementedError(
            "NeighborhoodPolicy has to be implemented by subclasses.")


class RandomNeighborhoodPolicy(NeighborhoodPolicy):
    def __init__(self, p: Problem):
        super().__init__(p)

    def get_neighborhood(self, block_size, cur_val=None, val_and_logit=None, *args, **kwargs) -> np.ndarray:
        """
        Get a partition of n_blocks based on the random model.
        """
        var_feat, _, _, _, _, _ = self.problem.get_data()
        n_vars = var_feat.shape[0]
        perm = np.random.permutation(n_vars)
        n_blocks = n_vars // block_size + 1
        neighborhoods = np.array_split(perm, n_blocks)
        binary_matrix = np.zeros((n_blocks, n_vars), dtype=int)
        for i, neighborhood in enumerate(neighborhoods):
            binary_matrix[i, neighborhood] = 1

        return binary_matrix

class MixedRepairPolicy(RepairPolicy):
        def __init__(self, p: Problem):
            super().__init__(p)
            self.quick_policy = QuickRepairPolicy(p)
            self.cautious_policy = CautiousRepairPolicy(p)
            self.counter = 0
    
        def get_repair(
            self,
            cur_sol: np.ndarray,
            neighborhood: np.ndarray,
            max_size: int
        ) -> np.ndarray:
            if self.counter == 0:
                neighborhood = self.cautious_policy.get_repair(cur_sol, neighborhood, max_size)
            else:
                neighborhood = self.quick_policy.get_repair(cur_sol, neighborhood, max_size)
                self.counter =(self.counter + 1) % 2
            return neighborhood

class ConstrRandomNeighborhoodPolicy(NeighborhoodPolicy):
    def __init__(self, p: Problem):
        super().__init__(p)

    def get_neighborhood(self, block_size, cur_val=None, val_and_logit=None, *args, **kwargs) -> np.ndarray:
        """
        Get a partition of n_blocks based on the random model.
        """
        var_feat, constr_feat, _, edge_index, _, _ = self.problem.get_data()
        n_vars = var_feat.shape[0]
        n_constrs = constr_feat.shape[0]
        perm = np.random.permutation(n_constrs)
        n_all = 0  # the number of elements in all neighborhooods

        constr_list = []
        for j in range(len(edge_index)):
            if edge_index[j][0] == n_vars + 2:
                index = j
                break
        for i in range(n_constrs):
            constr_list.append([])
            for j in range(index, len(edge_index)):
                if edge_index[j][0] == n_vars + i + 2:
                    if edge_index[j][2] == n_vars or edge_index[j][2] == n_vars + 1:
                        constr_list[i].append(edge_index[j][1])
                    else:
                        constr_list[i].append(edge_index[j][1])
                        constr_list[i].append(edge_index[j][2])
                else:
                    index = j
                    break
            constr_list[i] = list(set(constr_list[i]))  # remove duplication
            random.shuffle(constr_list[i])
            n_all += len(constr_list[i])

        n_blocks = n_all // block_size + 1
        binary_matrix = np.zeros((n_blocks, n_vars), dtype=int)
        count_var = np.zeros(n_blocks)
        neighborhood = 0
        for i in range(n_constrs):
            for j in constr_list[perm[i]]:
                if count_var[neighborhood] <= block_size:
                    binary_matrix[neighborhood, j] = 1
                    count_var[neighborhood] += 1
                else:
                    neighborhood += 1
                    binary_matrix[neighborhood, j] = 1
                    count_var[neighborhood] += 1

        return binary_matrix

class MixedNeighborhoodPolicy(NeighborhoodPolicy):
    def __init__(self, p: Problem):
        super().__init__(p)
        self.random_policy = RandomNeighborhoodPolicy(p)
        self.constr_random_policy = ConstrRandomNeighborhoodPolicy(p)
        self.counter = 0

    def get_neighborhood(self, block_size, *args, **kwargs) -> np.ndarray:
        self.counter += 1
        if self.counter <= 5:  # use random policy in the first 5 iterations
            neighborhood = self.random_policy.get_neighborhood(block_size, *args, **kwargs)
        else:
            if self.counter % 2 == 0:
                neighborhood = self.constr_random_policy.get_neighborhood(block_size, *args, **kwargs)
            else:
                neighborhood = self.random_policy.get_neighborhood(block_size, *args, **kwargs)        

        return neighborhood

def cross_neighborhood(p: Problem,
                       repair_policy: RepairPolicy,
                       block_size: int,
                       neighborhood_1: np.ndarray,  # the neighborhood whose cur_obj is better
                       neighborhood_2: np.ndarray,
                       cur_sol_1: np.ndarray,  # the solution whose cur_obj is better
                       cur_sol_2: np.ndarray,
                       cur_obj_1: np.ndarray,
                       cur_obj_2: np.ndarray):
    var_feat, constr_feat, obj_feat, edge_index, edge_attr, _ = p.get_data()
    n_vars = var_feat.shape[0]
    if (cur_obj_1 >= cur_obj_2 and p.obj_sense == Sense.MAXIMIZE) or (cur_obj_1 <= cur_obj_2 and p.obj_sense == Sense.MINIMIZE):
        cur_obj = cur_obj_1
        neighborhood = neighborhood_1
        neighborhood_ = neighborhood_2
        new_sol = cur_sol_1
        cur_sol = cur_sol_1
        cur_sol_ = cur_sol_2
    else:
        cur_obj = cur_obj_2
        neighborhood = neighborhood_2
        neighborhood_ = neighborhood_1
        new_sol = cur_sol_2
        cur_sol = cur_sol_2
        cur_sol_ = cur_sol_1
    for i in range(n_vars):
        if neighborhood[i] == 0 and neighborhood_[i] == 1:
            new_sol[i] = cur_sol_[i]
    new_neighborhood = repair_policy.get_repair(
        new_sol, neighborhood=None, max_size=block_size)
    if np.where(new_neighborhood == 1)[0].shape[0] > block_size:
        return (cur_sol, cur_obj, SolverStatus.SIZE_LIMIT)
    else:
        new_sol, new_obj, status = p.subproblem_solve(var_feat, constr_feat, obj_feat, edge_index, edge_attr,
                                                      new_sol, time_limit=CROSS_TIME_LIMIT, neighborhood=new_neighborhood)
        return (new_sol, new_obj, status)


neighborhood_policy_dict = {
    "random": RandomNeighborhoodPolicy,
    "constr_random": ConstrRandomNeighborhoodPolicy,
    "mixed": MixedNeighborhoodPolicy
}

repair_policy_dict = {
    "quick": QuickRepairPolicy,
    "cautious": CautiousRepairPolicy,
    "mixed": MixedRepairPolicy
}

initial_policy_dict = {
    "variable_relaxation": VariableRelaxationPolicy
}


def optimize(
    logger: logging.Logger,
    solver: str,
    problem: list[np.ndarray],
    output: np.ndarray,  # neural network output
    initial_policy: str,
    repair_policy: str,
    neighborhood_policy: str,
    time_limit: float,
    obj_limit: float,
    n_processes: int,
    block: float,
    crossover: bool = True,
    *args,
    **kwargs
):
    """
    Perform neighborhood search on the given problem.
    1. Predict a solution
    2. Get a feasible solution
    3. Generate the neighborhood of the solution
    4. solve
    """
    start_time = time.time()
    logger.info(f"time_limit {time_limit}")
    logger.info(f"n_processes {n_processes}")
    logger.info(f"block_size {block}")
    logger.info(f"repair_policy: {repair_policy}")
    logger.info(f"neighborhood_policy: {neighborhood_policy}")
    logger.info(f"crossover: {crossover}")
    p = Problem(solver, *problem)
    block_size = int(block * p.n_var)
    var_feat, constr_feat, obj_features, edge_index, edge_attr, _ = p.get_data()
    initial_policy = initial_policy_dict[initial_policy](p)
    neighborhood_policy = neighborhood_policy_dict[neighborhood_policy](p)
    repair_policy = repair_policy_dict[repair_policy](p)

    # step 1: get the predicted solution and logits
    val_and_logit = np.zeros((len(output), 3))
    val_and_logit[:, 0] = np.arange(len(output))
    val_and_logit[:, 1] = np.where(output >= 0, 1, 0).reshape(-1)
    val_and_logit[:, 2] = np.abs(output).reshape(-1)

    if np.sum(val_and_logit[:, 1]) == 0:
        logger.warning("All predicted solutions are zero")
    elif np.sum(val_and_logit[:, 1]) == p.n_var:
        logger.warning(" All predicted solution are one")

    # step 2: get a feasible solution
    cur_val, cur_obj = initial_policy.get_feasible_solution(
        logger, val_and_logit, repair_policy, alpha=0.1, alpha_ub=1.0, max_size=p.n_var)
    best_sol, best_obj = cur_val, cur_obj
    round = 0
    while time.time() - start_time < time_limit:
        # step 3: generate the neighborhood of the solution
        neighborhoods = neighborhood_policy.get_neighborhood(
            block_size, cur_val, val_and_logit)

        # step 4: solve the subproblems in parallel
        pool = multiprocessing.Pool(processes=n_processes)
        results_ = []
        results = []
        if time.time() - start_time >= time_limit:
            break
        for neighborhood in neighborhoods:
            results_.append(pool.apply_async(p.subproblem_solve, (var_feat, constr_feat,
                                                                  obj_features, edge_index, edge_attr, cur_val, SEARCH_TIME_LIMIT, neighborhood, )))
        pool.close()

        for result in results_:
            if time.time() - start_time < time_limit:
                results.append(result.get())
            else:
                break
        pool.terminate()

        try:
            sol, obj, status = zip(*results)
            sol, obj = np.array(sol), np.array(obj)
            if p.obj_sense == Sense.MAXIMIZE:
                best_obj = np.max(obj[obj != -1])
                best_sol = sol[np.argmax(obj[obj != -1])]
            else:
                best_obj = np.min(obj[obj != -1])
                best_sol = sol[np.argmin(obj[obj != -1])]
            cur_val = best_sol
            logger.info(f"Round {round} neighborhood search best obj: {best_obj}")
        except:
            logger.info("Objectives are invalid")
            logger.debug(f"Neighborhood search status:\n{pformat(status)}")
            logger.debug(f"Neighborhood search objective:\n{pformat(obj)}")
            break

        if p.obj_sense == Sense.MAXIMIZE:
            if obj_limit is not None and best_obj >= obj_limit:
                break
        else:
            if obj_limit is not None and best_obj <= obj_limit:
                break

        # find the best crossover solution
        if crossover:
            if time.time() - start_time >= time_limit:
                break
            n_neighbor = neighborhoods.shape[0]
            pool = multiprocessing.Pool(processes=n_processes)
            cross_results_ = []
            cross_results = []
            for i in range(n_neighbor//2):
                cur_sol_1, cur_sol_2 = sol[2*i], sol[2*i+1]
                cur_obj_1, cur_obj_2 = obj[2*i], obj[2*i+1]
                cross_results_.append(pool.apply_async(cross_neighborhood, (p, repair_policy, block_size, neighborhoods[2*i], neighborhoods[2*i+1], cur_sol_1,
                                                                            cur_sol_2, cur_obj_1, cur_obj_2)))
            pool.close()
            for result in cross_results_:
                if time.time() - start_time < time_limit:
                    cross_results.append(result.get())
                else:
                    break
            pool.terminate()

            try:
                cross_sol, cross_obj, cross_status = zip(*cross_results)
                cross_sol, cross_obj = np.array(cross_sol), np.array(cross_obj)
                new_obj = np.hstack((obj, cross_obj))
                new_sol = np.vstack((sol, cross_sol))
                if p.obj_sense == Sense.MAXIMIZE:
                    best_obj = np.max(new_obj[new_obj != -1])
                    best_sol = new_sol[np.argmax(new_obj[new_obj != -1])]
                else:
                    best_obj = np.min(new_obj[new_obj != -1])
                    best_sol = new_sol[np.argmin(new_obj[new_obj != -1])]
                cur_val = best_sol
                logger.info(f"Round {round} crossover best obj: {best_obj}")
            except:
                logger.info("Cross objectives are invalid")
                logger.debug(f"Crossover status:\n{pformat(cross_status)}")
                logger.debug(f"Crossover objective:\n{pformat(cross_obj)}")
                break

            if p.obj_sense == Sense.MAXIMIZE:
                if obj_limit is not None and best_obj >= obj_limit:
                    break
            else:
                if obj_limit is not None and best_obj <= obj_limit:
                    break
        round += 1

    duration = time.time() - start_time
    logger.info(f"Neighborhood optimization time used: {duration:.2f} seconds")

    return best_sol, best_obj, duration
