"""
Given a solution, we want to test the effectiveness of lns method

Steps:
1. Load config file
2. Generate initial solution
3. Load initial solution
4. Compare lns with benchmark methods, i.e., Gurobi and SCIP 
    4.1 run lns method for mutiple times
    4.2 run Gurobi for multiple times
    4.3 run SCIP for multiple times
5. Analyze the results and draw conclusions
"""

import os
import time
import hydra
import hydra.core
import hydra.core.utils
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
from typing import Dict, List, Union
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from pyscipopt import Model
import gurobipy as gp
import numpy as np
import json

from NeighborhoodSearch import *
from Encoding import file2graph_HG
from utils import MyLogger, gurobi_env

    
def cross_neighborhood(p: Problem,
                       repair_policy: RepairPolicy,
                       block_size: int,
                       cross_time_limit: float,
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
                                                      new_sol, time_limit=cross_time_limit, neighborhood=new_neighborhood)
        return (new_sol, new_obj, status)
    
    
def gurobi_solve(runs_dir: str,
                 lp_file: str,
                 init_sol: Dict[str, Union[int, float]] = None,
                 sol_no: int = 0,
                 nthread: int = 4,
                 time_limit: int = None,
                 gap_limit: float = None,
                 obj_limit: float = None,
                 overwrite: bool = False,
                 *args, **kwargs
                 ):
    t0 = time.time()
    m = gp.read(lp_file,env=gurobi_env)
    name = lp_file.rsplit('.', 1)[0].rsplit('/', 1)[-1]
    log_file = os.path.join(runs_dir, f"{name}_sol{sol_no}.log")
    sol_file = os.path.join(runs_dir, "sol", f"{name}_sol{sol_no}.sol")
    os.makedirs(runs_dir, exist_ok=True)
    if os.path.exists(sol_file) and not overwrite:
        return
    else:
        os.makedirs(os.path.dirname(sol_file), exist_ok=True)
    m.setAttr("ModelName", name)
    if os.path.exists(log_file):
        os.remove(log_file)
    m.setParam("LogFile", log_file)
    m.setParam("LogToConsole", 0)
    m.setParam("Threads", nthread)
    if time_limit is not None:
        m.setParam("TimeLimit", time_limit)
    if gap_limit is not None:
        m.setParam("MIPGap", gap_limit)
    if obj_limit is not None:
        m.setParam("BestObjStop", obj_limit)
    m.setParam("NonConvex", 2)
    if init_sol is not None:
        for key, val in init_sol.items():
            m.getVarByName(key).start = val
    m.update()
    m.optimize()
    duration = time.time() - t0
    m.write(sol_file)
    obj = m.objVal
    m.dispose()
    with open(log_file, 'a') as f:
        f.write(f"Total time: {duration:.2f}")
    return obj, duration


def scip_solve(runs_dir: str,
               lp_file: str,
               init_sol: Dict[str, Union[int, float]] = None,
               sol_no: int = 0,
               time_limit: int = None,
               gap_limit: float = None,
               overwrite: bool = False,
               *args, **kwargs
               ):
    t0 = time.time()
    m = Model()
    m.readProblem(lp_file)
    name = lp_file.rsplit('.', 1)[0].rsplit('/', 1)[-1]
    log_file = os.path.join(runs_dir, f"{name}_sol{sol_no}.log")
    sol_file = os.path.join(runs_dir,"sol", f"{name}_sol{sol_no}.sol")
    os.makedirs(runs_dir, exist_ok=True)
    if os.path.exists(sol_file) and not overwrite:
        return
    else:
        os.makedirs(os.path.dirname(sol_file), exist_ok=True)
    if os.path.exists(log_file):
        os.remove(log_file)
    m.setLogfile(log_file)
    if time_limit is not None:
        m.setRealParam("limits/time", time_limit)
    if gap_limit is not None:
        m.setRealParam("limits/gap", gap_limit)
    m.setIntParam("display/verblevel", 5)
    if init_sol is not None:
        vars = {var.name: var for var in m.getVars()}
        sol = m.createSol()
        for var_name, value in init_sol.items():
            var = vars.get(var_name)
            if var is not None:
                m.setSolVal(sol, var, value)
            else:
                raise ValueError(f"Variable {var_name} not found in the model.")
        added = m.addSol(sol)
        if not added:
            raise ValueError("Failed to add initial solution.")
            
        
    m.optimize()
    m.writeBestSol(filename=sol_file, write_zeros=True)
    duration = time.time() - t0
    with open(log_file, 'a') as f:
        f.write(f"Total time: {duration:.2f}")
    # return best obj and duration
    return m.getObjVal(), duration


def LNS_solve(
    config: DictConfig,
    lp_file: str,
    init_sol: Dict[str, Union[int, float]],
    lns_logger: logging.Logger,
    solver: str,
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
    assert config.lns.encoding == "HG"
    project_dir = Path(hydra.utils.get_original_cwd()).parent
    graph_dir = project_dir / "data" / "test" / "HG" / f"{config.run.problem}_graph" / f"{config.run.difficulty}"
    graph_dir.mkdir(exist_ok=True, parents=True)
    _, problem_data = file2graph_HG(path_to_file=lp_file, graph_dir=graph_dir)
    var_features, _, _, constr_features, obj_features, edge_features, edges = problem_data
    model = gp.read(lp_file, env=gurobi_env)
    init_sol_array = np.array([init_sol[var.varName] for var in model.getVars()])
    problem_data = var_features, constr_features, obj_features, edge_features, edges, init_sol_array
    assert len(problem_data) == 6, f"Problem data should have 6 elements, but got {len(problem_data)}"
    lns_logger.log(f"time_limit: {time_limit}")
    lns_logger.log(f"n_processes: {n_processes}")
    lns_logger.log(f"block_size: {block}")
    lns_logger.log(f"repair_policy: {repair_policy}")
    lns_logger.log(f"neighborhood_policy: {neighborhood_policy}")
    lns_logger.log(f"crossover: {crossover}")
    p = Problem(solver, *problem_data)
    block_size = int(block * p.n_var)
    var_feat, constr_feat, obj_features, edge_index, edge_attr, cur_val = p.get_data()
    neighborhood_policy = neighborhood_policy_dict[neighborhood_policy](p)
    repair_policy = repair_policy_dict[repair_policy](p)
    best_sol, best_obj = p.cur_val, p.cur_obj    
    start_time = time.time()
    round = 0
    while time.time() - start_time < time_limit:
        # step 3: generate the neighborhood of the solution
        neighborhoods = neighborhood_policy.get_neighborhood(block_size, cur_val)

        # step 4: solve the subproblems in parallel
        pool = multiprocessing.Pool(processes=n_processes)
        results_ = []
        results = []
        if time.time() - start_time >= time_limit:
            break
        for neighborhood in neighborhoods:
            results_.append(pool.apply_async(p.subproblem_solve, (var_feat, constr_feat,
                                                                  obj_features, edge_index, edge_attr, cur_val, config.lns.time_limit.search_time_limit, neighborhood, )))
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
            lns_logger.log(f"Round {round} neighborhood search best obj: {best_obj}")
        except:
            lns_logger.log("Objectives are invalid")
            lns_logger.log(f"Neighborhood search status:\n{pformat(status)}")
            lns_logger.log(f"Neighborhood search objective:\n{pformat(obj)}")
            break

        if p.obj_sense == Sense.MAXIMIZE:
            if obj_limit is not None and best_obj >= obj_limit:
                break
        else:
            if obj_limit is not None and best_obj <= obj_limit:
                break

        # find the best crossover solution
        if time.time() - start_time >= time_limit:
            break
        if crossover:
            n_neighbor = neighborhoods.shape[0]
            pool = multiprocessing.Pool(processes=n_processes)
            cross_results_ = []
            cross_results = []
            for i in range(n_neighbor//2):
                cur_sol_1, cur_sol_2 = sol[2*i], sol[2*i+1]
                cur_obj_1, cur_obj_2 = obj[2*i], obj[2*i+1]
                cross_results_.append(pool.apply_async(cross_neighborhood, (p, repair_policy, block_size, config.lns.time_limit.cross_time_limit,neighborhoods[2*i], neighborhoods[2*i+1], cur_sol_1,
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
                lns_logger.log(f"Round {round} crossover best obj: {best_obj}")
            except:
                lns_logger.log("Cross objectives are invalid")
                lns_logger.log(f"Cross objectives:\n{pformat(cross_obj)}")
                lns_logger.log(f"Crossover status:\n{pformat(cross_status)}")
                break

            if p.obj_sense == Sense.MAXIMIZE:
                if obj_limit is not None and best_obj >= obj_limit:
                    break
            else:
                if obj_limit is not None and best_obj <= obj_limit:
                    break
        round += 1

    duration = time.time() - start_time
    lns_logger.log(f"Neighborhood optimization time used: {duration:.2f} seconds")

    return best_obj, duration

def generate_init_sol(
    runs_dir: Union[str, Path],
    lp_file: Union[str, Path],
    n_solutions: int)-> List[Dict[str, Union[int, float]]]:
    lp_file, runs_dir = Path(lp_file), Path(runs_dir)
    log_file = runs_dir / f"{lp_file.stem}.log"
    runs_dir.mkdir(exist_ok=True, parents=True)
    m = gp.read(str(lp_file), env=gurobi_env)
    m.setParam("Threads", 4)
    m.setParam("SolutionLimit", n_solutions)
    m.setParam("NonConvex", 2)
    m.setParam("LogFile", str(log_file))
    m.update()
    m.optimize()
    sols = []
    for i in range(n_solutions):
        m.setParam("SolutionNumber", i)
        var_dict = {}
        for var in m.getVars():
            var_dict[var.varName] = var.x
        sols.append(var_dict)
    return sols

@hydra.main(config_path='../config', config_name='LNS',version_base="1.3")
def main(cfg: DictConfig):

    
    np.random.seed(cfg.run.seed)
    random.seed(cfg.run.seed)
    
    logger.debug(f"Config:\n{OmegaConf.to_yaml(cfg,resolve=True)}")
    project_dir = Path(hydra.utils.get_original_cwd()).parent
    data_dir = project_dir / 'data' / 'test'/ 'problem' / cfg.run.problem / str(cfg.run.difficulty)
    init_sol_dir = project_dir / 'data' / 'test' / 'init_sol' / cfg.run.problem / str(cfg.run.difficulty)
    init_sol_dir.mkdir(exist_ok=True, parents=True)
    runs_dir = Path().cwd()
    
    # load problems
    for lp_file in data_dir.glob('*.lp'):
        logger.info(f"Begin {lp_file.name}")
        # get initial solutions
        init_sol_file = init_sol_dir / lp_file.name.replace('.lp', '.json')
        if not init_sol_file.exists():
            logger.warning(f"\t{init_sol_file} not found")
            logger.info(f"\tGenerating {cfg.run.n_solutions} solutions")
            init_sols = generate_init_sol(runs_dir / 'init_sol', lp_file, cfg.run.n_solutions)
        else:
            with open(init_sol_file, 'r') as f:
                init_sols = json.load(f)
            logger.info(f"\tFound {len(init_sols)}/{cfg.run.n_solutions} initial solutions")
            if len(init_sols) < cfg.run.n_solutions:
                logger.warning(f"\tNot enough initial solutions, generating {cfg.run.n_solutions - len(init_sols)}")
                new_sols = generate_init_sol(runs_dir / 'init_sol', lp_file, cfg.run.n_solutions - len(init_sols))
                init_sols.extend(new_sols)
                with open(init_sol_file, 'w') as f:
                    json.dump(init_sols, f, indent=4)
            else:
                init_sols = init_sols[:cfg.run.n_solutions]

        logger.info(f"\t{len(init_sols)} initial solutions loaded")
        
        # run methods
        for i, init_sol in enumerate(init_sols):
            logger.info(f"\t\t{i+1}/{len(init_sols)} initial solution begin")
            method = cfg.run.method

            if method.lower() == 'lns':
                # logger.error("\t\t\t\tLNS method is not implemented yet")
                for j in range(cfg.lns.ntimes):
                    logger.info(f"\t\t\t{j+1}/{cfg.lns.ntimes} lns running for solution {i+1}")
                    lns_logger = MyLogger( f"{lp_file.stem}_sol{i+1}_run{j+1}.log")

                    obj,duration = LNS_solve(
                        config=cfg,
                        lp_file=str(lp_file),
                        init_sol=init_sol,
                        sol_no=i+1,
                        lns_logger=lns_logger,
                        solver=cfg.lns.solver,  # default is 4 threads
                        repair_policy=cfg.lns.policies.repair_policy,
                        neighborhood_policy=cfg.lns.policies.neighborhood_policy,
                        time_limit=cfg.run.time_limit,
                        obj_limit=cfg.lns.obj_limit,
                        n_processes=cfg.lns.nprocesses,
                        block=cfg.lns.block_size,
                        crossover=cfg.lns.policies.crossover,
                    )
                    lns_logger.dispose()

            elif method.lower() == 'gurobi':
                logger.info(f"\t\t\tgurobi running for solution {i+1}")
                obj,duration =gurobi_solve(
                    runs_dir=runs_dir,
                    lp_file=str(lp_file), 
                    init_sol=init_sol, 
                    sol_no=i+1, 
                    nthread=cfg.gurobi.nthread, 
                    time_limit=cfg.run.time_limit, 
                    gap_limit=cfg.gurobi.gap_limit, 
                    obj_limit=cfg.gurobi.obj_limit, 
                    overwrite=cfg.run.overwrite, 
                )

            elif method.lower() == 'scip':

                logger.info(f"\t\t\tscip running for solution {i+1}")
                obj, duration = scip_solve(
                    runs_dir=runs_dir,
                    lp_file=str(lp_file),
                    init_sol=init_sol,
                    sol_no=i+1,
                    time_limit=cfg.run.time_limit,
                    gap_limit=cfg.scip.gap_limit,
                    overwrite=cfg.run.overwrite,
                )
            else:
                logger.error(f"\t\t\t{method} invalid")
                obj, duration = -1, -1
        logger.info(f"\t{lp_file.name} finished")
    logger.info(f"{cfg.run.problem} - {cfg.run.difficulty} finished")
        
if __name__ == '__main__':
    main()