"""
Test neural prediction

1. Train (or load) a neural prediction model
2. (Optional) Tune the hyperparameters of the model
3. Test the model on a test set
    - Load the test set
    - Use the model to predict the output for problem in the test set
    - Compute the metrics
        - F1 score
        - Objective value
        - Constraint violation
    - Use the predicted solution to get feasible solution
    - Use Gurobi to get the first feasible solution

"""
import torch
import numpy as np
import pandas as pd
import gurobipy as gp
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from typing import Dict, List, Union,Tuple
import datetime as dt

from NeuralPrediction import *
from Encoding import file2graph_HG
from utils import gurobi_env
from NeighborhoodSearch import Problem, VariableRelaxationPolicy,CautiousRepairPolicy

def evaluate_prediction(
        var_feat: np.ndarray,  # Variable types and bounds
        constr_feat: np.ndarray,  # Constraint types and right-hand side
        obj_feat: np.ndarray,  # Objective function types
        edge_index: np.ndarray,  # Index of hyperedges corresponding to edges
        edge_attr: np.ndarray,  # Attributes of hyperedges corresponding to edge_features
        cur_sol: np.ndarray,  # Current solution
):
    """
    Evaluate the solution of the optimization problem using Gurobi.

    Parameters:
        logger (Logger): Logger
        var_feat (np.ndarray): Variable types and bounds
        constr_feat (np.ndarray): Constraint types and right-hand side
        obj_feat (np.ndarray): Objective function types
        edge_index (np.ndarray): Index of hyperedges corresponding to edges
        edge_attr (np.ndarray): Attributes of hyperedges corresponding to edge_features
        cur_sol (np.ndarray): Current solution
        time_limit (float): Time limit for optimization
        file_name (str): Name of the problem file
        
    Returns:
        violation_dict (dict): Constraint violation for each constraint
        obj_val (float): Objective function value
    """
    num_variables = var_feat.shape[0]
    num_constraints = constr_feat.shape[0]
    constr_dict = {i:0.0 for i in range(num_constraints)}
    for j in range(len(edge_index)):
        i = edge_index[j][0] - num_variables - 2
        if 0 <= i < num_constraints:
            if edge_index[j][2] == num_variables:
                constr_dict[i] += cur_sol[edge_index[j][1]] * edge_attr[j]
            elif edge_index[j][2] == num_variables + 1:
                constr_dict[i] += cur_sol[edge_index[j][1]] * cur_sol[edge_index[j][1]] * edge_attr[j]
            else:
                constr_dict[i] += cur_sol[edge_index[j][1]] * cur_sol[edge_index[j][2]] * edge_attr[j]
    
    violation_dict = {}
    for k in range(num_constraints):
            if (constr_feat[k][9:12] == [1, 0, 0]).all():  # <=
                violation_dict[k] = constr_dict[k] - constr_feat[k][12]
            elif (constr_feat[k][9:12] == [0, 1, 0]).all():  # >=
                violation_dict[k] = constr_feat[k][12] - constr_dict[k]
            else:
                violation_dict[k] = np.abs(constr_dict[k] - constr_feat[k][12])
    
    # evaluate objective function
    obj_val = 0.0
    for j in range(len(edge_index)):
        if edge_index[j][0] == num_constraints + num_variables + 2:
            if edge_index[j][2] == num_variables:
                obj_val += cur_sol[edge_index[j][1]] * edge_attr[j]
            elif edge_index[j][2] == num_variables + 1:
                obj_val += cur_sol[edge_index[j][1]] * cur_sol[edge_index[j][1]] * edge_attr[j]
            else:
                obj_val += cur_sol[edge_index[j][1]] * cur_sol[edge_index[j][2]] * edge_attr[j]
    
    return violation_dict, obj_val

def evaluate_initial_solution(
    name: str,
    solver: str,
    prediction: np.ndarray,
    var_feat: np.ndarray,
    constr_feat: np.ndarray,
    obj_feat: np.ndarray,
    edge_attr: np.ndarray,
    edge_index: np.ndarray,
    cur_sol: np.ndarray = None,
    cur_obj: float = None,
):
    problem = Problem(solver, var_feat, constr_feat, obj_feat, edge_attr, edge_index, cur_sol, cur_obj)
    initial_policy = VariableRelaxationPolicy(problem)
    repair_policy = CautiousRepairPolicy(problem)
    val_and_logit = np.zeros((len(prediction), 3))
    val_and_logit[:, 0] = np.arange(len(prediction))
    val_and_logit[:, 1] = np.where(prediction >= 0, 1, 0).reshape(-1)
    val_and_logit[:, 2] = np.abs(prediction).reshape(-1)
    start = dt.datetime.now()
    init_sol, init_obj = initial_policy.get_feasible_solution(
        logger=logging.getLogger(name=name),
        val_and_logit=val_and_logit,
        repair_policy=repair_policy,
        alpha=0.02,
        alpha_step=0.02,
        alpha_ub=1.0,
        max_size=10000
    )
    end = dt.datetime.now()
    
    return init_obj, (end - start).total_seconds()

def gurobi_solve(
        lp_file: Union[str, Path],
        log_file: Union[str, Path]
):
    model = gp.read(str(lp_file), env=gurobi_env)
    if log_file is not None:
        model.setParam('LogFile', str(log_file))
    model.setParam('SolutionLimit', 2) # the first initial solution tends to be 0, which is useless in practice
    model.setParam('TimeLimit', 10)
    model.setParam('NonConvex', 2)
    model.optimize()
    obj, runtime = model.ObjVal, model.Runtime
    model.dispose()
    return obj, runtime

def evaluate(graph_file: Union[str, Path], lp_file: Union[str, Path], prediction: np.ndarray) -> Dict:
    """
    Description:
        Evaluate the predicted solution using Gurobi. Compute the metrics
            - Objective value
            - Constraint violation

    Args:
        lp_file: The path to the LP file.
        graph_file: The path to the graph file.
        prediction: The predicted solution.

    Returns:
        A dictionary containing the evaluation results.
    """
    prediction_binary = (prediction > 0).astype(int)
    # load problem data
    with open(graph_file, "rb") as f:
        name, graph_data = pickle.load(f)
    var_feat,_,_, constr_feat, obj_feat, edge_attr, edge_index = graph_data
    violation_dict, obj_val = evaluate_prediction(var_feat, constr_feat, obj_feat, edge_index, edge_attr, prediction_binary)
    violation_count = sum(1 for v in violation_dict.values() if v > 0)
    lns_init_sol, lns_init_time = evaluate_initial_solution(
        name=name,
        solver="gurobi",
        prediction=prediction,
        var_feat=var_feat,
        constr_feat=constr_feat,
        obj_feat=obj_feat,
        edge_attr=edge_attr,
        edge_index=edge_index,
        cur_sol=prediction_binary,
        cur_obj=obj_val,
    )
    gurobi_log_dir = Path("./gurobi")
    gurobi_log_dir.mkdir(exist_ok=True)
    gurobi_obj, runtime = gurobi_solve(lp_file=lp_file, log_file= gurobi_log_dir / f"{name}.log")

    return {
        "name": name,
        "prediction_max": np.amax(prediction),
        "prediction_min": np.amin(prediction),
        "prediction_mean": np.mean(prediction),
        "prediction_std": np.std(prediction),
        "pos_percent": len(np.where(prediction_binary == 1)[0]) / len(prediction_binary),
        "neg_percent": len(np.where(prediction_binary == 0)[0]) / len(prediction_binary),
        "predicted_obj": obj_val,
        "max_violation": max(violation_dict.values()),
        "violation_count": violation_count,
        "violation_percent": violation_count / len(violation_dict),
        "init_obj": lns_init_sol,
        "init_runtime": lns_init_time,
        "gurobi_obj": gurobi_obj,
        "gurobi_runtime": runtime,
    }


def predict(cfg: DictConfig):
    
    # initialize model
    model = load_nn_model(cfg)
    device = cfg.run.device
    logger.info(f"{cfg.model.model} model initialized on {device}")
    # load test data
    test_problem, test_difficulty = str(cfg.run.test.problem), str(cfg.run.test.difficulty)
    test_data = load_test_data(test_problem, test_difficulty)
    logger.info(f"Test data loaded for {test_problem} {test_difficulty}")
    
    predictions: List[Tuple[str, np.ndarray]] = []
    for name, G in test_data:
        G = G.to(device)
        output = model(G)
        output = output[:len(G.opt_sol)]
        predictions.append((name, output.detach().cpu().numpy()))
    logger.info(f"Predictions for {len(predictions)} problems finished.")
    return predictions

def load_nn_model(cfg: DictConfig) -> HyperGraphModel:
    model = HyperGraphModel(cfg.model).to(cfg.run.device)
    global project_root
    project_root = Path(hydra.utils.get_original_cwd()).parent
    model_dir = project_root / "models"
    for m in model_dir.glob("*.pkl"):
        problem, difficulty = m.stem.split("_")
        if problem == cfg.run.train.problem and difficulty == str(cfg.run.train.difficulty):
            # load pickled model in .pkl file
            model.load_state_dict(torch.load(m))
            logger.debug(f"Model loaded from {m}")
            break

    return model

    
def load_test_data(
    problem: str,
    difficulty: str
) -> List[Tuple[str, HGData]]:
    logger.debug(f"Loading test data for {problem} {difficulty}")
    project_root = Path(hydra.utils.get_original_cwd()).parent
    processed_data_dir = project_root / "data" / "test" / "HG" / f"{problem}_graph" / difficulty
    if not processed_data_dir.exists():
        logger.debug(f"Processed data for {problem}-{difficulty} not found.")
        raw_data_dir = project_root / "data" /"test" / "problem" / problem / difficulty
        raw_data_files = [f for f in raw_data_dir.iterdir() if f.suffix == ".lp"]
        logger.debug(f"Encoding {len(raw_data_files)} files.")
        for f in raw_data_files:
            file2graph_HG(path_to_file=str(f),graph_dir=str(processed_data_dir))
    processed_data_files = [f for f in processed_data_dir.iterdir() if f.suffix == ".pkl"]
    graphs = []
    for f in processed_data_files:
        with open(f, "rb") as f:
            name,graph_data = pickle.load(f)
            graph_data.append(np.zeros(len(graph_data[0])))
            G = HGData(*graph_data)
            graphs.append((name, G))
    return graphs

@hydra.main(config_path='../config', config_name='NP',version_base="1.3")
def main(cfg: DictConfig):
    
    np.random.seed(cfg.run.seed)
    torch.manual_seed(cfg.run.seed)
    torch.cuda.manual_seed_all(cfg.run.seed)
    project_root = Path(hydra.utils.get_original_cwd()).parent
    logger.debug(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    predictions: List[Tuple[str, np.ndarray]] = predict(cfg)
    
    results = []
    for name, prediction in predictions:
        graph_file = project_root / "data" / "test" / "HG" / f"{cfg.run.test.problem}_graph" / str(cfg.run.test.difficulty) / f"{name}.pkl"
        lp_file = project_root / "data" / "test" / "problem" / cfg.run.test.problem / str(cfg.run.test.difficulty) / f"{name}.lp"
        logger.info(f"\tEvaluating {name}")
        result = evaluate(graph_file=graph_file, lp_file=lp_file, prediction=prediction)

        results.append(result)
    results_df = pd.DataFrame(results)
    results_df.sort_values(by='name', inplace=True)
    results_df.to_csv(f"results.csv")
    logger.info("Finished.")
    
if __name__ == '__main__':
    main()