# %%
from pathlib import Path
import numpy as np
import pandas as pd
import time
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

import NeuralPrediction
from NeighborhoodSearch import optimize
from Encoding import file2graph_HG
from utils import MyLogger


def _encoding(input_file, data_dir, encoding):
    name, graph_data = file2graph_HG(input_file, data_dir)
    n_vars = len(graph_data[0])
    sol_values = np.zeros(n_vars)
    graph_data.append(sol_values)
    return name, graph_data

@hydra.main(config_path="../config", config_name="main", version_base="1.3")
def main(cfg: DictConfig):
    """
    1. Encode the problem instances
    2. Neural prediction
    4. Get initial solution
    5. Neighborhood search
    """
    np.random.seed(cfg.run.seed)
    torch.manual_seed(cfg.run.seed)
    torch.cuda.manual_seed_all(cfg.run.seed)
    logger.debug(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    proj_dir = Path(hydra.utils.get_original_cwd()).parent
    T0 = time.time()
    problem, difficulty, encoding, model = str(cfg.run.test.problem), str(cfg.run.test.difficulty), str(cfg.run.encoding), str(cfg.model.model)
    input_dir = proj_dir / "data" / "test" / "problem" / problem / difficulty
    data_dir = proj_dir / "data" / "test" / encoding / f'{problem}_graph' / difficulty
    data_dir.mkdir(parents=True, exist_ok=True)
    results = []
    # load trained model
    device = torch.device(cfg.run.device)
    nnmodel = NeuralPrediction.all_models[cfg.model.model](cfg.model).to(device)
    model_path = proj_dir / "models" / f"{cfg.run.train.problem}_{cfg.run.train.difficulty}.pkl"
    logger.info(f"Loading model from {str(model_path)}")
    nnmodel.load_state_dict(torch.load(model_path))
    nnmodel.to(device)
    nnmodel.eval()
    logger.info(f"Model initialized on {device}.")
    instances = list(input_dir.glob("*.lp"))
    instances = sorted(instances, key=lambda x: int(x.stem.split("_")[-1]))
    instances = instances[:cfg.run.n_instances]
    logger.info(f"Testing on {len(instances)} instances.")
    for n, input_file in enumerate(instances):
        logger.info(f"No.{n} {input_file.name}")
        # step 1: encode the problem instances
        logger.info(f"\tEncoding {input_file.name}")
        name, graph_data = _encoding(str(input_file), str(data_dir), encoding)
        logger.info(f"\tOptimizing {name}")
        var_features, _, _, constr_features, obj_features, edge_features, edges, _ = graph_data
        problem_data = [var_features, constr_features, obj_features, edge_features, edges]
        for i in range(cfg.lns.n_times):
            logger.info(f"\t\t{i+1}th run begin")
            # step 2: neural prediction
            G = NeuralPrediction.all_data[encoding](*graph_data)
            G = G.to(device)
            output = nnmodel(G)
            output = output[:len(G.opt_sol)].cpu().detach()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            repair_policy, neighorhood_policy, crossover = cfg.lns.policies.repair_policy, cfg.lns.policies.neighborhood_policy, cfg.lns.policies.crossover
            start_time = time.time()

            try:
                sub_logger = MyLogger(f"{name}_{i}.log")
                cur_sol, cur_obj, _ = optimize(logger=sub_logger,
                                            solver=cfg.lns.solver,
                                            problem=problem_data,
                                            output=output.numpy(),
                                            initial_policy="variable_relaxation",
                                            repair_policy=repair_policy,
                                            neighborhood_policy=neighorhood_policy,
                                            time_limit=cfg.lns.time_limit,
                                            obj_limit=cfg.lns.obj_limit,
                                            n_processes=cfg.lns.n_processes,
                                            block=cfg.lns.block_size,
                                            crossover=crossover
                                            )
                sub_logger.dispose()
                duration = time.time() - start_time
                logger.info(f"\t\t\tFinished in {duration:.2f} seconds.")
                logger.info(f"\t\t\tIncumbent objective: {cur_obj:.2f}")
                results.append([name, i,cfg.run.train.difficulty, cfg.lns.solver, cfg.lns.block_size, cur_obj, duration])
            except:
                duration = time.time() - start_time
                logger.error(f"\t\t\tFailed in {duration:.2f} seconds.")
                
            # sub_logger = MyLogger(f"{name}_{i}.log")
            # cur_sol, cur_obj, _ = optimize(logger=sub_logger,
            #                             solver=cfg.lns.solver,
            #                             problem=problem_data,
            #                             output=output.numpy(),
            #                             initial_policy="variable_relaxation",
            #                             repair_policy=repair_policy,
            #                             neighborhood_policy=neighorhood_policy,
            #                             time_limit=cfg.lns.time_limit,
            #                             obj_limit=cfg.lns.obj_limit,
            #                             n_processes=cfg.lns.n_processes,
            #                             block=cfg.lns.block_size,
            #                             crossover=crossover
            #                             )
            # sub_logger.dispose()
            # duration = time.time() - start_time
            # logger.info(f"\t\t\tFinished in {duration:.2f} seconds.")
            # logger.info(f"\t\t\tIncumbent objective: {cur_obj:.2f}")
            # results.append([name, i,cfg.run.train.difficulty, cfg.lns.solver, cfg.lns.block_size, cur_obj, duration])
                
        logger.info(f"\t{name} finished.")
    logger.info(f"All {n+1} instances finished.")
    results = pd.DataFrame(results, columns=["name", "run", "train", "solver", "block_size", "obj", "time"])
    results.to_csv("results.csv", index=False)

# %%
if __name__ == "__main__":
    main()
