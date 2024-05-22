# %%
import os
from pathlib import Path
import gurobipy as gp
from pyscipopt import Model
import time
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
import hydra
from omegaconf import DictConfig, OmegaConf
from utils import gurobi_env

def gurobi_solve(runs_dir: str,
                 lp_file: str,
                 nthread: int = 4,
                 time_limit: int = None,
                 gap_limit: float = None,
                 obj_limit: float = None
                 ) -> None:
    t0 = time.time()
    m = gp.read(lp_file, env=gurobi_env)
    name = lp_file.rsplit('.', 1)[0].rsplit('/', 1)[-1]
    log_file = os.path.join(runs_dir, f"{name}.log")
    sol_file = os.path.join(runs_dir, f"{name}.sol")
    m.setAttr("ModelName", name)
    m.setParam("LogFile", log_file)
    m.setParam("Threads", nthread)
    if time_limit is not None:
        m.setParam("TimeLimit", time_limit)
    if gap_limit is not None:
        m.setParam("MIPGap", gap_limit)
    if obj_limit is not None:
        m.setParam("BestObjStop", obj_limit)
    m.setParam("NonConvex", 2)
    m.setParam("MIPFocus", 3)
    m.update()
    m.optimize()
    duration = time.time() - t0
    obj = m.getAttr("ObjVal")
    m.write(sol_file)
    m.dispose()
    with open(log_file, 'a') as f:
        f.write(f"Total time: {duration:.2f}")
    return obj


def scip_solve(runs_dir: str,
               lp_file: str,
               time_limit: int = None,
               gap_limit: float = None,
               obj_limit: float = None
               ) -> None:
    t0 = time.time()
    m = Model()
    m.readProblem(lp_file)
    name = lp_file.rsplit('.', 1)[0].rsplit('/', 1)[-1]
    log_file = os.path.join(runs_dir, f"{name}.log")
    sol_file = os.path.join(runs_dir, f"{name}.sol")
    m.setLogfile(log_file)
    if time_limit is not None:
        m.setRealParam("limits/time", time_limit)
    if gap_limit is not None:
        m.setRealParam("limits/gap", gap_limit)
    if obj_limit is not None:
        m.setRealParam("limits/objectivestop", obj_limit)
    m.setIntParam("display/verblevel", 5)
    m.optimize()
    m.writeBestSol(filename=sol_file, write_zeros=True)
    duration = time.time() - t0
    obj = m.getObjVal()
    with open(log_file, 'a') as f:
        f.write(f"Total time: {duration:.2f}")
    return obj



@hydra.main(config_path="../config", config_name="baseline",version_base="1.3")
def main(cfg: DictConfig):
    proj_dir = Path(hydra.utils.get_original_cwd()).parent
    data_dir = proj_dir / "data" / "test" / "problem" / cfg.run.problem / cfg.run.difficulty
    QPs = list(f for f in data_dir.glob("*.lp"))
    QPs = sorted(QPs, key=lambda x: int(x.stem.split("_")[-1]))
    QPs = QPs[:cfg.run.n_instances]
    logger.debug(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Solving {len(QPs)} QPs with {cfg.run.solver}")

    for i, qp in enumerate(QPs):
        logger.info(f"\t{i+1}/{len(QPs)}: {qp.stem}")
        start = time.time()
        try:
            if cfg.run.solver == 'gurobi':
                obj = gurobi_solve(
                    runs_dir=".",
                    lp_file=str(qp),
                    nthread=cfg.gurobi.nthread,
                    time_limit=cfg.run.time_limit,
                    gap_limit=cfg.run.gap_limit,
                    obj_limit=cfg.gurobi.obj_limit,
                )
            elif cfg.run.solver =='scip':
                obj = scip_solve(
                    runs_dir=".",
                    lp_file=str(qp),
                    time_limit=cfg.run.time_limit,
                    gap_limit=cfg.run.gap_limit,
                    obj_limit=cfg.scip.obj_limit
                )
            else:
                raise ValueError(f"Solver {cfg.run.solver} not supported")
            duration = time.time() - start
            logger.info(f"\t\tFinished in {duration:.2f} seconds")
        except Exception as e:
            logger.error(e)

    logger.info("Finished")


if __name__ == "__main__":
    main()
