# %%
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re
import datetime as dt
import pickle
from typing import Union
from omegaconf import OmegaConf

def parse_gurobi(file: Union[str, Path]):
    with open(file, "r") as f:
        log_data = f.read()
    pattern1 = re.compile(
        r"^[H*]\s+\d+\s+\d+\s+(-?[\d\.e\+\-]+)\s+(-?[\d\.e\+\-]+)\s+[\d\.]+%.*?(\d+s)"
    )
    pattern2 = re.compile(
        r"^\s*\d+\s+\d+\s+-?[\d\.e\+\-]+\s+\d+\s+\d+\s+(-?[\d\.e\+\-]+)\s+(-?[\d\.e\+\-]+)\s+[\d\.]+%.*?(\d+s)"
    )
    times = []
    incumbent_values = []
    best_bd_values = []

    for line in log_data.splitlines():
        match1 = pattern1.match(line)
        match2 = pattern2.match(line)
        if match1:
            incumbent_value = float(match1.group(1))
            best_bd_value = float(match1.group(2))
            time_value = int(match1.group(3)[:-1])

            times.append(time_value)
            incumbent_values.append(incumbent_value)
            best_bd_values.append(best_bd_value)
        elif match2:
            incumbent_value = float(match2.group(1))
            best_bd_value = float(match2.group(2))
            time_value = int(match2.group(3)[:-1])

            times.append(time_value)
            incumbent_values.append(incumbent_value)
            best_bd_values.append(incumbent_value)

    return times, incumbent_values


def parse_scip(file: Union[str, Path]):
    with open(file, "r") as f:
        log_data = f.read()

    pattern = re.compile(r"^\s*([a-zA-Z]*\s*\d+(\.\d+)?s)\s*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*([\d\.e\+\-]+)\s*\|\s*([\d\.e\+\-]+)\s*\|")
    solving_time = re.compile(r"^\s*Solving Time \(sec\)\s*:\s*([\d\.e\+\-]+)")
    primal = re.compile(r"^\s*Primal Bound\s*:\s*([\d\.e\+\-]+)")
    dual = re.compile(r"^\s*Dual Bound\s*:\s*([\d\.e\+\-]+)")
    times = []
    primal_bounds = []
    dual_bounds = []

    for line in log_data.splitlines():
        match = pattern.match(line)
        primal_match = primal.match(line)
        dual_match = dual.match(line)
        solving_time_match = solving_time.match(line)
        if match:
            time_str = match.group(1)
            time_value = float(re.sub(r"^[a-zA-Z]*", "", time_str[:-1])) 
            dual_bound = float(match.group(3))
            primal_bound = float(match.group(4))

            times.append(time_value)
            dual_bounds.append(dual_bound)
            primal_bounds.append(primal_bound)
        elif solving_time_match:
            solving_time_value = float(solving_time_match.group(1))
            times.append(solving_time_value)
        elif dual_match:
            dual_bounds.append(float(dual_match.group(1)))
        elif primal_match:
            primal_bounds.append(float(primal_match.group(1)))
        else:
            continue
            
    return times, primal_bounds


def parse_lns(log_file: Union[str, Path]):
    with open(log_file, "r") as f:
        log_data = f.read()

    pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\].*Round \d+ (neighborhood search|crossover) best obj: (-?\d+\.\d+)")

    start_time_pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]")
    start_time_match = start_time_pattern.search(log_data)
    start_time_str = start_time_match.group(1)
    start_time = dt.datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S.%f")

    times = []
    objs = []

    for line in log_data.splitlines():
        match = pattern.match(line)
        if match:
            time_str = match.group(1)
            current_time = dt.datetime.strptime(
                time_str, "%Y-%m-%d %H:%M:%S.%f")
            elapsed_time = (current_time - start_time).total_seconds()

            obj_value = float(match.group(3))

            times.append(elapsed_time)
            objs.append(obj_value)
    
    return times, objs

# %%
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    runs_dir_base = project_root / "runs" / "main"
    runs_dirs = [d for d in runs_dir_base.glob('*/*/*') if d.is_dir()]
    results = []
    data_file = project_root / "results" / "data.pkl"
        
    data_dict = {}
    for run_dir in runs_dirs:
        method = run_dir.parts[-2]
        problem, difficulty = run_dir.parts[-3].split("-")
        log_files = [f for f in run_dir.glob("*.log") if f.name != "main.log"]
        config_file = run_dir / ".hydra" / "config.yaml"
        config = OmegaConf.load(config_file)
        for log_file in tqdm(log_files, desc=f"Parsing {problem}-{difficulty}-{method}"):
            try:
                
                solver = block_size = run = train_on = None
                if method == "lns":
                    timestamps, obj_vals = parse_lns(log_file)
                    problem_name, run = log_file.stem.rsplit("_", 1)
                    if problem_name not in data_dict:
                        data_dict[problem_name] = {}
                    if method not in data_dict[problem_name]:
                        data_dict[problem_name][method] = []
                    solver, block_size, train_on = config.lns.solver, config.lns.block_size, config.run.train.difficulty
                    if config.lns.obj_limit == None:
                        limit = 'time'
                        data_dict[problem_name][method].append({"timestamps": timestamps, "obj_vals": obj_vals, "solver": solver, "block_size": block_size, "run": run, "train_on": train_on})
                    else:
                        limit = 'obj'
                elif method == "gurobi" or method == "scip":
                    timestamps, obj_vals = parse_gurobi(log_file) if method == "gurobi" else parse_scip(log_file)
                    problem_name = log_file.stem
                    #data_dict[problem_name][method] = {"timestamps": timestamps, "obj_vals": obj_vals}
                    if config.gurobi.obj_limit == None:
                        data_dict[problem_name][method] = {"timestamps": timestamps, "obj_vals": obj_vals}
                        limit = 'time'
                    else:
                        limit = 'obj'
                else:
                    raise ValueError(f"Unknown method: {method}")
                entry = (problem, difficulty, problem_name, limit, method, train_on, solver, block_size, run, obj_vals[-1], timestamps[-1])
                results.append(entry)
                
            except Exception as e:
                print(f"Error when parsing {log_file}: {e}")
                continue
    
    with open(data_file, "wb") as f:
        pickle.dump(data_dict, f)
    columns = ["problem", "difficulty", "problem_name", "limit", "method","train_on", "solver", "block_size", "run", "obj", "runtime"]
    df = pd.DataFrame(results, columns=columns)
    df.sort_values(by=["problem", "difficulty", "problem_name", "limit","method","train_on", "solver","block_size", "run"], inplace=True)
    df.to_csv(project_root / "results" / "Main_results.csv", index=False)
# %%