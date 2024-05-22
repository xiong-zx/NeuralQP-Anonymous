# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import datetime as dt
from collections import defaultdict
import pickle

def parse_gurobi(file: Path | str):
    with open(file, "r") as f:
        log_data = f.read()
    best_objective = None
    for line in log_data.splitlines():
        match = re.search(r'Best objective\s*([\d\.\+\-e]+)', line)
        if match:
            best_objective = float(match.group(1))
            break
    return best_objective


def parse_scip(file: Path | str):
    with open(file, "r") as f:
        log_data = f.read()
    primal = re.compile(r"^\s*Primal Bound\s*:\s*([\d\.e\+\-]+)")
    for line in log_data.splitlines(): 
        match = primal.match(line)
        if match:
            return float(match.group(1))
    return None

    
def parse_lns(log_file: Path | str):
    with open(log_file, "r") as f:
        log_data = f.read()

    pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]Round \d+ (neighborhood search|crossover) best obj: (-?\d+\.\d+)")

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
if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent
    runs_dir_base = project_root / 'runs' / 'testLNS'
    runs_dirs = [d for d in runs_dir_base.glob('*/*/*/*') if d.is_dir()]
    results = []
    for d in runs_dirs:
        problem, difficulty, method, time = d.parts[-4:]
        
        for log_file in tqdm(d.glob('*sol*.log'), desc=f"Parsing {problem}-{difficulty}-{method}-{time}"):

            if method.lower() == 'lns':
                _, obj_vals = parse_lns(log_file)
                obj = obj_vals[-1]
                problem_name,sol,run = log_file.stem.rsplit('_',2)
                sol = re.findall(r'\d+', sol)[0]
                run = re.findall(r'\d+', run)[0]
                with open(d / "main.log", "r") as log:
                    lines = log.readlines()
                for line in lines:
                    if "solver" in line:
                        solver = line.split()[1]
                        break
            elif method.lower() =='scip':
                obj = parse_scip(log_file)
                problem_name, sol = log_file.stem.rsplit('_',1)
                sol = re.findall(r'\d+', sol)[0]
                run = None
                solver = None
            elif method.lower() == 'gurobi':
                obj = parse_gurobi(log_file)
                problem_name, sol = log_file.stem.rsplit('_',1)
                sol = re.findall(r'\d+', sol)[0]
                run = None
                solver = None
                
            entry = (problem, difficulty, method, solver, problem_name, sol, run, obj) 
            results.append(entry)
                

    columns = ['problem', 'difficulty','method','solver', 'problem_name','sol', 'run', 'obj']
    df = pd.DataFrame(results, columns=columns)
    df.sort_values(by=['problem', 'difficulty','method', 'problem_name','sol', 'run'], inplace=True)
    df.to_csv(project_root / 'results' / 'LNS_results_raw.csv', index=False)
