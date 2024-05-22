# %%
import gurobipy as gp
import numpy as np
import pandas as pd
import argparse
import pickle
import os
import pprint
from utils import gurobi_env as env

__vtype__ = {'C': [1, 0, 0],
              'B': [0, 1, 0],
              'I': [0, 0, 1]}

__ctype__ = {'<': [1, 0, 0],
              '>': [0, 1, 0],
              '=': [0, 0, 1]}

__obj__ = {1: [1, 0],
            -1: [0, 1]}


def parse_solution(file) -> dict:
    """
    description:
        parse solution file
    parameters:
        file: path to the solution file
    return:
        solution_dict: a dictionary of variable-value pairs
    """
    solution_dict = {}
    if not os.path.exists(file):
        # print(f"Solution file {file} not found")
        raise FileNotFoundError(f"Solution file {file} not found")
    with open(file, "r") as f:
        for line in f:
            if not line.startswith('#') and not line.startswith('obj'):
                line = line.strip().split()
                solution_dict[line[0]] = float(line[1])
    return solution_dict


import os
import numpy as np
import gurobipy as gp

def get_solution_values(path_to_model, sol_dict) -> tuple:
    """
    Get solution values of full length

    Args:
    path_to_model (str): Path to the .lp file
    sol_dict (dict): A dictionary of variable-value pairs, returned by parse_solution

    Returns:
    tuple: sol_values (list): A list of variable values
           var_types (numpy.array): An array of variable types
    """
    if not os.path.exists(path_to_model):
        # If the file does not exist, raise FileNotFoundError
        raise FileNotFoundError(f"{path_to_model} file not found")
    # Read the model from the specified .lp file
    model = gp.read(path_to_model, env)
    # Get all decision variables of the model
    vars = model.getVars()
    sol_values = []
    for var in vars:
        # Check if the variable exists in the solution dictionary
        if var.VarName in sol_dict:
            sol_values.append(sol_dict[var.VarName])
        else:
            sol_values.append(0)
            sol_dict[var.VarName] = 0
    # Get the types of all variables in the model
    var_types = np.array(model.getAttr("VType"))
    return sol_values, var_types


def file2graph_HG(path_to_file: str, graph_dir: str) -> tuple:
    """
    parameters:
        file: path to the .lp file
        prob_type: 

    return:
        qp_name: name of the QP
        graph: a list of hypergraph data

    ======================
    encoding description:
    ======================
    total dimension: 7+1+1+4+2+1=16
    variable side (7):
    variable type: C,B,I
    variable lower bound
    variable upper bound
    variable lower bound infinity
    variable upper bound infinity
    constant 1 (1)
    square (1)
    constraint side (4):
    constraint sense: <, >, =
    constraint right-hand-side
    objective side (2):
    objective sense: min, max
    random number (1)        
    """

    if os.path.exists(path_to_file):
        pass
    else:
        raise FileNotFoundError(f"{path_to_file} file not found")

    file = path_to_file.split("/")[-1]
    file_base = file.rsplit('.', 1)[0]
    os.makedirs(graph_dir, exist_ok=True)
    if os.path.exists(os.path.join(graph_dir, f"{file_base}.pkl")):
        # print(f"File {file_base}.pkl already exists")
        name, data = pickle.load(
            open(os.path.join(graph_dir, f"{file_base}.pkl"), "rb"))
        if len(data) == 7:
            return (name, data)
        # else:
            # print(f"File {file_base}.pkl is invalid")
            # print("Re-encoding")

    model = gp.read(path_to_file, env)
    # print(f"Encoding {file_base}.lp")

    edge_features = []
    edges = []
    constr_features = []
    obj_features = []

    row = [var.VarName for var in model.getVars()] + ["1", "sqr"] + [constr.ConstrName for constr in model.getConstrs()] + \
        [Qconstr.QCName for Qconstr in model.getQConstrs()] + ["obj"]
    row_index = pd.Index(row)
    obj_index = row_index.get_loc("obj")
    one_index = row_index.get_loc("1")
    sqr_index = row_index.get_loc("sqr")

    # variable features
    var_types = np.array(model.getAttr("VType"))
    var_lb = np.array(model.getAttr("LB"))
    var_ub = np.array(model.getAttr("UB"))
    var_lb_inf = np.isinf(var_lb).astype(int)
    var_ub_inf = np.isinf(var_ub).astype(int)
    var_lb = np.where(np.isinf(var_lb), 0, var_lb)
    var_ub = np.where(np.isinf(var_ub), 0, var_ub)
    var_types = np.array([__vtype__[vtype] for vtype in var_types]).T

    edge_count = 0

    # constraint features
    for constr in model.getConstrs():
        b = constr.getAttr("RHS")
        sense = constr.getAttr("Sense")
        constr_features.append(__ctype__[sense] + [b])
        line_expr = model.getRow(constr)
        for i in range(line_expr.size()):
            var = line_expr.getVar(i)
            coeff = line_expr.getCoeff(i)
            constr_index = row_index.get_loc(constr.ConstrName)
            var_index = row_index.get_loc(var.VarName)
            edges.append([constr_index, var_index, one_index])
            edge_features.append(coeff)
            edge_count += 1

    for Qconstr in model.getQConstrs():
        b = Qconstr.getAttr("QCRHS")
        sense = Qconstr.getAttr("QCSense")
        constr_features.append(__ctype__[sense] + [b])
        quad_expr = model.getQCRow(Qconstr)
        quad_le = quad_expr.getLinExpr()
        for i in range(quad_le.size()):
            var = quad_le.getVar(i)
            coeff = quad_le.getCoeff(i)
            constr_index = row_index.get_loc(Qconstr.QCName)
            var_index = row_index.get_loc(var.VarName)
            edges.append([constr_index, var_index, one_index])
            edge_features.append(coeff)
            edge_count += 1

        for i in range(quad_expr.size()):
            var1 = quad_expr.getVar1(i)
            var2 = quad_expr.getVar2(i)
            coeff = quad_expr.getCoeff(i)
            var1_index = row_index.get_loc(var1.VarName)
            var2_index = row_index.get_loc(var2.VarName)
            qconstr_index = row_index.get_loc(Qconstr.QCName)
            edges.append([qconstr_index, var1_index,
                         sqr_index if var1_index == var2_index else var2_index])
            edge_features.append(coeff)
            edge_count += 1

    # objective features
    obj = model.getObjective()
    obj_features.append(__obj__[model.getAttr("ModelSense")])
    # judge whether the objective is quadratic
    if isinstance(obj, gp.QuadExpr):
        obj_le = obj.getLinExpr()
        for i in range(obj_le.size()):
            var = obj_le.getVar(i)
            coeff = obj_le.getCoeff(i)
            var_index = row_index.get_loc(var.VarName)
            edges.append([obj_index, var_index, one_index])
            edge_features.append(coeff)
            edge_count += 1

        for i in range(obj.size()):
            var1 = obj.getVar1(i)
            var2 = obj.getVar2(i)
            coeff = obj.getCoeff(i)
            var1_index = row_index.get_loc(var1.VarName)
            var2_index = row_index.get_loc(var2.VarName)
            edges.append(
                [obj_index, var1_index, sqr_index if var1_index == var2_index else var2_index])
            edge_features.append(coeff)
            edge_count += 1

    else:
        for i in range(obj.size()):
            var = obj.getVar(i)
            coeff = obj.getCoeff(i)
            var_index = row_index.get_loc(var.VarName)
            edges.append([obj_index, var_index, one_index])
            edge_features.append(coeff)
            edge_count += 1

    # Padding features to the same dimension
    constr_features = np.array(constr_features).reshape(-1, 4)
    obj_features = np.array(obj_features).reshape(-1, 2)
    var_features = np.vstack((var_types, var_lb, var_ub, var_lb_inf, var_ub_inf, np.zeros(
        (8, var_types.shape[1])), np.random.rand(1, var_types.shape[1]))).T
    constr_features = np.hstack((np.zeros((constr_features.shape[0], 9)), constr_features, np.zeros(
        (constr_features.shape[0], 2)), np.random.rand(constr_features.shape[0], 1)))
    obj_features = np.hstack((np.zeros(
        (obj_features.shape[0], 13)), obj_features, np.random.rand(obj_features.shape[0], 1)))

    one_features, sqr_features = np.zeros((1, 16)), np.zeros((1, 16))
    one_features[0, -1], sqr_features[0, -
                                      1] = np.random.rand(), np.random.rand()
    one_features[0, 7] = 1
    sqr_features[0, 8] = 1
    edge_features = np.array(edge_features)
    edges = np.array(edges, dtype=int)

    # print(f"Dumping {file_base}.pkl")
    with open(os.path.join(graph_dir, file.replace(".lp", ".pkl")), "wb") as f:
        pickle.dump((file_base, [var_features, one_features, sqr_features, constr_features,
                    obj_features, edge_features, edges]), f)
    # print(f"Finished encoding {file_base}")

    return (file_base, [var_features, one_features, sqr_features, constr_features, obj_features, edge_features, edges])


def initialize_dir(problem: str,difficulty: str) -> tuple:
    """
    Initialize the directory for the given problem.
    
    Args:
        problem: the name of the problem
        difficulty: difficulty of the problem
        encoding: the encoding method
    
    Returns:
        _root: the root directory of the project
        sol_dir: the directory of the solution files
        input_dir: the directory of the input .lp files
        output_dir: the directory of the output files for neural network training
        graph_dir: the directory of the graph data files
    """

    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    sol_dir = os.path.join(_root, "data", "train","problem", problem,difficulty)
    input_dir = os.path.join(_root, "data",  "train","problem", problem,difficulty)
    output_dir = os.path.join(_root, "data", "train", "HG", f"{problem}_train", difficulty)

    if not os.path.exists(input_dir):
        raise ValueError(f"Unsupported problem type {problem}")

    graph_dir = os.path.join(
        _root, "data", "train", "HG", f"{problem}_graph",difficulty)

    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    return (_root, sol_dir, input_dir, output_dir, graph_dir)


import os
import numpy as np
import pickle

def graph_encoding(problem: str, difficulty: str) -> None:
    """
    Function to generate graph data for the given problem and difficulty level.
    It processes input files, converts them into graph data, and saves the encoded data as pickle files.

    Args:
    - problem (str): The name of the problem.
    - difficulty (str): The difficulty level of the problem.

    Returns: None
    """

    # Initialize directories for input/output
    _root, sol_dir, input_dir, output_dir, graph_dir = initialize_dir(
        problem, difficulty)

    count = 0
    failures = {}
    # Iterate through input directory
    for file in os.listdir(input_dir):
        if file.endswith(".lp"):
            try:
                # Process the file and generate graph data
                file_base = file.rsplit('.', 1)[0]
                name, graph_data = file2graph_HG(os.path.join(
                    input_dir, file), graph_dir)
                # Parse solution file and get solution values
                sol_dict = parse_solution(
                    os.path.join(sol_dir, f"{file_base}.sol"))
                sol_values, var_types = get_solution_values(
                    os.path.join(input_dir, file), sol_dict)
                # Append solution values to graph data and save as pickle file
                graph_data.append(np.array(sol_values))
                pickle.dump((name, graph_data), open(
                    os.path.join(output_dir, f"{file_base}.pkl"), "wb"))
                print(f"Finished {file}\n")
                count += 1
            except Exception as e:
                # Handle file processing failures
                failures[file] = e
                print(f"Failed {file}\n")
                continue

    # Print summary of processing results
    print(f"Finished {count} {problem} files in total")
    print(f"Failed {len(failures)} {problem} files in total")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', '-p', type=str, required=True,help='problem type')
    parser.add_argument('--difficulty','-d',type=str,required=True,help='difficulty level')
    return parser.parse_args()


# %%
if __name__ == '__main__':
    args = parse_args()
    graph_encoding(**vars(args))
