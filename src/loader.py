# src/loader.py
import csv
import os

def load_structure(graph_path):
    parents = {}
    with open(graph_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            child = row['destination']
            parent = row['origin']
            if child not in parents:
                parents[child] = []
            parents[child].append(parent)
    return parents

def load_cpt(varname, data_dir):
    path = os.path.join(data_dir, f"{varname}.csv")
    with open(path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        table = [dict(zip(header, row)) for row in reader]
    return table

def get_variables(graph_path):
    variables = set()
    with open(graph_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            variables.add(row['origin'])
            variables.add(row['destination'])
    return list(variables)


def load_network(data_dir):
    graph_path = os.path.join(data_dir, "graph.csv")
    parents = load_structure(graph_path)
    variables = get_variables(graph_path)
    cpts = {}
    for var in variables:
        cpts[var] = load_cpt(var, data_dir)
    return variables, parents, cpts
