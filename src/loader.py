"""
Módulo para cargar la estructura de la red bayesiana y sus tablas de probabilidad.
"""
import csv
import os
import pandas as pd

def load_graph(graph_path):
    """
    Carga la estructura del grafo desde un archivo CSV.
    
    Args:
        graph_path (str): Ruta al archivo CSV con la estructura.
        
    Returns:
        list: Lista de tuplas (origen, destino) representando las aristas del grafo.
    """
    edges = []
    with open(graph_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            edges.append((row['origin'], row['destination']))
    return edges

def load_structure(graph_path):
    """
    Carga la estructura de padres para cada nodo.
    
    Args:
        graph_path (str): Ruta al archivo CSV con la estructura.
        
    Returns:
        dict: Diccionario donde las claves son nodos y los valores son listas de padres.
    """
    parents = {}
    edges = load_graph(graph_path)
    for origin, destination in edges:
        if destination not in parents:
            parents[destination] = []
        parents[destination].append(origin)
    return parents

def get_variables(graph_path):
    """
    Obtiene todas las variables en el grafo.
    
    Args:
        graph_path (str): Ruta al archivo CSV con la estructura.
        
    Returns:
        list: Lista de nombres de variables.
    """
    variables = set()
    with open(graph_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            variables.add(row['origin'])
            variables.add(row['destination'])
    return list(variables)

def load_probabilities(data_dir):
    """
    Carga todas las tablas de probabilidad de la red bayesiana.
    
    Args:
        data_dir (str): Directorio que contiene los archivos CSV de probabilidades.
        
    Returns:
        dict: Diccionario con las tablas de probabilidad para cada variable.
    """
    graph_path = os.path.join(data_dir, "graph.csv")
    variables = get_variables(graph_path)
    parents = load_structure(graph_path)
    
    probabilities = {}
    for var in variables:
        var_path = os.path.join(data_dir, f"{var}.csv")
        if os.path.exists(var_path):
            df = pd.read_csv(var_path)
            
            # Procesar tabla sin padres (solo probabilidades)
            if len(df.columns) == 2 and df.columns[0] == 'value' and df.columns[1] == 'prob':
                probabilities[var] = {row['value']: float(row['prob']) for _, row in df.iterrows() if pd.notna(row['value'])}
            # Procesar tabla con padres
            else:
                var_parents = parents.get(var, [])
                prob_table = []
                
                for _, row in df.iterrows():
                    if any(pd.isna(val) for val in row.values):
                        continue
                        
                    table_row = {}
                    for parent in var_parents:
                        if parent in df.columns:
                            table_row[parent] = row[parent]
                    
                    # Añadir las probabilidades para cada valor posible de la variable
                    for col in df.columns:
                        if col not in var_parents:
                            table_row[col] = float(row[col])
                    
                    prob_table.append(table_row)
                
                probabilities[var] = prob_table
    
    return probabilities

def get_variable_values(bn):
    """
    Extrae todos los valores posibles para cada variable de las tablas de probabilidad.
    
    Args:
        bn (dict): Diccionario con la estructura de la red bayesiana.
        
    Returns:
        dict: Diccionario con los valores posibles para cada variable.
    """
    values = {}
    for var, prob in bn['probabilities'].items():
        if isinstance(prob, dict):  # Variables sin padres
            values[var] = list(prob.keys())
        else:  # Variables con padres
            # Identifica columnas que no son padres
            possible_values = []
            parents = bn.get('parents', {}).get(var, [])
            
            # Extraer los valores del primer registro de la tabla
            first_row = prob[0]
            for key in first_row:
                if key not in parents:
                    possible_values.append(key)
            
            values[var] = possible_values
    
    return values

def build_bayesian_network(data_dir):
    """
    Construye la representación completa de la red bayesiana.
    
    Args:
        data_dir (str): Directorio con los archivos CSV.
        
    Returns:
        dict: Diccionario con la estructura completa de la red bayesiana.
    """
    graph_path = os.path.join(data_dir, "graph.csv")
    variables = get_variables(graph_path)
    parents = load_structure(graph_path)
    probabilities = load_probabilities(data_dir)
    
    bn = {
        "variables": variables,
        "parents": parents,
        "probabilities": probabilities
    }
    
    # Obtener los valores posibles para cada variable
    bn["values"] = {}
    for var in variables:
        bn["values"][var] = []
        if var in probabilities:
            if isinstance(probabilities[var], dict):  # Sin padres
                bn["values"][var] = list(probabilities[var].keys())
            else:  # Con padres
                # Encuentra las columnas que no son padres
                var_parents = parents.get(var, [])
                for col in probabilities[var][0].keys():
                    if col not in var_parents:
                        bn["values"][var].append(col)
    
    return bn