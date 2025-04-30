"""
Módulo para realizar inferencia por enumeración en redes bayesianas.
"""
from itertools import product

def enumeration_ask(X, evidence, bn, debug=False):
    """
    Calcula la distribución de probabilidad de la variable X dada la evidencia.
    
    Args:
        X (str): Variable de consulta.
        evidence (dict): Evidencia observada.
        bn (dict): Estructura de la red bayesiana.
        debug (bool): Si es True, muestra los pasos intermedios del cálculo.
    
    Returns:
        dict: Distribución de probabilidad condicional P(X|evidence).
    """
    Q = {}
    trace = []
    
    if debug:
        trace.append(f"Calculando P({X} | {evidence})")
        trace.append("Orden de variables para enumeración: " + ", ".join(bn['variables']))
    
    # Determinar el orden topológico para las variables
    variables = list(bn['variables'])
    
    # Calcular P(X=xi | evidence) para cada valor de X
    for xi in bn['values'][X]:
        if debug:
            trace.append(f"\nCalculando P({X}={xi} | {evidence})")
        
        # Extender la evidencia con X=xi
        extended_evidence = evidence.copy()
        extended_evidence[X] = xi
        
        # Enumerar todas las variables no observadas
        result = enumerate_all(variables, extended_evidence, bn, debug, trace, indent=2)
        Q[xi] = result
        
        if debug:
            trace.append(f"  P({X}={xi} | {evidence}) = {result:.6f}")
    
    if debug:
        return Q, "\n".join(trace)
    return Q

def enumerate_all(variables, evidence, bn, debug=False, trace=None, indent=0):
    """
    Enumera todas las variables no observadas para calcular la probabilidad conjunta.
    
    Args:
        variables (list): Variables ordenadas topológicamente.
        evidence (dict): Evidencia observada.
        bn (dict): Estructura de la red bayesiana.
        debug (bool): Si es True, muestra los pasos intermedios del cálculo.
        trace (list): Lista para almacenar los pasos del cálculo.
        indent (int): Nivel de indentación para la traza.
    
    Returns:
        float: Probabilidad conjunta.
    """
    if not variables:
        if debug and trace is not None:
            trace.append(f"{' ' * indent}No quedan variables por enumerar, retorno 1.0")
        return 1.0
    
    Y = variables[0]
    rest = variables[1:]
    
    # Obtener los padres de Y
    parents = bn['parents'].get(Y, [])
    
    if Y in evidence:
        # Variable Y está en la evidencia
        y_val = evidence[Y]
        parent_vals = {p: evidence.get(p) for p in parents if p in evidence}
        
        prob = probability(Y, y_val, parent_vals, bn)
        
        if debug and trace is not None:
            parent_str = ", ".join([f"{p}={parent_vals.get(p)}" for p in parents if p in parent_vals])
            if parent_str:
                trace.append(f"{' ' * indent}P({Y}={y_val} | {parent_str}) = {prob:.6f}")
            else:
                trace.append(f"{' ' * indent}P({Y}={y_val}) = {prob:.6f}")
        
        result = prob * enumerate_all(rest, evidence, bn, debug, trace, indent + 2)
        
        if debug and trace is not None:
            trace.append(f"{' ' * indent}Resultado parcial para {Y}={y_val}: {result:.6f}")
        
        return result
    else:
        # Variable Y no está en la evidencia, sumar sobre todos los valores
        total = 0
        
        if debug and trace is not None:
            trace.append(f"{' ' * indent}Sumando sobre todos los valores de {Y}: {bn['values'][Y]}")

        for y_val in bn['values'][Y]:
            extended = evidence.copy()
            extended[Y] = y_val
            
            parent_vals = {p: extended.get(p) for p in parents if p in extended}
            prob = probability(Y, y_val, parent_vals, bn)
            
            if debug and trace is not None:
                parent_str = ", ".join([f"{p}={parent_vals.get(p)}" for p in parents if p in parent_vals])
                if parent_str:
                    trace.append(f"{' ' * indent}  P({Y}={y_val} | {parent_str}) = {prob:.6f}")
                else:
                    trace.append(f"{' ' * indent}  P({Y}={y_val}) = {prob:.6f}")
            
            term = prob * enumerate_all(rest, extended, bn, debug, trace, indent + 4)
            
            if debug and trace is not None:
                trace.append(f"{' ' * indent}  Término para {Y}={y_val}: {term:.6f}")
            
            total += term
        
        if debug and trace is not None:
            trace.append(f"{' ' * indent}Suma total para {Y}: {total:.6f}")
        
        return total

def probability(var, value, parent_vals, bn):
    """
    Obtiene la probabilidad de una variable dado sus padres.
    
    Args:
        var (str): Nombre de la variable.
        value (str): Valor de la variable.
        parent_vals (dict): Valores de los padres.
        bn (dict): Estructura de la red bayesiana.
    
    Returns:
        float: Probabilidad P(var=value | padres).
    """
    table = bn['probabilities'][var]
    parents = bn['parents'].get(var, [])
    
    if not parents:  # Sin padres
        return table[value]
    
    # Con padres: buscar en la tabla de probabilidad condicional
    for row in table:
        # Verificar si esta fila coincide con los valores de los padres
        matches = True
        for parent in parents:
            if parent in parent_vals and parent in row and row[parent] != parent_vals[parent]:
                matches = False
                break
        
        if matches:
            return row[value]
    
    raise ValueError(f"No se encontró probabilidad para {var}={value} con padres {parent_vals}")