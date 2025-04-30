from itertools import product
import copy

def enumeration_ask(X, evidence, bn):
    Q = {}
    for xi in bn['values'][X]:
        extended_evidence = evidence.copy()
        extended_evidence[X] = xi
        Q[xi] = enumerate_all(bn['variables'], extended_evidence, bn)
    # Normalizar
    total = sum(Q.values())
    for key in Q:
        Q[key] /= total
    return Q

def enumerate_all(variables, evidence, bn):
    if not variables:
        return 1.0
    Y = variables[0]
    rest = variables[1:]
    parents = bn['parents'].get(Y, [])

    if Y in evidence:
        prob = probability(Y, evidence[Y], {p: evidence[p] for p in parents}, bn)
        return prob * enumerate_all(rest, evidence, bn)
    else:
        total = 0
        for y_val in bn['values'][Y]:
            extended = evidence.copy()
            extended[Y] = y_val
            prob = probability(Y, y_val, {p: extended[p] for p in parents}, bn)
            total += prob * enumerate_all(rest, extended, bn)
        return total

def probability(var, value, parent_vals, bn):
    table = bn['probabilities'][var]
    parents = bn['parents'].get(var, [])
    if not parents:
        return table[value]
    for row in table:
        if all(row[parent] == parent_vals[parent] for parent in parents):
            return row[value]
    raise ValueError(f"No match for {var} with {parent_vals}")
