from src.loader import load_graph, load_probabilities
from src.visualize import draw_graph
from src.inference import enumeration_ask

def build_bn(graph, probabilities):
    parents = {}
    variables = set()
    values = {}

    for origin, dest in graph:
        parents.setdefault(dest, []).append(origin)
        variables.add(origin)
        variables.add(dest)
    for var in probabilities:
        if isinstance(probabilities[var], dict):
            values[var] = list(probabilities[var].keys())
        else:
            values[var] = list(probabilities[var][0].keys() - set(parents.get(var, [])))
            variables.add(var)

    return {
        "variables": list(variables),
        "parents": parents,
        "probabilities": probabilities,
        "values": values
    }

def main():
    graph = load_graph("data/graph.csv")
    probabilities = load_probabilities("data")

    bn = build_bn(graph, probabilities)

    print("Red cargada correctamente:")
    print("Variables:", bn["variables"])
    print("Dependencias:", bn["parents"])

    draw_graph(graph)

    query = enumeration_ask("appointment", {"rain": "none"}, bn)
    print("\nResultado de inferencia: P(appointment | rain=none)")
    for k, v in query.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
