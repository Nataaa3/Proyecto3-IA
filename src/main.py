"""
Programa principal para ejecutar el sistema de inferencia bayesiana.
"""
import os
import sys
from src.loader import load_graph, build_bayesian_network
from src.visualize import draw_graph, draw_conditional_probabilities
from src.inference import enumeration_ask

def run_example_from_class(data_dir="data"):
    """
    Ejecuta el ejemplo visto en clase (rain-maintenance-train-appointment).
    
    Args:
        data_dir (str): Directorio con los archivos CSV.
    """
    print("===== EJEMPLO DE CLASE =====")
    
    # Cargar la red bayesiana
    graph = load_graph(os.path.join(data_dir, "graph.csv"))
    bn = build_bayesian_network(data_dir)
    
    print("\nRed cargada correctamente:")
    print("Variables:", bn["variables"])
    print("Dependencias:", bn["parents"])
    
    # Visualizar la red
    print("\nVisualizando la red bayesiana...")
    draw_graph(graph, "Red Bayesiana - Ejemplo de Clase")
    
    # Caso de inferencia 1: P(appointment | rain=none)
    print("\nCASO 1: P(appointment | rain=none)")
    evidence = {"rain": "none"}
    query, trace = enumeration_ask("appointment", evidence, bn, debug=True)
    
    print("\nTraza de la inferencia:")
    print(trace)
    
    print("\nResultado:")
    for k, v in query.items():
        print(f"P(appointment={k} | rain=none) = {v:.4f}")
    
    # Caso de inferencia 2: P(train | appointment=attend)
    print("\n\nCASO 2: P(train | appointment=attend)")
    evidence = {"appointment": "attend"}
    query, trace = enumeration_ask("train", evidence, bn, debug=True)
    
    print("\nTraza de la inferencia:")
    print(trace)
    
    print("\nResultado:")
    for k, v in query.items():
        print(f"P(train={k} | appointment=attend) = {v:.4f}")
    
    # Caso de inferencia 3: P(maintenance | train=delayed, rain=heavy)
    print("\n\nCASO 3: P(maintenance | train=delayed, rain=heavy)")
    evidence = {"train": "delayed", "rain": "heavy"}
    query, trace = enumeration_ask("maintenance", evidence, bn, debug=True)
    
    print("\nTraza de la inferencia:")
    print(trace)
    
    print("\nResultado:")
    for k, v in query.items():
        print(f"P(maintenance={k} | train=delayed, rain=heavy) = {v:.4f}")

    # Caso de inferencia 4: P(appointment=miss | rain=light, maintenance=no, train=delayed)
    print("\n\nCASO 4: P(appointment=miss | rain=light, maintenance=no, train=delayed)")
    evidence = {"rain": "light", "maintenance": "no", "train": "delayed"}
    query, trace = enumeration_ask("appointment",evidence, bn, debug=True)

    print("\nTraza de la inferencia:")
    print(trace)

    print("\nResultado:")
    for k, v in query.items():
        print(f"P(appointment={k} | rain=light, maintenance=no, train=delayed) = {v:.4f}")

def run_custom_example(data_dir="data_custom"):
    """
    Ejecuta el ejemplo personalizado con al menos 6 variables.
    
    Args:
        data_dir (str): Directorio con los archivos CSV del ejemplo personalizado.
    """
    print("\n\n===== EJEMPLO PERSONALIZADO =====")
    
    # Verificar si existe el directorio
    if not os.path.exists(data_dir):
        print(f"ERROR: El directorio {data_dir} no existe.")
        print("Por favor, crea primero el directorio y los archivos CSV necesarios.")
        return
    
    # Cargar la red bayesiana personalizada
    try:
        graph = load_graph(os.path.join(data_dir, "graph.csv"))
        bn = build_bayesian_network(data_dir)
    except Exception as e:
        print(f"ERROR al cargar la red bayesiana: {e}")
        return
    
    print("\nRed personalizada cargada correctamente:")
    print("Variables:", bn["variables"])
    print("Dependencias:", bn["parents"])
    
    # Verificar que tenga al menos 6 variables
    if len(bn["variables"]) < 6:
        print(f"ADVERTENCIA: La red personalizada tiene {len(bn['variables'])} variables, pero se requieren al menos 6.")
    
    # Verificar que al menos una variable dependa de 3 otras
    has_three_dependencies = False
    for node, parents in bn["parents"].items():
        if len(parents) >= 3:
            has_three_dependencies = True
            print(f"La variable '{node}' depende de {len(parents)} variables: {', '.join(parents)}")
            break
    
    if not has_three_dependencies:
        print("ADVERTENCIA: Ninguna variable depende de al menos 3 otras variables.")
    
    # Visualizar la red
    print("\nVisualizando la red bayesiana personalizada...")
    draw_graph(graph, "Red Bayesiana - Ejemplo Personalizado")
    
    # Ejecutar casos de prueba
    print("\n=== CASOS DE PRUEBA PERSONALIZADOS ===")
    
    # Caso 1
    print("\nCASO 1:")
    evidence1 = {"tiempo": "lluvioso", "transporte": "auto"}
    query_var1 = "llegada"
    query1, trace1 = enumeration_ask(query_var1, evidence1, bn, debug=True)
    
    print(f"\nTraza de P({query_var1} | tiempo=lluvioso, transporte=auto):")
    print(trace1)
    
    print("\nResultado:")
    for k, v in query1.items():
        print(f"P({query_var1}={k} | tiempo=lluvioso, transporte=auto) = {v:.4f}")
    
    # Caso 2
    print("\n\nCASO 2:")
    evidence2 = {"llegada": "tarde", "examen": "aprobado"}
    query_var2 = "estudio"
    query2, trace2 = enumeration_ask(query_var2, evidence2, bn, debug=True)
    
    print(f"\nTraza de P({query_var2} | llegada=tarde, examen=aprobado):")
    print(trace2)
    
    print("\nResultado:")
    for k, v in query2.items():
        print(f"P({query_var2}={k} | llegada=tarde, examen=aprobado) = {v:.4f}")
    
    # Caso 3
    print("\n\nCASO 3:")
    evidence3 = {"estudio": "mucho", "descanso": "suficiente"}
    query_var3 = "concentracion"
    query3, trace3 = enumeration_ask(query_var3, evidence3, bn, debug=True)
    
    print(f"\nTraza de P({query_var3} | estudio=mucho, descanso=suficiente):")
    print(trace3)
    
    print("\nResultado:")
    for k, v in query3.items():
        print(f"P({query_var3}={k} | estudio=mucho, descanso=suficiente) = {v:.4f}")

def main():
    """
    Función principal que ejecuta los ejemplos.
    """
    print("Sistema de Inferencia por Enumeración en Redes Bayesianas")
    print("--------------------------------------------------------")
    
    # Ejecutar el ejemplo de clase
    run_example_from_class()
    
    # Ejecutar el ejemplo personalizado
    run_custom_example()

if __name__ == "__main__":
    main()