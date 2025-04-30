"""
Módulo para visualizar la red bayesiana.
"""
import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(edges, title="Red Bayesiana"):
    """
    Dibuja el grafo de la red bayesiana.
    
    Args:
        edges (list): Lista de tuplas (origen, destino) representando las aristas.
        title (str): Título del gráfico.
    """
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)  # Posición fija para reproducibilidad
    
    # Dibujar nodos
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000, alpha=0.8)
    
    # Dibujar aristas
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, width=2)
    
    # Etiquetas de nodos
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')
    
    plt.title(title, fontsize=16)
    plt.axis('off')  # Sin ejes
    plt.tight_layout()
    plt.show()

def draw_conditional_probabilities(bn, node):
    """
    Visualiza las probabilidades condicionales de un nodo.
    
    Args:
        bn (dict): Estructura de la red bayesiana.
        node (str): Nombre del nodo a visualizar.
    """
    parents = bn['parents'].get(node, [])
    probs = bn['probabilities'][node]
    
    if not parents:  # Nodo sin padres
        plt.figure(figsize=(8, 6))
        values = list(probs.keys())
        probabilities = list(probs.values())
        
        plt.bar(values, probabilities, color='skyblue')
        plt.ylim(0, 1)
        plt.title(f'Probabilidad de {node}', fontsize=14)
        plt.ylabel('Probabilidad')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
    else:  # Nodo con padres
        if len(parents) == 1:  # Un solo padre
            parent = parents[0]
            
            # Agrupar por valor del padre
            parent_values = set()
            node_values = bn['values'][node]
            
            for row in probs:
                parent_values.add(row[parent])
            
            parent_values = sorted(list(parent_values))
            
            # Organizar datos para visualización
            data = {}
            for node_val in node_values:
                data[node_val] = []
                for parent_val in parent_values:
                    for row in probs:
                        if row[parent] == parent_val:
                            data[node_val].append(row[node_val])
                            break
            
            # Crear gráfico
            bar_width = 0.8 / len(node_values)
            
            plt.figure(figsize=(10, 6))
            
            for i, node_val in enumerate(node_values):
                positions = [p + i * bar_width for p in range(len(parent_values))]
                plt.bar(positions, data[node_val], width=bar_width, 
                        label=f'{node}={node_val}')
            
            plt.xticks([r + bar_width * (len(node_values) - 1) / 2 for r in range(len(parent_values))], 
                      [f'{parent}={val}' for val in parent_values])
            plt.ylim(0, 1)
            plt.title(f'Probabilidad de {node} dado {parent}', fontsize=14)
            plt.ylabel('Probabilidad')
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
        else:  # Múltiples padres (mostrar como tabla)
            print(f"Probabilidades para {node} con padres {', '.join(parents)}:")
            for row in probs:
                parent_vals = ", ".join([f"{p}={row[p]}" for p in parents])
                node_vals = ", ".join([f"{val}={row[val]}" for val in bn['values'][node]])
                print(f"  {parent_vals} -> {node_vals}")
    
    plt.tight_layout()
    plt.show()