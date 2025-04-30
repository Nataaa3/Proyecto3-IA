import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(edges):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, arrows=True, node_color='lightblue', node_size=2000)
    plt.title("Red Bayesiana")
    plt.show()
