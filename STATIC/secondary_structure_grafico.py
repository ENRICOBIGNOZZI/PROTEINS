import networkx as nx
import matplotlib.pyplot as plt

# Definire i nodi (strutture secondarie)
nodes = ['H1', 'E1', 'H2', 'E2', 'H3', 'E3']

# Definire le relazioni causali (direzione e valori TE tra donatore e accettore)
edges = [
    ('H1', 'E1', 0.006),   # H1 dona a E1
    ('E1', 'H2', 0.010),   # E1 dona a H2
    ('H2', 'E2', 0.008),   # H2 dona a E2
    ('E2', 'H3', 0.009),   # E2 dona a H3
    ('H3', 'E3', 0.006),   # H3 dona a E3
]

# Creare un grafo direzionale
G = nx.DiGraph()

# Aggiungere i nodi
G.add_nodes_from(nodes)

# Aggiungere gli archi (con direzione e peso)
for edge in edges:
    G.add_edge(edge[0], edge[1], weight=edge[2])

# Definire la posizione dei nodi per una buona visualizzazione
pos = nx.spring_layout(G)

# Disegnare i nodi del grafo
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=100)

# Disegnare le etichette dei nodi
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

# Disegnare gli archi (frecce direzionali) con frecce più grandi e visibili
nx.draw_networkx_edges(
    G, pos, edgelist=G.edges(data=True), arrowstyle='-|>', arrowsize=30, edge_color='black', width=[d['weight']*10 for (u, v, d) in G.edges(data=True)]
)

# Aggiungere etichette per i pesi (valori di TE)
edge_labels = {(u, v): f'{d["weight"]:.3f}' for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Titolo del grafo
plt.title('Causal Graph of Secondary Structure (Transfer Entropy Direction)')
plt.axis('off')  # Nascondere assi
plt.show()
