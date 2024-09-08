from Downlaod_data import PDBProcessor
from Visualize import Visualize
from funtions import plot_comparison
from matrix import GraphMatrixAnalyzer
import numpy as np
from causal_indicators_advances import TimeCorrelation, TransferEntropy, TimeResponse, CorrelationMatrixOperations, ResidualAnalysis
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.lines as mlines
import networkx as nx
from community import community_louvain

# Initialize PDBProcessor
pdb_processor = PDBProcessor(pdb_id="2m10")
pdb_processor.download_pdb()
pdb_processor.load_structure()

# Extract data
df1 = pdb_processor.secondary_structure()
df = pdb_processor.extract_atom_data()
df = df[df['Model ID'] == 0]
df = df[df['Atom Name'] == 'CA']

df = df.reset_index(drop=True)
concatenated_df = pd.concat([df1['Secondary Structure'], df], axis=1)
df = concatenated_df.dropna().reset_index(drop=True)

# Initialize Visualize and analyze graph
visualizer = Visualize(df)
raggio = visualizer.calculate_and_print_average_distance()
G = visualizer.create_and_print_graph(truncated=True, radius=8.0, plot=False, peso=20)

# Analisi del cammino più corto
shortest_path = nx.shortest_path(G, source=20, target=75)
print(f"Cammino più corto tra 21 e 76: {shortest_path}")
print(f"Lunghezza del cammino: {len(shortest_path) - 1}")

# Tutti i cammini semplici
all_paths = list(nx.all_simple_paths(G, source=20, target=75))
print(f"Numero di cammini semplici tra 21 e 76: {len(all_paths)}")

# Analisi dei percorsi pesati
def plot_weighted_paths(adjacency_matrix):
    G = nx.from_numpy_array(adjacency_matrix)
    num_nodes = G.number_of_nodes()
    
    weighted_paths = np.zeros(num_nodes)
    
    for source in range(num_nodes):
        for target in range(num_nodes):
            if source != target:
                paths = list(nx.all_simple_paths(G, source, target))
                for path in paths:
                    path_length = len(path) - 1
                    weighted_paths[source] += 1 / path_length
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(num_nodes), weighted_paths)
    plt.xlabel('Node')
    plt.ylabel('Weighted Number of Paths')
    plt.title('Number of Paths Connecting to Each Node (Weighted by Time)')
    plt.xticks(range(0, num_nodes, 5))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Initialize GraphMatrixAnalyzer
analyzer = GraphMatrixAnalyzer(G)

# Ottieni la matrice di adiacenza
adjacency_matrix = analyzer.get_adjacency_matrix()
plot_weighted_paths(adjacency_matrix)

# Calcola la matrice di Kirchhoff
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()

# Calcola autovalori e autovettori
autovalori, autovettori = np.linalg.eigh(kirchhoff_matrix)

# Nuove analisi avanzate
def advanced_network_analysis(adjacency_matrix, df, autovettori, autovalori):
    G = nx.from_numpy_array(adjacency_matrix)
    for (u, v, d) in G.edges(data=True):
        d['capacity'] = 1.0
    
    # 1. Analisi del flusso massimo
    def max_flow_analysis(G, top_k=5):
        betweenness = nx.betweenness_centrality(G)
        top_nodes = sorted(betweenness, key=betweenness.get, reverse=True)[:top_k]
        
        for i in range(len(top_nodes)):
            for j in range(i+1, len(top_nodes)):
                paths = list(nx.all_shortest_paths(G, top_nodes[i], top_nodes[j]))
                print(f"Percorsi più corti tra nodo {top_nodes[i]} e {top_nodes[j]}: {len(paths)}")
    
    # 2. Analisi delle comunità
    def community_analysis(G):
        partition = community_louvain.best_partition(G)
        nx.set_node_attributes(G, partition, 'community')
        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, node_color=list(partition.values()), with_labels=True, node_size=50)
        plt.title("Comunità nella rete proteica")
        plt.show()
    
    # 3. Analisi della robustezza
    def robustness_analysis(G, n_removals=10):
        initial_connectivity = nx.average_clustering(G)
        for _ in range(n_removals):
            if len(G) > 0:
                node_to_remove = max(G.degree(), key=lambda x: x[1])
                G.remove_node(node_to_remove[0])
                new_connectivity = nx.average_clustering(G)
                print(f"Rimosso nodo {node_to_remove[0]}, nuova connettività: {new_connectivity}")
            else:
                print("Tutti i nodi sono stati rimossi.")
                break
    
    # 4. Correlazione con dati strutturali
    def structure_correlation(G, df):
        sec_struct = df['Secondary Structure'].tolist()
        nx.set_node_attributes(G, dict(enumerate(sec_struct)), 'sec_struct')
        
        struct_colors = {'H': 'red', 'E': 'yellow', 'C': 'green'}
        node_colors = [struct_colors.get(G.nodes[n]['sec_struct'], 'blue') for n in G.nodes()]
        
        plt.figure(figsize=(12, 8))
        nx.draw(G, node_color=node_colors, with_labels=True, node_size=50)
        plt.title("Rete proteica colorata per struttura secondaria")
        plt.show()
    
    # 5. Analisi dinamica
    def dynamic_analysis(autovettori, autovalori, t_range=np.linspace(0, 1, 100)):
        dynamic_centrality = np.zeros((len(autovettori), len(t_range)))
        for i, t in enumerate(t_range):
            exp_lambda_t = np.exp(-autovalori * t)
            dynamic_centrality[:, i] = np.sum(autovettori**2 * exp_lambda_t, axis=1)
        
        plt.figure(figsize=(12, 8))
        for i in range(len(autovettori)):
            plt.plot(t_range, dynamic_centrality[i, :], label=f'Node {i}')
        plt.title("Centralità dinamica dei nodi")
        plt.xlabel("Tempo")
        plt.ylabel("Centralità")
        plt.legend()
        plt.show()
    
    # Esegui le analisi
    print("Analisi del flusso massimo:")
    max_flow_analysis(G)
    
    print("\nAnalisi delle comunità:")
    community_analysis(G)
    
    print("\nAnalisi della robustezza:")
    robustness_analysis(G.copy())
    
    print("\nCorrelazione con dati strutturali:")
    structure_correlation(G, df)
    
    print("\nAnalisi dinamica:")
    dynamic_analysis(autovettori, autovalori)

# Esegui l'analisi avanzata
advanced_network_analysis(adjacency_matrix, df, autovettori, autovalori)

# Analisi dei percorsi
def analyze_paths(adjacency_matrix):
    G = nx.from_numpy_array(adjacency_matrix)
    
    betweenness = nx.betweenness_centrality(G)
    
    top_nodes = sorted(betweenness, key=betweenness.get, reverse=True)[:10]
    
    print("Top 10 nodi per betweenness centrality:")
    for node in top_nodes:
        print(f"Nodo {node}: {betweenness[node]:.4f}")
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color='lightblue', 
            node_size=[v * 3000 for v in betweenness.values()],
            with_labels=True, font_size=8)
    plt.title("Rete proteica con dimensione dei nodi basata sulla betweenness centrality")
    plt.show()

    for i in range(min(5, len(top_nodes))):
        for j in range(i+1, min(5, len(top_nodes))):
            path = nx.shortest_path(G, source=top_nodes[i], target=top_nodes[j])
            print(f"Percorso più breve tra nodo {top_nodes[i]} e nodo {top_nodes[j]}: {path}")

# Esegui l'analisi dei percorsi
analyze_paths(adjacency_matrix)

# Inizializza ResidualAnalysis
residual_analysis = ResidualAnalysis(u=autovettori, lambdas=autovalori, mu=1, sec_struct_data=df)

# Perform correlation analysis
residual_analysis.plot_mfpt_matrix(adjacency_matrix)

# Analyze the Mean First Passage Time
residual_analysis.analyze_mfpt(adjacency_matrix)

# Calcola la distanza tra due residui
def calcola_distanza(residuo1, residuo2, df):
    coord1 = df.loc[df['Residue ID'] == residuo1, ['X', 'Y', 'Z']].values[0]
    coord2 = df.loc[df['Residue ID'] == residuo2, ['X', 'Y', 'Z']].values[0]
    distanza = np.linalg.norm(coord1 - coord2)
    return distanza

# Calcola la distanza tra il residuo 20 e il 40
distanza_20_40 = calcola_distanza(20, 40, df)
print(f"Distanza tra il residuo 20 e il 40: {distanza_20_40:.2f}")

# Analisi delle correlazioni e risposte temporali per specifici residui
lista = np.array([20,21,22,23,24])
t = [0.3]
time_idx = 0
for i in lista:
    residual_analysis.plot_residual_correlation_vs_j(i=i, t=t, time_idx=time_idx)
    residual_analysis.plot_residual_time_response_vs_j(i=i, t=t, time_idx=time_idx)
    residual_analysis.plot_residual_transfer_entropy_vs_j(i=i, t=t, time_idx=time_idx)

residual_analysis.plot_mean_quantity_over_segment(lista, t, time_idx, 'linear_response')

lista = np.array([71, 72, 73, 74, 75, 76, 77, 78, 79])
t = [0.3]
time_idx = 0
for i in lista:
    residual_analysis.plot_residual_correlation_vs_j(i=i, t=t, time_idx=time_idx)
    residual_analysis.plot_residual_time_response_vs_j(i=i, t=t, time_idx=time_idx)
    residual_analysis.plot_residual_transfer_entropy_vs_j(i=i, t=t, time_idx=time_idx)

residual_analysis.plot_mean_quantity_over_segment(lista, t, time_idx, 'correlation')
residual_analysis.plot_mean_quantity_over_segment(lista, t, time_idx, 'entropy')

