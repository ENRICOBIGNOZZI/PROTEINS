import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from funtions import build_graph_number_of_connections,euclidean_distance
import seaborn as sns
import os

class Visualize:
    def __init__(self, df):
        self.df = df
    def plot_different_models(self):
        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=self.df, x="X", y="Y", hue="Model ID", palette="tab20", style="Model ID", s=100)
        plt.title("Atomic Coordinates by Model ID")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend(title="Model ID", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
    def plot_connections_vs_radius(self):
        output_dir = 'docs/img'
        os.makedirs(output_dir, exist_ok=True)
        Raggi, numero_connessioni = build_graph_number_of_connections(self.df)
        plt.figure(figsize=(10, 6))
        plt.plot(Raggi, numero_connessioni, linestyle='-', color='b')
        plt.xlabel('Raggio')
        plt.ylabel('Numero di connessioni')
        plt.title('Numero di connessioni al variare del raggio')
        plt.grid(True)
        plt.show()
        output_file = os.path.join(output_dir, 'numero_connessioni_vs_raggio.png')
        plt.savefig(output_file)

    def calculate_and_print_average_distance(self):
        positions = self.df.loc[:, ['X', 'Y', 'Z']].values
        distances = []

        num_points = len(positions)
        for i in range(num_points):
            for j in range(i + 1, num_points):
                distance = euclidean_distance(positions[i], positions[j])
                distances.append(distance)

        media_distanza = np.mean(distances)
        print("Distanza media:", media_distanza)
        return media_distanza
    def calculate_weight(self,x):
        return (1/(x)**2)
    def calculate_weight_covalent(self, residue_id1, residue_id2,peso):
        return peso if abs(residue_id1 - residue_id2) == 1 else 1
    def create_and_print_graph(self, truncated,radius=None,plot=False,peso=1):
        positions = self.df.loc[:, ['X', 'Y', 'Z']].values
        residue_ids = self.df['Residue ID'].values
        G = nx.Graph()

        for i in range(len(positions)):
            for j in range(len(positions)):
                if i != j:
                    if truncated:
                        if euclidean_distance(positions[i], positions[j]) < radius:
                            weight = self.calculate_weight_covalent(residue_ids[i], residue_ids[j],peso)
                            G.add_edge(i, j, weight=weight)
                    else:
                        weight = self.calculate_weight(euclidean_distance(positions[i], positions[j]),peso)
                        G.add_edge(i, j, weight=weight)
        if plot:
            plt.figure(figsize=(12, 10))
            if truncated is not True:
                pos = {i: (positions[i][0], positions[i][1]) for i in range(len(positions))}
                nx.draw_networkx_nodes(G, pos, node_size=400, node_color='lightblue')

                nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='gray')
                # Definisci le posizioni dei nodi per la visualizzazione
                
                nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)})
                nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
            else:
                nx.draw(G, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold', edge_color='gray')
            plt.title("Visualizzazione del Grafo")
            plt.show()
        #print("Nodi del grafo:", G.nodes())
        #print("Archi del grafo:", G.edges())
        return G



