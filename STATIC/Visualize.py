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
    def create_and_print_graph(self, truncated, radius=8.0, plot=False, peso=1):
        positions = self.df.loc[:, ['X', 'Y', 'Z']].values
        residue_ids = self.df['Residue ID'].values
        G = nx.Graph()
        print(len(residue_ids))
        for i in range(len(residue_ids)):
            G.add_node(residue_ids[i])
            for j in range(len(residue_ids)):  # Consider all pairs, not just upper triangle
                G.add_node(residue_ids[j])
                if i != j:  # Avoid self-loops
                    distance = euclidean_distance(positions[i], positions[j])
                    if distance <= radius:
                        if abs(residue_ids[i] - residue_ids[j]) == 1:
                            # Covalent bond
                            weight = peso
                        else:
                            # Non-covalent interaction
                            weight = 1
                        G.add_edge(residue_ids[i], residue_ids[j], weight=weight)

        if plot:
            plt.figure(figsize=(12, 10))
            nx.draw(G, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold', edge_color='gray')
            plt.title("Protein Contact Map")
            plt.show()

        return G



