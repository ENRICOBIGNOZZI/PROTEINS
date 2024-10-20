from Downlaod_data import PDBProcessor
from Visualize import Visualize

from matrix import GraphMatrixAnalyzer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
class BaseCorrelationAnalysis:
    def __init__(self, u, lambdas, mu, sec_struct_data,stringa,Q):
        self.u = u  # Ensure u is at least 2D
        self.lambdas = np.array(lambdas)
        self.mu = mu
        self.sec_struct_data = sec_struct_data
        self.name=stringa
        self.Q=Q

class CorrelationMatrixOperations(BaseCorrelationAnalysis):
    def compute_static_correlation_matrix(self):
        correlation = np.zeros((len(self.lambdas), len(self.lambdas)))
        
        
        for i in range(len(self.lambdas)):
            for j in range(len(self.lambdas)):
                sum_result = 0.0
                for k in range(1, len(self.lambdas)):
                    for p in range(1, len(self.lambdas)):
                        term = (self.u[i, k] * self.Q[k, p] * self.u[j, p]) / (self.lambdas[k] + self.lambdas[p])
                        sum_result += term
                correlation[i][j] = sum_result
        return correlation

    def split_correlation_matrix(self, correlation_matrix):
        positive_matrix = np.where(correlation_matrix > 0, correlation_matrix, 0)
        negative_matrix = np.where(correlation_matrix < 0, correlation_matrix, 0)
        return positive_matrix, negative_matrix


    def plot_correlation_matrix_nan(self, correlation_matrix, kirchhoff_matrix, secondary_structure, positive_only,epsilon):
        plt.figure(figsize=(10, 10))

        masked_matrix = np.where(correlation_matrix > 0, correlation_matrix, np.nan) if positive_only else np.where(correlation_matrix < 0, correlation_matrix, np.nan)

        # Plotta la matrice di correlazione
        plt.imshow(masked_matrix, cmap='coolwarm', interpolation='none', origin='lower', alpha=0.4)
        cbar = plt.colorbar()
        cbar.set_label('Correlation')

        # Sovrappone la matrice di contatti (Kirchhoff matrix)
        binary_matrix = np.where(kirchhoff_matrix != 0, 1, 0)
        rows, cols = np.where(binary_matrix == 1)

        # Plotta i punti della matrice di Kirchhoff
        plt.scatter(cols, rows, color='black', alpha=0.4, s=10, zorder=2)

        # Aggiungi rettangoli o patch (esempio)
        rectangle1 = mpatches.Rectangle((19, 71), 5, 9, linewidth=2, edgecolor='r', facecolor='none')
        rectangle2 = mpatches.Rectangle((71, 19), 9, 5, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rectangle1)
        plt.gca().add_patch(rectangle2)

        # Segnamenti su assi x e y basati sulla struttura secondaria
        start = 0
        current_structure = secondary_structure[0][0] if len(secondary_structure) >= 2 else secondary_structure[0]

        for i, structure in enumerate(secondary_structure):
            structure = structure[0] if len(structure) >= 2 else structure

            if structure != current_structure:
                if current_structure == 'H' or current_structure == 'E':  # Plot solo per eliche e foglietti beta
                    color = 'red' if current_structure == 'H' else 'blue'
                    plt.plot([start, i], [-0.5, -0.5], color=color, linewidth=8)  # Riga orizzontale (sopra)
                    plt.plot([-0.5, -0.5], [start, i], color=color, linewidth=8)  # Riga verticale (a sinistra)
                    plt.text((start + i) / 2, -1, current_structure, ha='center', va='top', fontsize=12, fontweight='bold')  # Etichetta in basso
                    plt.text(-1, (start + i) / 2, current_structure, ha='right', va='center', fontsize=12, fontweight='bold')  # Etichetta a sinistra
                start = i
                current_structure = structure

        # Plot per l'ultimo segmento
        if current_structure == 'H' or current_structure == 'E':
            color = 'red' if current_structure == 'H' else 'blue'
            plt.plot([start, i + 1], [-0.5, -0.5], color=color, linewidth=8)
            plt.plot([-0.5, -0.5], [start, i + 1], color=color, linewidth=8)
            plt.text((start + i + 1) / 2, -1, current_structure, ha='center', va='top', fontsize=12, fontweight='bold')
            plt.text(-1, (start + i + 1) / 2, current_structure, ha='right', va='center', fontsize=12, fontweight='bold')

        # Aggiungi legenda per Helix e Beta Sheet
        handles = [
            mpatches.Patch(color='red', label='Helix'),
            mpatches.Patch(color='blue', label='Beta Sheet')
        ]
        plt.legend(handles=handles, loc='upper right')

        # Etichette degli assi
        plt.xlabel('Index j')
        plt.ylabel('Index i')

        # Salva la figura
        if not os.path.exists(f'images/{self.name}/2_temperature_sferical/Matrici_Correlazione/'):
            os.makedirs(f'images/{self.name}/2_temperature_sferical/Matrici_Correlazione/')

        plt.savefig(f'images/{self.name}/2_temperature_sferical/Matrici_Correlazione/Correlation_MatrixNan_{positive_only}_{epsilon}.png')
        # plt.show()

stringa="3LNX"
pdb_processor = PDBProcessor(pdb_id="3LNX")#2m07
pdb_processor.download_pdb()
pdb_processor.load_structure()
df1 = pdb_processor.secondary_structure()
df = pdb_processor.extract_atom_data()
df = df[df['Model ID'] == 0]
df = df[df['Atom Name'] == 'CA']
df = df[df['Chain ID'] == 'A']
df = df.reset_index(drop=True)
concatenated_df = pd.concat([df1['Secondary Structure'], df], axis=1)
df = concatenated_df.dropna().reset_index(drop=True)
df = df.loc[:,~df.T.duplicated()]
df = concatenated_df.dropna().reset_index(drop=True)
df = df.T.drop_duplicates().T
df=df.dropna().reset_index(drop=True)
visualizer = Visualize(df)
print(df)
raggio=visualizer.calculate_and_print_average_distance()
#visualizer.plot_connections_vs_radius()
G = visualizer.create_and_print_graph(truncated=True, radius=8.0, plot=False, peso=1)  # Adjust radius as needed
analyzer = GraphMatrixAnalyzer(G)
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
contatti_diagonale = kirchhoff_matrix.diagonal()
df[['X', 'Y', 'Z']] = df[['X', 'Y', 'Z']].apply(pd.to_numeric, errors='coerce')
centro_massa = df[['X', 'Y', 'Z']].mean().values
distanze = np.sqrt(((df[['X', 'Y', 'Z']] - centro_massa) ** 2).sum(axis=1))



for epsilon in [1,0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
    # Parametri per la temperatura radiale
    T0 = epsilon  # Temperatura al centro
    Tb = 1  # Temperatura al bordo
    R = distanze.max()  # Raggio massimo
    temperatura = T0 + (Tb - T0) / R * distanze

   
    G = visualizer.create_and_print_graph(truncated=True, radius=8.0, plot=False, peso=20)  # Adjust radius as needed
    analyzer = GraphMatrixAnalyzer(G)
    kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
    B=np.sqrt(temperatura)
    B = np.diag(B)

    lambdaa, U = np.linalg.eig(kirchhoff_matrix)
    BBT = B @ B.T
    Q = U @ BBT  @ U.T



    function= CorrelationMatrixOperations(u=U, lambdas=lambdaa, mu=0, sec_struct_data=df, stringa=stringa, Q=Q)
    correlation_matrix=function.compute_static_correlation_matrix()
    autocorrelations_back = function.plot_correlation_matrix_nan(correlation_matrix, kirchhoff_matrix, df['Secondary Structure'], positive_only=True,epsilon=epsilon)
    autocorrelations_back = function.plot_correlation_matrix_nan(correlation_matrix, kirchhoff_matrix,  df['Secondary Structure'], positive_only=False,epsilon=epsilon)