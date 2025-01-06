import numpy as np
import matplotlib.pyplot as plt
from Downlaod_data import PDBProcessor
from Visualize import Visualize
from matrix import GraphMatrixAnalyzer
import pandas as pd
import matplotlib.patches as patches
import os
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

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


contatti_somma_righe = kirchhoff_matrix.sum(axis=1)
contatti = contatti_diagonale/2


temperatura = np.where(contatti >= 5, 0.5, 1)
G = visualizer.create_and_print_graph(truncated=True, radius=8.0, plot=False, peso=20)  # Adjust radius as needed
analyzer = GraphMatrixAnalyzer(G)
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()

B=np.sqrt(temperatura)
B = np.diag(B)

lambdaa, U = np.linalg.eig(kirchhoff_matrix)
BBT = B @ B.T
Q = U @ BBT  @ U.T
print(Q)
t=np.linspace(0., 2, 300)  # Time points

correlation=np.zeros((len(lambdaa),len(lambdaa),len(t)))
for tau in t:
    for i in range(len(lambdaa)):
        for j in range(len(lambdaa)):
            sum_result = 0.0
            for k in range(1, len(lambdaa)):
                for p in range(1, len(lambdaa)):
                    term = (U[i, k] * Q[k, p] * U[j, p]) / (lambdaa[k] + lambdaa[p])*np.exp(-lambdaa[k] * tau)
                    sum_result += term
            correlation[i,j,t] = sum_result