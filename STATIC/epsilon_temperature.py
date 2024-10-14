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



import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_residual_correlation_vs_j(df, i, t, s, time_idx, nome, Q, lambdaa, U, color, label=None):
    correlation_i = np.zeros((len(lambdaa)))
    z = np.array(t) - np.array(s)
    
    for tau in z:
        for j in range(len(lambdaa)):
            sum_result = 0.0
            for k in range(1, len(lambdaa)):
                for p in range(1, len(lambdaa)):
                    term = (U[i, k] * Q[k, p] * U[j, p]) / (lambdaa[k] + lambdaa[p])
                    if tau > 0:
                        term *= np.exp(-lambdaa[k] * tau)
                    else:
                        term *= np.exp(lambdaa[p] * tau)
                    sum_result += term
            correlation_i[j] = sum_result

    # Plot the main matrix with the specified label
    plt.plot(range(len(correlation_i)), correlation_i, marker='o', linestyle='-', alpha=0.7, color=color, label=label)
    
    # Call the secondary structure plotting function
    

def _plot_secondary_structure(sec_struct_data):
    sec_struct_info = sec_struct_data['Secondary Structure']
    residue_ids = sec_struct_data['Residue ID'].astype(int)

    # Colors only for 'H' (alpha-helix) and 'E' (beta-sheet)
    colors = {'H': 'red', 'E': 'blue'}
    sec_struct_colors = ['white'] * len(residue_ids)  # Default to white for all residues

    # Assign colors to residues based on their secondary structure
    for idx, rid in enumerate(residue_ids):
        struct = sec_struct_info.get(rid, 'C')  # Default to 'C' if not found
        if struct in colors:
            sec_struct_colors[idx] = colors[struct]

    # Plot the secondary structure bands
    current_color = 'white'
    start_idx = 0
    for idx, resid in enumerate(residue_ids):
        if sec_struct_colors[idx] != current_color:
            if idx > 0 and current_color in colors.values():
                plt.axvspan(start_idx, idx, color=current_color, alpha=0.2)
            current_color = sec_struct_colors[idx]
            start_idx = idx

    # Plot the last segment
    if current_color in colors.values():
        plt.axvspan(start_idx, len(residue_ids), color=current_color, alpha=0.2)

    return colors  # Return colors for the legend

def main_plot(df, kirchhoff_matrix, contatti, t, s, time_idx, nome, lista):
    # Creazione della figura e dell'asse per sovrapporre i grafici
    plt.figure(figsize=(12, 8))
    
    # Store colors and labels for the legend
    legend_handles = []
    secondary_colors = {'H': 'red', 'E': 'blue'}  # Colors for secondary structures

    for epsilon in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        # Definisco la temperatura con la variazione di epsilon
        temperatura = np.where(contatti >= 5, epsilon, 1)

        # Definisco B
        B = np.sqrt(temperatura)
        B = np.diag(B)

        # Eigen decomposition
        lambdaa, U = np.linalg.eig(kirchhoff_matrix)
        BBT = B @ B.T

        # Q calculation
        Q = U @ BBT @ U.T

        # Ciclo for per generare i plot sovrapposti
        for i in lista:  # O il numero desiderato di iterazioni
            color = plt.cm.viridis((epsilon + 0.1) / 1.0)  # Get a color based on epsilon
            label = f'Epsilon = {epsilon}'
            plot_residual_correlation_vs_j(df, i, t, s, time_idx, nome, Q, lambdaa, U, color=color, label=label)

            # Add to legend handles only once
            if label not in [h.get_label() for h in legend_handles]:
                legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=label))
    _plot_secondary_structure(df)
    # Add secondary structure legend handles
    for struct, color in secondary_colors.items():
        legend_handles.append(mpatches.Patch(color=color, label=struct, alpha=0.2))

    # Mostra la legenda finale
    plt.legend(handles=legend_handles, title='Legenda', loc='upper right')

    # Salva il grafico finale combinato con tutti i plot sovrapposti
    plt.xlabel('Residue Index')
    plt.ylabel('Correlation')
    plt.title('Combined Correlation Plots')
    plt.grid(True)
    plt.savefig(f'images/{nome}/2_temperature_cutoff/combined_correlation_plots_epsilon.png')
    plt.show()

# Esempio di chiamata della funzione
# main_plot(df, kirchhoff_matrix, contatti, t, s, time_idx, nome, lista)

# Esempio di chiamata della funzione
# main_plot(df, kirchhoff_matrix, contatti, t, s, time_idx, nome, lista)

# Esempio di chiamata della funzione
# main_plot(df, kirchhoff_matrix, contatti, t, s, time_idx, nome, lista)

# Usa la funzione `main_plot` per eseguire l'intero processo con variazione di epsilon

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


temperatura = np.where(contatti >= 5, 0.9, 1)
G = visualizer.create_and_print_graph(truncated=True, radius=8.0, plot=False, peso=20)  # Adjust radius as needed
analyzer = GraphMatrixAnalyzer(G)
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()


t = np.linspace(0., 2, 300)  # Time points

tau_mean=0.1845


t=[tau_mean]
s=[0,0,0]
time_idx = 0

lista = np.array([23])
main_plot(df, kirchhoff_matrix, contatti, t, s, time_idx, stringa,lista)

