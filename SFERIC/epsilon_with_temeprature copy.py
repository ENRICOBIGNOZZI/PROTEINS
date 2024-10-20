import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
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




def plot_residual_correlation_vs_j(df, i, t, s, time_idx, nome, Q, lambdaa, U, color, label=None, ax=None):
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

    ax.plot(range(len(correlation_i)), correlation_i, marker='o', linestyle='-', alpha=0.7, color=color, label=label)

def _plot_secondary_structure(sec_struct_data, ax):
    sec_struct_info = sec_struct_data['Secondary Structure']
    residue_ids = sec_struct_data['Residue ID'].astype(int)

    colors = {'H': 'red', 'E': 'blue'}
    sec_struct_colors = ['white'] * len(residue_ids)

    for idx, rid in enumerate(residue_ids):
        struct = sec_struct_info.get(rid, 'C')
        if struct in colors:
            sec_struct_colors[idx] = colors[struct]

    current_color = 'white'
    start_idx = 0
    for idx, resid in enumerate(residue_ids):
        if sec_struct_colors[idx] != current_color:
            if idx > 0 and current_color in colors.values():
                ax.axvspan(start_idx, idx, color=current_color, alpha=0.2)
            current_color = sec_struct_colors[idx]
            start_idx = idx

    if current_color in colors.values():
        ax.axvspan(start_idx, len(residue_ids), color=current_color, alpha=0.2)

    return colors

def main_plot(df, kirchhoff_matrix, contatti, t, s, time_idx, nome, lista):
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 12))

    legend_handles = []

    df[['X', 'Y', 'Z']] = df[['X', 'Y', 'Z']].apply(pd.to_numeric, errors='coerce')
    centro_massa = df[['X', 'Y', 'Z']].mean().values
    distanze = np.sqrt(((df[['X', 'Y', 'Z']] - centro_massa) ** 2).sum(axis=1))
    for epsilon in [1,0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        # Definisco la temperatura con la variazione di epsilon
        T0 = epsilon  # Temperatura al centro
        Tb = 1  # Temperatura al bordo
        R = distanze.max()  # Raggio massimo
        temperatura = T0 + (Tb - T0) / R * distanze
        B = np.sqrt(temperatura)
        B = np.diag(B)

        lambdaa, U = np.linalg.eig(kirchhoff_matrix)
        BBT = B @ B.T
        Q = U @ BBT @ U.T

        for i in lista:
            color = plt.cm.viridis((epsilon + 0.1) / 1.0)
            label = f'Temperature = {epsilon}'
            plot_residual_correlation_vs_j(df, i, t, s, time_idx, nome, Q, lambdaa, U, color=color, label=label, ax=ax1)

            if label not in [h.get_label() for h in legend_handles]:
                legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=label))

    # Configurazione del primo sottomodulo
    ax1.legend(handles=legend_handles, title='Legenda Epsilon', loc='upper right')
    ax1.set_xlabel('Residue Index')
    ax1.set_ylabel('Correlation')
    ax1.set_title('Correlazioni Residue')
    ax1.grid(True)

    # Tracciare le temperature nel secondo sottomodulo
    for epsilon in [1.,0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        T0 = epsilon  # Temperatura al centro
        Tb = 1  # Temperatura al bordo
        R = distanze.max()  # Raggio massimo
        temperatura = T0 + (Tb - T0) / R * distanze

        # Usa un colore diverso per ogni epsilon
        color = plt.cm.viridis((epsilon + 0.1) / 1.0)
        ax2.plot(range(len(contatti)), temperatura, linestyle='-', alpha=0.7, color=color, label=f'Temperatura ={epsilon}')

    # Configurazione del secondo sottomodulo
    ax2.set_xlabel('Residue Index')
    ax2.set_ylabel('Temperatura')
    ax2.set_title('Variazione della Temperatura')
    ax2.grid(True)
    ax2.legend(title='Temperature', loc='upper right')

    # Plot della struttura secondaria
    _plot_secondary_structure(df, ax1)

    # Salvataggio del grafico
    plt.tight_layout()
    plt.savefig(f'images/{nome}/2_temperature_sferical/combined_correlation_temperature_plots.png')
    plt.show()






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

