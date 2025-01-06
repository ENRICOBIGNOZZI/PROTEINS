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
    for i in lista:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 12))

        legend_handles = []

        # Tracciare i grafici nel primo sottomodulo
        for epsilon in [1,0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
            temperatura = np.where(contatti >= 5, epsilon, 1)
            B = np.sqrt(2*temperatura)
            B = np.diag(B)

            lambdaa, U = np.linalg.eig(kirchhoff_matrix)
            BBT = B @ B.T
            Q = U @ BBT @ U.T

            
            color = plt.cm.viridis((epsilon + 0.1) / 1.0)
            label = f'Temperature = {epsilon}'
            plot_residual_transfer_entropy_vs_j_donatore(df, i, t, s, time_idx, nome, Q, lambdaa, U, color=color, label=label,ax=ax1)

            if label not in [h.get_label() for h in legend_handles]:
                legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=label))

        # Configurazione del primo sottomodulo
        ax1.legend(handles=legend_handles, title='Legenda Epsilon', loc='upper right')
        ax1.set_xlabel("Residue Index")
        ax1.set_ylabel(f'Transfer entropy {i}')
        ax1.set_title(f'Transfer entropy {i}')
        ax1.grid(True)

        # Tracciare le temperature nel secondo sottomodulo
        for epsilon in [1.,0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
            temperatura = np.where(contatti >= 5, epsilon, 1)
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
        plt.savefig(f'images/{nome}/2_temperature_cutoff/combined_entropy_temperature_donatore_{i}_plots.png')
        plt.show()
        plt.close()

def transfer_entropy(C,C_statica, i, j):
    C_ii_0 = C_statica[i, i]
    C_jj_0 = C_statica[j, j]
    C_ii_t = C[i, i]
    C_ij_0 = C_statica[i, j]
    C_ij_t = C[i, j]
    alpha_ij_t = (C_ii_0 * C_ij_t - C_ij_0 * C_ii_t) ** 2
    beta_ij_t = (C_ii_0 * C_jj_0-(C_ij_0**2)) * (C_ii_0**2- C_ii_t ** 2)
    ratio = np.clip(alpha_ij_t / beta_ij_t, 0, 1 - 1e-10)
    return -0.5 * np.log(1 - ratio)

def plot_residual_transfer_entropy_vs_j_accettore(df, i, t, s, time_idx, nome, Q, lambdaa, U, color, label=None, ax=None):
    correlation_i = np.zeros((len(lambdaa),len(lambdaa)))
    lista=i
    correlation_i_zero = np.zeros((len(lambdaa),len(lambdaa)))
    z = np.array(t) - np.array(s)
    
    for tau in z:
        for i in range(len(lambdaa)):
            for j in range(len(lambdaa)):
                sum_result = 0.0
                sum_result2 = 0.0
                for k in range(1, len(lambdaa)):
                    for p in range(1, len(lambdaa)):
                        term = (U[i, k] * Q[k, p] * U[j, p]) / (lambdaa[k] + lambdaa[p])
                        term2=term
                        term *= np.exp(-lambdaa[k] * tau)
                        
                        sum_result += term
                        sum_result2+=term2
                correlation_i[i][j] = sum_result
                correlation_i_zero[i][j] = sum_result2
    
                
    i=lista
    transfer_entropy_matrix =  np.zeros(len(lambdaa))
    for j in range(len(lambdaa)):
        if j==i:
            transfer_entropy_matrix[j]=0
        else:
            transfer_entropy_matrix[j] = transfer_entropy(correlation_i,correlation_i_zero,i, j)

    
    # Plot the main matrix with the specified label
    ax.plot(range(len(transfer_entropy_matrix)), transfer_entropy_matrix, marker='o', linestyle='-', alpha=0.7, color=color, label=label)
    
    # Call the secondary structure plotting function



def plot_residual_transfer_entropy_vs_j_donatore(df, i, t, s, time_idx, nome, Q, lambdaa, U, color, label=None, ax=None):
    correlation_i = np.zeros((len(lambdaa),len(lambdaa)))
    lista=i
    correlation_i_zero = np.zeros((len(lambdaa),len(lambdaa)))
    z = np.array(t) - np.array(s)
    
    for tau in z:
        for i in range(len(lambdaa)):
            for j in range(len(lambdaa)):
                sum_result = 0.0
                sum_result2 = 0.0
                for k in range(1, len(lambdaa)):
                    for p in range(1, len(lambdaa)):
                        term = (U[i, k] * Q[k, p] * U[j, p]) / (lambdaa[k] + lambdaa[p])
                        term2=term
                        term *= np.exp(-lambdaa[k] * tau)
                        
                        sum_result += term
                        sum_result2+=term2
                correlation_i[i][j] = sum_result
                correlation_i_zero[i][j] = sum_result2
    
                
    i=lista
    transfer_entropy_matrix =  np.zeros(len(lambdaa))
    for j in range(len(lambdaa)):
        if j==i:
            transfer_entropy_matrix[j]=0
        else:
            transfer_entropy_matrix[j] = transfer_entropy(correlation_i,correlation_i_zero,j, i)

    
    # Plot the main matrix with the specified label
    ax.plot(range(len(transfer_entropy_matrix)), transfer_entropy_matrix, marker='o', linestyle='-', alpha=0.7, color=color, label=label)
    
import numpy as np

'''def transfer_entropy(C, C_statica, i, j):
    C_ii_0 = C_statica[i, i]
    C_jj_0 = C_statica[j, j]
    C_ii_t = C[i, i]
    C_ij_0 = C_statica[i, j]
    C_ij_t = C[i, j]
    alpha_ij_t = (C_ii_0 * C_ij_t - C_ij_0 * C_ii_t) ** 2
    beta_ij_t = (C_ii_0 * C_jj_0 - (C_ij_0 ** 2)) * (C_ii_0 ** 2 - C_ii_t ** 2)
    ratio = np.clip(alpha_ij_t / beta_ij_t, 0, 1 - 1e-10)
    return -0.5 * np.log(1 - ratio)

def plot_residual_transfer_entropy_vs_j_accettore(df, i, t, s, time_idx, nome, Q, lambdaa, U, color, label=None, ax=None):
    n = len(lambdaa)
    correlation_i = np.zeros((n, n))
    correlation_i_zero = np.zeros((n, n))
    z = np.array(t) - np.array(s)
    
    # Precompute U[i, k] * Q[k, p] * U[j, p] for all i, j, k, p
    UQ = np.einsum('ik,kp,jp->ijkp', U, Q, U)
    
    # Precompute lambdaa[k] + lambdaa[p] for all k, p
    lambda_sum = lambdaa[:, None] + lambdaa[None, :]
    
    # Precompute the term (U[i, k] * Q[k, p] * U[j, p]) / (lambdaa[k] + lambdaa[p])
    term = UQ / lambda_sum[None, None, :, :]
    
    for tau in z:
        # Compute the exponential term for all k
        exp_term = np.exp(-lambdaa * tau)  # Shape: (n,)
        
        # Multiply by the exponential term and sum over k and p
        correlation_i += np.sum(term * exp_term[None, None, :, None], axis=(2, 3))
        correlation_i_zero += np.sum(term, axis=(2, 3))
    
    transfer_entropy_matrix = np.zeros(n)
    for j in range(n):
        if j != i:
            transfer_entropy_matrix[j] = transfer_entropy(correlation_i, correlation_i_zero, i, j)
            print(transfer_entropy_matrix[j])
    # Plot the main matrix with the specified label
    ax.plot(range(len(transfer_entropy_matrix)), transfer_entropy_matrix, marker='o', linestyle='-', alpha=0.7, color=color, label=label)
'''
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




tau_mean=0.1845


t=[tau_mean]
s=[0,0,0]
time_idx = 0

lista=[30,72]
# Loop da 0 a 94 prendendo uno ogni 3
#for i in range(0, 95, 3):  # i va da 0 a 94 con step di 3
#   # Selezioniamo l'indice modulo della lunghezza della lista
#   lista.append(i)
main_plot(df, kirchhoff_matrix, contatti, t, s, time_idx, stringa,lista)

