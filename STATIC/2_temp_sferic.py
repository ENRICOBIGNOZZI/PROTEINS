import numpy as np
import matplotlib.pyplot as plt
from Downlaod_data import PDBProcessor
from Visualize import Visualize
from matrix import GraphMatrixAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import os
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

raggio=visualizer.calculate_and_print_average_distance()
#visualizer.plot_connections_vs_radius()
G = visualizer.create_and_print_graph(truncated=True, radius=8.0, plot=False, peso=1)  # Adjust radius as needed
analyzer = GraphMatrixAnalyzer(G)
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
df[['X', 'Y', 'Z']] = df[['X', 'Y', 'Z']].apply(pd.to_numeric, errors='coerce')
centro_massa = df[['X', 'Y', 'Z']].mean().values
distanze = np.sqrt(((df[['X', 'Y', 'Z']] - centro_massa) ** 2).sum(axis=1))

# Parametri per la temperatura radiale
T0 = 0.5  # Temperatura al centro
Tb = 1.0  # Temperatura al bordo
R = distanze.max()  # Raggio massimo

# Calcolo della temperatura radiale
temperatura_radiale = T0 + (Tb - T0) / R * distanze

def calculate_Q(U, B):
    B_transpose = np.transpose(B)
    U_transpose = np.transpose(U)
    Q = np.dot(U, np.dot(B, np.dot(B_transpose, U_transpose)))
    return Q

eigenvalues, eigenvectors = np.linalg.eig(kirchhoff_matrix)

Q=calculate_Q(  eigenvectors,temperatura_radiale)



residui = range(1, len(temperatura_radiale) + 1)

plt.figure(figsize=(12, 6))
plt.scatter(residui, temperatura_radiale, c=temperatura_radiale, cmap='coolwarm', marker='o')
plt.colorbar(label='Temperatura')
plt.title('Grafico della temperatura radiale al variare del residuo')
plt.xlabel('Numero del residuo')
plt.ylabel('Temperatura')
plt.grid(True)
plt.tight_layout()
if not os.path.exists(f'images/{stringa}/2_temperature_sferical/'):
    os.makedirs(f'images/{stringa}/2_temperature_sferical/')


# Save the figure
plt.savefig(f'images/{stringa}/2_temperature_sferical/temperature_sferic.png')

# Grafico 3D della proteina colorata per temperatura
def plot_matrix( matrix, secondary_structure, nome, title="Matrix"):
    binary_matrix=matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create a binary matrix: 1 where connections exist, 0 otherwise
    
    
    # Get the indices of non-zero elements
    
    
    # Plot the dots
    cax = ax.imshow(matrix, cmap='viridis')

    # Aggiunta della barra del colore (opzionale)
    fig.colorbar(cax)

    # Add rectangles
    rectangle1 = patches.Rectangle((19, 71), 5, 9, linewidth=2, edgecolor='r', facecolor='none')
    rectangle2 = patches.Rectangle((71, 19), 9, 5, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rectangle1)
    ax.add_patch(rectangle2)
    
    # Create segments on x and y axes based on secondary structure
    start = 0
    current_structure = secondary_structure[0]
    if len(secondary_structure) >= 2:
        current_structure = current_structure[0]
    
    for i, structure in enumerate(secondary_structure):
        if len(structure) >= 2:
            structure = structure[0]
    
        if structure != current_structure:
            if current_structure == 'H' or current_structure == 'E':  # Plot only for Helix and Beta Sheet
                color = 'red' if current_structure == 'H' else 'blue'
                ax.plot([start, i], [0, 0], color=color, linewidth=8)  # Increase line thickness
                ax.plot([0, 0], [start, i], color=color, linewidth=8)  # Increase line thickness
                ax.text((start+i)/2, -0.5, current_structure, ha='center', va='top', fontsize=12, fontweight='bold')
                ax.text(-0.5, (start+i)/2, current_structure, ha='right', va='center', fontsize=12, fontweight='bold')
            start = i
            current_structure = structure
    if current_structure == 'H' or current_structure == 'E':  # Plot final segment if it is Helix or Beta Sheet
        color = 'red' if current_structure == 'H' else 'blue'
        ax.plot([start, i+1], [0, 0], color=color, linewidth=8)  # Increase line thickness
        ax.plot([0, 0], [start, i+1], color=color, linewidth=8)  # Increase line thickness
        ax.text((start+i+1)/2, -0.5, current_structure, ha='center', va='top', fontsize=12, fontweight='bold')
        ax.text(-0.5, (start+i+1)/2, current_structure, ha='right', va='center', fontsize=12, fontweight='bold')
    
    # Add legend for Helix and Beta Sheet only
    handles = [
        patches.Patch(color='red', label='Helix'),
        patches.Patch(color='blue', label='Beta Sheet')
    ]
    ax.legend(handles=handles, loc='upper right')
    
    #ax.set_title(title)
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('Residue Index')
    ax.set_ylim(0, binary_matrix.shape[0])
    ax.set_xlim(0, binary_matrix.shape[1])
    
    plt.tight_layout()
    if not os.path.exists(f'images/{stringa}/2_temperature_sferical/'):
        os.makedirs(f'images/{stringa}/2_temperature_sferical/')
    plt.savefig(f'images/{stringa}/2_temperature_sferical/correlation.png')

import scipy.integrate as integrate

# Assicuriamoci che temperatura_radiale sia un array numpy
temperatura_radiale = np.array(temperatura_radiale)


# Funzione per calcolare l'integrale


# Calcolo dell'integrale
K = 1.0  # Puoi modificare questo valore secondo le tue necessità

def calculate_Cij_matrix_static(u, Q, lambdaa,t,s):#questa è quella corretta
    if t>s:
        Cij= np.dot(u,np.dot(Q,u))/ np.sum(lambdaa + lambdaa)*np.exp(-lambdaa*(t-s))
    else:
        Cij= np.dot(u,np.dot(Q,u))/ np.sum(lambdaa + lambdaa)*np.exp(-lambdaa*(s-t))
    return Cij
# Calcola la matrice Cij per tutti gli ij
#Cij_matrix = calculate_Cij_matrix(eigenvectors, Q, eigenvalues, s=0, t=1)

t=0.1
s=0
Cij_matrix = calculate_Cij_matrix_static(eigenvectors, Q, eigenvalues,t,s)
plot_matrix( Cij_matrix ,  df['Secondary Structure'].values, stringa, title="Matrix_sferic_temperature")
'''plt.figure(figsize=(12, 6))
plt.imshow(Cij_matrix, cmap='hot', interpolation='nearest')
plt.colorbar(label='Correlation Matrix t={}, s={}'.format(t,s))  
plt.title('Correlation Matrix')
if not os.path.exists(f'images/{stringa}/2_temperature_sferical/'):
    os.makedirs(f'images/{stringa}/2_temperature_sferical/')
plt.savefig(f'images/{stringa}/2_temperature_sferical/correlation.png')'''


def plot_residual_correlation_vs_j(self, i, t, time_idx):
    residual_correlation_matrix = self.compute_residual_correlation_matrix(t, i, time_idx)
    print("inzio correlation")
    print(residual_correlation_matrix.shape)
    print("fine correlation")
    self._plot_with_secondary_structure(residual_correlation_matrix, f'Correlation with i={i}', f'Residual Correlation C_ij for i={i} as a function of j at time index {time_idx}')
def compute_residual_correlation_matrix(self, t,i, time_idx):
    """Calcola la matrice di correlazione dei residui per tutti i j e per un intervallo di tempo specificato."""
    n = self.u.shape[0]
    t_subset=t
    # Inizializza una matrice per le correlazioni temporali
    residual_correlation_matrix = np.zeros((n, n, len(t_subset)))

    for j in range(n):
        residual_correlation_matrix[i, j, :] = self.time_correlation_2(i, j, t_subset)
    return residual_correlation_matrix[i,:,:]

def _plot_with_secondary_structure(self, matrix, ylabel, title):
        sec_struct_info = self.sec_struct_data['Secondary Structure']
        residue_ids = self.sec_struct_data['Residue ID'].astype(int)

        # Colors only for 'H' (alpha-helix) and 'E' (beta-sheet)
        colors = {'H': 'red', 'E': 'blue'}
        sec_struct_colors = ['white'] * len(residue_ids)  # Default to white for all residues

        # Assign colors to residues based on their secondary structure
        for idx, rid in enumerate(residue_ids):
            struct = sec_struct_info.get(rid, 'C')  # Default to 'C' if not found
            if struct in colors:
                sec_struct_colors[idx] = colors[struct]

        plt.figure(figsize=(12, 8))
        plt.plot(range(len(matrix)), matrix, marker='o', linestyle='-', alpha=0.7)

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
        legend_handles = [
            mpatches.Patch(color='red', label='Helix (H)', alpha=0.2),  # Alpha uniforme
            mpatches.Patch(color='blue', label='Beta sheet (E)', alpha=0.2)  # Alpha uniforme
        ]
        plt.legend(handles=legend_handles, loc='upper right')
        # Create custom legend handles only for 'H' and 'E'
        handles = [mlines.Line2D([0], [0], color=color, lw=4, label=struct, alpha=0.2) for struct, color in colors.items()]
    
        plt.legend(handles=handles, title='Secondary Structure', loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)

        plt.xlabel('Residue Index')
        plt.ylabel(ylabel)
        plt.grid(True)

        if not os.path.exists(f'images/{self.name}/Time_indicators/'):
            os.makedirs(f'images/{self.name}/Time_indicators/')
        
        # Save the figure
        plt.savefig(f'images/{self.name}/Time_indicators/{title}.png')
