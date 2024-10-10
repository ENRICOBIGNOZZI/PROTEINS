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

class BaseCorrelationAnalysis:
    def __init__(self, u, lambdas, mu, sec_struct_data,stringa,Q):
        self.u = u  # Ensure u is at least 2D
        self.lambdas = np.array(lambdas)
        self.mu = mu
        self.sec_struct_data = sec_struct_data
        self.name=stringa
        self.Q=Q


class TimeCorrelation(BaseCorrelationAnalysis):

    def time_correlation(self, i, j, t):
        # No changes made here
        C_ij_t = np.zeros(len(t))
        a=0
        for tau in t:
            sum_result = 0.0
            for k in range(1, len(self.lambdas)):
                for p in range(1,len(lambdaa)):
                    term= (self.u[i, k] * self.Q[k, p] * self.u[j, p]) / (self.lambdas[k] + self.lambdas[p])
                    if tau > 0:
                        term *= np.exp(-lambdaa[p] * tau)
                    else:
                        term *= np.exp(lambdaa[k] * tau)
                    sum_result += term
                C_ij_t[a] = sum_result
            a+=1
        return C_ij_t
    def normalize_autocorrelations(self, C_ii_t):
        C_ii_0 = C_ii_t[0]  # Primo valore di C_ii_t
        return C_ii_t / C_ii_0
    def estimate_tau_2(self, t, normalized_autocorrelations):
        def find_1_e_crossing(t, y):
            # Find the index where y crosses 1/e
            idx = np.argmin(np.abs(y - np.exp(-1)))
            return t[idx]

        tau_values = []
        for i in range(normalized_autocorrelations.shape[0]):
            tau = find_1_e_crossing(t, normalized_autocorrelations[i,:])
            tau_values.append(tau)
        
        
        tau_mean = np.mean(tau_values)
        return tau_mean, tau_values

    def plot_tau_histogram(self, t, normalized_autocorrelations):
        tau_values = self.estimate_tau_2(t, normalized_autocorrelations)
        
        plt.figure(figsize=(10, 6))
        tau_values =np.array(tau_values, dtype=object)
        flattened_tau = [tau_values[0]] + tau_values[1]

        # Crea un array NumPy unidimensionale
        tau = np.array(flattened_tau)

        plt.hist(np.array(tau), bins=7, edgecolor='black')
        plt.xlabel('Tau (time to reach 1/e)')
        plt.ylabel('Frequency')
        #plt.title('Histogram of Tau Values')
        #plt.grid(True, alpha=0.3)
    
        # Check if the 'images' directory exists, if not, create it
        if not os.path.exists(f'images/{self.name}/2_temperature_cutoff/Stima_tau/'):
            os.makedirs(f'images/{self.name}/2_temperature_cutoff/Stima_tau/')

        # Save the figure in the 'images' directory
        plt.savefig(f'images/{self.name}/2_temperature_cutoff/Stima_tau/tau_histogram.png')



    def plot_autocorrelation_fits(self, t, normalized_autocorrelations):
        plt.figure(figsize=(12, 6))
        for i in range(normalized_autocorrelations.shape[0]):
            plt.plot(t, normalized_autocorrelations[i, :], label=f'Residuo {i}')
            tau_estimated = self.estimate_tau_2(t, normalized_autocorrelations)[0]
            plt.plot(t, np.exp(-t / tau_estimated), '--', label=f'Fit Residuo {i}')
        plt.xlabel('Time')
        plt.ylabel('Normalized Autocorrelation')
        #plt.title('Autocorrelation and Fits')
        #plt.legend()
        plt.grid(True)
        if not os.path.exists(f'images/{self.name}/2_temperature_cutoff/Stima_tau/'):
            os.makedirs(f'images/{self.name}/2_temperature_cutoff/Stima_tau/')
        plt.savefig(f'images/{self.name}/2_temperature_cutoff/Stima_tau/autocorrelation_fits.png')
        

    def plot_time_correlation(self, i, j, t):
        C_ij_t = self.time_correlation(i, j, t)
        plt.figure(figsize=(8, 6))
        plt.plot(t, C_ij_t, label=f'Time Correlation C({i},{j})')
        plt.xlabel('Time')
        plt.ylabel('Correlation')
        #plt.title(f'Time Correlation between {i} and {j}')
        plt.legend()
        plt.grid(True)
        if not os.path.exists(f'images/{self.name}/2_temperature_cutoff/'):
            os.makedirs(f'images/{self.name}/2_temperature_cutoff/')
        plt.savefig(f'images/{self.name}/2_temperature_cutoff/Time Correlation C({i},{j}).png')

    def compute_residual_transfer_entropy_matrix_accettore(self, t, i, time_idx):
        n = self.u.shape[0]
        t_subset=t
        transfer_entropy_matrix =  np.zeros((n, n, len(t_subset)))
        for j in range(n):
            if j==i:
                transfer_entropy_matrix[i, j, :]=0
            else:
                transfer_entropy_matrix[i,j,:] = self.transfer_entropy(i, j, t_subset)
        return transfer_entropy_matrix[i,:,:]
    
    def transfer_entropy(self, i, j, t):
        C_ii_0 = self.time_correlation(i, i,t=[0])
        C_jj_0 = self.time_correlation(j, j,t=[0])
        C_ii_t = self.time_correlation(i, i, t)

        C_ij_0 = self.time_correlation(i, j,t=[0])
        C_ij_t = self.time_correlation(i, j, t)

        alpha_ij_t = (C_ii_0 * C_ij_t - C_ij_0 * C_ii_t) ** 2
        beta_ij_t = (C_ii_0 * C_jj_0-(C_ij_0**2)) * (C_ii_0**2- C_ii_t ** 2)


        ratio = np.clip(alpha_ij_t / beta_ij_t, 0, 1 - 1e-10)
        return -0.5 * np.log(1 - ratio)
    def plot_residual_transfer_entropy_vs_j_accettore(self, i, t, time_idx,nome):
        transfer_entropy_matrix = self.compute_residual_transfer_entropy_matrix_accettore(t, i, time_idx)
        _plot_with_secondary_structure(df,transfer_entropy_matrix, f'Transfer Entropy with i={i}',nome, f'Transfer Entropy TE_ij for i={i} as a function of j ')



def plot_residual_correlation_vs_j(df, i, t,s, time_idx,nome,Q,lambdaa,U):
    correlation_i=np.zeros((len(lambdaa),len(t)))
    a=0
    z=np.array(t)-np.array(s)
    for tau in z:
        sum_result = 0.0
        for j in range(0,len(lambdaa)): 
            for k in range(1,len(lambdaa)):
                for p in range(1,len(lambdaa)):
                    term = (U[i, k] * Q[k, p] * U[j, p]) / (lambdaa[k] + lambdaa[p])
                    if tau > 0:
                        term *= np.exp(-lambdaa[p] * tau)
                    else:
                        term *= np.exp(lambdaa[k] * tau)
                    sum_result += term
            correlation_i[j,a] = sum_result
        a+=1
            
    _plot_with_secondary_structure(df,correlation_i, f'Correlation with i={i}',nome, f'Residual Correlation with 2 temperature C_ij for i={i} as a function of j at time index {time_idx}')



def plot_beta_factors(df, nome,Q,lambdaa,U):
    correlation =np.zeros((len(lambdaa),len(lambdaa)))
    for i in range(0,len(lambdaa)):
        for j in range(0,len(lambdaa)): 
            sum_result = 0.0
            for k in range(1,len(lambdaa)):
                for p in range(1,len(lambdaa)):
                    term = (U[i, k] * Q[k, p] * U[j, p]) / (lambdaa[k] + lambdaa[p])
                    sum_result += term
            correlation[i][j] = sum_result

    
    diagonale = np.diagonal(correlation)
    compare_b_factors_with_sec_structure(df['B-Factor'], diagonale, df,nome)



def _plot_with_secondary_structure(sec_struct_data, matrix, ylabel, name, title):
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

        if not os.path.exists(f'images/{name}/2_temperature_cutoff/'):
            os.makedirs(f'images/{name}/2_temperature_cutoff/')
        
        # Save the figure
        plt.savefig(f'images/{name}/2_temperature_cutoff/{title}.png')
def plot_matrix( matrix, secondary_structure, nome, title="Matrix"):
    binary_matrix=matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(matrix, cmap='viridis')
    fig.colorbar(cax)
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
    if not os.path.exists(f'images/{stringa}/2_temperature_cutoff/'):
        os.makedirs(f'images/{stringa}/2_temperature_cutoff/')
    plt.savefig(f'images/{stringa}/2_temperature_cutoff/{nome}.png')




def compare_b_factors_with_sec_structure(actual_b_factors, predicted_b_factors, sec_struct_data, name):
    # Scala i fattori B predetti per farli corrispondere all'intervallo dei fattori B reali
    scale_factor = np.mean(actual_b_factors) / np.mean(predicted_b_factors)
    predicted_b_factors_scaled = predicted_b_factors * scale_factor

    
    # Calcola la deviazione quadratica media (RMSD)
    rmsd = np.sqrt(np.mean((actual_b_factors - predicted_b_factors_scaled)**2))

    # Crea il plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot dei fattori B reali
    ax.set_xlabel('Indice del Residuo')
    ax.set_ylabel('Fattori B')
    ax.plot(actual_b_factors, label='Beta Factors', color='blue')

    # Plot dei fattori B predetti
    ax.plot(predicted_b_factors_scaled, label='Predicted Beta Factors', color='red')

    # Aggiungi il titolo e la legenda
    #plt.title('Confronto tra Fattori B Reali e Predetti')
    ax.legend(loc="upper right")

    # Aggiungi una griglia
    ax.grid(True, alpha=0.3)

    # Plot della struttura secondaria
    sec_struct_info = sec_struct_data['Secondary Structure']
    residue_ids = sec_struct_data['Residue ID'].astype(int)


    colors = {'H': 'red', 'E': 'blue'}  # Escludiamo 'C' dalla mappatura dei colori

    # Mappa i colori o None se la struttura è 'C'
    sec_struct_colors = [colors.get(sec_struct_info.get(rid), None) for rid in residue_ids]

    # Aggiungi le bande colorate per la struttura secondaria, escludendo 'C'
    current_color = None
    start_idx = 0
    a=0.2

    for idx, resid in enumerate(residue_ids):
        if sec_struct_colors[idx] != current_color:
            # Se non siamo nel primo residuo e l'attuale colore non è None, disegna una banda
            if idx > 0 and current_color is not None:
                ax.axvspan(start_idx, idx, color=current_color, alpha=a)  # Alpha uniforme
            # Cambia il colore attuale e aggiorna l'indice di partenza
            current_color = sec_struct_colors[idx]
            start_idx = idx

    # Aggiungi l'ultimo segmento se non è 'C' (None)
    if current_color is not None:
        ax.axvspan(start_idx, len(residue_ids), color=current_color, alpha=a)  # Alpha uniforme

    # Creazione delle legende con lo stesso alpha per consistenza
    legend_handles = [
        mpatches.Patch(color='red', label='Helix (H)', alpha=a),  # Alpha uniforme
        mpatches.Patch(color='blue', label='Beta sheet (E)', alpha=a)  # Alpha uniforme
    ]

    # Aggiungi la legenda con i nuovi handle
    ax.legend(handles=legend_handles, loc='upper right')

    # Crea la legenda personalizzata per la struttura secondaria
    handles = [mlines.Line2D([0], [0], color=color, lw=4, label=struct, alpha=a) for struct, color in colors.items()]
    ax.legend(handles=handles, title='Secondary Structure', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    
    plt.tight_layout()
    plt.legend()


    if not os.path.exists(f'images/{name}/2_temperature_cutoff/beta_factors/'):
        os.makedirs(f'images/{name}/2_temperature_cutoff/beta_factors/')

    # Salva la figura
    plt.savefig(f'images/{name}/2_temperature_cutoff/beta_factors/Confronto_con_Struttura_Secondaria.png')

    print(f"Deviazione Quadratica Media (RMSD): {rmsd:.4f}")

    return rmsd

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
temperatura = np.where(contatti >= 5, 1, 1)
G = visualizer.create_and_print_graph(truncated=True, radius=8.0, plot=False, peso=20)  # Adjust radius as needed
analyzer = GraphMatrixAnalyzer(G)
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
residui= np.arange(1, len(temperatura)+1)
plt.figure(figsize=(12, 6))
plt.plot(residui, temperatura, marker='o')
plt.title('Grafico della temperatura al variare del residuo')
plt.xlabel('Numero del residuo')
plt.ylabel('Temperatura')
plt.yticks([0.5, 1.0])
plt.grid(True)
plt.tight_layout()
if not os.path.exists(f'images/{stringa}/2_temperature_cutoff/'):
    os.makedirs(f'images/{stringa}/2_temperature_cutoff/')

# Save the figure
plt.savefig(f'images/{stringa}/2_temperature_cutoff/temperatures.png')

B=np.sqrt(temperatura)
B = np.diag(B)
B[0, 0] = 100
print(B)
lambdaa, U = np.linalg.eig(kirchhoff_matrix)
BBT = B @ B.T
Q = U @ BBT  @ U.T


Z

