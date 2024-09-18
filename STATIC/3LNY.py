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
from beta_functions import analyze_b_factors
from secondary_structure import analyze_secondary_structure_transfer_entropy
from multiple_time_response import plot_time_response_multiple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def plot_correlation_matrix(self, matrix, secondary_structure, nome, title="Correlation Matrix"):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Crea una matrice binaria per correlazioni positive e negative
    pos_matrix = np.where(matrix > 0, 1, 0)
    neg_matrix = np.where(matrix < 0, 1, 0)
    
    # Indici degli elementi non zero per le correlazioni positive e negative
    pos_rows, pos_cols = np.where(pos_matrix == 1)
    neg_rows, neg_cols = np.where(neg_matrix == 1)
    
    # Plot dei punti: verde per correlazioni positive, viola per correlazioni negative
    ax.scatter(pos_cols, pos_rows, s=30, c='green', label='Positive Correlation')  # Correlazioni positive in verde
    ax.scatter(neg_cols, neg_rows, s=30, c='purple', label='Negative Correlation')  # Correlazioni negative in viola

    # Aggiungi i rettangoli come indicato nel codice originale
    rectangle1 = patches.Rectangle((19, 71), 5, 9, linewidth=2, edgecolor='r', facecolor='none')
    rectangle2 = patches.Rectangle((71, 19), 9, 5, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rectangle1)
    ax.add_patch(rectangle2)
    
    # Creazione delle sezioni per la struttura secondaria sugli assi x e y
    start = 0
    current_structure = secondary_structure[0] if len(secondary_structure) >= 2 else secondary_structure[0]
    
    for i, structure in enumerate(secondary_structure):
        if len(structure) >= 2:
            structure = structure[0]
        
        if structure != current_structure:
            if current_structure == 'H' or current_structure == 'E':  # Solo per Elica e Foglietto Beta
                color = 'red' if current_structure == 'H' else 'blue'
                ax.plot([start, i], [0, 0], color=color, linewidth=8)
                ax.plot([0, 0], [start, i], color=color, linewidth=8)
                ax.text((start+i)/2, -0.5, current_structure, ha='center', va='top', fontsize=12, fontweight='bold')
                ax.text(-0.5, (start+i)/2, current_structure, ha='right', va='center', fontsize=12, fontweight='bold')
            start = i
            current_structure = structure
    if current_structure == 'H' or current_structure == 'E':
        color = 'red' if current_structure == 'H' else 'blue'
        ax.plot([start, i+1], [0, 0], color=color, linewidth=8)
        ax.plot([0, 0], [start, i+1], color=color, linewidth=8)
        ax.text((start+i+1)/2, -0.5, current_structure, ha='center', va='top', fontsize=12, fontweight='bold')
        ax.text(-0.5, (start+i+1)/2, current_structure, ha='right', va='center', fontsize=12, fontweight='bold')
    
    # Aggiungi la legenda per correlazioni positive e negative, eliche e foglietti Beta
    handles = [
        patches.Patch(color='green', label='Positive Correlation'),
        patches.Patch(color='purple', label='Negative Correlation'),
        patches.Patch(color='red', label='Helix'),
        patches.Patch(color='blue', label='Beta Sheet')
    ]
    ax.legend(handles=handles, loc='upper right')
    
    ax.set_title(title)
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('Residue Index')
    ax.set_ylim(0, matrix.shape[0])
    ax.set_xlim(0, matrix.shape[1])
    
    plt.tight_layout()
    
    # Creazione delle cartelle per salvare le immagini se non esistono
    if not os.path.exists(f'images/{nome}/correlazioni/'):
        os.makedirs(f'images/{nome}/correlazioni/')
    
    # Salva la figura nella directory 'images'
    plt.savefig(f'images/{nome}/correlazioni/{title}.png')
    plt.show()

raggio=8.0
# Initialize PDBProcessor
stringa="3LNY"
pdb_processor = PDBProcessor(pdb_id="3LNY")#2m07
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
df['Residue ID'].values[-6:] = [95, 96, 97, 98, 99,100] 
print(df)
print( df['Residue ID'].values)
# Initialize Visualize and analyze graph
visualizer = Visualize(df)
raggio=visualizer.calculate_and_print_average_distance()
#visualizer.plot_connections_vs_radius()
G = visualizer.create_and_print_graph(truncated=True, radius=8, plot=False, peso=20)  # Adjust radius as needed

# Initialize GraphMatrixAnalyzer
analyzer = GraphMatrixAnalyzer(G)

# Calcola la matrice di Kirchhoff
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
predicted_b_factors, correlation, rmsd = analyze_b_factors(df, analyzer,name=stringa)
# Calcola autovalori e autovettori
autovalori, autovettori = np.linalg.eigh(kirchhoff_matrix)
print(autovalori.shape)
print(autovettori.shape)
#analyzer.plot_eigenvalues(autovalori)

# Ordina autovalori e autovettori
#idx = autovalori.argsort()[::-1]
#autovalori = autovalori[idx]
#autovettori = autovettori[:, idx]

# Normalizza gli autovettori
#autovettori = autovettori / np.linalg.norm(autovettori, axis=0)

# Escludi i primi 6 modi (o 3 per proteine più piccole)
#autovalori = autovalori[3:]
#autovettori = autovettori[:, 3:]


# Ottieni la matrice di adiacenza
adjacency_matrix = analyzer.get_adjacency_matrix()


# Se vuoi anche visualizzare la matrice di Kirchhoff
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()

pseudo_inverse = analyzer.get_pseudo_inverse()
adjacency_matrix = analyzer.get_adjacency_matrix()

kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
eigenvalues = analyzer.get_eigenvalues_adjacency()
eigenvectors = analyzer.get_eigenvectors_adjacency()
eigenvectors=eigenvectors.T

secondary_structure = df['Secondary Structure'].values

analyzer.plot_matrix(kirchhoff_matrix, secondary_structure, title="Matrice di Kirchhoff della Proteina",nome=stringa)


# Perform Eigenvalue Decomposition
autovalori, autovettori = np.linalg.eig(kirchhoff_matrix)
#np.linalg.eigh(kirchhoff_matrix)  # Usa eigh invece di eig

#analyzer.plot_eigenvectors(autovettori[0:1,1:7])
# Ordina gli autovalori e gli autovettori
#idx = autovalori.argsort()[::-1]   
#autovalori = autovalori[idx]
#autovettori = autovettori[:, idx]

# Normalizza gli autovettori
#autovettori = autovettori / np.linalg.norm(autovettori, axis=0)

# Aggiungi questi controlli

# Parameters
k_B = 1  # Boltzmann constant (J/K)
T = 1  # Temperature (K)
g = 1  # A constant for simplicity
mu = 1  # Time scaling factor
t = np.linspace(0.01, 0.5, 300)  # Time points
# Initialize Analysis
time_correlation = TimeCorrelation(u=autovettori, lambdas=autovalori, mu=mu, sec_struct_data=df,stringa=stringa)
autocorrelations = time_correlation.time_correlation(0, 1, t)  # Example indices
normalized_autocorrelations = autocorrelations / autocorrelations[0]  # Normalize example
#t = np.array([0.20, 0.25, 0.30, 0.35])

# Calcola le autocorrelazioni e normalizzale
normalized_autocorrelations = np.zeros((97, len(t)))
for i in range(97):
    C_ii_t = time_correlation.time_correlation(i, i, t)
    normalized_autocorrelations[i, :] = time_correlation.normalize_autocorrelations(C_ii_t)

# Calcola e stampa i tempi caratteristici
tau_mean, taus = time_correlation.estimate_tau(t, normalized_autocorrelations)
print(f"Tempo caratteristico medio: {tau_mean:.4f}")
tau_mean, taus = time_correlation.estimate_tau_2(t, normalized_autocorrelations)
print(f"Tempo caratteristico medio: {tau_mean:.4f}")
#time_correlation.plot_tau_histogram( t, normalized_autocorrelations)
#time_correlation.plot_autocorrelation_fits(t, normalized_autocorrelations)



#time_correlation.plot_time_correlation(0, 1, t)





transfer_entropy = TransferEntropy(u=autovettori, lambdas=autovalori, mu=mu, sec_struct_data=df,stringa=stringa)
TE_ij = transfer_entropy.transfer_entropy(0, 1, t)  # Example indices
#transfer_entropy.plot_transfer_entropy(0, 1, t)







time_response = TimeResponse(u=autovettori, lambdas=autovalori, mu=mu, sec_struct_data=df,stringa=stringa)
# Perform Time Response analysis
R_ij_t = time_response.time_response(0, 71, t)  # Example indices

#time_response.plot_time_response(1, 71, t)


# Calcola e stampa i tempi caratteristici
tau_mean, taus = time_correlation.estimate_tau(t, normalized_autocorrelations)
print(f"Tempo caratteristico medio: {tau_mean:.4f}")
tau_mean, taus = time_correlation.estimate_tau_2(t, normalized_autocorrelations)
print(f"Tempo caratteristico medio: {tau_mean:.4f}")
#time_correlation.plot_tau_histogram( t, normalized_autocorrelations)
#time_correlation.plot_autocorrelation_fits(t, normalized_autocorrelations)



time_correlation.plot_time_correlation(20, 70, t)





transfer_entropy = TransferEntropy(u=autovettori, lambdas=autovalori, mu=mu, sec_struct_data=df,stringa=stringa)
TE_ij = transfer_entropy.transfer_entropy(0, 1, t)  # Example indices
transfer_entropy.plot_transfer_entropy(20, 70, t)







time_response = TimeResponse(u=autovettori, lambdas=autovalori, mu=mu, sec_struct_data=df,stringa=stringa)
# Perform Time Response analysis
R_ij_t = time_response.time_response(0, 71, t)  # Example indices

time_response.plot_time_response(20, 70, t)






matrix_operations = CorrelationMatrixOperations(u=autovettori, lambdas=autovalori, mu=mu, sec_struct_data=df,stringa=stringa)



matrix_operations = CorrelationMatrixOperations(u=autovettori, lambdas=autovalori, mu=mu, sec_struct_data=df,stringa=stringa)
# Perform Correlation Matrix Operations


#correlation_matrix = matrix_operations.compute_static_correlation_matrix()
#positive_matrix, negative_matrix = matrix_operations.split_correlation_matrix(correlation_matrix)
#matrix_operations.plot_correlation_matrix(positive_matrix, 'Correlation Matrix')
#matrix_operations.plot_correlation_matrix_nan(correlation_matrix, title='Correlation Matrix', positive_only=False)
#matrix_operations.plot_correlation_matrix(kirchhoff_matrix, title='Correlation Matrix')
#matrix_operations.plot_correlation_matrix_nan(kirchhoff_matrix, title='Correlation Matrix', positive_only=False)

residual_analysis = ResidualAnalysis(u=autovettori, lambdas=autovalori, mu=mu, sec_struct_data=df,stringa=stringa)

# Perform correlation analysis



# Definisci i punti temporali
#pairs = [(70, 20), (70, 23), (71, 24), (72, 25), (73, 26), (74, 27)]
#residual_analysis.plot_multiple_time_correlations(pairs, t)

# Esegui l'analisi e visualizza i risultati

'''lista = np.array([21,22, 23, 24])
t=[tau_mean-0.1,tau_mean,tau_mean+0.1,2*tau_mean,3*tau_mean]
time_idx = 3
residual_analysis.plot_mean_quantity_over_segment(lista, t, time_idx,'correlation')#'correlation','linear_response','entropy'
residual_analysis.plot_mean_quantity_over_segment(lista, t, time_idx,'linear_response')
residual_analysis.plot_mean_quantity_over_segment(lista, t, time_idx,'entropy')
'''
def calcola_distanza(residuo1, residuo2, df):
    coord1 = df.loc[df['Residue ID'] == residuo1, ['X', 'Y', 'Z']].values[0]
    coord2 = df.loc[df['Residue ID'] == residuo2, ['X', 'Y', 'Z']].values[0]
    distanza = np.linalg.norm(coord1 - coord2)
    return distanza

# Calcola la distanza tra il residuo 20 e il 40
distanza_20_40 = calcola_distanza(20, 40, df)
print(f"Distanza tra il residuo 20 e il 40: {distanza_20_40:.2f}")
lista = np.array([21,22, 23, 24])
lista = np.array([20,21,22, 23, 24])
t=[tau_mean]#[tau_mean]
time_idx = 0
for i in range(len(lista)):
    residual_analysis.plot_residual_correlation_vs_j(i=lista[i], t=t, time_idx=time_idx)
    residual_analysis.plot_residual_time_response_vs_j(i=lista[i], t=t, time_idx=time_idx)
    residual_analysis.plot_residual_transfer_entropy_vs_j(i=lista[i], t=t, time_idx=time_idx)

#residual_analysis.plot_mean_quantity_over_segment(lista, t, time_idx,'correlation')#'correlation','linear_response','entropy'
residual_analysis.plot_mean_quantity_over_segment(lista, t, time_idx,'linear_response')
#residual_analysis.plot_mean_quantity_over_segment(lista, t, time_idx,'entropy')

lista = np.array([71, 72, 73, 74, 75, 76, 77, 78, 79])
t=[tau_mean]
time_idx = 0
for i in range(len(lista)):
    residual_analysis.plot_residual_correlation_vs_j(i=lista[i], t=t, time_idx=time_idx)
    #residual_analysis.plot_residual_time_response_vs_j(i=lista[i], t=t, time_idx=time_idx)
    residual_analysis.plot_residual_transfer_entropy_vs_j(i=lista[i], t=t, time_idx=time_idx)

residual_analysis.plot_mean_quantity_over_segment(lista, t, time_idx,'correlation')#'correlation','linear_response','entropy'
#residual_analysis.plot_mean_quantity_over_segment(lista, t, time_idx,'linear_response')
residual_analysis.plot_mean_quantity_over_segment(lista, t, time_idx,'entropy')


te_matrix, segments = analyze_secondary_structure_transfer_entropy(stringa, raggio, tau_mean)



residue_pairs = [(20, 24), (20, 40), (20, 75)]
t = np.linspace(0.01, 0.5, 300)  
plot_time_response_multiple(time_response, residue_pairs, t, 'Time Response for Selected Residue Pairs',name=stringa)
