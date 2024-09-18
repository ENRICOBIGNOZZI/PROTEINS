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
raggio=8.0
# Initialize PDBProcessor
stringa="3LNX"
pdb_processor = PDBProcessor(pdb_id="3LNY")#2m07
pdb_processor.download_pdb()
pdb_processor.load_structure()

# Extract data
df1 = pdb_processor.secondary_structure()
df = pdb_processor.extract_atom_data()
print(df)
df = df[df['Model ID'] == 0]
print(df)
df = df[df['Atom Name'] == 'CA']
df = df[df['Chain ID'] == 'A']
df = df.reset_index(drop=True)
concatenated_df = pd.concat([df1['Secondary Structure'], df], axis=1)
df = concatenated_df.dropna().reset_index(drop=True)
df = df.loc[:,~df.T.duplicated()]
print(df)
print(df.columns)
# Initialize Visualize and analyze graph
visualizer = Visualize(df)
raggio=visualizer.calculate_and_print_average_distance()
#visualizer.plot_connections_vs_radius()
G = visualizer.create_and_print_graph(truncated=True, radius=8, plot=False, peso=20)  # Adjust radius as needed

# Initialize GraphMatrixAnalyzer
analyzer = GraphMatrixAnalyzer(G)
concatenated_df = pd.concat([df1['Secondary Structure'], df], axis=1)
df = concatenated_df.dropna().reset_index(drop=True)

df = df.T.drop_duplicates().T
# Calcola la matrice di Kirchhoff
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
predicted_b_factors, correlation, rmsd = analyze_b_factors(df, analyzer,name=stringa)
# Calcola autovalori e autovettori
autovalori, autovettori = np.linalg.eigh(kirchhoff_matrix)
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

# Plotta la mappa dei contatti
#analyzer.plot_matrix(adjacency_matrix, title="Mappa dei Contatti della Proteina")

# Se vuoi anche visualizzare la matrice di Kirchhoff
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
#analyzer.plot_matrix(kirchhoff_matrix, title="Matrice di Kirchhoff della Proteina")

pseudo_inverse = analyzer.get_pseudo_inverse()
adjacency_matrix = analyzer.get_adjacency_matrix()

kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
eigenvalues = analyzer.get_eigenvalues_adjacency()
eigenvectors = analyzer.get_eigenvectors_adjacency()
eigenvectors=eigenvectors.T
secondary_structure = df['Secondary Structure'].values

analyzer.plot_matrix(kirchhoff_matrix, secondary_structure, title="Matrice di Kirchhoff della Proteina",nome=stringa)

df = df.T.drop_duplicates().T


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

secondary_structure = df['Secondary Structure'].values

analyzer.plot_matrix(kirchhoff_matrix, secondary_structure, title="Matrice di Kirchhoff della Proteina",nome=stringa)
df = df.T.drop_duplicates().T

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
normalized_autocorrelations = np.zeros((94, len(t)))
for i in range(94):
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


df = df.T.drop_duplicates().T


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




