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
from multiple_time_response import plot_time_response_multiple, plot_time_correlation_multiple, plot_time_entropy_multiple
raggio=8.0
# Initialize PDBProcessor
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
visualizer = Visualize(df)
print(df)

#raggio=visualizer.calculate_and_print_average_distance()
G = visualizer.create_and_print_graph(truncated=True, radius=raggio, plot=False, peso=20)  # Adjust radius as needed
#visualizer.plot_connections_vs_radius()




'''import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd

# Calcolo della betweenness centrality
betweenness = nx.betweenness_centrality(G, normalized=True, weight='weight')

# Creazione di un DataFrame ordinato
betweenness_df = pd.DataFrame(list(betweenness.items()), columns=["Node", "Betweenness Centrality"])
betweenness_df = betweenness_df.sort_values(by="Betweenness Centrality", ascending=False)[0:20]

# Creazione della colormap 'plasma'
cmap = cm.get_cmap('plasma')
norm = mcolors.Normalize(vmin=betweenness_df["Betweenness Centrality"].min(), 
                         vmax=betweenness_df["Betweenness Centrality"].max())
colors = [cmap(norm(value)) for value in betweenness_df["Betweenness Centrality"]]

# Plot ordinato della betweenness centrality
plt.figure(figsize=(12, 6))
plt.bar(betweenness_df["Node"].astype(str), betweenness_df["Betweenness Centrality"], color=colors)
plt.xlabel("Residue (Node)")
plt.ylabel("Betweenness Centrality")
plt.title("Nodes Ordered by Betweenness Centrality")
plt.xticks(rotation=90)  # Ru

# Calcolo della betweenness centrality
betweenness = nx.betweenness_centrality(G, normalized=True, weight='weight')

# Creazione di un mapping per il colore
node_sizes = [betweenness[node] * 1000 for node in G.nodes()]
node_colors = [betweenness[node] for node in G.nodes()]

# Creazione della mappa dei colori
cmap = plt.cm.plasma
norm = mcolors.Normalize(vmin=min(node_colors), vmax=max(node_colors))
sm = cm.ScalarMappable(cmap=cmap, norm=norm)

# Creazione della figura e dell'Axes
fig, ax = plt.subplots(figsize=(12, 8))

# Disegno del grafo
pos = nx.spring_layout(G, seed=42)
nx.draw(
    G, pos, with_labels=True, node_size=node_sizes, node_color=node_colors,
    cmap=cmap, edge_color='gray', font_weight='bold', ax=ax
)

# Aggiunta della colorbar esplicita all'Axes
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Betweenness Centrality")

# Titolo e visualizzazione finale
plt.title("Betweenness Centrality in the Protein Graph")
plt.show()'''






analyzer = GraphMatrixAnalyzer(G)

concatenated_df = pd.concat([df1['Secondary Structure'], df], axis=1)

pseudo_inverse = analyzer.get_pseudo_inverse()
adjacency_matrix = analyzer.get_adjacency_matrix()
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
secondary_structure = df['Secondary Structure'].values
predicted_b_factors, correlation, rmsd = analyze_b_factors(df, analyzer,df,stringa)
analyzer.plot_matrix(kirchhoff_matrix, secondary_structure, title="Matrice di Kirchhoff della Proteina",nome=stringa)
autovalori, autovettori = np.linalg.eig(kirchhoff_matrix)

# Parameters
k_B = 1  # Boltzmann constant (J/K)
T = 1  # Temperature (K)
g = 1  # A constant for simplicity
mu = 1  # Time scaling factor
t = np.linspace(0., 2, 300)  # Time points
time_correlation = TimeCorrelation(u=autovettori, lambdas=autovalori, mu=mu, sec_struct_data=df,stringa=stringa)
autocorrelations = time_correlation.time_correlation(0, 1, t)  # Example indices
#time_correlation.plot_time_correlation(0, 1, t)
normalized_autocorrelations = autocorrelations / autocorrelations[0]  # Normalize example
normalized_autocorrelations = np.zeros((94, len(t)))
for i in range(94):
    C_ii_t = time_correlation.time_correlation(i, i, t)
    normalized_autocorrelations[i, :] = time_correlation.normalize_autocorrelations(C_ii_t)

# Calcola e stampa i tempi caratteristici
tau_mean, taus = time_correlation.estimate_tau_2(t, normalized_autocorrelations)
print(f"Tempo caratteristico medio: {tau_mean:.4f}")
#time_correlation.plot_tau_histogram( t, normalized_autocorrelations)
#time_correlation.plot_autocorrelation_fits(t, normalized_autocorrelations)

transfer_entropy = TransferEntropy(u=autovettori, lambdas=autovalori, mu=mu, sec_struct_data=df,stringa=stringa)
#TE_ij = transfer_entropy.transfer_entropy(0, 1, t)  # Example indices
#transfer_entropy.plot_transfer_entropy(1, 20, t)

time_response = TimeResponse(u=autovettori, lambdas=autovalori, mu=mu, sec_struct_data=df,stringa=stringa)
#R_ij_t = time_response.time_response(0, 1, t)  # Example indices
#time_response.plot_time_response(0, 1, t)

matrix_operations = CorrelationMatrixOperations(u=autovettori, lambdas=autovalori, mu=mu, sec_struct_data=df,stringa=stringa)

correlation_matrix = matrix_operations.compute_static_correlation_matrix()
positive_matrix, negative_matrix = matrix_operations.split_correlation_matrix(correlation_matrix)

matrix_operations.plot_correlation_matrix_nan(correlation_matrix, kirchhoff_matrix,secondary_structure, positive_only=False)
matrix_operations.plot_correlation_matrix_nan(correlation_matrix,kirchhoff_matrix, secondary_structure, positive_only=True)
#matrix_operations.plot_correlation_matrix(kirchhoff_matrix, title='Correlation Matrix')
#matrix_operations.plot_correlation_matrix_nan(kirchhoff_matrix, title='Correlation Matrix', positive_only=False)

residual_analysis = ResidualAnalysis(u=autovettori, lambdas=autovalori, mu=mu, sec_struct_data=df,stringa=stringa)
#residual_analysis.analyze_mfpt(adjacency_matrix,kirchhoff_matrix, secondary_structure )

#lista = np.array([20,21,22, 23, 24])
# Prepare data for 3D plot
from mpl_toolkits.mplot3d import Axes3D
t = np.linspace(0., 2, 300) 
num_residues=len(autovalori)
z_axis = []
x_axis = []
y_axis = []

for i in range(num_residues):
    for j in range(num_residues):
        correlations, _ = residual_analysis._calculate_correlation_cost(i, j, t)
        z_axis.extend(correlations)
        x_axis.extend([i] * len(t))
        y_axis.extend(t)

# Create 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_axis, y_axis, z_axis, c=z_axis, cmap='viridis', marker='o')

ax.set_xlabel("alpha carbon Index")
ax.set_ylabel("Time")
ax.set_zlabel("Covariance")
ax.set_title("3D CovarianceAnalysis Over Residues and Time")
plt.savefig("3D_plot.png")
plt.close()
'''lista=[]
# Loop da 0 a 94 prendendo uno ogni 3
for i in range(0, 95, 3):  # i va da 0 a 94 con step di 3
    # Selezioniamo l'indice modulo della lunghezza della lista
    lista.append(i)

t=[tau_mean-1/2*tau_mean,tau_mean,tau_mean+1/2*tau_mean]
residue_pairs = [(20, 30), (20, 75),(20,72),(24,72),(14,44),(28,30),(30,72),(27,30),(27,72)]
t = np.linspace(0., 2, 300) 
plot_time_response_multiple(time_response, residue_pairs, t, 'Time Response for Selected Residue Pairs',name=stringa)
plot_time_correlation_multiple(time_correlation, residue_pairs, t, 'Time Correlation for Selected Residue Pairs',name=stringa)
plot_time_entropy_multiple(transfer_entropy, residue_pairs, t, 'Time Transfer entropy TE_{i,j} for Selected Residue Pairs',name=stringa)


residual_analysis.analyze_mfpt(adjacency_matrix,kirchhoff_matrix, secondary_structure )

 
time_idx = 0
for i in range(len(lista)):
    residual_analysis.plot_residual_correlation_vs_j(i=lista[i], t=t, time_idx=time_idx)
    residual_analysis.plot_residual_time_response_vs_j(i=lista[i], t=t, time_idx=time_idx)
    residual_analysis.plot_residual_transfer_entropy_vs_j_accettore(i=lista[i], t=t, time_idx=time_idx)
    residual_analysis.plot_residual_transfer_entropy_vs_j_donatore(i=lista[i], t=t, time_idx=time_idx)
    residual_analysis.plot_time_matrix_i_j(i=lista[i],adjacency_matrix=adjacency_matrix)
    residual_analysis.plot_time_matrix_j_i(i=lista[i],adjacency_matrix=adjacency_matrix)
    residual_analysis.plot_time_matrix_i_j_plus_response(i=lista[i],adjacency_matrix=adjacency_matrix,t=t)


#residual_analysis.plot_mean_quantity_over_segment(lista, t, time_idx,'correlation')#'correlation','linear_response','entropy'
#residual_analysis.plot_mean_quantity_over_segment(lista, t, time_idx,'linear_response')
#residual_analysis.plot_mean_quantity_over_segment(lista, t, time_idx,'entropy')'''

'''lista = np.array([71, 72, 73, 74, 75, 76, 77, 78, 79])
t=[tau_mean-1/2*tau_mean,tau_mean,tau_mean+1/2*tau_mean]
time_idx = 0
for i in range(len(lista)):
    residual_analysis.plot_residual_correlation_vs_j(i=lista[i], t=t, time_idx=time_idx)
    residual_analysis.plot_residual_time_response_vs_j(i=lista[i], t=t, time_idx=time_idx)
    residual_analysis.plot_residual_transfer_entropy_vs_j_accettore(i=lista[i], t=t, time_idx=time_idx)
    residual_analysis.plot_residual_transfer_entropy_vs_j_donatore(i=lista[i], t=t, time_idx=time_idx)
    residual_analysis.plot_time_matrix_i_j(i=lista[i],adjacency_matrix=adjacency_matrix)
    residual_analysis.plot_time_matrix_j_i(i=lista[i],adjacency_matrix=adjacency_matrix)
    residual_analysis.plot_time_matrix_i_j_plus_response(i=lista[i],adjacency_matrix=adjacency_matrix,t=t)

#residual_analysis.plot_mean_quantity_over_segment(lista, t, time_idx,'correlation')#'correlation','linear_response','entropy'
residual_analysis.plot_mean_quantity_over_segment(lista, t, time_idx,'linear_response')
#residual_analysis.plot_mean_quantity_over_segment(lista, t, time_idx,'entropy')
te_matrix, segments = analyze_secondary_structure_transfer_entropy(stringa, raggio, tau_mean)

residue_pairs = [(20, 24),(20, 30), (20, 60), (20, 75),(20,72),(24,72),(14,44)]
t = np.linspace(0., 2, 300) 
plot_time_response_multiple(time_response, residue_pairs, t, 'Time Response for Selected Residue Pairs',name=stringa)
plot_time_correlation_multiple(time_correlation, residue_pairs, t, 'Time Correlation for Selected Residue Pairs',name=stringa)
plot_time_entropy_multiple(transfer_entropy, residue_pairs, t, 'Time Transfer entropy TE_{i,j} for Selected Residue Pairs',name=stringa)


residual_analysis.analyze_mfpt(adjacency_matrix,kirchhoff_matrix, secondary_structure )'''

 