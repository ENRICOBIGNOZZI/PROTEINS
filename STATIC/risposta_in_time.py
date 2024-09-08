import numpy as np
import matplotlib.pyplot as plt
from causal_indicators_advances import TimeResponse

def plot_time_response_multiple(time_response, residue_pairs, t, title):
    plt.figure(figsize=(12, 8))
    for i, j in residue_pairs:
        R_ij_t = time_response.time_response(i, j, t)
        plt.plot(t, R_ij_t, label=f'R({i},{j})')
    
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


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
# Initialize PDBProcessor
pdb_processor = PDBProcessor(pdb_id="2m10")
pdb_processor.download_pdb()
pdb_processor.load_structure()

# Extract data
df1 = pdb_processor.secondary_structure()
df = pdb_processor.extract_atom_data()
print(df)
df = df[df['Model ID'] == 0]
df = df[df['Atom Name'] == 'CA']


df = df.reset_index(drop=True)
concatenated_df = pd.concat([df1['Secondary Structure'], df], axis=1)
df = concatenated_df.dropna().reset_index(drop=True)
print(df)
# Initialize Visualize and analyze graph
visualizer = Visualize(df)
raggio=visualizer.calculate_and_print_average_distance()
#visualizer.plot_connections_vs_radius()
G = visualizer.create_and_print_graph(truncated=True, radius=8.0, plot=False, peso=20)  # Adjust radius as needed
# Assumendo che G sia il tuo grafo

# Initialize GraphMatrixAnalyzer
analyzer = GraphMatrixAnalyzer(G)

# Calcola la matrice di Kirchhoff
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
adjacency_matrix = analyzer.get_adjacency_matrix()
# Plotta la mappa dei contatti
#analyzer.plot_matrix(adjacency_matrix, title="Mappa dei Contatti della Proteina")

# Calcola autovalori e autovettori
autovalori, autovettori = np.linalg.eigh(kirchhoff_matrix)
mu=1
# Inizializza TimeResponse
time_response = TimeResponse(u=autovettori, lambdas=autovalori, mu=mu, sec_struct_data=df)

# Definisci l'intervallo di tempo
t = np.linspace(0.01, 2, 300)  # Puoi modificare questo intervallo se necessario
contatti_residuo_21 = []
for i, contatto in enumerate(adjacency_matrix[20]):  # Indice 20 perché gli array in Python iniziano da 0
    if contatto == 1:
        contatti_residuo_21.append(i + 1)  # Aggiungiamo 1 per ottenere il numero del residuo reale

print(f"Contatti del residuo 21: {contatti_residuo_21}")

# Definisci le coppie di residui
residue_pairs = [(20, 24), (20, 40), (20, 75)]

# Plotta la risposta nel tempo per tutte le coppie
plot_time_response_multiple(time_response, residue_pairs, t, 'Time Response for Selected Residue Pairs')

# Se vuoi plottare ogni coppia separatamente, puoi usare questo ciclo
for i, j in residue_pairs:
    time_response.plot_time_response(i, j, t)

# Dopo aver calcolato la matrice di adiacenza
