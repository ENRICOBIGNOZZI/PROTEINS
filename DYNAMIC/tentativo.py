import numpy as np
from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt
from Downlaod_data import PDBProcessor
from Visualize import Visualize
from matrix import GraphMatrixAnalyzer
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.lines as mlines
from scipy.stats import pearsonr
import matplotlib.lines as mlines
import os
import matplotlib.patches as mpatches
from numba import njit, prange
import math

# Esempio di utilizzo
stringa="3LNX"
pdb_processor = PDBProcessor(pdb_id="3LNX")#2m07
pdb_processor.download_pdb()
pdb_processor.load_structure()
df1 = pdb_processor.secondary_structure()
df = pdb_processor.extract_atom_data()
#df = df[df['Atom Name'] == 'CA']

#df = df[df['Atom Name'] == 'CA'].drop_duplicates(subset=['Atom Name'], keep='first')

#df = df.groupby('Residue ID', as_index=False).second()

df = df[df['Model ID'] == 0]

df = df[df['Chain ID'] == 'A']
df = df[df['Atom Name'] == 'CA']

# Mantiene il primo 'CA' per ogni combinazione di 'Residue Name' e 'Residue ID'
#df = df.drop_duplicates(subset=['Residue Name', 'Residue ID'], keep='first')
#df = df.groupby(['Residue Name', 'Residue ID']).nth(1)
df = df.reset_index(drop=True)
concatenated_df = pd.concat([df1['Secondary Structure'], df], axis=1)
df = concatenated_df.dropna().reset_index(drop=True)
df = df.loc[:,~df.T.duplicated()]
df = concatenated_df.dropna().reset_index(drop=True)
df = df.T.drop_duplicates().T
# Specifica il percorso del file
file_path = '1pdz.ca'
with open(file_path, 'r') as file:
    lines = file.readlines()
data = [line.split() for line in lines if not line.startswith("#") and line.strip() != ""]
df2= pd.DataFrame(data)
df2.columns = df.columns

df[['X', 'Y', 'Z','Residue ID']] = df2[['X', 'Y', 'Z','Residue ID']].astype(float)
visualizer = Visualize(df)

#raggio=visualizer.calculate_and_print_average_distance()
G = visualizer.create_and_print_graph(truncated=True, radius=8, plot=False, peso=20)  # Adjust radius as needed
analyzer = GraphMatrixAnalyzer(G)
concatenated_df = pd.concat([df1['Secondary Structure'], df], axis=1)



pseudo_inverse = analyzer.get_pseudo_inverse()
adjacency_matrix = analyzer.get_adjacency_matrix()
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()

#print(kirchhoff_matrix)
#for i in range(kirchhoff_matrix.shape[0]):
#    print(kirchhoff_matrix[:,i])

secondary_structure = df['Secondary Structure'].values

autovalori, autovettori = np.linalg.eig(kirchhoff_matrix)
norms = np.linalg.norm(autovettori, axis=0)




positions = df[['X', 'Y', 'Z']].values

N = autovalori.shape[0]  # number of residues
K = kirchhoff_matrix
# Parametri
T = 100000  # Numero di passi temporali

dt = 0.01  # Intervallo temporale

omega = 2*np.pi # Frequenza della forza oscillante
a, b = 20, 75  # Indici delle particelle che ricevono la forza oscillante
epsilon_0=1
k_b=1
Temperature=1
# Generazione del rumore bianco per tutte le particelle
#np.random.seed(0)
eta = np.random.normal(0, 1, (T, N))  # Rumore bianco di dimensione (T, N)

# Inizializzazione della matrice X di dimensione N x T
X = np.zeros((N, T))

# Simulazione del processo stocastico per N particelle
for t in range(1, T):
    oscillating_term = np.zeros(N)
    oscillating_term[a] = epsilon_0*(1 - np.cos(omega * t * dt))  # Forza oscillante per la particella a
    oscillating_term[b] = -(epsilon_0 * (1 - np.cos(omega * t * dt)))  # Forza oscillante per la particella b
    
    # Aggiornamento simultaneo di tutte le particelle
    X[:, t] = X[:, t-1] - np.dot(K, X[:, t-1]) * dt + np.sqrt(2*k_b*Temperature)*eta[t] * dt + oscillating_term * dt


def autocorrelation(x):
    # Calcolo della correlazione con se stesso
    result = np.correlate(x, x, mode='full')
    # Normalizzazione per la varianza e la lunghezza del segnale
    maxcorr = np.argmax(result)
    #print 'maximum = ', result[maxcorr]
    result = result / result[maxcorr]     # <=== normalization
    return result[result.size // 2:] #/ (np.var(x) * len(x))

# Funzione per calcolare la cross-correlazione normalizzata
def cross_correlation(x, y):
    # Calcolo della correlazione incrociata tra x e y
    result = np.correlate(x, y, mode='full')
    

    # Normalizzazione per le deviazioni standard e la lunghezza dei segnali
    return result[result.size // 2:] #/ (np.std(x) * np.std(y) * len(x))
# Calcolare l'autocorrelazione per le particelle nelle posizioni i=20, 30, 40
empirical_autocorrs = []
indices = [40, 50, 70]
for idx in indices:
    empirical_autocorr = autocorrelation(X[idx, :])
    empirical_autocorrs.append(empirical_autocorr)

# Calcolare la cross-correlazione tra due particelle specifiche (ad esempio, x_20 e x_30)
cross_corr_20_30 = cross_correlation(X[20, :], X[30, :])

# Plot dei risultati
plt.figure(figsize=(10, 6))

# Grafico dell'autocorrelazione per i=40, 50, 70
for idx, empirical_autocorr in zip(indices, empirical_autocorrs):
    plt.plot(np.arange(0, T) * dt, empirical_autocorr, label=f'Autocorrelazione Empirica x_{idx}')


plt.title('Autocorrelazione e Cross-Correlazione dei Processi Stocastici con Forza Oscillante')
plt.xlabel('Tempo (t)')
plt.ylabel('Correlazione')
plt.legend()
plt.grid(True)
plt.show()
