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
df = df[df['Model ID'] == 0]
df = df[df['Chain ID'] == 'A']
df = df[df['Atom Name'] == 'CA']

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
secondary_structure = df['Secondary Structure'].values
autovalori, autovettori = np.linalg.eig(kirchhoff_matrix)
norms = np.linalg.norm(autovettori, axis=0)
positions = df[['X', 'Y', 'Z']].values

N = autovalori.shape[0]  # number of residues
K = kirchhoff_matrix
# Parametri
T = 4000#5050  # Piu' è piccola piu' cade piu' velocemente
gamma=1
dt = 0.02 # Piu' è grande piu' l'autoccorelazione empirica decade lentamente
omega = 2*np.pi 
a, b = 20, 75  
epsilon_0=3#5
k_b=1
Temperature=1.#4#1
N_process=1

X = np.zeros((N_process, N, T))
eta = np.random.normal(0, 1, (N_process, T, N))  # Rumore bianco per N_process simulazioni

# Simulazione del processo stocastico per ogni realizzazione
for n in range(N_process):
    for t in range(0, T):
        oscillating_term = np.zeros(N)
        oscillating_term[a] = epsilon_0 * (1 - np.cos(omega * t * dt))
        oscillating_term[b] = -epsilon_0 * (1 - np.cos(omega * t * dt))
        X[n, :, t] = X[n, :, t - 1] - np.dot(K, X[n, :, t - 1]) * dt  + oscillating_term * dt#+ np.sqrt(2 * k_b * Temperature) * eta[n, t] * dt

# Funzione per calcolare l'autocorrelazione di una serie temporale
def autocorrelation(x):
    result = np.correlate(x, x, mode='full')
    #maxcorr = np.argmax(result)
    #result = result / result[maxcorr]
    return result[result.size // 2:]

# Calcolo della correlazione empirica mediata su tutte le realizzazioni
empirical_autocorrs = []
for idx in [40]:  # Seleziona la particella da analizzare
    autocorr_sum = np.zeros(T)
    for n in range(N_process):
        autocorr_sum += autocorrelation(X[n, idx, :])#/len(X[n, idx, :])
    avg_autocorr = autocorr_sum / N_process
    empirical_autocorrs.append(avg_autocorr)

# Calcolo della correlazione teorica
'''teoretical_corrs = []
for t in np.arange(0, T * dt, dt):
    teoretical_corr = teoretical_C_i_j(autovalori, autovettori, 40, 40, gamma, k_b, Temperature, 0, t, omega, epsilon_0)
    teoretical_corrs.append(teoretical_corr)'''
# Calcolo della correlazione teorica mediata su diverse realizzazioni
# Calcolo della correlazione teorica per ogni tau

@njit
def teoretical_C_i_j(autovalori, autovettori, i, j, gamma, k_b, T, s, t, omega, epsilon_0):
    # Inizializza la correlazione
    Cij = 0

    # Ciclo sui modi (autovalori) per calcolare la risposta completa
    for k in range(len(autovalori)):
        # Termini oscillanti, f_k(t) che dipendono dalla coseno e dal seno
        f_k_t = (1 / autovalori[k]) - (autovalori[k] * np.cos(omega * t) + gamma * omega * np.sin(omega * t)) / \
                (autovalori[k] ** 2 + (gamma * omega) ** 2)

        # Sommiamo il contributo per la correlazione
        Cij += autovettori[i, k] * (k_b * T / autovalori[k]) * np.exp(-autovalori[k] * np.abs(t - s)) * autovettori[k, j]

        # Forza oscillante per il calcolo della correlazione tra stati distanti
        B_a_b_k = epsilon_0 * (autovettori[20, k] - autovettori[75, k])

        for p in range(len(autovalori)):  # Sommiamo il contributo per ogni p
            B_a_b_p = epsilon_0 * (autovettori[20, p] - autovettori[75, p])
            f_p_s = (1 / autovalori[p]) - (autovalori[p] * np.cos(omega * s) + gamma * omega * np.sin(omega * s)) / \
                    (autovalori[p] ** 2 + (gamma * omega) ** 2)

            Cij += (autovettori[i, k] * B_a_b_p * B_a_b_k * f_p_s * f_k_t * autovettori[p, j])

    return Cij

# Calcolo della correlazione teorica per ogni tau
teoretical_corrs = []
for tau in np.arange(0, T * dt, dt):
    teoretical_corr_sum = 0
    count = 0
    for n in range(N_process):  # Iteriamo su N_process realizzazioni
        for s in np.arange(0, T * dt - tau, dt):  # Consideriamo solo (s, s + tau)
            t = s + tau
            # Calcola la correlazione teorica per ogni tau
            teoretical_corr_sum += teoretical_C_i_j(autovalori, autovettori, 40, 40, gamma, k_b, Temperature, s, t, omega, epsilon_0)
            count += 1
    teoretical_corrs.append(teoretical_corr_sum / count)  # Media su tutte le realizzazioni e posizioni s per ogni tau

# Normalizzazione della correlazione teorica e empirica
avg_autocorr = np.array(empirical_autocorrs[0])  # Considera solo la prima realizzazione per il grafico
avg_autocorr = avg_autocorr    #/ avg_autocorr[0]  # Normalizzazione della correlazione empirica

teoretical_corrs = np.array(teoretical_corrs)
teoretical_corrs = teoretical_corrs #/ teoretical_corrs[0]  # Normalizzazione della correlazione teorica

# Plot dei risultati
plt.figure(figsize=(10, 6))
for idx, avg_autocorr in zip([40], empirical_autocorrs):
    plt.plot(np.arange(0, T) * dt, avg_autocorr, label=f'Autocorrelazione Empirica - x_{idx}')
#plt.plot(np.arange(0, T) * dt, teoretical_corrs, label='Correlazione Teorica - x_40', linestyle='-')

plt.title('Autocorrelazione e Correlazione Teorica del Processo Stocastico con Forza Oscillante')
plt.xlabel('Tempo (t)')
plt.ylabel('Correlazione')
plt.legend()
plt.grid(True)
plt.show()
