from scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from Downlaod_data import PDBProcessor
from Visualize import Visualize
from matrix import GraphMatrixAnalyzer
import pandas as pd
import os
from scipy.optimize import curve_fit
import matplotlib.lines as mlines
from scipy.stats import pearsonr
import matplotlib.patches as mpatches
# Funzione del processo stocastico che hai fornito
def stochastic_process_1d(K, epsilon_0, omega, dt, T, k_b, gamma, MaxTime, N, distance):
    t = np.arange(0, MaxTime, dt)
    r_history = np.zeros((len(t), N))
    r = np.zeros(N)
    epsilon = np.zeros((len(t), N))

    for n in range(1, len(t)):
        epsilon_t = epsilon_0 * (1 - np.cos(omega * t[n]))
        dH_dr = np.zeros(N)

        for i in range(N):
            dH_dr[i] = np.sum(K[i, :] * r)
            if i == 20:
                dH_dr[i] -= epsilon_t * (r[20] - r[75])
            elif i == 75:
                dH_dr[i] += epsilon_t * (r[20] - r[75])

        eta = np.random.normal(0, 0.1, N)
        r = r - dH_dr * dt + np.sqrt(2 * k_b * T * gamma * dt) * eta
        r_history[n] = r
        epsilon[n] = epsilon_t

    return t, r_history, epsilon
def compute_evolution_operator_stochastic(r_history, t, dt):
    dR = np.diff(r_history, axis=0) / dt  # Derivata di r(t)
    
    # Calcolo dell'operatore A stimato
    A_estimated = np.zeros((N, N))
    for i in range(1, len(dR)):
        A_estimated += np.outer(dR[i], r_history[i])  # Prodotto esterno per approssimare l'operatore di transizione
    
    A_estimated /= len(dR)  # Media degli operatori calcolati
    
    # Operatore di evoluzione U(t, s)
    U = expm(A_estimated * dt)
    return U



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
raggio=visualizer.calculate_and_print_average_distance()
#visualizer.plot_connections_vs_radius()
G = visualizer.create_and_print_graph(truncated=True, radius=8.0, plot=False, peso=20)  # Adjust radius as needed
analyzer = GraphMatrixAnalyzer(G)
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
autovalori, autovettori = np.linalg.eig(kirchhoff_matrix)
# Parametri del processo
positions = df[['X', 'Y', 'Z']].values
position_20 = df.loc[df['Residue ID'] == 21, ['X', 'Y', 'Z']].values[0]
position_75 = df.loc[df['Residue ID'] == 76, ['X', 'Y', 'Z']].values[0]
distance = np.linalg.norm(position_20 - position_75)
# Example usage:
N = positions.shape[0]  # number of residues
K = kirchhoff_matrix
epsilon_0 =2#0.1
omega = 2*np.pi
dt = 0.00001#0.00001
T = 0.001
k_b = 1
gamma = 1.
MaxTime =20#*np.pi*5#5.1#25*4
t, r_history,epsilon = stochastic_process_1d(K, epsilon_0, omega, dt, T, k_b, gamma, MaxTime, N,distance)
U_t_s_stochastic = compute_evolution_operator_stochastic(r_history, t, dt)
eigenvalues, _ = np.linalg.eig(U_t_s_stochastic)
# Calcolo del modulo degli autovalori
moduli_autovalori = np.abs(eigenvalues)

# Analisi della stabilità
stabili = moduli_autovalori < 1
instabili = moduli_autovalori > 1

print("Autovalori stabili:", moduli_autovalori[stabili])
print("Autovalori instabili:", moduli_autovalori[instabili])



# Disegna l'istogramma dei valori di r(t) alla fine della simulazione
plt.figure(figsize=(10, 6))
plt.hist(r_history[-100000:,20], bins=30, density=True)
plt.title('Istogramma dei valori di r(t) (ultimi 1000 valori)')
plt.xlabel('Valori di r(t)')
plt.ylabel('Densità')
plt.grid(True)
plt.show()

# 2. Tracciamento della media mobile
window_size = 50  # Dimensione della finestra per la media mobile
moving_average = np.convolve(r_history[-100000:,20].flatten(), np.ones(window_size)/window_size, mode='valid')

# Disegna la media mobile nel tempo
plt.figure(figsize=(10, 6))
plt.plot(moving_average, label='Media Mobile', color='b')
plt.title('Media Mobile dei Valori di r(t)')
plt.xlabel('Tempo')
plt.ylabel('Valori di r(t)')
plt.grid(True)
plt.legend()
plt.show()

# Funzione per calcolare l'autocorrelazione
def autocorrelation(x):
    n = len(x)
    result = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')
    return result[result.size // 2:] / (np.var(x) * n)

# Calcola l'autocorrelazione
autocorr_values = autocorrelation(r_history[:,20].flatten())

# Disegna l'autocorrelazione
plt.figure(figsize=(10, 6))
plt.plot(autocorr_values, label='Autocorrelazione', color='orange')
plt.title('Autocorrelazione di r(t)')
plt.xlabel('Lag')
plt.ylabel('Autocorrelazione')
plt.grid(True)
plt.legend()
plt.show()
