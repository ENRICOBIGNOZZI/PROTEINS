import numpy as np
import matplotlib.pyplot as plt
from Downlaod_data import PDBProcessor
from Visualize import Visualize
from matrix import GraphMatrixAnalyzer
import pandas as pd
import os
def stochastic_process_1d(K, epsilon_0, omega, dt, T, k_b, gamma, MaxTime, N):
    t = np.arange(0, MaxTime, dt)
    r_history = np.zeros((len(t), N))
    r = np.zeros(N)  # Inizializza tutte le posizioni a zero
    epsilon= np.zeros((len(t), N))
    p=0
    for n in range(1, len(t)):
        epsilon_t = epsilon_0 * (1 + np.cos(omega * t[n])) / 2
        if epsilon_t==0:
            p+=1
            print(p)
        #if epsilon_t<=0.0001 and epsilon_t>=-0.0001:
        #    print("periodo")
        

        dH_dr = np.zeros(N)
        for i in range(N):
            dH_dr[i] = np.sum(K[i, :] * r)
            if i == 20:  # index 20 corresponds to residue 21
                dH_dr[i] += 2 * epsilon_t * (r[20] - r[75])
            elif i == 75:  # index 75 corresponds to residue 76
                dH_dr[i] -= 2 * epsilon_t * (r[20] - r[75])
        
        eta = np.random.normal(0, 1, N)
        r = r - dH_dr * dt + np.sqrt(2 * k_b * T * gamma * dt) * eta
        r_history[n] = r
        epsilon[n]=epsilon_t


    return t, r_history,epsilon
stringa="2m10"
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
# Initialize Visualize and analyze graph
visualizer = Visualize(df)
raggio=visualizer.calculate_and_print_average_distance()
#visualizer.plot_connections_vs_radius()
G = visualizer.create_and_print_graph(truncated=True, radius=8.0, plot=False, peso=20)  # Adjust radius as needed
analyzer = GraphMatrixAnalyzer(G)
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()

def calculate_time_average_x_squared(r_history):
    return np.mean(r_history**2, axis=0)
# Extract positions
positions = df[['X', 'Y', 'Z']].values

# Example usage:
N = positions.shape[0]  # number of residues
K = kirchhoff_matrix
epsilon_0 =0.1
omega = 2*np.pi
dt = 0.01
T = 1
k_b = 1
gamma = 1.
MaxTime =5.1#25*4

t, r_history,epsilon = stochastic_process_1d(K, epsilon_0, omega, dt, T, k_b, gamma, MaxTime, N)
time_avg_x_squared = calculate_time_average_x_squared(r_history)
# Estrai le traiettorie dei due residui
residue1_trajectory = r_history[:, 20]
residue2_trajectory = r_history[:, 75]


# Calcola la media di ogni traiettoria
mean1 = np.mean(epsilon)
mean2 = np.mean(epsilon)

# Sottrai la media da ogni traiettoria
residue1_trajectory -= mean1
residue2_trajectory -= mean2

# Inizializza un array vuoto per la correlazione
correlation = np.zeros_like(epsilon)

# Calcola la correlazione
for i in range(len(epsilon)):
    for j in range(len(epsilon) - i):
        correlation[i] += epsilon[j] * epsilon[i + j]

# Normalizza la correlazione per avere valori tra -1 e 1
correlation = correlation / np.max(correlation)

# Plotta la correlazione
plt.figure(figsize=(12, 6))
plt.plot(correlation)
plt.xlabel('Time Lag')
plt.ylabel('Correlation')
plt.title('Correlation between Residue 21 and Residue 76')
plt.tight_layout()

if not os.path.exists(f'images/{stringa}/dynamic/'):
    os.makedirs(f'images/{stringa}/dynamic/')

# Salva la figura
plt.savefig(f'images/{stringa}/dynamic/Correlation.png')
# Plot the time average of x(t)^2 for each residue
plt.figure(figsize=(12, 6))
plt.plot(range(1, N+1), time_avg_x_squared, 'b-')
plt.xlabel('Residue Number')
plt.ylabel('Time Average of x(t)^2')
plt.title('Time Average of x(t)^2 for Each Residue')
plt.tight_layout()
if not os.path.exists(f'images/{stringa}/dynamic/'):
    os.makedirs(f'images/{stringa}/dynamic/')

# Save the figure
plt.savefig(f'images/{stringa}/dynamic/Stima beta factors.png')

# Select residues to plot, including 20 and 75
selected_residues = [20, 75]  # You can add more if needed

# Plot the positions of selected residues over time
plt.figure(figsize=(12, 8))

for residue in selected_residues:
    plt.plot(t, r_history[:, residue], label=f'Residue {residue+1}')

plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Displacements of Selected Residues over Time (1D)')
plt.legend()
plt.tight_layout()
if not os.path.exists(f'images/{stringa}/dynamic/'):
    os.makedirs(f'images/{stringa}/dynamic/')

# Save the figure
plt.savefig(f'images/{stringa}/dynamic/Processo_stocastico.png')

# Plot the t

# Dopo aver creato l'istanza di GraphMatrixAnalyzer
analyzer = GraphMatrixAnalyzer(G)

# Analizza la stabilità del sistema
analyzer.analyze_stability()

# Visualizza le frequenze naturali
analyzer.natural_frequencies()

# Calcola e visualizza i moltiplicatori di Floquet
T = 2 * np.pi / omega  # Periodo della perturbazione
analyzer.plot_floquet_multipliers(T)
