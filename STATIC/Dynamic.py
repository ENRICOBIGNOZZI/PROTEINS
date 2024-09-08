import numpy as np
import matplotlib.pyplot as plt
from Downlaod_data import PDBProcessor
from Visualize import Visualize
from matrix import GraphMatrixAnalyzer
import pandas as pd

def stochastic_process(r, K, epsilon_0, omega, dt, T, k_b, gamma, MaxTime):
    N = r.shape[0]
    t = np.arange(0, MaxTime, dt)
    r_history = np.zeros((len(t), N, 3))
    r_history[0] = r

    for n in range(1, len(t)):
        epsilon_t = epsilon_0 * (1 + np.cos(omega * t[n])) / 2
        
        dH_dr = np.zeros((N, 3))
        for i in range(N):
            for j in range(N):
                dH_dr[i] = (K[i,j]* r[j])
            if i == 20:  # index 20 corresponds to residue 21
                dH_dr[i] += 2 * epsilon_t * (r[20] - r[75])
            elif i == 75:  # index 75 corresponds to residue 76
                dH_dr[i] -= 2 * epsilon_t * (r[20] - r[75])
        
        eta = np.random.normal(0, 1, (N, 3))
        r = r - dH_dr * dt + np.sqrt(2 * k_b * T * gamma) * eta
        r_history[n] = r

    return t, r_history

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

# Extract positions
positions = df[['X', 'Y', 'Z']].values

# Example usage:
N = positions.shape[0]  # number of residues
K = kirchhoff_matrix
r0 = positions  # initial positions
epsilon_0 = 0.05
omega = 0.5
dt = 0.001
T = 1
k_b = 1
gamma = 1.
MaxTime = omega*2*np.pi*5*4

t, r_history = stochastic_process(r0, K, epsilon_0, omega, dt, T, k_b, gamma, MaxTime)

# Select 10 residues to plot, including 20 and 75
selected_residues = [20, 75]#[10, 20, 30, 40, 50, 60, 70, 75, 80, 90]

# Plot the positions of selected residues over time
plt.figure(figsize=(12, 8))

for i, residue in enumerate(selected_residues):
    plt.plot(t, r_history[:, residue, 0], label=f'Residue {residue+1} (X)')
    #plt.plot(t, r_history[:, residue, 1], label=f'Residue {residue+1} (Y)')
    #plt.plot(t, r_history[:, residue, 2], label=f'Residue {residue+1} (Z)')

plt.xlabel('Time')
plt.ylabel('Coordinate')
plt.title('Coordinates of Selected Residues over Time')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot the trajectory of a specific residue (e.g., residue 21)
residue_index = 20  # Index 20 corresponds to residue 21
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(r_history[:, residue_index, 0], r_history[:, residue_index, 1], r_history[:, residue_index, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Trajectory of Residue 21')
plt.show()

# Dopo aver creato l'istanza di GraphMatrixAnalyzer
analyzer = GraphMatrixAnalyzer(G)

# Analizza la stabilità del sistema
analyzer.analyze_stability()

# Visualizza le frequenze naturali
analyzer.natural_frequencies()

# Calcola e visualizza i moltiplicatori di Floquet
T = 2 * np.pi / omega  # Periodo della perturbazione
analyzer.plot_floquet_multipliers(T)
