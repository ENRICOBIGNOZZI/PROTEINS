import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
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
# Initialize Visualize and analyze graph
visualizer = Visualize(df)
raggio=visualizer.calculate_and_print_average_distance()
#visualizer.plot_connections_vs_radius()
G = visualizer.create_and_print_graph(truncated=True, radius=8.0, plot=False, peso=20)  # Adjust radius as needed
analyzer = GraphMatrixAnalyzer(G)
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
autovalori, autovettori = np.linalg.eig(kirchhoff_matrix)



# Crea il grafico degli autovalori
'''plt.figure(figsize=(10, 6))
plt.plot(autovalori_ordinati, 'bo-', markersize=8, label='Autovalori')
plt.title('Autovalori della Matrice di Kirchhoff')
plt.xlabel('Indice degli Autovalori')
plt.ylabel('Valore degli Autovalori')
plt.axhline(0, color='red', linestyle='--', linewidth=1)  # Linea orizzontale a zero
plt.grid(True)
plt.legend()
plt.show()
num_autovettori_da_visualizzare = min(5, autovettori.shape[1])  # Visualizza al massimo 5 autovettori

# Creazione del grafico
plt.figure(figsize=(12, 8))
for i in range(num_autovettori_da_visualizzare):
    plt.plot(autovettori[:, i], marker='o', label=f'Autovettore {i+1}')

plt.title('Autovettori della Matrice di Kirchhoff')
plt.xlabel('Indice')
plt.ylabel('Valore dell\'Autovettore')
plt.axhline(0, color='red', linestyle='--', linewidth=1)  # Linea orizzontale a zero
plt.grid(True)
plt.legend()
plt.show()'''
def calculate_time_average_x_squared(r_history):
    return np.mean(r_history**2, axis=0)
# Extract positions
positions = df[['X', 'Y', 'Z']].values
print(df)
position_20 = df.loc[df['Residue ID'] == 21, ['X', 'Y', 'Z']].values[0]
position_75 = df.loc[df['Residue ID'] == 76, ['X', 'Y', 'Z']].values[0]

# Calcola la distanza euclidea tra le due posizioni
G = visualizer.create_and_print_graph(truncated=True, radius=8.0, plot=False, peso=20)  # Adjust radius as needed
analyzer = GraphMatrixAnalyzer(G)
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
distance = np.linalg.norm(position_20 - position_75)
contatti_somma_righe = kirchhoff_matrix.sum(axis=1)
contatti_diagonale = kirchhoff_matrix.diagonal()
contatti = contatti_diagonale/2
temperatura = np.where(contatti >= 5, 0.9, 1)

# Example usage:
N = positions.shape[0]  # number of residues
K = kirchhoff_matrix
dt = 0.01#0.00001
k_b = 0.5

gamma = 1#0.5
T =10#*np.pi*5#5.1#25*4
A = -K /gamma
#B = (np.sqrt(2 * gamma * k_b * temperatura))
B = np.sqrt(2 * gamma * k_b * temperatura)[:, np.newaxis] * np.eye(N)

steps = int(T / dt)
x = np.zeros((steps, N))
x[0] = np.random.uniform(-1, 1, size=N)  # Random initial conditions
for t in range(1, steps):
    noise = np.random.normal(0, np.sqrt(dt), size=N)
    x[t] = x[t-1] + dt * (A @ x[t-1]) + B @ noise



# Compute theoretical and experimental entropy rates
cov = np.cov(x.T)

# Ensure positive-definiteness of covariance matrix
cov += np.eye(N) * 1e-8
def entropy_rate(A, cov, B):
    entropy_prod = np.trace(A.T @ np.linalg.inv(cov) @ A @ cov + np.linalg.inv(cov) @ B @ B.T)
    entropy_flow = np.trace(A)
    return (entropy_prod - entropy_flow) / 2

S_rate_theoretical = entropy_rate(A, cov, B)
print("Theoretical entropy rate:", S_rate_theoretical)

# Visualize the stochastic process (example for first two residues)
plt.figure(figsize=(10, 5))
plt.plot(np.linspace(0, T, steps), x[:, 0], label='X1')
#plt.plot(np.linspace(0, T, steps), x[:, 1], label='X2')
plt.title("Stochastic Process Simulation (First Two Residues)")
plt.xlabel("Time")
plt.ylabel("State")
plt.legend()
plt.show()

# Compute experimental entropy rate (from simulated data)
def experimental_entropy_rate(data, dt):
    increments = np.diff(data, axis=0)
    cov_increments = np.cov(increments.T) / dt
    return np.trace(np.linalg.inv(cov) @ cov_increments) / 2

S_rate_experimental = experimental_entropy_rate(x, dt)
print("Experimental entropy rate:", S_rate_experimental)

# Equilibrium distribution (PDF visualization for first two residues)
'''plt.figure(figsize=(8, 6))
plt.hist2d(x[:, 0], x[:, 1], bins=50, density=True, cmap='viridis')
plt.colorbar(label='Probability Density')
plt.title("Equilibrium Distribution (First Two Residues)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()'''



