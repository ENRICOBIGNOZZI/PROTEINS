import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt
from Downlaod_data import PDBProcessor
from Visualize import Visualize
from matrix import GraphMatrixAnalyzer
import pandas as pd
import matplotlib.patches as patches
import os
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
# Parameters
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
df=df.dropna().reset_index(drop=True)
visualizer = Visualize(df)
print(df)
raggio=visualizer.calculate_and_print_average_distance()
#visualizer.plot_connections_vs_radius()
G = visualizer.create_and_print_graph(truncated=True, radius=8.0, plot=False, peso=1)  # Adjust radius as needed
analyzer = GraphMatrixAnalyzer(G)
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
contatti_diagonale = kirchhoff_matrix.diagonal()


contatti_somma_righe = kirchhoff_matrix.sum(axis=1)
contatti = contatti_diagonale/2


temperatura = np.where(contatti >= 5, 0.5, 1)
G = visualizer.create_and_print_graph(truncated=True, radius=8.0, plot=False, peso=20)  # Adjust radius as needed
analyzer = GraphMatrixAnalyzer(G)
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
residui= np.arange(1, len(temperatura)+1)


gamma = 0.5  # friction coefficient (s^-1)
k_B = 1.0    # Boltzmann constant (J/K)
T_high = 2.0 # high temperature (K)
T_low = 1.0  # low temperature (K)
B_high = np.sqrt(2 * gamma * k_B * T_high)
B_low = np.sqrt(2 * gamma * k_B * T_low)

# System matrix (assuming a simple linear system)
K = np.array([[1.0, 0], [0, 1.5]])  # Spring constants (arbitrary)
A = -K / gamma

# Noise term (position-dependent temperature)
B = np.diag([B_low, B_high])

# Simulation parameters
dt = 0.01  # time step
T = 10.0   # total simulation time
steps = int(T / dt)

# Initial conditions
x0 = np.array([1.0, 0.5])

# Simulate Langevin dynamics using Euler-Maruyama method
x = np.zeros((steps, len(x0)))
x[0] = x0

for t in range(1, steps):
    noise = B @ np.random.normal(0, np.sqrt(dt), size=x0.shape)
    x[t] = x[t-1] + dt * (A @ x[t-1]) + noise

# Compute theoretical and experimental entropy rates
cov = np.cov(x.T)

def entropy_rate(A, cov, B):
    entropy_prod = np.trace(A.T @ np.linalg.inv(cov) @ A @ cov + np.linalg.inv(cov) @ B @ B.T)
    entropy_flow = np.trace(A)
    return (entropy_prod - entropy_flow) / 2

S_rate_theoretical = entropy_rate(A, cov, B)
print("Theoretical entropy rate:", S_rate_theoretical)

# Visualize the stochastic process
plt.figure(figsize=(10, 5))
plt.plot(np.linspace(0, T, steps), x[:, 0], label='X1')
plt.plot(np.linspace(0, T, steps), x[:, 1], label='X2')
plt.title("Stochastic Process Simulation")
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

# Equilibrium distribution (PDF visualization)
plt.figure(figsize=(8, 6))
plt.hist2d(x[:, 0], x[:, 1], bins=50, density=True, cmap='viridis')
plt.colorbar(label='Probability Density')
plt.title("Equilibrium Distribution")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
