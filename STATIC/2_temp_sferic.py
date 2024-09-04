import numpy as np
import matplotlib.pyplot as plt
from Downlaod_data import PDBProcessor
from Visualize import Visualize
from matrix import GraphMatrixAnalyzer
import pandas as pd
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
G = visualizer.create_and_print_graph(truncated=True, radius=8.0, plot=False, peso=1)  # Adjust radius as needed
analyzer = GraphMatrixAnalyzer(G)
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()



centro_massa = df[['X', 'Y', 'Z']].mean().values

# Calcolo delle distanze dal centro di massa
distanze = np.sqrt(((df[['X', 'Y', 'Z']] - centro_massa) ** 2).sum(axis=1))

# Parametri per la temperatura radiale
T0 = 0.5  # Temperatura al centro
Tb = 1.0  # Temperatura al bordo
R = distanze.max()  # Raggio massimo

# Calcolo della temperatura radiale
temperatura_radiale = T0 + (Tb - T0) / R * distanze

print("Centro di massa:", centro_massa)
print("Raggio massimo:", R)
print("Primi 5 valori di distanza:", distanze[:5])
print("Primi 5 valori di temperatura radiale:", temperatura_radiale[:5])

# Grafico della temperatura radiale
residui = range(1, len(temperatura_radiale) + 1)

plt.figure(figsize=(12, 6))
plt.scatter(residui, temperatura_radiale, c=temperatura_radiale, cmap='coolwarm', marker='o')
plt.colorbar(label='Temperatura')
plt.title('Grafico della temperatura radiale al variare del residuo')
plt.xlabel('Numero del residuo')
plt.ylabel('Temperatura')
plt.grid(True)
plt.tight_layout()
plt.show()

# Grafico 3D della proteina colorata per temperatura
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['X'], df['Y'], df['Z'], c=temperatura_radiale, cmap='coolwarm')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.colorbar(scatter, label='Temperatura')
plt.title('Struttura 3D della proteina colorata per temperatura radiale')
plt.tight_layout()
plt.show()
# ... existing code ...

import scipy.integrate as integrate

# Assicuriamoci che temperatura_radiale sia un array numpy
temperatura_radiale = np.array(temperatura_radiale)

# Funzione da integrare
def integrando(u, K, T):
    exp_Ku = np.exp(K * u)
    return exp_Ku * T[:, np.newaxis] * T[np.newaxis, :] * exp_Ku

# Funzione per calcolare l'integrale
def calcola_integrale(K, T, limite_inferiore=-100, num_punti=1000):
    def func(u):
        return integrando(u, K, T)
    
    risultato, _ = integrate.quad_vec(func, limite_inferiore, 0, limit=num_punti)
    return risultato

# Calcolo dell'integrale
K = 1.0  # Puoi modificare questo valore secondo le tue necessità
risultato_integrale = calcola_integrale(K, temperatura_radiale)

print("Dimensioni del risultato dell'integrale:", risultato_integrale.shape)
print("Primi 5x5 elementi del risultato:")
print(risultato_integrale[:5, :5])

# Visualizzazione del risultato come heatmap
plt.figure(figsize=(10, 8))
plt.imshow(risultato_integrale, cmap='viridis', aspect='auto')
plt.colorbar(label='Valore dell\'integrale')
plt.title(f'Heatmap del risultato dell\'integrale (K={K})')
plt.xlabel('Indice del residuo')
plt.ylabel('Indice del residuo')
plt.tight_layout()
plt.show()