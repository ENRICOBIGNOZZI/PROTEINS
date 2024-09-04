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

# Aggiungi queste stampe di debug
print("Dimensioni della matrice di Kirchhoff:", kirchhoff_matrix.shape)
print("Somma di tutti gli elementi della matrice:", kirchhoff_matrix.sum())
print("Primi 5 elementi della diagonale:", kirchhoff_matrix.diagonal()[:5])
print("Primi 5 elementi della somma delle righe:", kirchhoff_matrix.sum(axis=1)[:5])

contatti_diagonale = kirchhoff_matrix.diagonal()
contatti_somma_righe = kirchhoff_matrix.sum(axis=1)

print("Somma dei contatti dalla diagonale:", contatti_diagonale.sum())
print("Somma dei contatti dalle righe:", contatti_somma_righe.sum())
contatti = contatti_diagonale/2

print("Numero totale di contatti:", contatti.sum())
print("Primi 5 valori di contatti:", contatti[:5])

# Definizione del vettore temperatura
temperatura = np.where(contatti >= 5, 0.5, 1.0)

print("Primi 5 valori di temperatura:", temperatura[:5])
print("Numero di residui con temperatura 0.5:", np.sum(temperatura == 0.5))
print("Numero di residui con temperatura 1.0:", np.sum(temperatura == 1.0))

# Grafico dei contatti
residui = range(1, len(contatti) + 1)

plt.figure(figsize=(12, 6))
plt.plot(residui, contatti, marker='o')
plt.title('Grafico dei contatti al variare del residuo')
plt.xlabel('Numero del residuo')
plt.ylabel('Numero di contatti')
plt.grid(True)
plt.tight_layout()
plt.show()

# Grafico della temperatura
plt.figure(figsize=(12, 6))
plt.plot(residui, temperatura, marker='o')
plt.title('Grafico della temperatura al variare del residuo')
plt.xlabel('Numero del residuo')
plt.ylabel('Temperatura')
plt.yticks([0.5, 1.0])
plt.grid(True)
plt.tight_layout()
plt.show()

import scipy.integrate as integrate

# Assicuriamoci che temperatura_radiale sia un array numpy
temperatura_radiale = np.array(temperatura)

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