import numpy as np
import matplotlib.pyplot as plt
from Downlaod_data import PDBProcessor
from Visualize import Visualize
from matrix import GraphMatrixAnalyzer
import pandas as pd
import os
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
def calculate_Cij(u, Q,lambdaa, i,j,s,t):
    Cij = np.zeros((u.shape[0], u.shape[0]))
    if s>t:
        for k in range(1,u.shape[0]):
            for p in range(1,u.shape[0]):
                Cij[i][j] += ((u[i][k] * Q[k][p] * u[j][p]) / (lambdaa[k] + lambdaa[p]))*np.exp(-lambdaa[p]*(s-t))
    if s<t:
        for k in range(1,u.shape[0]):
            for p in range(1,u.shape[0]):
                Cij[i][j] += ((u[i][k] * Q[k][p] * u[j][p]) / (lambdaa[k] + lambdaa[p]))*np.exp(-lambdaa[k]*(t-s))
    return Cij

import numpy as np

def calculate_Q(U, B):
    B_transpose = np.transpose(B)
    U_transpose = np.transpose(U)
    Q = np.dot(U, np.dot(B, np.dot(B_transpose, U_transpose)))
    return Q

eigenvalues, eigenvectors = np.linalg.eig(kirchhoff_matrix)

Q=calculate_Q(  eigenvectors,temperatura_radiale)
print(Q)
print(Q.shape)
Cij=calculate_Cij(eigenvectors, Q, eigenvalues, i=20,j=70,s=0,t=1)

def calculate_Cij_matrix_static(u, Q, lambdaa,t,s):#questa è quella corretta
    print(np.dot(u,np.dot(Q,u)))
    print(np.dot(u,np.dot(Q,u)).shape)
    print(np.sum(lambdaa + lambdaa))
    print(np.sum(lambdaa + lambdaa).shape)
    if t>s
        Cij= np.dot(u,np.dot(Q,u))/ np.sum(lambdaa + lambdaa)*np.exp(-lambdaa*(t-s))
    else:
        Cij= np.dot(u,np.dot(Q,u))/ np.sum(lambdaa + lambdaa)*np.exp(-lambdaa*(s-t))
    return Cij
# Calcola la matrice Cij per tutti gli ij
#Cij_matrix = calculate_Cij_matrix(eigenvectors, Q, eigenvalues, s=0, t=1)

Cij_matrix = calculate_Cij_matrix_static(eigenvectors, Q, eigenvalues)
print(Cij_matrix)
print(Cij_matrix.shape)
# Crea un plot della matrice di correlazione
plt.figure(figsize=(10, 10))
plt.imshow(Cij_matrix, cmap='hot', interpolation='nearest')
plt.colorbar(label='Correlation')
plt.title('Correlation Matrix')
plt.show()


'''residui = range(1, len(temperatura_radiale) + 1)

plt.figure(figsize=(12, 6))
plt.scatter(residui, temperatura_radiale, c=temperatura_radiale, cmap='coolwarm', marker='o')
plt.colorbar(label='Temperatura')
plt.title('Grafico della temperatura radiale al variare del residuo')
plt.xlabel('Numero del residuo')
plt.ylabel('Temperatura')
plt.grid(True)
plt.tight_layout()
if not os.path.exists(f'images/{stringa}/2_temperature_sferical/'):
    os.makedirs(f'images/{stringa}/2_temperature_sferical/')


# Save the figure
plt.savefig(f'images/{stringa}/2_temperature_sferical/temperature_sferic.png')

# Grafico 3D della proteina colorata per temperatura


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
if not os.path.exists(f'images/{stringa}/2_temperature_sferical/'):
    os.makedirs(f'images/{stringa}/2_temperature_sferical/')

# Save the figure
plt.savefig(f'images/{stringa}/2_temperature_sferical/correlation.png')'''


