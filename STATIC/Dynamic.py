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
def stochastic_process_1d(K, epsilon_0, omega, dt, T, k_b, gamma, MaxTime, N,distance):
    t = np.arange(0, MaxTime, dt)
    r_history = np.zeros((len(t), N))
    r = np.zeros( N)
    epsilon= np.zeros((len(t), N))
    #p=0
    for n in range(1, len(t)):
        epsilon_t = epsilon_0 * (1 - np.cos(omega * t[n])) #/ 2
        #if epsilon_t==0:
        #    p+=1
            #print(p)
        #if epsilon_t<=0.0001 and epsilon_t>=-0.0001:
            #print("periodo")
        

        dH_dr = np.zeros(N)
        
        for i in range(N):
            dH_dr[i] = np.sum(K[i, :] * r)
            #print(r[20] - r[75])
            # print(2 * epsilon_t )
            #print(np.sum(K[i, :] * r))
            if i == 20:  # index 20 corresponds to residue 21
                dH_dr[i] -=  epsilon_t * (r[20] - r[75])
            elif i == 75:  # index 75 corresponds to residue 76
                dH_dr[i] +=  epsilon_t * (r[20] - r[75])
       
        eta = np.random.normal(0, 0.1, N)
        #print("rumore",np.sqrt(2 * k_b * T * gamma * dt) * eta[20])
        #print("epsilon",2 * epsilon_t * (r[20] - r[75]))
        
        #print("K", np.sum(K[i, :] * r))
        r = r - dH_dr * dt + np.sqrt(2 * k_b * T * gamma * dt) * eta
        r_history[n] = r
        epsilon[n]=epsilon_t

    #print(np.max(epsilon))
    return t, r_history,epsilon
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

# Ordina gli autovalori
autovalori_ordinati = np.sort(autovalori)

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
distance = np.linalg.norm(position_20 - position_75)
# Example usage:
N = positions.shape[0]  # number of residues
K = kirchhoff_matrix
epsilon_0 =2.5#0.1
omega = 2*np.pi
dt = 0.00001#0.00001
T = 0.001
k_b = 1
gamma = 1.
MaxTime =10#*np.pi*5#5.1#25*4

t, r_history,epsilon = stochastic_process_1d(K, epsilon_0, omega, dt, T, k_b, gamma, MaxTime, N,distance)
time_avg_x_squared = calculate_time_average_x_squared(r_history)
# Calculate the modulus of the displacement (dx)
dx = np.linalg.norm(r_history, axis=1)

# Plotting the intensity of the modulus of dx as a function of time
'''plt.figure(figsize=(10, 6))
plt.plot(t, dx, label='Intensity of |dx|', color='b')
plt.title('Intensity of Modulus of Displacement (|dx|) vs Time')
plt.xlabel('Time')
plt.ylabel('Intensity of |dx|')
plt.grid(True)
plt.legend()
plt.show()'''
def autocorrelation(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]

r_21_history = r_history[:, 20]  # Storia di r[20]
autocorr = autocorrelation(r_21_history)

# Normalizzazione dell'autocorrelazione
autocorr /= autocorr[0]

# Plot della correlazione
plt.plot(autocorr)
plt.title("Autocorrelazione di r[20] (Residuo 21)")
plt.xlabel("Lag (tau)")
plt.ylabel("Autocorrelazione")
plt.show()
# Calcola la correlazione tra le traiettorie dei due residui
residue1_trajectory = r_history[:, 20]
residue2_trajectory = r_history[:, 75]

# Calcola la media di ogni traiettoria
mean1 = np.mean(residue1_trajectory)
mean2 = np.mean(residue2_trajectory)

# Sottrai la media da ogni traiettoria
residue1_trajectory -= mean1
residue2_trajectory -= mean2

# Calcola la correlazione utilizzando np.correlate
correlation = np.correlate(residue1_trajectory, residue2_trajectory, mode='full')

# Normalizza la correlazione per avere valori tra -1 e 1
correlation = correlation / np.max(np.abs(correlation))

cos_signal = 1 - np.cos(omega * t)

# Plotta la correlazione e il segnale basato su 1 - cos(omega * t) con lo stesso asse temporale
plt.figure(figsize=(12, 6))

# Plotta la correlazione tra i residui
plt.plot(t[:len(correlation)], correlation[:len(t)], label='Correlation (Residue 21 vs Residue 76)', color='b')

# Plotta il segnale 1 - cos(omega * t)
plt.plot(t, cos_signal, label='Signal: 1 - cos(omega * t)', color='r', linestyle='--')

# Imposta etichette e titolo
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Correlation between Residue 21 and Residue 76 and Signal (1 - cos(omega * t))')
plt.legend()

# Imposta il layout e salva il plot
plt.tight_layout()

# Crea la directory se non esiste
if not os.path.exists(f'images/{stringa}/dynamic/'):
    os.makedirs(f'images/{stringa}/dynamic/')

# Salva la figura
plt.savefig(f'images/{stringa}/dynamic/Correlation_and_Cosine_Signal.png')
plt.show()


def compare_b_factors_with_sec_structure(actual_b_factors, predicted_b_factors, sec_struct_data, name):
    # Scala i fattori B predetti per farli corrispondere all'intervallo dei fattori B reali
    scale_factor = np.mean(actual_b_factors) / np.mean(predicted_b_factors)
    predicted_b_factors_scaled = predicted_b_factors * scale_factor

    # Calcola il coefficiente di correlazione
    #correlation, _ = pearsonr(actual_b_factors, predicted_b_factors_scaled)

    # Calcola la deviazione quadratica media (RMSD)
    rmsd = np.sqrt(np.mean((actual_b_factors - predicted_b_factors_scaled)**2))

    # Crea il plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot dei fattori B reali
    ax.set_xlabel('Indice del Residuo')
    ax.set_ylabel('Fattori B')
    ax.plot(actual_b_factors, label='Beta Factors', color='blue')

    # Plot dei fattori B predetti
    ax.plot(predicted_b_factors_scaled, label='Predicted Beta Factors', color='red')

    # Aggiungi il titolo e la legenda
    #plt.title('Confronto tra Fattori B Reali e Predetti')
    ax.legend(loc="upper right")

    # Aggiungi una griglia
    ax.grid(True, alpha=0.3)

    # Plot della struttura secondaria
    sec_struct_info = sec_struct_data['Secondary Structure']
    residue_ids = sec_struct_data['Residue ID'].astype(int)


    colors = {'H': 'red', 'E': 'blue'}  # Escludiamo 'C' dalla mappatura dei colori

    # Mappa i colori o None se la struttura è 'C'
    sec_struct_colors = [colors.get(sec_struct_info.get(rid), None) for rid in residue_ids]

    # Aggiungi le bande colorate per la struttura secondaria, escludendo 'C'
    current_color = None
    start_idx = 0
    a=0.2

    for idx, resid in enumerate(residue_ids):
        if sec_struct_colors[idx] != current_color:
            # Se non siamo nel primo residuo e l'attuale colore non è None, disegna una banda
            if idx > 0 and current_color is not None:
                ax.axvspan(start_idx, idx, color=current_color, alpha=a)  # Alpha uniforme
            # Cambia il colore attuale e aggiorna l'indice di partenza
            current_color = sec_struct_colors[idx]
            start_idx = idx

    # Aggiungi l'ultimo segmento se non è 'C' (None)
    if current_color is not None:
        ax.axvspan(start_idx, len(residue_ids), color=current_color, alpha=a)  # Alpha uniforme

    # Creazione delle legende con lo stesso alpha per consistenza
    legend_handles = [
        mpatches.Patch(color='red', label='Helix (H)', alpha=a),  # Alpha uniforme
        mpatches.Patch(color='blue', label='Beta sheet (E)', alpha=a)  # Alpha uniforme
    ]

    # Aggiungi la legenda con i nuovi handle
    ax.legend(handles=legend_handles, loc='upper right')

    # Crea la legenda personalizzata per la struttura secondaria
    handles = [mlines.Line2D([0], [0], color=color, lw=4, label=struct, alpha=a) for struct, color in colors.items()]
    ax.legend(handles=handles, title='Secondary Structure', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    
    plt.tight_layout()
    plt.legend()

    if not os.path.exists(f'images/{name}/dynamic/'):
        os.makedirs(f'images/{name}/dynamic/')

    # Salva la figura
    plt.savefig(f'images/{name}/dynamic/Confronto_Beta_con_Struttura_Secondaria.png')

    #print(f"Coefficiente di correlazione tra fattori B reali e predetti: {correlation:.4f}")
    print(f"Deviazione Quadratica Media (RMSD): {rmsd:.4f}")

    return correlation, rmsd



correlation,rmsd=compare_b_factors_with_sec_structure(df['B-Factor'].values, time_avg_x_squared, df, stringa)


import numpy as np
import matplotlib.pyplot as plt







# Select residues to plot, including 20 and 75
selected_residues = [20, 75]  # You can add more if needed
epsilon
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
plt.figure(figsize=(12, 8))

for residue in selected_residues:
    plt.plot(t, r_history[:, residue], label=f'Residue {residue+1}')
plt.plot(t, epsilon[:,0], label='Epsilon')  # Plot epsilon

plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Displacements of Selected Residues over Time (1D)')
plt.legend()
plt.tight_layout()
if not os.path.exists(f'images/{stringa}/dynamic/'):
    os.makedirs(f'images/{stringa}/dynamic/')

# Save the figure
plt.savefig(f'images/{stringa}/dynamic/Processo_stocastico_con_sengale.png')
# Plot the t

# Dopo aver creato l'istanza di GraphMatrixAnalyzer
analyzer = GraphMatrixAnalyzer(G)

eigenvalues = analyzer.get_eigenvalues_kirchhoff()
eigenvectors = analyzer.get_eigenvectors_kirchhoff()

# Get actual B-factors from the dataframe
actual_b_factors = df['B-Factor'].values
def predict_b_factors(eigenvalues, eigenvectors, temperature=1):
    k_B = 1  # Boltzmann constant in J/K
    T = temperature  # Temperature in Kelvin

    # Rimuovi l'autovalore più piccolo (solitamente vicino a zero) e il corrispondente autovettore
    eigenvalues = eigenvalues[1:]
    eigenvectors = eigenvectors[:, 1:]

    predicted_b_factors = np.zeros(eigenvectors.shape[0])
    for i in range(eigenvectors.shape[0]):
        predicted_b_factors[i] = (8 * np.pi**2 * k_B * T) / 3 * np.sum(
            eigenvectors[i, :]**2 / eigenvalues[:]
        )

    return predicted_b_factors
# Predict B-factors
predicted_b_factors = predict_b_factors(eigenvalues, eigenvectors)

def compare_predicted_b_factors_with_sec_structure(actual_b_factors, predicted_b_factors, sec_struct_data, name):
    # Scala i fattori B predetti per farli corrispondere all'intervallo dei fattori B reali
    scale_factor = np.mean(actual_b_factors) / np.mean(predicted_b_factors)
    predicted_b_factors_scaled = predicted_b_factors * scale_factor

    # Calcola il coefficiente di correlazione
    #correlation, _ = pearsonr(actual_b_factors, predicted_b_factors_scaled)

    # Calcola la deviazione quadratica media (RMSD)
    rmsd = np.sqrt(np.mean((actual_b_factors - predicted_b_factors_scaled)**2))

    # Crea il plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot dei fattori B reali
    ax.set_xlabel('Indice del Residuo')
    ax.set_ylabel('Fattori B')
    ax.plot(actual_b_factors, label='Beta Factors predicted by Kirchhoff', color='blue')

    # Plot dei fattori B predetti
    ax.plot(predicted_b_factors_scaled, label='Beta Factors predicted by <x^2>', color='red')

    # Aggiungi il titolo e la legenda
    #plt.title('Confronto tra Fattori B Reali e Predetti')
    ax.legend(loc="upper right")

    # Aggiungi una griglia
    ax.grid(True, alpha=0.3)

    # Plot della struttura secondaria
    sec_struct_info = sec_struct_data['Secondary Structure']
    residue_ids = sec_struct_data['Residue ID'].astype(int)


    colors = {'H': 'red', 'E': 'blue'}  # Escludiamo 'C' dalla mappatura dei colori

    # Mappa i colori o None se la struttura è 'C'
    sec_struct_colors = [colors.get(sec_struct_info.get(rid), None) for rid in residue_ids]

    # Aggiungi le bande colorate per la struttura secondaria, escludendo 'C'
    current_color = None
    start_idx = 0
    a=0.2

    for idx, resid in enumerate(residue_ids):
        if sec_struct_colors[idx] != current_color:
            # Se non siamo nel primo residuo e l'attuale colore non è None, disegna una banda
            if idx > 0 and current_color is not None:
                ax.axvspan(start_idx, idx, color=current_color, alpha=a)  # Alpha uniforme
            # Cambia il colore attuale e aggiorna l'indice di partenza
            current_color = sec_struct_colors[idx]
            start_idx = idx

    # Aggiungi l'ultimo segmento se non è 'C' (None)
    if current_color is not None:
        ax.axvspan(start_idx, len(residue_ids), color=current_color, alpha=a)  # Alpha uniforme

    # Creazione delle legende con lo stesso alpha per consistenza
    legend_handles = [
        mpatches.Patch(color='red', label='Helix (H)', alpha=a),  # Alpha uniforme
        mpatches.Patch(color='blue', label='Beta sheet (E)', alpha=a)  # Alpha uniforme
    ]

    # Aggiungi la legenda con i nuovi handle
    ax.legend(handles=legend_handles, loc='upper right')

    # Crea la legenda personalizzata per la struttura secondaria
    handles = [mlines.Line2D([0], [0], color=color, lw=4, label=struct, alpha=a) for struct, color in colors.items()]
    ax.legend(handles=handles, title='Secondary Structure', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    
    plt.tight_layout()
    plt.legend()

    if not os.path.exists(f'images/{name}/dynamic/'):
        os.makedirs(f'images/{name}/dynamic/')

    # Salva la figura
    plt.savefig(f'images/{name}/dynamic/Confronto_Beta_Predizioni.png')

    #print(f"Coefficiente di correlazione tra fattori B reali e predetti: {correlation:.4f}")
    print(f"Deviazione Quadratica Media (RMSD): {rmsd:.4f}")

    return correlation, rmsd

correlation,rmsd=compare_predicted_b_factors_with_sec_structure(predicted_b_factors, time_avg_x_squared, df, stringa)

