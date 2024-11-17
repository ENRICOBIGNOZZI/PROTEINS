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

# Rumore gaussiano
def gaussian_noise(N, dt):
    return np.sqrt(dt) * np.random.normal(0, 1, (N,))


def stochastic_process_1d(K, epsilon_0, omega, dt, T, k_b, gamma, MaxTime, N):
    t = np.arange(dt, MaxTime, dt)
    r_history = np.zeros((len(t), N))
    r = np.zeros( N)
    epsilon= np.zeros((len(t), N))
    dx_dt= np.zeros((len(t), N))
    F= np.zeros((len(t), N))
    for n in range(1, len(t)):
        epsilon_t = epsilon_0 * (1 - np.cos(omega * t[n]))/gamma
        dH_dr = np.zeros(N)
        for i in range(N):
            dH_dr[i] = np.sum(K[i, :] * r)/gamma
            if i == 20:  # index 20 corresponds to residue 21
                dH_dr[i] -=  epsilon_t #* (r[20] - r[75])
            if i == 75:  # index 75 corresponds to residue 76
                dH_dr[i] +=  epsilon_t #* (r[20] - r[75])
        eta= gaussian_noise(N, dt)
        
        r = r - dH_dr * dt + np.sqrt(2 * k_b * T/gamma) * eta* dt
        F[n]=-dH_dr
        r_history[n] = r
        dx_dt[n]= (r_history[n]-r_history[n-1])/dt
        epsilon[n]=epsilon_t
    S_derivata=np.cumsum(dt*np.sum(F*dx_dt,axis=1))/ t
    return t, r_history,epsilon,S_derivata,dx_dt

def heat_current(velocita,posizione,K,t,dt):
    energy=np.zeros((posizione.shape[1],len(t)))
    h=np.zeros((len(t)))
    derivata_E=np.zeros((posizione.shape[1],len(t)))
    for k in range(len(t)):
        for i in range(posizione.shape[1]):
            energy[i,k]=0.5*posizione[k,i]*sum(K[i,:]*posizione[k,:])
   
            derivata_E[i,k]=(energy[i,k]-energy[i,k-1])/dt
            h[k]+=velocita[k,i]*energy[i,k]+posizione[k,i]*derivata_E[i,k]
    return h


def z(p,autovalori,gamma):
    return autovalori[p]/gamma

def teoretical_C_i_j(autovalori,autovettori,i,j,gamma,k_b,T,s,t,omega,epsilon_0): 
    esponenziale=0
    oscillatorio=0
    for k in range(1,len(autovalori)):
        
        for p in range(1,len(autovalori)):
            if k==p:
                esponenziale+=k_b*T*np.exp(-((autovalori[k])/gamma)*(t-s))/(autovalori[k])*autovettori[i,k]*autovettori[p,j]
            

            B_a_b_k=epsilon_0*(autovettori[20,k]-autovettori[75,k])/gamma
            B_a_b_p_transp=epsilon_0*(autovettori[p,20]-autovettori[p,75])/gamma
            G_k=-gamma/autovalori[k]-(((autovalori[k]/gamma)*np.cos(omega*(t))+(omega*np.sin(omega*(t))))/((omega**2)+((autovalori[k]**2)/(gamma**2))))
            G_p=-gamma/autovalori[p]-(((autovalori[p]/gamma)*np.cos(omega*(t))+(omega*np.sin(omega*(t))))/((omega**2)+((autovalori[p]**2)/(gamma**2))))
            oscillatorio+=autovettori[i,k]*autovettori[p,j]*B_a_b_k*B_a_b_p_transp*G_k*G_p
            #print(oscillatorio)
            
    return esponenziale+oscillatorio

def teoretical_C_i_j_TIME(autovalori,autovettori,i,j,gamma,k_b,T,s,t,omega,epsilon_0): 
    CIJ=np.zeros(len(t))
    for f in range(len(t)):
        CIJ[f]=teoretical_C_i_j(autovalori=autovalori,autovettori=autovettori,i=i,j=j,gamma=gamma,k_b=k_b,T=T,s=0,t=t[f],omega=omega,epsilon_0=epsilon_0)
    return CIJ

import numpy as np

def correlazione_connessa(X, Y):
    """
    Calcola la cross-covarianza senza sottrarre la media per lag k >= 0.
    
    Parametri:
    X (array-like): Prima serie temporale
    Y (array-like): Seconda serie temporale
    max_lag (int): Massimo lag da calcolare (positivo)

    Ritorna:
    dict: Dizionario con lag come chiavi e la cross-covarianza come valori
    """
    N = len(X)
    max_lag=len(X)
    covariances = np.zeros(max_lag )
    
    for k in range(0,max_lag ):
        cov = np.sum(X[:N-k] * Y[k:]) / (N - k)
        covariances[k] = cov

    return covariances



# Calcoliamo la correlazione connessa



def autocorrelation(x):
    result = np.correlate(x, x, mode='full')
    total_size = result.size
    return result[total_size // 2 : total_size // 2 + total_size // 2]

def calculate_time_average_x_squared(r_history):
    return np.mean(r_history**2, axis=0)
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
G = visualizer.create_and_print_graph(truncated=True, radius=8.0, plot=False, peso=1)  # Adjust radius as needed
analyzer = GraphMatrixAnalyzer(G)
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
autovalori, autovettori = np.linalg.eig(kirchhoff_matrix)
autovalori = np.sort(autovalori)
autovettori = autovettori[:, np.argsort(autovalori)]

# Extract positions
positions = df[['X', 'Y', 'Z']].values
N = positions.shape[0]  # number of residues
K = kirchhoff_matrix
epsilon_0 =1#.001#0.001#0.5#0.1
omega = 2*np.pi
dt =0.005 #0.01#0.0001
T =0.001
k_b = 1
gamma = 2.
MaxTime =20##*np.pi*5#5.1#25*4


t, r_history,epsilon,S,velocita = stochastic_process_1d(K, epsilon_0, omega, dt, T, k_b, gamma, MaxTime, N)
h=heat_current(velocita,r_history,K,t,dt)
plt.plot(h)
plt.title("Heat current")
plt.xlabel("time")
plt.ylabel("Heat currentn")
if not os.path.exists(f'images/{stringa}/dynamic/'):
    os.makedirs(f'images/{stringa}/dynamic/')
plt.savefig(f'images/{stringa}/dynamic/Heat current.png')

plt.figure(figsize=(12, 6))
plt.plot(S)
plt.title("Entropy production")
plt.xlabel("time")
plt.ylabel("Entropy production")
if not os.path.exists(f'images/{stringa}/dynamic/'):
    os.makedirs(f'images/{stringa}/dynamic/')
plt.savefig(f'images/{stringa}/dynamic/Enropy_production.png')

time_avg_x_squared = calculate_time_average_x_squared(r_history)
r_21_history = r_history[:, 20]  # Storia di r[20]
CIJ_T=teoretical_C_i_j_TIME(autovalori=autovalori,autovettori=autovettori,i=20,j=20,gamma=gamma,k_b=k_b,T=T,s=0,t=np.arange(dt, MaxTime, dt),omega=omega,epsilon_0=epsilon_0)
autocorr = correlazione_connessa(r_21_history,r_21_history)
print(autocorr)
autocorr /= autocorr[0]
autocorr=autocorr[:int(len(autocorr)/4)]

#/np.max(np.abs(correlation))
print(CIJ_T)
CIJ_T/= CIJ_T[0]#np.max(np.abs(CIJ_T))#

# Calcola la correlazione tra le traiettorie dei due residui
residue1_trajectory = r_history[:, 20]#r_history[:, 20]
residue2_trajectory = r_history[:, 75]#r_history[:, 75]

# Calcola la media di ogni traiettoria
#mean1 = np.mean(residue1_trajectory)
#mean2 = np.mean(residue2_trajectory)

# Sottrai la media da ogni traiettoria
#residue1_trajectory -= mean1
#residue2_trajectory -= mean2
correlation,rmsd=compare_b_factors_with_sec_structure(df['B-Factor'].values, time_avg_x_squared, df, stringa)









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

# Predict B-factors
predicted_b_factors = predict_b_factors(eigenvalues, eigenvectors)


correlation,rmsd=compare_predicted_b_factors_with_sec_structure(predicted_b_factors, time_avg_x_squared, df, stringa)

