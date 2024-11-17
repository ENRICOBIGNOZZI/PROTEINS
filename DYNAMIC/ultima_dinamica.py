import numpy as np
from numba import njit, prange
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
from numba import njit, prange
import math
def plot_residual_autocorrelation(sperimentale, autovalori, autovettori, gamma, k_b, T, MaxTime, dt, omega, epsilon_0, stringa, i):
    """
    Calcola e confronta l'autocorrelazione empirica di un residuo con la sua autocorrelazione teorica per un dato indice.
    
    Args:
        r_history_mean (np.ndarray): Media della storia di r.
        autovalori, autovettori: Parametri teorici per il calcolo di CIJ_T.
        gamma, k_b, T (float): Costanti per la funzione teorica.
        MaxTime, dt (float): Parametri temporali.
        omega, epsilon_0 (float): Altri parametri per il modello teorico.
        stringa (str): Nome per il percorso di salvataggio delle immagini.
        i (int): Indice del residuo da analizzare. Default è 20.
    
    """

    # Calcolo dell'autocorrelazione teorica
    CIJ_T = teoretical_C_i_j_TIME(autovalori=autovalori, autovettori=autovettori, i=i, j=i,
                                  gamma=gamma, k_b=k_b, T=T, s=0, t=np.arange(0, MaxTime, dt),
                                  omega=omega, epsilon_0=epsilon_0)
    autocorr = calculate_autocorrelation_for_index(sperimentale, i)
    autocorr=autocorr[::-1]
    lungheza=int(len(autocorr) / 2)
    autocorr = autocorr[:lungheza]#[0]#[:int(len(autocorr) / 10)]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(autocorr, color='blue', label=f"Autocorrelazione di Residuo {i}")
    ax1.set_xlabel("Lag (tau)")
    ax1.set_ylabel("Autocorrelazione Empirica", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()  # Condivide lo stesso asse x, ma ha un asse y separato
    ax2.plot(CIJ_T[:lungheza], color='red', label="Autocorrelazione Teorica")
    ax2.set_ylabel("Autocorrelazione Teorica", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax1.set_title(f"Autocorrelazione di Residuo {i}")
    fig.tight_layout()  # Ottimizza il layout per evitare sovrapposizioni

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    
    output_path = f'images/{stringa}/dynamic/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(f'{output_path}/AutoCorrelation_and_Cosine_Signal_{i}_vs_Teorico.png')
    plt.close()
def teoretical_C_i_j_TIME(autovalori,autovettori,i,j,gamma,k_b,T,s,t,omega,epsilon_0):
    


    CIJ=np.zeros(len(t))
    
    for f in range(len(t)):
        CIJ[f]=teoretical_C_i_j(autovalori=autovalori,autovettori=autovettori,i=i,j=j,gamma=gamma,k_b=k_b,T=T,s=0,t=t[f],omega=omega,epsilon_0=epsilon_0)
        
    return CIJ
#@njit
def stochastic_process_1d_optimized(K, epsilon_0, omega, dt, T, k_b, gamma, MaxTime, N):
    t = np.arange(0, MaxTime, dt)

    r_history = np.zeros((len(t), N), dtype=np.float64)
    r = np.zeros(N, dtype=np.float64)
    epsilon = np.zeros(len(t), dtype=np.float64)
    dx_dt = np.zeros((len(t), N), dtype=np.float64)
    F = np.zeros((len(t), N), dtype=np.float64)
    
    K = K.astype(np.float64)
    if K.shape != (N, N):
        raise ValueError("La matrice K deve avere dimensioni (N, N)")

    sqrt_term = np.sqrt(2 * k_b * T / gamma)
    
    for n in range(1, len(t)):
        
        epsilon_t = epsilon_0 * (1 - np.cos(omega * t[n])) / gamma
        for i in range(len(r)):
            step=0
            dH_dr=0
        
            dH_dr += np.sum((K[i,:] * r_history[n-1,:]))/ gamma  # Operazione vettoriale
            
            

            if i==20:
                dH_dr += epsilon_t
            if i==75:
                dH_dr -= epsilon_t

            eta = np.random.normal(0, 1, 1)
            step = - dH_dr * dt + sqrt_term * eta * np.sqrt(dt) 
            
            r_history[n,i] += step
        #print(r_history[0])
        dx_dt[n] = (r_history[n] - r_history[n-1]) / dt
        F[n] = -dH_dr
   
    S_derivata = 0#np.cumsum(dt * np.sum(F * dx_dt, axis=1)) / t
    return r_history, epsilon, S_derivata, dx_dt
def calculate_autocorrelation_for_index(r_history, index):
    """
    Calcola l'autocorrelazione temporale per un residuo specifico (indice) in un processo stocastico.

    Parameters:
        r_history (numpy.ndarray): Un array di dimensione (len(t), N) che rappresenta la storia delle posizioni del sistema.
        index (int): L'indice del residuo per il quale calcolare l'autocorrelazione.

    Returns:
        autocorr (numpy.ndarray): Un array che contiene l'autocorrelazione temporale per il residuo selezionato.
    """
    len_t = r_history.shape[0]
    autocorr = np.zeros(len_t, dtype=np.float64)
    
    for tau in range(len_t):
        autocorr[tau] = np.mean(r_history[tau:, index] * r_history[:len_t-tau, index])
    
    return autocorr
def average_stochastic_process(K, epsilon_0, omega, dt, T, k_b, gamma, MaxTime, N, num_realizations):
    realizations = np.zeros((num_realizations, int(MaxTime/dt), N), dtype=np.float64)
    
    for i in range(num_realizations):
        np.random.seed(i) 
        r_history, _, _, _ = stochastic_process_1d_optimized(K, epsilon_0, omega, dt, T, k_b, gamma, MaxTime, N)
        realizations[i] = r_history


    # Calcolo della media lungo l'asse 0
    #print(realizations.shape)
    r_history_mean = np.mean(realizations, axis=0)
    #first_step = realizations[:, 1:2, :]  # Estraggo il primo oggetto di forma (N, 1, M)
    #print(first_step)
    
    #print(r_history_mean.shape)
    # Calcolo del prodotto elemento per elemento
    #autocorrelation = calculate_autocorrelation(r_history_mean)#realizations* first_step  # Prodotto broadcasting, risulterà in (N, Z, M)
    
    #print(autocorrelation.shape)
    # Calcolo della media lungo l'asse N
    #autocorrelation = np.mean(autocorrelation, axis=0)  # Risultato finale con forma (Z, M)
  

    
    
    return r_history_mean#,(autocorrelation)
@njit
def teoretical_C_i_j(autovalori,autovettori,i,j,gamma,k_b,T,s,t,omega,epsilon_0): 
    Cij=0

    
    for k in range(1,len(autovalori)):
        z_k=autovalori[k]/gamma
        f_k_t_num=-(+math.pow(z_k, 2)*np.cos(omega*t)+omega*z_k*np.sin(omega*t))#+(math.pow(z_k, 2)+math.pow(omega, 2))
        f_k_t_den=z_k*(+math.pow(omega, 2)+math.pow(z_k, 2))
        

        f_k_t=f_k_t_num/f_k_t_den
        Cij+=autovettori[i,k]*(k_b*T/(z_k*gamma))*np.exp(-z_k*np.abs(t-s))*autovettori[j,k]
        #print(Cij)


        B_a_b_k=epsilon_0/gamma*(autovettori[20,k]-autovettori[75,k])
        for p in range(1,len(autovalori)):    
           
            z_p=autovalori[p]/gamma
            
            B_a_b_p=epsilon_0/gamma*(autovettori[20,p]-autovettori[75,p])
            
            
            f_p_s_num=-(+math.pow(z_p, 2)*np.cos(omega*s)+omega*z_p*np.sin(omega*s))#+(math.pow(z_p, 2)+math.pow(omega, 2))
            

            f_p_s_den=z_p*(+math.pow(omega, 2)+math.pow(z_p, 2))

            f_p_s=f_p_s_num/f_p_s_den
            
            Cij=Cij+(autovettori[i,k]*B_a_b_p*B_a_b_k*f_p_s*f_k_t*autovettori[p,j])  
            
    #print(Cij)
    return Cij
stringa="3LNX"
pdb_processor = PDBProcessor(pdb_id="3LNX")#2m07
pdb_processor.download_pdb()
pdb_processor.load_structure()
df1 = pdb_processor.secondary_structure()
df = pdb_processor.extract_atom_data()
df = df[df['Model ID'] == 0]
df = df[df['Chain ID'] == 'A']
df = df[df['Atom Name'] == 'CA']
df = df.reset_index(drop=True)
concatenated_df = pd.concat([df1['Secondary Structure'], df], axis=1)
df = concatenated_df.dropna().reset_index(drop=True)
df = df.loc[:,~df.T.duplicated()]
df = concatenated_df.dropna().reset_index(drop=True)
df = df.T.drop_duplicates().T
file_path = '1pdz.ca'
with open(file_path, 'r') as file:
    lines = file.readlines()
data = [line.split() for line in lines if not line.startswith("#") and line.strip() != ""]
df2= pd.DataFrame(data)
df2.columns = df.columns
df[['X', 'Y', 'Z','Residue ID']] = df2[['X', 'Y', 'Z','Residue ID']].astype(float)
visualizer = Visualize(df)
G = visualizer.create_and_print_graph(truncated=True, radius=8, plot=False, peso=20)  # Adjust radius as needed
analyzer = GraphMatrixAnalyzer(G)
concatenated_df = pd.concat([df1['Secondary Structure'], df], axis=1)
pseudo_inverse = analyzer.get_pseudo_inverse()
adjacency_matrix = analyzer.get_adjacency_matrix()
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
secondary_structure = df['Secondary Structure'].values
autovalori, autovettori = np.linalg.eig(kirchhoff_matrix)
norms = np.linalg.norm(autovettori, axis=0)
positions = df[['X', 'Y', 'Z']].values

N = autovalori.shape[0]  # number of residues
K = kirchhoff_matrix
epsilon_0 = 0#0.4#0.7
omega = 1#2*np.pi#4*np.pi#np.pi
dt = 0.00
T = 1
k_b = 1
gamma = 1
MaxTime = 30
num_realizations =10
sperimentale= average_stochastic_process(K, epsilon_0, omega, dt, T, k_b, gamma, MaxTime, N, num_realizations)
t = np.arange(0, MaxTime, dt)
plot_residual_autocorrelation(sperimentale, autovalori, autovettori, gamma, k_b, T, MaxTime, dt, omega, epsilon_0, stringa, i=10)
plot_residual_autocorrelation(autocorrelazione_sperimentale, autovalori, autovettori, gamma, k_b, T, MaxTime, dt, omega, epsilon_0, stringa, i=20)
plot_residual_autocorrelation(autocorrelazione_sperimentale, autovalori, autovettori, gamma, k_b, T, MaxTime, dt, omega, epsilon_0, stringa, i=21)
