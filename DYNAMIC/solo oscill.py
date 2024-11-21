import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit

# Parametri
K = 1  # Costante di dissipazione
epsilon_0 = 100  # Intensità del forcing oscillante
omega = 2 * np.pi  # Frequenza angolare
a, b = 20, 75  # Indici dei residui i = a, b
T = 15000  # Numero di passi temporali
dt = 0.05  # Intervallo di tempo
N_process = 5  # Numero di realizzazioni
gamma = 1  # Dissipazione (per lo smorzamento)

# Funzione per la derivata dx/dt
def dynamics(x, t, K, omega, epsilon_0, a, b):
    dxdt = -K * x 
    forcing = epsilon_0 * (1 - np.cos(omega * t))
   
    return dxdt + forcing

# Delta di Kronecker per selezionare i residui a e b
def delta(t, i):
    if t == i:
        return 1
    else:
        return 0

# Simulazione di X con equazione differenziale
def simulate_process():
    x_init = np.zeros(1)  # Inizializzazione del sistema (se è un vettore di una sola variabile)
    t = np.arange(0, T * dt, dt)  # Tempo discreto
    X = np.zeros((N_process, T))  # Memorizza i risultati delle N_process realizzazioni

    # Simulazione del processo stocastico per ogni realizzazione
    for n in range(N_process):
        X[n, :] = odeint(dynamics, x_init, t, args=(K, omega, epsilon_0, a, b)).flatten()

    return X, t

# Funzione per calcolare l'autocorrelazione empirica
def autocorrelation(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]

# Funzione di fitting per la correlazione teorica
def fit_theoretical_autocorr(tau, A, B):
    return A * np.exp(-tau / B)  # Fitting esponenziale come esempio

# Calcolo dell'autocorrelazione empirica e teorica
X, t = simulate_process()

empirical_autocorr = []
for idx in [a]:  # Seleziona il residuo da analizzare
    autocorr_sum = np.zeros(T)
    for n in range(N_process):
        autocorr_sum += autocorrelation(X[n, :]) / len(X[n, :])  # Correlazione per ogni processo
    avg_autocorr = autocorr_sum / N_process  # Media sull'autocorrelazione di tutte le realizzazioni
    empirical_autocorr.append(avg_autocorr)

# Autocorrelazione teorica (approssimazione)
teoretical_corrs = []
for tau in np.arange(0, T * dt, dt):
    # Si suppone una dinamica esponenziale per la correlazione teorica
    teoretical_corrs.append(np.exp(-tau / (gamma * omega)))  # Correlazione esponenziale

# Fitting della correlazione teorica
tau = np.arange(0, T * dt, dt)
popt, _ = curve_fit(fit_theoretical_autocorr, tau, teoretical_corrs, p0=[1, 100])

# Plot dei risultati
plt.figure(figsize=(10, 6))
for idx, avg_autocorr in zip([a], empirical_autocorr):
    plt.plot(t, avg_autocorr / avg_autocorr[0], label=f'Autocorrelazione Empirica - x_{idx}')
plt.plot(tau, fit_theoretical_autocorr(tau, *popt), label='Autocorrelazione Teorica Fit', linestyle='-')
plt.title('Autocorrelazione Empirica e Teorica')
plt.xlabel('Tempo (t)')
plt.ylabel('Correlazione')
plt.legend()
plt.grid(True)
plt.show()
