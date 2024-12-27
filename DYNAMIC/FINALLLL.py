import numpy as np
import matplotlib.pyplot as plt

# Parametri del problema
N = 25  # Numero di residui
a, b = 10, 20  # Indici di a e b
omega = 1.0  # Frequenza
epsilon = 0.1  # Ampiezza del rumore
T = 10000  # Passi temporali
dt = 0.01  # Incremento temporale
M = 20  # Numero di realizzazioni indipendenti

# Generazione di K come matrice simmetrica non diagonale
np.random.seed(42)  # Fissiamo il seme per riproducibilità
K_random = np.random.rand(N, N)
K = (K_random + K_random.T) / 2  # Rendiamo K simmetrica
lambda_k, V = np.linalg.eigh(K)  # Calcolo di autovalori e autovettori

# Tempo
times = np.arange(0, T*dt, dt)
lag_times = np.linspace(-T*dt, T*dt, 2*len(times)-1)

# Inizializzazione per cross-correlazione empirica media
cross_corr_empirical_sum = np.zeros(2*len(times)-1)

# Funzione delta di Kronecker
def delta_kronecker(i, j):
    return 1 if i == j else 0

# Simulazione di M processi indipendenti
for m in range(M):
    x = np.zeros((N, len(times)))
    eta = np.random.normal(0, 1, size=(N, len(times)))  # Rumore gaussiano
    
    for t in range(1, len(times)):
        for i in range(N):
            force = epsilon * (1 - np.cos(omega * times[t])) * (delta_kronecker(i, a) - delta_kronecker(i, b))
            x[i, t] = x[i, t-1] - sum(K[i, :] * x[:, t-1]) * dt + force * dt + np.sqrt(dt) * eta[i, t]
    
    # Centrare i segnali rimuovendo la media
    x10_centered = x[10, :] - np.mean(x[10, :])
    x20_centered = x[20, :] - np.mean(x[20, :])
    
    # Calcolo della cross-correlazione empirica per questa realizzazione
    cross_corr_empirical = np.correlate(x10_centered, x20_centered, mode='full')
    #cross_corr_empirical /= (len(times) * np.std(x10_centered) * np.std(x20_centered))
    
    # Sommare per ottenere la media successivamente
    cross_corr_empirical_sum += cross_corr_empirical

# Media della cross-correlazione empirica
cross_corr_empirical_mean = cross_corr_empirical_sum / (M*100)

# Calcolo della cross-correlazione teorica
def cross_correlation_theoretical(t, s, lambdas, omega, epsilon, V,i=a,j=b):
    corr = 0.0
    print(t)
    for k in range(N):
        lambda_k = lambdas[k]
        V_ik, V_jk = V[i, k], V[k, j]

        # Contributo del rumore
        noise_contrib = (np.exp(-lambda_k * abs(t - s)) - np.exp(-lambda_k * (t + s))) / (2 * lambda_k)

        # Contributo oscillante
        force_contrib = epsilon * (V[a, k] - V[b, k]) / (lambda_k**2 + omega**2)
        cos_term = lambda_k * np.cos(omega * t) * np.cos(omega * s)
        sin_term = omega * np.sin(omega * t) * np.sin(omega * s)
        oscillatory_contrib = force_contrib * (cos_term + sin_term)

        # Somma dei contributi
        corr += V_ik * V_jk * (noise_contrib + oscillatory_contrib)

    return corr




cross_corr_theoretical = [
    cross_correlation_theoretical(t, 0, lambda_k, omega, epsilon, V) for t in times
]

# Plot della cross-correlazione empirica media e teorica
plt.figure(figsize=(8, 6))
plt.plot(lag_times, cross_corr_empirical_mean, label="Cross-Correlazione Empirica Media", linestyle='--')
#plt.plot(times, cross_corr_theoretical, label="Cross-Correlazione Teorica")
plt.title("Cross-Correlazione tra residui 10 e 20 (Media su M processi)")
plt.xlabel("Tempo")
plt.ylabel("Cross-Correlazione")
plt.legend()
plt.grid()
plt.show()
