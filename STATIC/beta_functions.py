import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
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

def compare_b_factors(actual_b_factors, predicted_b_factors,name):
    # Scala i fattori B predetti per farli corrispondere all'intervallo dei fattori B reali
    scale_factor = np.mean(actual_b_factors) / np.mean(predicted_b_factors)
    predicted_b_factors_scaled = predicted_b_factors * scale_factor

    # Calcola il coefficiente di correlazione
    correlation, _ = pearsonr(actual_b_factors, predicted_b_factors_scaled)

    # Calcola la deviazione quadratica media (RMSD)
    rmsd = np.sqrt(np.mean((actual_b_factors - predicted_b_factors_scaled)**2))

    # Crea il plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot dei fattori B reali
    ax.set_xlabel('Indice del Residuo')
    ax.set_ylabel('Fattori B')
    ax.plot(actual_b_factors, label='Fattori B Reali', color='blue')

    # Plot dei fattori B predetti
    ax.plot(predicted_b_factors_scaled, label='Fattori B Predetti', color='red')

    # Aggiungi il titolo e la legenda
    plt.title('Confronto tra Fattori B Reali e Predetti')
    ax.legend(loc="upper right")

    # Aggiungi una griglia
    ax.grid(True, alpha=0.3)

    # Mostra il plot
    plt.tight_layout()
    if not os.path.exists('images'):
        os.makedirs('images')

    # Save the figure in the 'images' directory
    plt.savefig(f'images/{name}Confronto tra Fattori B Reali e Predetti.png')
   

    print(f"Coefficiente di correlazione tra fattori B reali e predetti: {correlation:.4f}")
    print(f"Deviazione Quadratica Media (RMSD): {rmsd:.4f}")

    return correlation, rmsd

# This function can be called from your main script
def analyze_b_factors(df, analyzer,name):
    # Get eigenvalues and eigenvectors of the Kirchhoff matrix
    eigenvalues = analyzer.get_eigenvalues_kirchhoff()
    eigenvectors = analyzer.get_eigenvectors_kirchhoff()

    # Get actual B-factors from the dataframe
    actual_b_factors = df['B-Factor'].values

    # Predict B-factors
    predicted_b_factors = predict_b_factors(eigenvalues, eigenvectors)

    # Compare and plot results
    correlation, rmsd = compare_b_factors(actual_b_factors, predicted_b_factors,name)

    return predicted_b_factors, correlation, rmsd