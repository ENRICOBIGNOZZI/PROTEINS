import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

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

def compare_b_factors(actual_b_factors, predicted_b_factors):
    # Scala i fattori B predetti per farli corrispondere all'intervallo dei fattori B reali
    scale_factor = np.mean(actual_b_factors) / np.mean(predicted_b_factors)
    predicted_b_factors_scaled = predicted_b_factors * scale_factor

    # Calcola il coefficiente di correlazione
    correlation, _ = pearsonr(actual_b_factors, predicted_b_factors_scaled)

    # Calcola la deviazione quadratica media (RMSD)
    rmsd = np.sqrt(np.mean((actual_b_factors - predicted_b_factors_scaled)**2))

    # Crea il plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot dei fattori B reali (asse sinistro)
    ax1.set_xlabel('Indice del Residuo')
    ax1.set_ylabel('Fattori B Reali', color='blue')
    ax1.plot(actual_b_factors, label='Fattori B Reali', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Crea un secondo asse y per i fattori B predetti
    ax2 = ax1.twinx()
    ax2.set_ylabel('Fattori B Predetti', color='red')
    ax2.plot(predicted_b_factors_scaled, label='Fattori B Predetti', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Aggiungi il titolo e la legenda
    plt.title('Confronto tra Fattori B Reali e Predetti')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

    # Aggiungi una griglia
    ax1.grid(True, alpha=0.3)

    # Mostra il plot
    plt.tight_layout()
    plt.show()

    print(f"Coefficiente di correlazione tra fattori B reali e predetti: {correlation:.4f}")
    print(f"Deviazione Quadratica Media (RMSD): {rmsd:.4f}")

    return correlation, rmsd

# This function can be called from your main script
def analyze_b_factors(df, analyzer):
    # Get eigenvalues and eigenvectors of the Kirchhoff matrix
    eigenvalues = analyzer.get_eigenvalues_kirchhoff()
    eigenvectors = analyzer.get_eigenvectors_kirchhoff()

    # Get actual B-factors from the dataframe
    actual_b_factors = df['B-Factor'].values

    # Predict B-factors
    predicted_b_factors = predict_b_factors(eigenvalues, eigenvectors)

    # Compare and plot results
    correlation, rmsd = compare_b_factors(actual_b_factors, predicted_b_factors)

    return predicted_b_factors, correlation, rmsd

from Downlaod_data import PDBProcessor
from Visualize import Visualize
from funtions import plot_comparison
from matrix import GraphMatrixAnalyzer
import numpy as np
from causal_indicators_advances import TimeCorrelation, TransferEntropy, TimeResponse, CorrelationMatrixOperations, ResidualAnalysis
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.lines as mlines
# Initialize PDBProcessor
pdb_processor = PDBProcessor(pdb_id="2m10")
pdb_processor.download_pdb()
pdb_processor.load_structure()

df1 = pdb_processor.secondary_structure()
df = pdb_processor.extract_atom_data()
print(df)
df = df[df['Model ID'] == 0]
df = df[df['Atom Name'] == 'CA']


df = df.reset_index(drop=True)
concatenated_df = pd.concat([df1['Secondary Structure'], df], axis=1)
df = concatenated_df.dropna().reset_index(drop=True)
print(df)
# Initialize Visualize and analyze graph
visualizer = Visualize(df)
raggio=visualizer.calculate_and_print_average_distance()
#visualizer.plot_connections_vs_radius()
G = visualizer.create_and_print_graph(truncated=True, radius=8.0, plot=False, peso=20)  # Adjust radius as needed
# Assumendo che G sia il tuo grafo

# Initialize GraphMatrixAnalyzer
analyzer = GraphMatrixAnalyzer(G)

# Analyze B-factors
predicted_b_factors, correlation, rmsd = analyze_b_factors(df, analyzer)

print(f"B-factor analysis complete. Correlation: {correlation:.4f}, RMSD: {rmsd:.4f}")
