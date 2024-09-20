import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib.lines as mlines
import os
import matplotlib.patches as mpatches
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
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('Beta Factors')
    ax.plot(actual_b_factors, label='Beta Factors', color='blue')

    # Plot dei fattori B predetti
    ax.plot(predicted_b_factors_scaled, label='Predicted Beta Factors', color='red')

    # Aggiungi il titolo e la legenda
    #plt.title('Confronto tra Fattori B Reali e Predetti')
    ax.legend(loc="upper right")

    # Aggiungi una griglia
    ax.grid(True, alpha=0.3)

    # Mostra il plot
    plt.tight_layout()
    if not os.path.exists(f'images/{name}/beta_factors/'):
        os.makedirs(f'images/{name}/beta_factors/')

    # Save the figure in the 'images' directory
    plt.savefig(f'images/{name}/beta_factors/Confronto tra Fattori B Reali e Predetti.png')
   

    print(f"Coefficiente di correlazione tra fattori B reali e predetti: {correlation:.4f}")
    print(f"Deviazione Quadratica Media (RMSD): {rmsd:.4f}")

    return correlation, rmsd


def compare_b_factors_with_sec_structure(actual_b_factors, predicted_b_factors, sec_struct_data, name):
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

    if not os.path.exists(f'images/{name}/beta_factors/'):
        os.makedirs(f'images/{name}/beta_factors/')

    # Salva la figura
    plt.savefig(f'images/{name}/beta_factors/Confronto_con_Struttura_Secondaria.png')

    print(f"Coefficiente di correlazione tra fattori B reali e predetti: {correlation:.4f}")
    print(f"Deviazione Quadratica Media (RMSD): {rmsd:.4f}")

    return correlation, rmsd

# This function can be called from your main script
def analyze_b_factors(df, analyzer, sec_struct_data,name):
    # Get eigenvalues and eigenvectors of the Kirchhoff matrix
    eigenvalues = analyzer.get_eigenvalues_kirchhoff()
    eigenvectors = analyzer.get_eigenvectors_kirchhoff()

    # Get actual B-factors from the dataframe
    actual_b_factors = df['B-Factor'].values

    # Predict B-factors
    predicted_b_factors = predict_b_factors(eigenvalues, eigenvectors)

    # Compare and plot results
    correlation, rmsd = compare_b_factors_with_sec_structure(actual_b_factors, predicted_b_factors, sec_struct_data, name)

    return predicted_b_factors, correlation, rmsd