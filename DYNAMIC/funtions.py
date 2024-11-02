import numpy as np
import matplotlib.pyplot as plt
def euclidean_distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("The points must have the same number of dimensions")

    squared_distance = np.sum((np.array(point1) - np.array(point2)) ** 2)
    distance = np.sqrt(squared_distance)
    return distance

def build_graph_number_of_connections(df):
    positions = df.loc[:, ['X', 'Y', 'Z']].values
    Raggi = np.linspace(0, 45, 100)
    numero_connessioni = []

    for r in Raggi:
        N = 0
        for i in range(len(positions)):
            for j in range(len(positions)):
                if i != j:
                    if euclidean_distance(positions[i], positions[j]) < r:
                        N += 1
        numero_connessioni.append(N)

    return Raggi, numero_connessioni

def plot_comparison(df, pseudo_inverse):

    diagonale = np.diag(pseudo_inverse)
    
    # Creare una figura e un primo asse
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df['B-Factor'].values / (4 * np.pi * np.pi), label='B_i', color='blue')
    ax1.set_xlabel('x')
    ax1.set_ylabel('B-Factors', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)

    # Secondo asse per la diagonale della matrice pseudo-inversa
    ax2 = ax1.twinx()
    ax2.plot(diagonale, label='Stima', color='red')
    ax2.set_ylabel('Forecasting B-Factors', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Titolo e legenda
    plt.title('Confronto tra stima e realtà')
    fig.tight_layout()  # Per evitare sovrapposizioni di etichette
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))

    # Mostrare il grafico
    plt.show()



