import numpy as np
import networkx as nx
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
class GraphMatrixAnalyzer:
    def __init__(self, graph):
        self.graph = graph
        self.adjacency_matrix = None
        self.kirchhoff_matrix = None
        self.pseudo_inverse = None
        self.eigenvalues_adjacency = None
        self.eigenvectors_adjacency = None
        self.eigenvalues_kirchhoff = None
        self.eigenvectors_kirchhoff = None
        self.eigenvalues_pseudo_inverse = None
        self.eigenvectors_pseudo_inverse = None
        
        self._compute_matrices()
        self._compute_eigenvalues_and_eigenvectors()

    def _compute_matrices(self):
        # Compute adjacency matrix
        self.adjacency_matrix = nx.adjacency_matrix(self.graph).todense()
        
        # Compute degree matrix
        degree_matrix = np.diag(np.sum(self.adjacency_matrix, axis=1))
        
        # Compute Kirchhoff matrix (Laplacian)
        self.kirchhoff_matrix = degree_matrix - self.adjacency_matrix
        
        # Compute pseudo-inverse of Kirchhoff matrix
        self.pseudo_inverse = np.linalg.pinv(self.kirchhoff_matrix)

    def _compute_eigenvalues_and_eigenvectors(self):
        # Calcolare gli autovalori e autovettori della matrice di adiacenza
        self.eigenvalues_adjacency, self.eigenvectors_adjacency = np.linalg.eigh(self.adjacency_matrix)
        
        # Calcolare gli autovalori e autovettori della matrice di Kirchhoff
        self.eigenvalues_kirchhoff, self.eigenvectors_kirchhoff = eigh(self.kirchhoff_matrix)
        
        # Calcolare gli autovalori e autovettori della pseudo-inversa della matrice di Kirchhoff
        self.eigenvalues_pseudo_inverse, self.eigenvectors_pseudo_inverse = np.linalg.eigh(self.pseudo_inverse)

    def get_pseudo_inverse(self):
        return self.pseudo_inverse

    def get_adjacency_matrix(self):
        return self.adjacency_matrix

    def get_kirchhoff_matrix(self):
        return self.kirchhoff_matrix

    def get_eigenvalues_adjacency(self):
        return self.eigenvalues_adjacency

    def get_eigenvectors_adjacency(self):
        return self.eigenvectors_adjacency

    def get_eigenvalues_kirchhoff(self):
        return self.eigenvalues_kirchhoff

    def get_eigenvectors_kirchhoff(self):
        return self.eigenvectors_kirchhoff

    def get_eigenvalues_pseudo_inverse(self):
        return self.eigenvalues_pseudo_inverse

    def get_eigenvectors_pseudo_inverse(self):
        return self.eigenvectors_pseudo_inverse
    

    def plot_matrix(self, matrix, secondary_structure, nome,title="Matrix"):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create a binary matrix: 1 where connections exist, 0 otherwise
        binary_matrix = np.where(matrix != 0, 1, 0)
        
        # Get the indices of non-zero elements
        rows, cols = np.where(binary_matrix == 1)
        
        # Plot the dots
        ax.scatter(cols, rows, s=10, c='black')

        # Add rectangles
        rectangle1 = patches.Rectangle((19, 71), 5, 9, linewidth=1, edgecolor='r', facecolor='none')
        rectangle2 = patches.Rectangle((71, 19), 9, 5, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rectangle1)
        ax.add_patch(rectangle2)
        
        # Create segments on x and y axes based on secondary structure
        start = 0
        current_structure = secondary_structure[0]
        if len(secondary_structure) >= 2:
            current_structure = current_structure[0]
        
        for i, structure in enumerate(secondary_structure):
            if len(structure) >= 2:
                structure = structure[0]
           
            if structure != current_structure:
                color = 'red' if current_structure == 'H' else 'blue' if current_structure == 'E' else 'green'
                ax.plot([start, i], [0, 0], color=color, linewidth=5)
                ax.plot([0, 0], [start, i], color=color, linewidth=5)
                #ax.text((start+i)/2, -0.5, current_structure, ha='center', va='top')
                #ax.text(-0.5, (start+i)/2, current_structure, ha='right', va='center')
                ax.text((start+i)/2, 0, current_structure, ha='center', va='top')
                ax.text(0, (start+i)/2, current_structure, ha='right', va='center')
                start = i
                current_structure = structure
        color = 'red' if current_structure == 'H' else 'blue' if current_structure == 'E' else 'green'
        ax.plot([start, i+1], [0, 0], color=color, linewidth=5)
        ax.plot([0, 0], [start, i+1], color=color, linewidth=5)
        ax.text((start+i+1)/2, 0, current_structure, ha='center', va='top')
        ax.text(0, (start+i+1)/2, current_structure, ha='right', va='center')
        ax.set_title(title)
        ax.set_xlabel('Residue Index')
        ax.set_ylabel('Residue Index')
        ax.set_ylim(0, binary_matrix.shape[0])  # This line sets y-axis from 0 to max
        ax.set_xlim(0, binary_matrix.shape[1] )
        
        plt.tight_layout()
        if not os.path.exists('images'):
            os.makedirs('images')
        # Save the figure in the 'images' directory
        plt.savefig(f'images/{nome}_{title}.png')

    def plot_eigenvalues(self, eigenvalues, title="Autovalori"):
        plt.figure(figsize=(10, 6))
        plt.plot(eigenvalues, 'o', linestyle='-', color='b')
        plt.xlabel('Indice')
        plt.ylabel('Autovalore')
        plt.title(title)
        plt.grid(True)
        plt.show()

    def plot_eigenvectors(self, eigenvectors, title="Autovettori"):
        plt.figure(figsize=(10, 6))
        num_eigenvectors = eigenvectors.shape[1]
        for i in range(num_eigenvectors):
            plt.plot(eigenvectors[:, i], label=f'Autovettore {i}')
        plt.xlabel('Indice')
        plt.ylabel('Valore')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_all_matrices(self):
        self.plot_matrix(self.get_adjacency_matrix(), title="Matrice di Adiacenza")
        self.plot_matrix(self.get_kirchhoff_matrix(), title="Matrice di Kirchhoff")
        self.plot_matrix(self.get_pseudo_inverse(), title="Pseudo-inversa della Matrice di Kirchhoff")
        
        # Plotta gli autovalori di tutte le matrici
        self.plot_eigenvalues(self.get_eigenvalues_adjacency(), title="Autovalori della Matrice di Adiacenza")
        self.plot_eigenvalues(self.get_eigenvalues_kirchhoff(), title="Autovalori della Matrice di Kirchhoff")
        self.plot_eigenvalues(self.get_eigenvalues_pseudo_inverse(), title="Autovalori della Pseudo-inversa della Matrice di Kirchhoff")
        
        # Plotta gli autovettori di tutte le matrici
        self.plot_eigenvectors(self.get_eigenvectors_adjacency(), title="Autovettori della Matrice di Adiacenza")
        self.plot_eigenvectors(self.get_eigenvectors_kirchhoff(), title="Autovettori della Matrice di Kirchhoff")
        self.plot_eigenvectors(self.get_eigenvectors_pseudo_inverse(), title="Autovettori della Pseudo-inversa della Matrice di Kirchhoff")

    def compute_floquet_multipliers(self, T):
        """
        Calcola i moltiplicatori di Floquet.
        
        :param T: Periodo della perturbazione
        :return: Array dei moltiplicatori di Floquet
        """
        return np.exp(self.eigenvalues_kirchhoff * T)

    def plot_floquet_multipliers(self, T):
        """
        Visualizza i moltiplicatori di Floquet nel piano complesso.
        
        :param T: Periodo della perturbazione
        """
        multipliers = self.compute_floquet_multipliers(T)
        plt.figure(figsize=(10, 8))
        plt.scatter(multipliers.real, multipliers.imag)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title("Moltiplicatori di Floquet")
        plt.xlabel("Parte Reale")
        plt.ylabel("Parte Immaginaria")
        plt.grid(True)
        plt.show()

    def analyze_stability(self):
        """
        Analizza la stabilità del sistema basandosi sugli autovalori.
        """
        stable_modes = np.sum(self.eigenvalues_kirchhoff < 0)
        unstable_modes = np.sum(self.eigenvalues_kirchhoff > 0)
        critical_modes = np.sum(np.isclose(self.eigenvalues_kirchhoff, 0))
        
        print(f"Modi stabili: {stable_modes}")
        print(f"Modi instabili: {unstable_modes}")
        print(f"Modi critici: {critical_modes}")

    def natural_frequencies(self):
        """
        Calcola e visualizza le frequenze naturali del sistema.
        """
        frequencies = np.sqrt(np.abs(self.eigenvalues_kirchhoff))
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, 'o-')
        plt.title("Frequenze Naturali del Sistema")
        plt.xlabel("Indice del Modo")
        plt.ylabel("Frequenza")
        plt.grid(True)
        plt.show()



