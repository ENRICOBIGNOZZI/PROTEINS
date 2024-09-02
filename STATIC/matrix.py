import numpy as np
import networkx as nx
from scipy.linalg import eigh
import matplotlib.pyplot as plt

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

    def plot_matrix(self, matrix, title="Matrix"):
        plt.figure(figsize=(10, 8))
        
        # Create a binary matrix: 1 where connections exist, 0 otherwise
        binary_matrix = np.where(matrix != 0, 1, 0)
        
        # Get the indices of non-zero elements
        rows, cols = np.where(binary_matrix == 1)
        
        # Plot the dots
        plt.scatter(cols, rows, s=1, c='black')
        
        plt.title(title)
        plt.xlabel('Residue Index')
        plt.ylabel('Residue Index')
        plt.ylim(-0.5, binary_matrix.shape[0] - 0.5)  # This line sets y-axis from 0 to max
        plt.xlim(-0.5, binary_matrix.shape[1] - 0.5)
        plt.gca().invert_yaxis()  # This line inverts the y-axis
        plt.tight_layout()
        plt.show()

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





