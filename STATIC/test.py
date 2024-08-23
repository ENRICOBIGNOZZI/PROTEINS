from Downlaod_data import PDBProcessor
from Visualize import Visualize
from funtions import plot_comparison
from matrix import GraphMatrixAnalyzer
import numpy as np
from causal_indicators import CorrelationAnalysis
pdb_processor = PDBProcessor("2m10")
pdb_processor.download_pdb()
pdb_processor.load_structure()

# Extract atomic data and create a DataFrame
'''df = pdb_processor.extract_atom_data()
df=df[df['Atom Name']=='CA']
visualizer = Visualize(df)
visualizer.plot_different_models()
df=df[df['Model ID']==0]
df=df.reset_index(drop=True)
visualizer = Visualize(df)

#visualizer.plot_connections_vs_radius()
#visualizer.calculate_and_print_average_distance()
#visualizer.create_and_print_graph(truncated=True, radius=10,plot=True)  # Usa il raggio che preferisci
G=visualizer.create_and_print_graph(truncated=False, radius=None,plot=False)  # Usa il raggio che preferisci
# Creiamo un'istanza della classe
analyzer = GraphMatrixAnalyzer(G)

# Otteniamo le matrici e gli autovalori/autovettori
pseudo_inverse = analyzer.get_pseudo_inverse()
adjacency_matrix = analyzer.get_adjacency_matrix()
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
eigenvalues = analyzer.get_eigenvalues_adjacency()
eigenvectors = analyzer.get_eigenvectors_adjacency()
#nalyzer.plot_all_matrices()
plot_comparison(df, pseudo_inverse)
visualizer = Visualize(df)
#visualizer.plot_connections_vs_radius()
#visualizer.calculate_and_print_average_distance()
#visualizer.create_and_print_graph(truncated=True, radius=10,plot=True)  # Usa il raggio che preferisci
G=visualizer.create_and_print_graph(truncated=False, radius=None,plot=False)  # Usa il raggio che preferisci
# Creiamo un'istanza della classe
analyzer = GraphMatrixAnalyzer(G)

# Otteniamo le matrici e gli autovalori/autovettori
pseudo_inverse = analyzer.get_pseudo_inverse()
adjacency_matrix = analyzer.get_adjacency_matrix()
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
eigenvalues = analyzer.get_eigenvalues_adjacency()
eigenvectors = analyzer.get_eigenvectors_adjacency()
#nalyzer.plot_all_matrices()


#plot_comparison(df, pseudo_inverse)
autovalori, autovettori = np.linalg.eig(kirchhoff_matrix)  # Ensure kirchhoff_matrix is defined
k_B = 1  # Boltzmann constant (J/K)
T = 1  # Temperature (K)
g = 1  # A constant for simplicity
mu = 1 # Time scaling factor
t = np.linspace(0.1, 25, 1000)  # Time points

# Initialize the analysis class
analysis = CorrelationAnalysis(u=autovettori,lambdas=autovalori,mu= mu)
correlation_matrix = analysis.compute_correlation_matrix()

# Split into positive and negative correlations
positive_matrix, negative_matrix = analysis.split_correlation_matrix(correlation_matrix)


# Plot the matrices
analysis.plot_correlation_matrix(correlation_matrix, title='Total Correlation Matrix')
analysis.plot_correlation_matrix(positive_matrix, title='Positive Correlation Matrix')
analysis.plot_correlation_matrix(negative_matrix, title='Negative Correlation Matrix')

print(df)'''
df = pdb_processor.extract_atom_data()
df=df[df['Atom Name']=='CA']
df=pdb_processor.ensamble(df)
print(df)

# Visualizzazione
visualizer = Visualize(df)
visualizer.plot_connections_vs_radius()
#visualizer.calculate_and_print_average_distance()
#visualizer.create_and_print_graph(truncated=True, radius=10,plot=True)  # Usa il raggio che preferisci
G=visualizer.create_and_print_graph(truncated=False, radius=None,plot=False)  # Usa il raggio che preferisci
# Creiamo un'istanza della classe
analyzer = GraphMatrixAnalyzer(G)

# Otteniamo le matrici e gli autovalori/autovettori
pseudo_inverse = analyzer.get_pseudo_inverse()
adjacency_matrix = analyzer.get_adjacency_matrix()
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
eigenvalues = analyzer.get_eigenvalues_adjacency()
eigenvectors = analyzer.get_eigenvectors_adjacency()
#nalyzer.plot_all_matrices()


#plot_comparison(df, pseudo_inverse)
autovalori, autovettori = np.linalg.eig(kirchhoff_matrix)  # Ensure kirchhoff_matrix is defined
k_B = 1  # Boltzmann constant (J/K)
T = 1  # Temperature (K)
g = 1  # A constant for simplicity
mu = 1 # Time scaling factor
t = np.linspace(0.1, 25, 100)  # Time points

# Initialize the analysis class
analysis = CorrelationAnalysis(u=autovettori,lambdas=autovalori,mu= mu)
correlation_matrix = analysis.compute_correlation_matrix()

# Split into positive and negative correlations
positive_matrix, negative_matrix = analysis.split_correlation_matrix(correlation_matrix)


# Plot the matrices
#analysis.plot_correlation_matrix(correlation_matrix, title='Total Correlation Matrix')
#analysis.plot_correlation_matrix(positive_matrix, title='Positive Correlation Matrix')
#analysis.plot_correlation_matrix(negative_matrix, title='Negative Correlation Matrix')
#analysis.plot_correlation_matrix_2(correlation_matrix,  positive_only=False)
#analysis.plot_correlation_matrix_2(correlation_matrix,  positive_only=True)
#analysis.plot_time_correlation(0, 1, t)
#analysis.plot_time_response(0, 1, t)
#analysis.plot_transfer_entropy(0, 1, t)
i=10
#analysis.plot_fixed_i_correlations(i=i)
#analysis.plot_fixed_i_time_response(i)
analysis.plot_fixed_i_transfer_entropy(i)
# Compute and plot the time response matrix



