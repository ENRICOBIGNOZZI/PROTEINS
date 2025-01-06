# Protein - Library for Simulating static and dynamic of protein

# Protein is a Python libraryfor Simulating static and dynamic of protein.

## Installation
To install it in `dev` environnement:
```bash
pip install -e <path_to_STATIC>
```
option `-e` it's helpfull to install an editable version. To install in `prod`:
```bash
pip install <path_to_STATIC>
```

## Example of Usage Package for downaloding data from Protein Databank

```python
from Downlaod_data import PDBProcessor
name_of_protein="2m10"
pdb_processor = PDBProcessor(name_of_protein) 
pdb_processor.download_pdb()
pdb_processor.load_structure()
df = pdb_processor.extract_atom_data()
print(df)
```
## exemple of output:
    Residue ID          X          Y          Z  B-Factor  Model ID
0            1 -13.238699  -2.230200 -13.072400   33.6995         0
1            2 -10.020250  -3.929400 -12.479899   37.0770         0
2            3 -10.055600  -7.570600 -12.013100   36.0820         0
3            4  -8.739800  -9.402349  -9.057950   36.5790         0
4            5  -5.009550  -9.626349  -9.308800   43.7120         0
..         ...        ...        ...        ...       ...       ...
## `PDBProcessor` parameters
- `pdb_id`: name of protein to download.



## Example of Usage Package for choosing wich is the best way to legate amino acids

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from Downlaod_data import PDBProcessor
from Visualize import Visualize
from matrix import GraphMatrixAnalyzer
from causal_indicators_advances import TimeCorrelation, TransferEntropy, TimeResponse, CorrelationMatrixOperations, ResidualAnalysis, plot_time_response_multiple, plot_time_correlation_multiple, plot_time_entropy_multiple,plot_3d_correlation
from beta_functions import analyze_b_factors
warnings.filterwarnings("ignore")

raggio=8.0
stringa="3LNX"
pdb_processor = PDBProcessor(pdb_id=stringa)
pdb_processor.download_pdb()
pdb_processor.load_structure()
df1 = pdb_processor.secondary_structure()
df = pdb_processor.extract_atom_data()
df = df[df['Model ID'] == 0]
df = df[df['Atom Name'] == 'CA']
df = df[df['Chain ID'] == 'A']
df = df.reset_index(drop=True)
concatenated_df = pd.concat([df1['Secondary Structure'], df], axis=1)
df = concatenated_df.dropna().reset_index(drop=True)
df = df.loc[:,~df.T.duplicated()]
df = concatenated_df.dropna().reset_index(drop=True)
df = df.T.drop_duplicates().T
visualizer = Visualize(df)
print(df)


G = visualizer.create_and_print_graph(truncated=True, radius=raggio, plot=False, peso=20)  # Adjust radius as needed
raggio=visualizer.calculate_and_print_average_distance()
visualizer.plot_connections_vs_radius()

analyzer = GraphMatrixAnalyzer(G)

concatenated_df = pd.concat([df1['Secondary Structure'], df], axis=1)
pseudo_inverse = analyzer.get_pseudo_inverse()
adjacency_matrix = analyzer.get_adjacency_matrix()
kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
secondary_structure = df['Secondary Structure'].values
predicted_b_factors, correlation, rmsd = analyze_b_factors(df, analyzer,df,stringa)
analyzer.plot_matrix(kirchhoff_matrix, secondary_structure, title="Matrice di Kirchhoff della Proteina",nome=stringa)
autovalori, autovettori = np.linalg.eig(kirchhoff_matrix)

# Parameters
k_B = 1  # Boltzmann constant (J/K)
T = 1  # Temperature (K)
g = 1  # A constant for simplicity
mu = 1  # Time scaling factor
t = np.linspace(0., 2, 100)  # Time points


time_correlation = TimeCorrelation(u=autovettori, lambdas=autovalori, mu=mu, sec_struct_data=df,stringa=stringa)
transfer_entropy = TransferEntropy(u=autovettori, lambdas=autovalori, mu=mu, sec_struct_data=df,stringa=stringa)
time_response = TimeResponse(u=autovettori, lambdas=autovalori, mu=mu, sec_struct_data=df,stringa=stringa)
matrix_operations = CorrelationMatrixOperations(u=autovettori, lambdas=autovalori, mu=mu, sec_struct_data=df,stringa=stringa)
residual_analysis = ResidualAnalysis(u=autovettori, lambdas=autovalori, mu=mu, sec_struct_data=df,stringa=stringa)

normalized_autocorrelations = np.zeros((94, len(t)))
for i in range(94):
    C_ii_t = time_correlation.time_correlation(i, i, t)
    normalized_autocorrelations[i, :] = time_correlation.normalize_autocorrelations(C_ii_t)

# Calcola e stampa i tempi caratteristici
tau_mean, taus = time_correlation.estimate_tau_2(t, normalized_autocorrelations)
print(f"Tempo caratteristico medio: {tau_mean:.4f}")
time_correlation.plot_tau_histogram( t, normalized_autocorrelations)
time_correlation.plot_autocorrelation_fits(t, normalized_autocorrelations)


correlation_matrix = matrix_operations.compute_static_correlation_matrix()
positive_matrix, negative_matrix = matrix_operations.split_correlation_matrix(correlation_matrix)

matrix_operations.plot_correlation_matrix_nan(correlation_matrix, kirchhoff_matrix,secondary_structure, positive_only=False)
matrix_operations.plot_correlation_matrix_nan(correlation_matrix,kirchhoff_matrix, secondary_structure, positive_only=True)

residual_analysis.analyze_mfpt(adjacency_matrix,kirchhoff_matrix, secondary_structure )


num_residues=len(autovalori)



plot_3d_correlation(t, num_residues,residual_analysis)

lista=[]
for i in range(0, 95, 3):  
    lista.append(i)

t_car=[tau_mean-1/2*tau_mean,tau_mean,tau_mean+1/2*tau_mean]
residue_pairs = [(20, 30), (20, 75),(20,72),(24,72),(14,44),(28,30),(30,72),(27,30),(27,72)]


residual_analysis.analyze_mfpt(adjacency_matrix,kirchhoff_matrix, secondary_structure )

 
time_idx = 0
for i in range(len(lista)):
    residual_analysis.plot_residual_correlation_vs_j(i=lista[i], t=t_car, time_idx=time_idx)
    residual_analysis.plot_residual_time_response_vs_j(i=lista[i], t=t_car, time_idx=time_idx)
    residual_analysis.plot_residual_transfer_entropy_vs_j_accettore(i=lista[i], t=t_car, time_idx=time_idx)
    residual_analysis.plot_residual_transfer_entropy_vs_j_donatore(i=lista[i], t=t_car, time_idx=time_idx)
    residual_analysis.plot_time_matrix_i_j(i=lista[i],adjacency_matrix=adjacency_matrix)
    residual_analysis.plot_time_matrix_j_i(i=lista[i],adjacency_matrix=adjacency_matrix)
    residual_analysis.plot_time_matrix_i_j_plus_response(i=lista[i],adjacency_matrix=adjacency_matrix,t=t)


residue_pairs = [(20, 24),(20, 30), (20, 60), (20, 75),(20,72),(24,72),(14,44)]
t = np.linspace(0., 2, 300) 
plot_time_response_multiple(time_response, residue_pairs, t, 'Time Response for Selected Residue Pairs',name=stringa)
plot_time_correlation_multiple(time_correlation, residue_pairs, t, 'Time Correlation for Selected Residue Pairs',name=stringa)
plot_time_entropy_multiple(transfer_entropy, residue_pairs, t, 'Time Transfer entropy TE_{i,j} for Selected Residue Pairs',name=stringa)


residual_analysis.analyze_mfpt(adjacency_matrix,kirchhoff_matrix, secondary_structure )

 
```




