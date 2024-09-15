import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Downlaod_data import PDBProcessor
from Visualize import Visualize
from matrix import GraphMatrixAnalyzer
from causal_indicators_advances import ResidualAnalysis
import pandas as pd
import os
def identify_secondary_structure_segments(sec_struct_data):
    segments = []
    current_struct = sec_struct_data.iloc[0]
    start = 0
    for i, struct in enumerate(sec_struct_data):
        if struct != current_struct:
            segments.append((start, i-1, current_struct))
            start = i
            current_struct = struct
    segments.append((start, len(sec_struct_data)-1, current_struct))
    return segments

def calculate_segment_transfer_entropy(seg1, seg2, residual_analysis, t, time_idx):
    te_values = []
    for i in range(seg1[0], seg1[1]+1):
        for j in range(seg2[0], seg2[1]+1):
            te = residual_analysis.transfer_entropy(i, j, t[time_idx:time_idx+1])[0]
            if not np.isnan(te) and te > 1e-10:  # Ignora i valori NaN e quelli troppo piccoli
                te_values.append(te)
    return np.mean(te_values) if te_values else np.nan  # Restituisci NaN se non ci sono valori validi

def analyze_secondary_structure_transfer_entropy(pdb_id, radius=8.0, tau_mean=None):
    # Initialize PDBProcessor and load data
    pdb_processor = PDBProcessor(pdb_id=pdb_id)
    pdb_processor.download_pdb()
    pdb_processor.load_structure()

    # Extract data
    df1 = pdb_processor.secondary_structure()
    df = pdb_processor.extract_atom_data()
    df = df[df['Model ID'] == 0]
    df = df[df['Atom Name'] == 'CA']
    if pdb_id=="3LNX":
        df = df[df['Chain ID'] == 'A']
    df = df.reset_index(drop=True)
    concatenated_df = pd.concat([df1['Secondary Structure'], df], axis=1)
    df = concatenated_df.dropna().reset_index(drop=True)
    if pdb_id=="3LNY":
        df['Residue ID'].values[-6:] = [95, 96, 97, 98, 99,100]

    # Create graph
    visualizer = Visualize(df)
    G = visualizer.create_and_print_graph(truncated=True, radius=radius, plot=False)

    # Initialize GraphMatrixAnalyzer
    analyzer = GraphMatrixAnalyzer(G)
    kirchhoff_matrix = analyzer.get_kirchhoff_matrix()
    autovalori, autovettori = np.linalg.eigh(kirchhoff_matrix)

    # Initialize ResidualAnalysis
    residual_analysis = ResidualAnalysis(u=autovettori, lambdas=autovalori, mu=1, sec_struct_data=df,stringa=pdb_id)

    # Identify segments
    segments = identify_secondary_structure_segments(df['Secondary Structure'])

    # Calculate transfer entropy matrix
    n_segments = len(segments)
    te_matrix = np.zeros((n_segments, n_segments))

    if tau_mean is None:
        tau_mean = 1.0  # Default value if not provided

    t = [tau_mean]
    time_idx = 0

    for i in range(n_segments):
        for j in range(n_segments):
            te_matrix[i, j] = calculate_segment_transfer_entropy(segments[i], segments[j], residual_analysis, t, time_idx)

    # Visualize transfer entropy matrix
    plt.figure(figsize=(12, 10))
    mask = np.isnan(te_matrix) | (te_matrix <= 1e-10)
    sns.heatmap(te_matrix, annot=True, cmap='YlOrRd', xticklabels=[f"{s[2]}({s[0]}-{s[1]})" for s in segments], 
                yticklabels=[f"{s[2]}({s[0]}-{s[1]})" for s in segments], mask=mask)
    plt.title(f'Transfer Entropy between Secondary Structure Segments - {pdb_id}')
    plt.xlabel('Acceptor Segment')
    plt.ylabel('Donor Segment')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if not os.path.exists('images'):
        os.makedirs('images')

    # Save the figure in the 'images' directory
    plt.savefig(f'images/{pdb_id}analyze_secondary_structure_transfer_entropy.png')

    # Identify main donors and acceptors
    donor_scores = np.nansum(te_matrix, axis=1)  # Usa nansum per ignorare i NaN
    acceptor_scores = np.nansum(te_matrix, axis=0)

    print("Main information donors:")
    for i in np.argsort(donor_scores)[::-1]:
        print(f"Segment {segments[i][2]} (residues {segments[i][0]}-{segments[i][1]}): {donor_scores[i]:.4e}")

    print("\nMain information acceptors:")
    for i in np.argsort(acceptor_scores)[::-1]:
        print(f"Segment {segments[i][2]} (residues {segments[i][0]}-{segments[i][1]}): {acceptor_scores[i]:.4e}")

    return te_matrix, segments
