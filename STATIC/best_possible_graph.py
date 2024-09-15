import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx
from scipy.spatial import distance_matrix
from Downlaod_data import PDBProcessor
import pandas as pd
import optuna
from optuna.trial import Trial
import matplotlib.pyplot as plt  # Assicurati di importare matplotlib

# Load protein data (2m10)
pdb_processor = PDBProcessor(pdb_id="2m10")
pdb_processor.download_pdb()
pdb_processor.load_structure()

df1 = pdb_processor.secondary_structure()
df = pdb_processor.extract_atom_data()
df = df[df['Atom Name'] == 'CA']
df = df.reset_index(drop=True)
concatenated_df = pd.concat([df1['Secondary Structure'], df], axis=1)
df = concatenated_df.dropna().reset_index(drop=True)

def create_graph(df):
    G = nx.Graph()
    for i, row in df.iterrows():
        G.add_node(i, pos=(row['X'], row['Y'], row['Z']), 
                   sec_struct=row['Secondary Structure'],
                   b_factor=row['B-Factor'])
    
    # Create a fully connected graph
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            G.add_edge(i, j)
    
    return G

class GATWithLearnableAsymmetricEdgeWeights(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_heads=1):
        super(GATWithLearnableAsymmetricEdgeWeights, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=num_heads, edge_dim=1)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, edge_dim=1)
        self.lin = torch.nn.Linear(hidden_channels, 1)
        self.edge_weight = torch.nn.Linear(3, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index):
        edge_attr_input = torch.cat([
            torch.arange(edge_index.size(1), dtype=torch.float32).unsqueeze(1),
            edge_index[0].float().unsqueeze(1),
            edge_index[1].float().unsqueeze(1),
        ], dim=1)
        edge_attr = self.sigmoid(self.edge_weight(edge_attr_input))
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.lin(x)
        return x, edge_attr

def evaluate_graph(G, df, num_node_features, hidden_channels, num_heads, learning_rate):
    data = from_networkx(G)
    data.x = torch.tensor(df[['X', 'Y', 'Z']].values, dtype=torch.float)
    data.y = torch.tensor(df['B-Factor'].values, dtype=torch.float).unsqueeze(1)
    
    sec_struct_mapping = {'H': 0, 'E': 1, 'C': 2}
    sec_struct_encoded = torch.tensor([sec_struct_mapping.get(s, 2) for s in df['Secondary Structure']], dtype=torch.long)
    sec_struct_one_hot = F.one_hot(sec_struct_encoded, num_classes=3)
    data.x = torch.cat([data.x, sec_struct_one_hot.float()], dim=1)
    
    # Use only the first num_node_features
    data.x = data.x[:, :num_node_features]
    
    num_nodes = data.num_nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:int(0.8 * num_nodes)] = True
    test_mask = ~train_mask
    
    model = GATWithLearnableAsymmetricEdgeWeights(num_node_features=num_node_features, hidden_channels=hidden_channels, num_heads=num_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.L1Loss()  # MAE
    
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index)
        loss = loss_fn(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        out, learned_weights = model(data.x, data.edge_index)
        test_loss = loss_fn(out[test_mask], data.y[test_mask])
    
    for (u, v), weight in zip(G.edges(), learned_weights):
        G[u][v]['weight'] = weight.item()
        if not G.is_directed():
            G[v][u]['weight'] = weight.item()

    return test_loss.item(), G

def objective(trial: Trial):
    num_node_features = 6  # Fixed to 6 (3 spatial + 3 one-hot encoded)
    hidden_channels = trial.suggest_int('hidden_channels', 8, 64)
    num_heads = trial.suggest_int('num_heads', 1, 8)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

    G = create_graph(df)
    test_loss, _ = evaluate_graph(G, df, num_node_features, hidden_channels, num_heads, learning_rate)
    
    return test_loss

# Run hyperparameter optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)  # Adjust n_trials as needed

# Print best hyperparameters and loss
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Use best hyperparameters to create and evaluate final model
best_hidden_channels = trial.params['hidden_channels']
best_num_heads = trial.params['num_heads']
best_learning_rate = trial.params['learning_rate']

G = create_graph(df)
final_loss, G_with_weights = evaluate_graph(G, df, 6, best_hidden_channels, best_num_heads, best_learning_rate)

print(f"Final Loss with best hyperparameters: {final_loss}")
print("Example of learned weights:")
for i, (u, v, data) in enumerate(G_with_weights.edges(data=True)):
    if i < 5:  # Show only the first 5 for brevity
        print(f"Edge ({u}, {v}): weight = {data['weight']}")
    else:
        break

# Plot dei legami del nodo 0
edges_to_plot = [(0, v) for u, v in G_with_weights.edges() if u == 0 or v == 0]

# Creazione del grafico
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G_with_weights)  # Posizione dei nodi
nx.draw(G_with_weights, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
nx.draw_networkx_edges(G_with_weights, pos, edgelist=edges_to_plot, edge_color='red', width=2)

plt.title("Edges of Node 0 with Other Nodes")
plt.show()

# Here you can add code to save or visualize the graph with weights
