import torch
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from sklearn.model_selection import train_test_split
import dgl
from model import GraphTransformerModel  # Assurez-vous que le modèle est dans un fichier model.py
from data_processing import preprocess_data  # Fonction pour charger et prétraiter les données

def train_model(graph, model, epochs=50, lr=0.001):
    optimizer = Adam(model.parameters(), lr=lr)
    
    # Correctement formater edge_index
    edge_index = torch.stack(graph.edges(etype='registered_in'), dim=0)
    print(f"edge_index shape: {edge_index.shape}")  # Debugging
    
    x = graph.nodes['student'].data['features']
    y = torch.randint(0, 2, (x.shape[0],))  # Exemple de labels aléatoires
    
    train_idx, val_idx = train_test_split(torch.arange(x.size(0)), test_size=0.2, random_state=42)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = cross_entropy(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model


if __name__ == "__main__":
  

    print("Loading and preprocessing data...")
    graph = preprocess_data()
    dgl.save_graphs('data/processed/oulad_graph_with_features.bin', [graph])
    print(graph.edges(etype='registered_in'))
    print("Initializing model...")
    input_dim = graph.nodes['student'].data['features'].shape[1]
    hidden_dim = 128
    output_dim = 2  # Tâche binaire
    model = GraphTransformerModel(input_dim, hidden_dim, output_dim)

    print("Training model...")
    trained_model = train_model(graph, model, epochs=100)
    torch.save(trained_model.state_dict(), 'graph_transformer.pth')
    print("Model saved ")
