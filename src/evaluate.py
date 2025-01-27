import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import dgl
from model import GraphTransformerModel  # Assurez-vous que le modèle est dans un fichier model.py
from data_processing import preprocess_data  # Fonction pour charger et prétraiter les données

def evaluate_model(graph, model):
    """
    Évalue les performances du modèle sur l'ensemble de test.
    """
    model.eval()
    edge_index = torch.stack(graph.edges(etype='registered_in'), dim=0)
    x = graph.nodes['student'].data['features']
    y = torch.randint(0, 2, (x.shape[0],))  # Exemple de labels aléatoires (à remplacer par vos vrais labels)
    _, test_idx = train_test_split(torch.arange(x.size(0)), test_size=0.2, random_state=42)

    with torch.no_grad():
        out = model(x, edge_index)
        predictions = out[test_idx].argmax(dim=1).cpu().numpy()
        labels = y[test_idx].cpu().numpy()

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        conf_matrix = confusion_matrix(labels, predictions)

        print(f"Accuracy: {acc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)

    return acc, f1, conf_matrix

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    graph = preprocess_data()
    model = GraphTransformerModel(input_dim=graph.nodes['student'].data['features'].shape[1],
                                   hidden_dim=128,
                                   output_dim=2)

    print("Loading model...")
    model.load_state_dict(torch.load('graph_transformer.pth'))
    print("Evaluating model...")
    evaluate_model(graph, model)
