import torch
from model import GNN, GraphTransformer
from data_processing import load_data
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)
        accuracy = accuracy_score(data.y.cpu(), pred.cpu())
        f1 = f1_score(data.y.cpu(), pred.cpu(), average="weighted")
        cm = confusion_matrix(data.y.cpu(), pred.cpu())
        return accuracy, f1, cm

# Load data
data = load_data()

# Model parameters
in_dim = data.x.size(1)
hidden_dim = 64
out_dim = 2
num_heads = 4

# Initialize models
gnn_model = GNN(in_dim, hidden_dim, out_dim)
gt_model = GraphTransformer(in_dim, hidden_dim, out_dim, num_heads)

# Evaluate models
gnn_accuracy, gnn_f1, gnn_cm = evaluate(gnn_model, data)
gt_accuracy, gt_f1, gt_cm = evaluate(gt_model, data)

# Display results
print(f"GNN - Accuracy: {gnn_accuracy}, F1: {gnn_f1}\nConfusion Matrix:\n{gnn_cm}")
print(f"Graph Transformer - Accuracy: {gt_accuracy}, F1: {gt_f1}\nConfusion Matrix:\n{gt_cm}")
