import torch
from torch.optim.lr_scheduler import StepLR
from model import GNN
from data_processing import load_data, create_graph

def train_gnn():
    data = load_data()
    G = create_graph(data)
    edge_index = torch.tensor([(u, v) for u, v in G.edges()], dtype=torch.long).t().contiguous()
    x = torch.randn(G.number_of_nodes(), 64)  # Node features can be improved based on the graph

    graph_data = Data(x=x, edge_index=edge_index)
    model = GNN(in_dim=64, hidden_dim=128, out_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Early stopping
    best_loss = float('inf')
    patience, patience_counter = 10, 0

    y = torch.randint(0, 2, (G.number_of_nodes(),))  # Replace with actual labels

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    train_gnn()
