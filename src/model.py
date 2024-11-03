import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class GNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphTransformer(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GraphTransformer, self).__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=num_heads)
        self.gat2 = GATConv(hidden_dim * num_heads, out_dim, heads=num_heads, concat=False)

    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = self.gat2(h, edge_index)
        return F.log_softmax(h, dim=1)
