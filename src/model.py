import torch
from torch.nn import Module, Linear
from torch_geometric.nn import TransformerConv

class GraphTransformerModel(Module):
    """
    Modèle Graph Transformer pour les graphes hétérogènes.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, num_heads=4, dropout=0.2):
        super(GraphTransformerModel, self).__init__()
        self.layers = torch.nn.ModuleList()

        # Ajouter les couches TransformerConv
        for i in range(num_layers):
            self.layers.append(
                TransformerConv(
                    in_channels=input_dim if i == 0 else hidden_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=False  # Pour garder la dimension constante
                )
            )

        # Couche Fully Connected pour la classification
        self.classifier = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        """
        Passe avant du modèle.
        Args:
            x: caractéristiques des nœuds (features)
            edge_index: indices des arêtes du graphe
        Returns:
            logits des classes de sortie
        """
        for layer in self.layers:
            x = layer(x, edge_index).relu()

        # Appliquer la couche de classification
        x = self.classifier(x)
        return x
