import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
from dgl import load_graphs
import time


class ImprovedGraphTransformer(nn.Module):
    def init(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout, temporal_dim=16):
        super(ImprovedGraphTransformer, self).init()
        self.input_projection = nn.Linear(input_dim + temporal_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True
            ) for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)  # Add LayerNorm for stability
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, temporal_encodings):
        # Input projection
        # Concatenate the original features with the temporal encoding along the feature dimension
        x = torch.cat([features, temporal_encodings], dim=1)
        x = self.input_projection(x)
        x = self.layer_norm(x)
        x = self.dropout(x)

        x = x.unsqueeze(1) 
        # Transformer Encoder Layers
        for layer in self.layers:
            x = layer(x)
        x = x.squeeze(1)
        
        # Output projection
        x = self.output_projection(x)
        return x


def train_model(graph, input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout, lr, epochs, batch_size, temporal_dim=16):
    print("Preparing training...")

    # Prepare data
    train_node_ids = torch.arange(graph.num_nodes('student'))
    features = graph.nodes['student'].data['features']
    labels = graph.nodes['student'].data['labels']

    # Filter out nodes with invalid labels (-1)
    valid_mask = labels != -1
    train_node_ids = train_node_ids[valid_mask]
    features = features[valid_mask]
    labels = labels[valid_mask]

    temporal_encodings = graph.nodes['student'].data['temporal_encoding']
    temporal_encodings = temporal_encodings[valid_mask]
    dataset = TensorDataset(features, temporal_encodings, labels)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Define model, loss function, and optimizer
    model = ImprovedGraphTransformer(input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout, temporal_dim=temporal_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)  # Use AdamW for better optimization
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    # Training loop
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        model.train()

        # Training phase
        for batch_features, batch_labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features.unsqueeze(1), batch_temporal)  # Add a sequence dimension
            loss = criterion(outputs.squeeze(), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_features,  batch_temporal, batch_labels in val_dataloader:
                outputs = model(batch_features.unsqueeze(1), batch_temporal).squeeze()
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

                # Accuracy calculation
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)

        val_loss /= len(val_dataloader)
        accuracy = correct / total
        scheduler.step()
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, Time: {epoch_time:.2f}s")

    print("Training completed.")
    return model


if _name_ == "_main_":
    print("Loading graph...")
    graphs, _ = load_graphs('data/processed/oulad_graph_with_features.bin')
    graph = graphs[0]

    # Define model parameters
    input_dim = graph.nodes['student'].data['features'].shape[1]
    hidden_dim = 128
    output_dim = 4  # Assuming 4 classes (Pass, Fail, Distinction, Withdrawn)
    num_heads = 4
    num_layers = 2
    dropout = 0.1
    lr = 0.001
    epochs = 100  # Increased epochs for better learning
    batch_size = 64
    temporal_dim = 16

    print("Starting training...")
    trained_model = train_model(
        graph, input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout, lr, epochs, batch_size, temporal_dim
    )

    # Save the trained model
    torch.save(trained_model.state_dict(), 'graph_transformer_model.pth')
    print("Model saved successfully!")
