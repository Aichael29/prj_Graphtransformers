import torch
import pandas as pd
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph

def load_data():
    # Load the dataset files
    student_info = pd.read_csv("data/raw/studentInfo.csv")
    student_info = student_info.fillna(0)

    # Extract features and labels
    features = student_info[['studied_credits', 'num_of_prev_attempts']].values
    labels = student_info['final_result'].apply(lambda x: 1 if x == 'Pass' else 0).values

    # Convert to tensors
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # Use k-nearest neighbors to create a sparse edge index
    k = 5  # Define the number of neighbors
    adjacency_matrix = kneighbors_graph(features, k, mode='connectivity', include_self=False)
    edge_index = torch.tensor(adjacency_matrix.nonzero(), dtype=torch.long)

    # Create the Data object for torch_geometric
    data = Data(x=X, edge_index=edge_index, y=y)
    return data
