import torch
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool, BatchNorm
from torch.nn import Sequential, Linear, ReLU

class ImprovedNNConv(torch.nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim):
        super(ImprovedNNConv, self).__init__()

        nn1 = Sequential(Linear(edge_dim, 32), ReLU(), Linear(32, hidden_dim * input_dim))
        self.conv1 = NNConv(input_dim, hidden_dim, nn1, aggr='mean')
        self.bn1 = BatchNorm(hidden_dim)

        nn2 = Sequential(Linear(edge_dim, 32), ReLU(), Linear(32, hidden_dim * hidden_dim))
        self.conv2 = NNConv(hidden_dim, hidden_dim, nn2, aggr='mean')
        self.bn2 = BatchNorm(hidden_dim)

        self.dropout = torch.nn.Dropout(p=0.3)
        self.lin = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

    def extract_embedding(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = global_mean_pool(x, batch)
        return x