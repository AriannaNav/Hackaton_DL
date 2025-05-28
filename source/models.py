import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, Dropout, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import GATConv, global_mean_pool

class ImprovedGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p=0.2, heads=4):
        super(ImprovedGAT, self).__init__()

        self.x_encoder = Linear(input_dim, hidden_dim)

        self.convs = ModuleList()
        self.bns = ModuleList()

        # Primo GAT layer
        self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True))
        self.bns.append(BatchNorm1d(hidden_dim))

        # Secondo GAT layer
        self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True))
        self.bns.append(BatchNorm1d(hidden_dim))

        # Terzo GAT layer
        self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True))
        self.bns.append(BatchNorm1d(hidden_dim))

        self.dropout = Dropout(p=dropout_p)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.x_encoder(x)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x

    def extract_embedding(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.x_encoder(x)

        for conv, _ in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        return x
    