import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, Dropout, BatchNorm1d
from torch_geometric.nn import GATConv, global_mean_pool

class ImprovedGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=6, dropout_p=0.3, heads=8):
        super(ImprovedGAT, self).__init__()

        self.x_encoder = Linear(input_dim, hidden_dim)

        self.convs = ModuleList()
        self.bns = ModuleList()

        for _ in range(3):  # 3 GAT layers
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True))
            self.bns.append(BatchNorm1d(hidden_dim))

        self.dropout = Dropout(p=dropout_p)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.x_encoder(x)

        for conv, bn in zip(self.convs, self.bns):
            residual = x
            x = conv(x, edge_index)
            x = F.elu(x)        # ELU per stabilità in GAT
            x = bn(x)
            x = self.dropout(x)
            x = x + residual    # residual connection

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        return x

    def extract_embedding(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.x_encoder(x)

        for conv, bn in zip(self.convs, self.bns):
            residual = x
            x = conv(x, edge_index)
            x = F.elu(x)
            x = bn(x)
            x = x + residual

        x = global_mean_pool(x, batch)
        return x
    