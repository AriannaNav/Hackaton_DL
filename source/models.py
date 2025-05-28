import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, Dropout, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import GINEConv, global_mean_pool

class ImprovedGINE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=6, edge_dim=7, dropout_p=0.2):
        super(ImprovedGINE, self).__init__()

        self.x_encoder = Linear(input_dim, hidden_dim)
        self.edge_encoder = Linear(edge_dim, hidden_dim)

        self.convs = ModuleList()
        self.bns = ModuleList()

        for _ in range(3):  # 3 GINE layers
            nn = Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(nn))
            self.bns.append(BatchNorm1d(hidden_dim))

        self.dropout = Dropout(p=dropout_p)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.x_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for conv, bn in zip(self.convs, self.bns):
            residual = x
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + residual  # Residual connection

        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x

    def extract_embedding(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.x_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for conv, _ in zip(self.convs, self.bns):
            residual = x
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = x + residual

        x = global_mean_pool(x, batch)
        return x
