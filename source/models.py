import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, Dropout, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import GINEConv, global_mean_pool

class ImprovedGINE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=6, edge_dim=7, dropout_p=0.2, num_layers=4):
        super().__init__()

        self.x_encoder = Linear(input_dim, hidden_dim)
        self.edge_encoder = Linear(edge_dim, hidden_dim)

        self.convs = ModuleList()
        self.norms = ModuleList()

        for _ in range(num_layers):
            nn = Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim)
            )
            conv = GINEConv(nn)
            self.convs.append(conv)
            self.norms.append(BatchNorm1d(hidden_dim))

        self.dropout = Dropout(dropout_p)

        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)
        self.bn_final = BatchNorm1d(hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.x_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.bn_final.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.x_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index, edge_attr)
            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)
            x = x + h  # residual connection

        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        x = self.bn_final(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.lin2(x)

    def extract_embedding(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.x_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index, edge_attr)
            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)
            x = x + h

        x = global_mean_pool(x, batch)
        return x