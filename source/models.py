import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, Sequential, ReLU, Dropout, BatchNorm1d
from torch_geometric.nn import NNConv, global_mean_pool

class ImprovedNNConv(torch.nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim=256, output_dim=6, dropout_p=0.5):
        super(ImprovedNNConv, self).__init__()

        self.x_encoder = Linear(input_dim, hidden_dim)

        # Crea una edge_network MLP per ogni layer
        self.edge_mlp1 = Sequential(Linear(edge_dim, hidden_dim * hidden_dim), ReLU())
        self.edge_mlp2 = Sequential(Linear(edge_dim, hidden_dim * hidden_dim), ReLU())
        self.edge_mlp3 = Sequential(Linear(edge_dim, hidden_dim * hidden_dim), ReLU())

        self.convs = ModuleList([
            NNConv(hidden_dim, hidden_dim, self.edge_mlp1, aggr='mean'),
            NNConv(hidden_dim, hidden_dim, self.edge_mlp2, aggr='mean'),
            NNConv(hidden_dim, hidden_dim, self.edge_mlp3, aggr='mean')
        ])

        self.bns = ModuleList([BatchNorm1d(hidden_dim) for _ in range(3)])

        self.dropout = Dropout(p=dropout_p)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.x_encoder(x)

        for conv, bn in zip(self.convs, self.bns):
            residual = x
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = bn(x)
            x = self.dropout(x)
            if x.shape == residual.shape:
                x = x + residual

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        return self.lin2(x)

def extract_embedding(self, data):
    x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
    x = self.x_encoder(x)

    for conv, bn in zip(self.convs, self.bns):
        residual = x
        x = conv(x, edge_index, edge_attr)
        x = F.relu(x)
        x = bn(x)
        if x.shape == residual.shape:
            x = x + residual

    x = global_mean_pool(x, batch)
    return x

    