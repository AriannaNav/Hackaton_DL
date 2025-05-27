import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, Dropout, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import GINEConv, global_mean_pool

class ImprovedGINE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim=7, dropout_p=0.4):
        super(ImprovedGINE, self).__init__()
        
        self.convs = ModuleList()
        self.bns = ModuleList()

        nn1 = Sequential(Linear(edge_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.convs.append(GINEConv(nn1))
        self.bns.append(BatchNorm1d(hidden_dim))

        nn2 = Sequential(Linear(edge_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.convs.append(GINEConv(nn2))
        self.bns.append(BatchNorm1d(hidden_dim))

        self.dropout = Dropout(p=dropout_p)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
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
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        for conv, _ in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        return x