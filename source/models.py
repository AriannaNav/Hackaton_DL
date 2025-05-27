# ✅ FILE: models.py (OTTIMIZZATO)
import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, Dropout
from torch_geometric.nn import GCNConv, global_mean_pool

class ImprovedGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ImprovedGCN, self).__init__()
        self.convs = ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.dropout = Dropout(p=0.3)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
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

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        return x