# models.py
import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, Dropout, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import GINEConv, GCNConv, SAGEConv, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=6):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)

    def extract_embedding(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return global_mean_pool(x, batch)

class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=6):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)

    def extract_embedding(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return global_mean_pool(x, batch)

class ImprovedGINE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=6, edge_dim=7, dropout_p=0.2):
        super(ImprovedGINE, self).__init__()
        self.x_encoder = Linear(input_dim, hidden_dim)
        self.edge_encoder = Linear(edge_dim, hidden_dim)
        self.convs = ModuleList()
        self.bns = ModuleList()
        for _ in range(4):
            nn = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            self.convs.append(GINEConv(nn))
            self.bns.append(BatchNorm1d(hidden_dim))
        self.dropout = Dropout(p=dropout_p)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.bn_final = BatchNorm1d(hidden_dim)
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
            x = x + residual
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
        for conv, bn in zip(self.convs, self.bns):
            residual = x
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + residual
        return global_mean_pool(x, batch)

class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=6):
        super(MLPClassifier, self).__init__()
        self.model = Sequential(
            Linear(input_dim, hidden_dim), ReLU(), Dropout(0.3),
            Linear(hidden_dim, hidden_dim), ReLU(), Dropout(0.3),
            Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
