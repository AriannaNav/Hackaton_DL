import gzip
import json
import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
import numpy as np

class GraphDataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None):
        self.raw = filename
        self.graphs = self.loadGraphs(self.raw)
        super().__init__(None, transform, pre_transform)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

    @staticmethod
    def loadGraphs(path):
        print(f"Loading graphs from {path}...")
        print("This may take a few minutes, please wait...")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            graphs_dicts = json.load(f)
        graphs = []
        for graph_dict in tqdm(graphs_dicts, desc="Processing graphs", unit="graph"):
            graphs.append(dictToGraphObject(graph_dict))
        return graphs

# Global normalization parameters for edge attributes (calculated from explore_data)
EDGE_ATTR_MEAN = np.array([0.11952845, 0.00841562, 0.14141378, 0.16347325, 0.17529258, 0.2773405, 0.23347288])
EDGE_ATTR_STD = np.array([0.17995964, 0.07121123, 0.20729883, 0.24505588, 0.26747337, 0.2651396, 0.20596924])

def dictToGraphObject(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    # Normalize edge_attr if present
    if graph_dict["edge_attr"]:
        edge_attr = np.array(graph_dict["edge_attr"], dtype=np.float32)
        edge_attr = (edge_attr - EDGE_ATTR_MEAN) / EDGE_ATTR_STD
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_attr = None
    num_nodes = graph_dict["num_nodes"]
    y = torch.tensor(graph_dict["y"][0], dtype=torch.long) if graph_dict["y"] is not None else None

    # Compute node degree as simple nodal feature
    deg = torch.zeros(num_nodes, dtype=torch.float)
    for src in edge_index[0]:
        deg[src] += 1.0

    # Use degree as node feature x
    x = deg.unsqueeze(1)  # (num_nodes, 1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)