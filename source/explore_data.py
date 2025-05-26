from source.utils import loadGraphs
import numpy as np
from collections import Counter

def analyze_graphs(graphs):
    num_nodes = [g.num_nodes for g in graphs]
    num_edges = [g.edge_index.shape[1] for g in graphs]
    edge_attrs = [g.edge_attr.numpy() if g.edge_attr is not None else np.array([]) for g in graphs]

    print(f"Number of graphs: {len(graphs)}")
    print(f"Avg number of nodes per graph: {np.mean(num_nodes):.2f}")
    print(f"Avg number of edges per graph: {np.mean(num_edges):.2f}")
    print(f"Number of graphs with edge attributes: {sum([1 for e in edge_attrs if e.size > 0])}")

    # Edge attributes stats
    if any(e.size > 0 for e in edge_attrs):
        all_edge_attrs = np.vstack([e for e in edge_attrs if e.size > 0])
        print(f"Edge attributes shape: {all_edge_attrs.shape}")
        print(f"Edge attributes mean: {np.mean(all_edge_attrs, axis=0)}")
        print(f"Edge attributes std: {np.std(all_edge_attrs, axis=0)}")
    else:
        print("No edge attributes found.")

    # Controlla features nodali
    node_features = [g.x for g in graphs if g.x is not None]
    if node_features:
        shapes = [f.shape for f in node_features]
        print(f"Number of graphs with node features: {len(node_features)}")
        print(f"Example node feature shape: {shapes[0]}")
    else:
        print("No node features found in any graph.")

    # Label distribution
    labels = [g.y.item() for g in graphs if g.y is not None]
    if labels:
        print("\nLabel distribution:")
        label_counts = Counter(labels)
        for label, count in sorted(label_counts.items()):
            print(f"Class {label}: {count} samples")
    else:
        print("No labels found.")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "./datasets/A/train.json.gz"
    graphs = loadGraphs(path)
    analyze_graphs(graphs)