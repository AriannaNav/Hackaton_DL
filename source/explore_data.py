from load_data import GraphDataset
import numpy as np
from collections import Counter

def analyze_graphs(graphs):
    num_nodes = [g.num_nodes for g in graphs]
    num_edges = [g.edge_index.shape[1] for g in graphs]
    edge_attrs = [g.edge_attr.numpy() if g.edge_attr is not None else np.array([]) for g in graphs]

    print("======== GENERAL INFO ========")
    print(f"Number of graphs: {len(graphs)}")
    print(f"Avg number of nodes per graph: {np.mean(num_nodes):.2f}")
    print(f"Avg number of edges per graph: {np.mean(num_edges):.2f}")
    print(f"Graphs with edge attributes: {sum(e.size > 0 for e in edge_attrs)} / {len(graphs)}")

    # Edge attributes stats
    if any(e.size > 0 for e in edge_attrs):
        all_edge_attrs = np.vstack([e for e in edge_attrs if e.size > 0])
        print("\n======== EDGE ATTRIBUTES ========")
        print(f"Edge attr shape: {all_edge_attrs.shape}")
        print(f"Mean: {np.mean(all_edge_attrs, axis=0)}")
        print(f"Std: {np.std(all_edge_attrs, axis=0)}")
    else:
        print("\nNo edge attributes found.")

    # No node features in questo dataset, quindi skip

    # Labels
    labels = [g.y.item() for g in graphs if g.y is not None]
    if labels:
        print("\n======== LABEL DISTRIBUTION ========")
        label_counts = Counter(labels)
        for label, count in sorted(label_counts.items()):
            print(f"Class {label}: {count} samples ({(count / len(labels)) * 100:.2f}%)")
    else:
        print("\nNo labels found.")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "./datasets/A/train.json.gz"
    dataset = GraphDataset(path)
    analyze_graphs(dataset.graphs)