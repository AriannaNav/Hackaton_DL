import argparse
import os
import torch
import pandas as pd
from torch import nn
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from source.load_data import GraphDataset
from source.models import ImprovedGINE, MLPClassifier
from source.utils import set_seed, add_node_features, train, evaluate, save_top_checkpoints


def extract_embeddings(model, data_loader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            emb = model.extract_embedding(data)
            embeddings.append(emb.cpu())
            if data.y is not None:
                labels.append(data.y.cpu())
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0) if labels else None


def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_set_name = os.path.basename(os.path.dirname(args.test_path))

    input_dim, hidden_dim, output_dim = 4, 64, 6
    model = ImprovedGINE(input_dim, hidden_dim, output_dim).to(device)

    # Load datasets
    train_dataset = GraphDataset(args.train_path, transform=add_node_features) if args.train_path else None
    test_dataset = GraphDataset(args.test_path, transform=add_node_features)
    batch_size = 32

    if train_dataset:
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    else:
        train_loader = val_loader = criterion = optimizer = None

    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Training + Save Top 5
    top_checkpoints = []
    checkpoints_dir = os.path.join("checkpoints", test_set_name)
    os.makedirs(checkpoints_dir, exist_ok=True)

    if train_loader:
        for epoch in range(args.epochs):
            train_loss, train_acc = train(train_loader, model, optimizer, criterion, device)
            val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(val_loader, model, device, criterion, True)

            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

            top_checkpoints = save_top_checkpoints(model, val_acc, epoch, checkpoints_dir, top_checkpoints)

    # Load best checkpoint
    if top_checkpoints:
        model.load_state_dict(torch.load(top_checkpoints[0][2]))
        print(f"\nLoaded best checkpoint: {top_checkpoints[0][2]}")

    # Extract embeddings
    train_embeddings, train_labels = extract_embeddings(model, DataLoader(train_dataset, batch_size=batch_size), device)
    test_embeddings, test_labels = extract_embeddings(model, test_loader, device)

    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(train_embeddings), dtype=torch.float32).to(device)
    X_test = torch.tensor(scaler.transform(test_embeddings), dtype=torch.float32).to(device)
    y_train = train_labels.to(device)

    # Train MLP classifier
    clf = MLPClassifier(X_train.shape[1], 128, output_dim).to(device)
    clf.train()
    clf_optim = torch.optim.Adam(clf.parameters(), lr=1e-3)
    clf_criterion = nn.CrossEntropyLoss()

    for _ in range(50):
        clf_optim.zero_grad()
        loss = clf_criterion(clf(X_train), y_train)
        loss.backward()
        clf_optim.step()

    clf.eval()
    with torch.no_grad():
        y_pred = clf(X_test).argmax(dim=1).cpu()

    if test_labels is not None:
        print("\nClassification Report:\n", classification_report(test_labels, y_pred))

    os.makedirs("submission", exist_ok=True)
    pd.DataFrame({"id": list(range(len(y_pred))), "pred": y_pred}).to_csv(f"submission/testset_{test_set_name}.csv", index=False)
    print(f"Predictions saved to submission/testset_{test_set_name}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./datasets/A/train.json.gz")
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    main(args)