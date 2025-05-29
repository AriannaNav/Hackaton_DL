import argparse
import os
import torch
import pandas as pd
from torch import nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from source.load_data import GraphDataset
from source.models import GCN, GraphSAGE, ImprovedGINE, MLPClassifier
from source.utils import set_seed, add_node_features, train, evaluate, save_top_checkpoints


def extract_embeddings(model, data_loader, device):
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            emb = model.extract_embedding(data)
            embeddings.append(emb.cpu())
            if data.y is not None:
                labels.append(data.y.cpu())
    return torch.cat(embeddings), torch.cat(labels) if labels else None


def select_model(dataset_name, input_dim, hidden_dim, output_dim):
    if dataset_name == "A":
        return GCN(input_dim, hidden_dim, output_dim)
    elif dataset_name == "B":
        return GraphSAGE(input_dim, hidden_dim, output_dim)
    elif dataset_name == "C":
        return ImprovedGINE(input_dim, hidden_dim, output_dim)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_set_name = args.test_path.split("/")[-2]  # A, B, C

    input_dim, hidden_dim, output_dim = 4, 64, 6
    model = select_model(test_set_name, input_dim, hidden_dim, output_dim).to(device)

    # === Load Datasets ===
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
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    else:
        train_loader = val_loader = criterion = optimizer = None

    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # === Logging and Checkpoints ===
    checkpoints_dir = os.path.join("checkpoints", test_set_name)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_file = open(os.path.join("logs", f"{test_set_name}.log"), "w")
    top_checkpoints = []
    MAX_TOP = 5

    if train_loader:
        for epoch in range(args.epochs):
            train_loss, train_acc = train(train_loader, model, optimizer, criterion, device)
            val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(val_loader, model, device, criterion, calculate_metrics=True)

            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")

            if (epoch + 1) % 10 == 0:
                log_file.write(f"{epoch+1},{train_loss:.4f},{val_loss:.4f},{val_acc:.4f}\n")
                log_file.flush()

            top_checkpoints = save_top_checkpoints(model, val_acc, epoch, checkpoints_dir, top_checkpoints, MAX_TOP)
        log_file.close()

    # === Load Best Model ===
    if top_checkpoints:
        best_model_path = top_checkpoints[0][2]
        model.load_state_dict(torch.load(best_model_path))
        print(f"\nBest model loaded from: {best_model_path}")
    else:
        print("Warning: No best model found, using last model state.")

    # === Extract Embeddings & Train Classifier ===
    print("Extracting embeddings...")
    train_embeddings, train_labels = extract_embeddings(model, DataLoader(train_dataset, batch_size=batch_size), device)
    test_embeddings, test_labels = extract_embeddings(model, test_loader, device)

    scaler = StandardScaler()
    train_embeddings = scaler.fit_transform(train_embeddings)
    test_embeddings = scaler.transform(test_embeddings)

    clf = MLPClassifier(input_dim=train_embeddings.shape[1], hidden_dim=128, output_dim=6).to(device)
    criterion_clf = nn.CrossEntropyLoss()
    optimizer_clf = torch.optim.Adam(clf.parameters(), lr=0.001)

    X_train = torch.tensor(train_embeddings, dtype=torch.float32).to(device)
    y_train = train_labels.to(device)
    X_test = torch.tensor(test_embeddings, dtype=torch.float32).to(device)

    for _ in range(50):
        optimizer_clf.zero_grad()
        output = clf(X_train)
        loss = criterion_clf(output, y_train)
        loss.backward()
        optimizer_clf.step()

    clf.eval()
    with torch.no_grad():
        y_pred = clf(X_test).argmax(dim=1).cpu()

    if test_labels is not None:
        report = classification_report(test_labels, y_pred)
        print("\nClassification Report:\n", report)

    os.makedirs("submission", exist_ok=True)
    df = pd.DataFrame({"id": list(range(len(y_pred))), "pred": y_pred})
    df.to_csv(f"submission/testset_{test_set_name}.csv", index=False)
    print(f"Predictions saved to submission/testset_{test_set_name}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./datasets/A/train.json.gz")
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    main(args)
    