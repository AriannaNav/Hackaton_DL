import argparse
import os
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from source.load_data import GraphDataset
from source.models import ImprovedGINE
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
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0) if len(labels) > 0 else None


def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_set_name = args.test_path.split("/")[-2]

    input_dim = 4
    hidden_dim = 64
    output_dim = 6
    model = ImprovedGINE(input_dim, hidden_dim, output_dim).to(device)

    train_dataset = GraphDataset(args.train_path, transform=add_node_features) if args.train_path else None
    test_dataset = GraphDataset(args.test_path, transform=add_node_features)
    batch_size = 32

    if train_dataset:
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=2)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    else:
        train_loader = val_loader = criterion = optimizer = scheduler = None

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)

    top_checkpoints = []
    MAX_TOP = 5
    checkpoints_dir = os.path.join("checkpoints", test_set_name)
    os.makedirs(checkpoints_dir, exist_ok=True)

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{test_set_name}.log")
    log_file = open(log_path, "w")

    if train_loader:
        for epoch in range(args.epochs):
            train_loss, train_acc = train(train_loader, model, optimizer, criterion, device)
            val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(val_loader, model, device, criterion, calculate_metrics=True)

            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")

            if (epoch + 1) % 10 == 0:
                log_file.write(f"{epoch+1},{train_loss:.4f},{val_loss:.4f},{val_acc:.4f},{val_f1:.4f},{val_prec:.4f},{val_rec:.4f}\n")
                log_file.flush()

            top_checkpoints = save_top_checkpoints(model, val_f1, epoch, checkpoints_dir, top_checkpoints, MAX_TOP, dataset_name=test_set_name)

        log_file.close()

    best_model_path = top_checkpoints[0][2] if top_checkpoints else None
    if best_model_path and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"\nBest model loaded from: {best_model_path}")
    else:
        print("Warning: No best model found, using last model state.")

    print("Extracting embeddings...")
    train_embeddings, train_labels = extract_embeddings(model, DataLoader(train_dataset, batch_size=batch_size, num_workers=2), device)
    test_embeddings, test_labels = extract_embeddings(model, test_loader, device)

    scaler = StandardScaler()
    train_embeddings = scaler.fit_transform(train_embeddings)
    test_embeddings = scaler.transform(test_embeddings)

    clf = LogisticRegression(max_iter=1000, penalty='l2', C=0.01, solver='lbfgs', class_weight='balanced')
    clf.fit(train_embeddings, train_labels.numpy())
    y_pred = clf.predict(test_embeddings)

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
    