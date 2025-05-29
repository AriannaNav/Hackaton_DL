import argparse
import os
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from source.load_data import GraphDataset
from source.models import ImprovedGINE as ImprovedNNConv
from source.utils import set_seed, add_node_features, train, evaluate
from collections import Counter

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

def compute_class_weights(dataset, num_classes=6):
    labels = [data.y.item() for data in dataset if data.y is not None]
    label_counts = Counter(labels)
    total = sum(label_counts.values())
    freqs = torch.tensor([label_counts.get(i, 0) / total for i in range(num_classes)], dtype=torch.float32)
    weights = 1.0 / (freqs + 1e-8)
    weights = weights / weights.sum()
    return weights

def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_set_name = args.test_path.split("/")[-2]  # es: "A"

    # === Model setup ===
    input_dim = 4
    hidden_dim = 64
    output_dim = 6
    model = ImprovedNNConv(input_dim, hidden_dim, output_dim).to(device)
    
    # === Load test dataset ===
    test_dataset = GraphDataset(args.test_path, transform=add_node_features)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=2)

    # === Inference-only mode ===
    if not args.train_path:
        checkpoints_dir = os.path.join("checkpoints", test_set_name)
        best_model_path = sorted(os.listdir(checkpoints_dir))[-1]
        best_model_path = os.path.join(checkpoints_dir, best_model_path)
        model.load_state_dict(torch.load(best_model_path))
        print(f"\n[Inference] Loaded model from: {best_model_path}")
        
        test_embeddings, _ = extract_embeddings(model, test_loader, device)
        scaler = StandardScaler()
        test_embeddings = scaler.fit_transform(test_embeddings)  # dummy fit
        y_pred = [0] * len(test_embeddings)  # dummy output (you can change this)

        os.makedirs("submission", exist_ok=True)
        df = pd.DataFrame({"id": list(range(len(y_pred))), "pred": y_pred})
        df.to_csv(f"submission/testset_{test_set_name}.csv", index=False)
        print(f"Predictions saved to submission/testset_{test_set_name}.csv")
        return

    # === Full training mode ===
    train_dataset = GraphDataset(args.train_path, transform=add_node_features)
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=32, num_workers=2)

    # === Compute class weights from training set ===
    weights = compute_class_weights(train_set)
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))

    best_val_acc = 0.0
    top_checkpoints = []
    MAX_TOP = 5
    checkpoints_dir = os.path.join("checkpoints", test_set_name)
    os.makedirs(checkpoints_dir, exist_ok=True)

    logs_to_save = []

    for epoch in range(args.epochs):
        train_loss, train_acc = train(train_loader, model, optimizer, criterion, device)
        val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(val_loader, model, device, criterion, calculate_metrics=True)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")

        # Save logs every 10 epochs
        if (epoch + 1) % 10 == 0:
            logs_to_save.append([epoch+1, train_loss, val_loss, val_acc])

        # Save top 5 checkpoints
        if val_acc > best_val_acc or len(top_checkpoints) < MAX_TOP:
            model_name = f"model_{test_set_name}_epoch_{epoch+1}.pth"
            checkpoint_path = os.path.join(checkpoints_dir, model_name)
            torch.save(model.state_dict(), checkpoint_path)
            top_checkpoints.append((val_acc, epoch+1, checkpoint_path))
            top_checkpoints = sorted(top_checkpoints, key=lambda x: x[0], reverse=True)[:MAX_TOP]
            best_val_acc = top_checkpoints[0][0]
            print(f"Checkpoint salvato: {checkpoint_path}")

    print("\nTop 5 checkpoints:")
    for acc, ep, path in top_checkpoints:
        print(f" - Epoch {ep} | Acc: {acc:.4f} | File: {path}")

    # Save logs
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"logs_{test_set_name}.csv")
    logs_df = pd.DataFrame(logs_to_save, columns=["epoch", "train_loss", "val_loss", "val_acc"])
    logs_df.to_csv(log_path, index=False)
    print(f"\nLog salvato in: {log_path}")

    # Load best checkpoint
    best_model_path = top_checkpoints[0][2]
    model.load_state_dict(torch.load(best_model_path))
    print(f"\nBest model loaded from: {best_model_path}")

    # Embedding + Classifier
    print("Extracting embeddings...")
    train_embeddings, train_labels = extract_embeddings(model, DataLoader(train_dataset, batch_size=32, num_workers=2), device)
    test_embeddings, test_labels = extract_embeddings(model, test_loader, device)

    scaler = StandardScaler()
    train_embeddings = scaler.fit_transform(train_embeddings)
    test_embeddings = scaler.transform(test_embeddings)

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=40,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    clf.fit(train_embeddings, train_labels)
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
    