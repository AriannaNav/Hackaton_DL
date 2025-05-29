import argparse
import os
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # === Dataset loading ===
    train_dataset = GraphDataset(args.train_path, transform=add_node_features) if args.train_path else None
    test_dataset = GraphDataset(args.test_path, transform=add_node_features)
    batch_size = 32

    if train_dataset:
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=2)

        # === Compute class weights from training set ===
        weights = compute_class_weights(train_set)
        criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        train_loader = val_loader = None
        criterion = None

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)

    # === Training ===
    best_val_acc = 0.0
    if train_loader:
        for epoch in range(args.epochs):
            train_loss, train_acc = train(train_loader, model, optimizer, criterion, device)

            if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
                val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(val_loader, model, device, criterion, calculate_metrics=True)
                print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")
            else:
                val_loss, val_acc, _, _, _ = evaluate(val_loader, model, device, criterion, calculate_metrics=True)
                print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # === Save Best Checkpoint ===
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoints_dir = os.path.join("checkpoints", test_set_name)
                os.makedirs(checkpoints_dir, exist_ok=True)
                best_model_path = os.path.join(checkpoints_dir, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"🟢 Best model updated! Saved to {best_model_path}")

    # === Load Best Checkpoint (if exists) ===
    best_model_path = os.path.join("checkpoints", test_set_name, "best_model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"\n🔁 Best model loaded from {best_model_path}")
    else:
        print(f"⚠️ Warning: Best model not found, using last model state.")

    # === Embedding + Classifier ===
    print("Extracting embeddings...")
    train_embeddings, train_labels = extract_embeddings(model, DataLoader(train_dataset, batch_size=batch_size, num_workers=2), device)
    test_embeddings, test_labels = extract_embeddings(model, test_loader, device)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(train_embeddings, train_labels)
    y_pred = clf.predict(test_embeddings)

    if test_labels is not None:
        report = classification_report(test_labels, y_pred)
        print("\nClassification Report:\n", report)

    # === Save Submission ===
    os.makedirs("submission", exist_ok=True)
    df = pd.DataFrame({"id": list(range(len(y_pred))), "pred": y_pred})
    df.to_csv(f"submission/testset_{test_set_name}.csv", index=False)
    print(f"Predictions saved to submission/testset_{test_set_name}.csv")