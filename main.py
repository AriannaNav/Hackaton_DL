import argparse
import os
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.linear_model import LogisticRegression
from source.load_data import GraphDataset
from source.models import ImprovedNNConv
from source.utils import set_seed, add_node_features, train, evaluate, save_predictions, plot_training_progress

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
    embeddings = torch.cat(embeddings, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy() if labels else None
    return embeddings, labels

def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Modello ===
    input_dim = 4         # 4 node features
    edge_dim = 7          # 7 edge features
    hidden_dim = 256
    output_dim = 6
    model = ImprovedNNConv(input_dim, edge_dim, hidden_dim, output_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # === Dataset ===
    train_dataset = GraphDataset(args.train_path, transform=add_node_features) if args.train_path else None
    test_dataset = GraphDataset(args.test_path, transform=add_node_features)

    batch_size = 64

    if train_dataset is not None:
        # Split in train/validation
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        train_loader = val_loader = None

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # === Training ===
    if train_loader is not None:
        checkpoint_folder = os.path.join(os.getcwd(), "checkpoints")
        os.makedirs(checkpoint_folder, exist_ok=True)

        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []

        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(args.epochs):
            train_loss, train_acc = train(train_loader, model, optimizer, criterion, device)
            val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(val_loader, model, device, criterion, calculate_metrics=True)

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            print(
                f"Epoch {epoch+1}/{args.epochs}\n"
                f" Train  - Loss: {train_loss:.4f} Acc: {train_acc:.4f}\n"
                f" Valid  - Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} Precision: {val_prec:.4f} Recall: {val_rec:.4f}\n"
            )

            if (epoch + 1) % 4 == 0 or (epoch + 1) == args.epochs:
                checkpoint_file = os.path.join(checkpoint_folder, f"model_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), checkpoint_file)
                print(f"Saved checkpoint at epoch {epoch+1}")

        # === Miglior modello ===
        best_epoch = val_accuracies.index(max(val_accuracies)) + 1
        best_model_path = os.path.join(checkpoint_folder, f"model_epoch_{best_epoch}.pth")
        model.load_state_dict(torch.load(best_model_path))

        plot_training_progress(train_losses, train_accuracies, val_losses, val_accuracies, output_dir=os.getcwd())

        # === Embedding + Logistic Regression ===
        train_embeddings, train_labels = extract_embeddings(model, train_loader, device)
        val_embeddings, val_labels = extract_embeddings(model, val_loader, device)

        clf = LogisticRegression(max_iter=2000, class_weight='balanced')
        clf.fit(train_embeddings, train_labels)

        val_preds = clf.predict(val_embeddings)
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        acc = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds, average='macro')
        precision = precision_score(val_labels, val_preds, average='macro')
        recall = recall_score(val_labels, val_preds, average='macro')
        print(f"Logistic Regression Validation - Acc: {acc:.4f} F1: {f1:.4f} Precision: {precision:.4f} Recall: {recall:.4f}")

        # === Predizioni finali ===
        test_embeddings, _ = extract_embeddings(model, test_loader, device)
        test_preds = clf.predict(test_embeddings)
    else:
        test_preds = evaluate(test_loader, model, device, criterion=None, calculate_metrics=False)[1]

    save_predictions(test_preds, args.test_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate NNConv GNN with logistic regression on graph datasets.")
    parser.add_argument("--train_path", type=str, default=None, help="Path to training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test dataset.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    args = parser.parse_args()
    main(args)