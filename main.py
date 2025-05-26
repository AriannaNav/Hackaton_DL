# main.py
import argparse
import os
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
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
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0) if len(labels) > 0 else None

def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = 4
    edge_dim = 7
    hidden_dim = 64
    output_dim = 6
    model = ImprovedNNConv(input_dim, edge_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    train_dataset = GraphDataset(args.train_path, transform=add_node_features) if args.train_path else None
    test_dataset = GraphDataset(args.test_path, transform=add_node_features)

    batch_size = 4

    if train_dataset:
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)
    else:
        train_loader = val_loader = None

    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    if train_loader:
            criterion = torch.nn.CrossEntropyLoss()
            for epoch in range(args.epochs):
                train_loss, train_acc = train(train_loader, model, optimizer, criterion, device)
                val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(val_loader, model, device, criterion, calculate_metrics=True)
                print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    
                save_every = 5  
                if (epoch + 1) == 1 or (epoch + 1) % save_every == 0:
                    checkpoints_dir = os.path.join("checkpoints", test_set_name)
                    os.makedirs(checkpoints_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoints_dir, f"model_epoch_{epoch+1:02d}.pt")
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f" Checkpoint salvato: {checkpoint_path}")

    print("Extracting embeddings...")
    train_embeddings, train_labels = extract_embeddings(model, DataLoader(train_dataset, batch_size=batch_size), device)
    test_embeddings, test_labels = extract_embeddings(model, test_loader, device)

    clf = LogisticRegression(C=0.01, penalty='l2', solver='lbfgs', max_iter=1000)
    clf.fit(train_embeddings, train_labels)
    y_pred = clf.predict(test_embeddings)

    if test_labels is not None:
        report = classification_report(test_labels, y_pred)
        print("\nClassification Report:\n", report)

    test_set_name = args.test_path.split("/")[-2]  # es: "A"
    os.makedirs("submission", exist_ok=True)
    df = pd.DataFrame({"id": list(range(len(y_pred))), "pred": y_pred})
    df.to_csv(f"submission/testset_{test_set_name}.csv", index=False)
    print(f"Predictions saved to submission/testset_{test_set_name}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./datasets/A/train.json.gz")
    parser.add_argument("--test_path", type=str, default="./datasets/A/test.json.gz")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args)