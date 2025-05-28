import argparse
import os
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report
from source.load_data import GraphDataset
from source.models import ImprovedGAT as ImprovedNNConv
from source.utils import (
    set_seed,
    add_node_features,
    train,
    evaluate,
    FocalLoss,
    compute_class_weights,
    make_balanced_sampler,
    save_predictions,
    plot_training_progress
)
from torch.optim.lr_scheduler import StepLR


def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_set_name = args.test_path.split("/")[-2]

    input_dim = 20  
    hidden_dim = 64
    output_dim = 6
    model = ImprovedNNConv(input_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    train_dataset = GraphDataset(args.train_path, transform=add_node_features) if args.train_path else None
    test_dataset = GraphDataset(args.test_path, transform=add_node_features)
    batch_size = 32

    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    if train_dataset:
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        sampler = make_balanced_sampler(train_set)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=2)

        weights = compute_class_weights(train_set)
        criterion = FocalLoss(alpha=1, gamma=2)
    else:
        train_loader = val_loader = None
        criterion = None

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)

    best_val_acc = 0.0
    best_model_path = ""

    if train_loader:
        for epoch in range(args.epochs):
            train_loss, train_acc = train(train_loader, model, optimizer, criterion, device)
            scheduler.step()

            train_losses.append(train_loss)
            train_accs.append(train_acc)

            if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
                val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(val_loader, model, device, criterion, calculate_metrics=True)
                print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")

                val_losses.append(val_loss)
                val_accs.append(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    checkpoints_dir = os.path.join("checkpoints", test_set_name)
                    os.makedirs(checkpoints_dir, exist_ok=True)
                    best_model_path = os.path.join(checkpoints_dir, f"best_model.pt")
                    torch.save(model.state_dict(), best_model_path)
                    print(f"● Best model updated! Saved to {best_model_path} (Acc: {best_val_acc:.4f})")
            else:
                val_loss, _ = evaluate(val_loader, model, device, criterion, calculate_metrics=False)
                print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Plot training progress
        plot_training_progress(train_losses, train_accs, val_losses, val_accs, output_dir="results")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"\n🔄 Loaded best model from {best_model_path}")

    print("Predicting on test set...")
    model.eval()
    all_preds = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            logits = model(data)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())

    save_predictions(all_preds, args.test_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./datasets/A/train.json.gz")
    parser.add_argument("--test_path", type=str, default="./datasets/A/test.json.gz")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    main(args)
