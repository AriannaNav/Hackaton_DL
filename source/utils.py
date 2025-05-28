import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
from collections import Counter
from torch.utils.data import WeightedRandomSampler

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def add_node_features(data):
    row, col = data.edge_index
    deg = torch.bincount(row, minlength=data.num_nodes).float().view(-1, 1)
    deg = deg / (deg.max() + 1e-5)

    in_deg = torch.bincount(col, minlength=data.num_nodes).float().view(-1, 1)
    in_deg = in_deg / (in_deg.max() + 1e-5)

    out_deg = torch.bincount(row, minlength=data.num_nodes).float().view(-1, 1)
    out_deg = out_deg / (out_deg.max() + 1e-5)

    if hasattr(data, 'x') and data.x is not None:
        data.x = torch.cat([data.x, deg, in_deg, out_deg], dim=1)
    else:
        data.x = torch.cat([deg, in_deg, out_deg], dim=1)

    return data

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def compute_class_weights(dataset, num_classes=6):
    labels = [data.y.item() for data in dataset if data.y is not None]
    label_counts = Counter(labels)
    total = sum(label_counts.values())
    freqs = torch.tensor([label_counts.get(i, 0) / total for i in range(num_classes)], dtype=torch.float32)
    weights = 1.0 / (freqs + 1e-8)
    weights = weights / weights.sum()
    return weights

def make_balanced_sampler(dataset, num_classes=6):
    labels = [data.y.item() for data in dataset if data.y is not None]
    label_counts = Counter(labels)
    weights_per_class = {cls: 1.0 / count for cls, count in label_counts.items()}
    sample_weights = [weights_per_class[data.y.item()] for data in dataset]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

def train(data_loader, model, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in tqdm(data_loader, desc="Training batches"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        pred = output.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs

    avg_loss = total_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc

def evaluate(data_loader, model, device, criterion=None, calculate_metrics=False):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            if calculate_metrics:
                all_labels.extend(data.y.cpu().tolist())
            if criterion is not None:
                loss = criterion(output, data.y)
                total_loss += loss.item() * data.num_graphs

    avg_loss = total_loss / len(data_loader.dataset) if criterion is not None else None

    if calculate_metrics and all_labels:
        f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        accuracy = accuracy_score(all_labels, all_preds)
        return avg_loss, accuracy, f1, precision, recall

    return avg_loss, all_preds

def save_predictions(preds, test_path):
    test_dir_name = os.path.basename(os.path.dirname(test_path))
    output_dir = "submission"
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, f"testset_{test_dir_name}.csv")
    df = pd.DataFrame({"id": list(range(len(preds))), "pred": preds})
    df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

def plot_training_progress(train_losses, train_acc, val_losses, val_acc, output_dir):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over epochs")

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over epochs")

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/training_progress.png")
    plt.close()
