import os
import random
import torch
import numpy as np
from torch.nn.functional import cross_entropy
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_node_features(data):
    # Presuppone che `data` abbia già `x` come feature dei nodi, oppure si può generare qui
    if not hasattr(data, 'x') or data.x is None:
        data.x = torch.ones((data.num_nodes, 4))  # Dummy features se mancanti
    return data


def train(loader, model, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs

    return total_loss / total, correct / total


def evaluate(loader, model, device, criterion=None, calculate_metrics=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            if criterion:
                loss = criterion(out, data.y)
                total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.num_graphs
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    acc = correct / total
    avg_loss = total_loss / total if criterion else None

    if calculate_metrics:
        f1 = f1_score(y_true, y_pred, average='macro')
        prec = precision_score(y_true, y_pred, average='macro')
        rec = recall_score(y_true, y_pred, average='macro')
        return avg_loss, acc, f1, prec, rec
    else:
        return avg_loss, acc


def save_top_checkpoints(model, val_acc, epoch, checkpoints_dir, top_checkpoints, max_top=5):
    ckpt_path = os.path.join(checkpoints_dir, f"epoch{epoch+1}_acc{val_acc:.4f}.pt")
    torch.save(model.state_dict(), ckpt_path)

    top_checkpoints.append((val_acc, epoch, ckpt_path))
    top_checkpoints = sorted(top_checkpoints, key=lambda x: x[0], reverse=True)[:max_top]

    # Delete checkpoints not in top
    saved_paths = [ckpt[2] for ckpt in top_checkpoints]
    for filename in os.listdir(checkpoints_dir):
        full_path = os.path.join(checkpoints_dir, filename)
        if full_path not in saved_paths:
            os.remove(full_path)

    return top_checkpoints
