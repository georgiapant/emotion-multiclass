import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np


def monitor_metrics(logits, labels_true):
    if labels_true is None:
        return {}

    preds = torch.argmax(logits, dim=1).flatten()
    # Calculate the accuracy rate
    accuracy = (preds == labels_true).cpu().numpy().mean() * 100
    # Calculate macro f1
    fscore_macro = f1_score(labels_true.cpu(), preds.cpu(), average='macro')
    return accuracy, fscore_macro


def evaluate(probs, y_true, use_preds=False):
    if use_preds:
        y_pred = probs
    else:
        y_pred = []

        for i in range(len(probs)):
            pred = np.where(probs[i] == np.amax(probs[i]))[0][0]
            y_pred.append(pred)

    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%', flush=True)
    print(f'F1-score macro: {f1_score(y_true, y_pred, average="macro")}', flush=True)
    print(f'Classification report: {classification_report(y_true, y_pred)}', flush=True)
