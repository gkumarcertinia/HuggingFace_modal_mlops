import torch
from sklearn.metrics import accuracy_score, f1_score

class MyDataset(torch.utils.data.Dataset):
    """Custom Dataset wrapper for HuggingFace model ingestion."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    """Computes accuracy and weighted F1 for the Trainer."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {
        'accuracy': acc,
        'f1': f1,
    }

def create_label_maps(labels):
    """Generates unique bidirectional mappings for targets."""
    unique_labels = sorted(list(set(labels)))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label
