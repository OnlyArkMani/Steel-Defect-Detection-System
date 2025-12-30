import torch
from torch.utils.data import Dataset, DataLoader
import timm
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    accuracy_score
)
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(r"C:\Projects\CV_SDT")
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
CONFIG = {
    'model_name': 'convnextv2_tiny.fcmae_ft_in22k_in1k',
    'num_classes': 6,
    'image_size': 224,
    'batch_size': 32,
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

DEFECT_CLASSES = [
    'Crazing',
    'Inclusion',
    'Patches',
    'Pitted Surface',
    'Rolled-in Scale',
    'Scratches'
]


class SteelDefectDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        label = self.data.iloc[idx]['label']

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label


def get_transforms():
    return A.Compose([
        A.Resize(CONFIG['image_size'], CONFIG['image_size']),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def load_model(checkpoint_path):
    model = timm.create_model(
        CONFIG['model_name'],
        pretrained=False,
        num_classes=CONFIG['num_classes']
    )

    checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(CONFIG['device'])
    model.eval()

    return model


def evaluate_model(model, loader):
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(CONFIG['device'])

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )


def plot_confusion(y_true, y_pred, save_path, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2%' if normalize else 'd',
        cmap='Blues',
        xticklabels=DEFECT_CLASSES,
        yticklabels=DEFECT_CLASSES
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_per_class_metrics(report_dict, save_path):
    metrics = ['precision', 'recall', 'f1-score']
    data = {m: [] for m in metrics}

    for cls in DEFECT_CLASSES:
        for m in metrics:
            data[m].append(report_dict[cls][m])

    x = np.arange(len(DEFECT_CLASSES))
    width = 0.25

    plt.figure(figsize=(14, 8))
    plt.bar(x - width, data['precision'], width, label='Precision')
    plt.bar(x, data['recall'], width, label='Recall')
    plt.bar(x + width, data['f1-score'], width, label='F1')

    plt.xticks(x, DEFECT_CLASSES, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def calculate_roc_auc(y_true, y_proba, save_path):
    y_onehot = np.eye(CONFIG['num_classes'])[y_true]
    scores = {}

    for i, cls in enumerate(DEFECT_CLASSES):
        scores[cls] = roc_auc_score(y_onehot[:, i], y_proba[:, i])

    with open(save_path, 'w') as f:
        for cls, score in scores.items():
            f.write(f"{cls}: {score:.4f}\n")
        f.write(f"\nMacro Average: {np.mean(list(scores.values())):.4f}\n")

    return scores


def save_summary(accuracy, report_dict, roc_auc_scores, save_path):
    summary = {
        'accuracy': float(accuracy),
        'roc_auc': roc_auc_scores,
        'per_class': {
            cls: {
                'precision': report_dict[cls]['precision'],
                'recall': report_dict[cls]['recall'],
                'f1': report_dict[cls]['f1-score']
            } for cls in DEFECT_CLASSES
        }
    }

    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=4)


def main():
    model_files = list(MODELS_DIR.glob('best_model_*.pth'))
    if not model_files:
        return

    model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    model = load_model(model_path)

    test_dataset = SteelDefectDataset(
        PROCESSED_DATA_DIR / 'test_split.csv',
        transform=get_transforms()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers']
    )

    y_true, y_pred, y_proba = evaluate_model(model, test_loader)
    accuracy = accuracy_score(y_true, y_pred)

    plot_confusion(
        y_true, y_pred,
        RESULTS_DIR / 'confusion_matrix.png',
        normalize=False
    )

    plot_confusion(
        y_true, y_pred,
        RESULTS_DIR / 'confusion_matrix_normalized.png',
        normalize=True
    )

    report = classification_report(
        y_true, y_pred,
        target_names=DEFECT_CLASSES,
        output_dict=True
    )

    with open(RESULTS_DIR / 'classification_report.txt', 'w') as f:
        f.write(classification_report(
            y_true, y_pred,
            target_names=DEFECT_CLASSES
        ))

    plot_per_class_metrics(
        report,
        RESULTS_DIR / 'per_class_metrics.png'
    )

    roc_auc_scores = calculate_roc_auc(
        y_true, y_proba,
        RESULTS_DIR / 'roc_auc_scores.txt'
    )

    save_summary(
        accuracy,
        report,
        roc_auc_scores,
        RESULTS_DIR / 'evaluation_summary.json'
    )


if __name__ == "__main__":
    main()
