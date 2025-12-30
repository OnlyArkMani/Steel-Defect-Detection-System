"""
Steel Defect Detection - Step 3: Model Training (FIXED VERSION)

Using ConvNeXt V2 (2024's state-of-the-art CNN architecture)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import timm
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime
import json

# Configuration
PROJECT_ROOT = Path(r"C:\Projects\CV_SDT")
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Training hyperparameters
CONFIG = {
    'model_name': 'convnextv2_tiny.fcmae_ft_in22k_in1k',
    'num_classes': 6,
    'image_size': 224,
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'mixed_precision': True,
    'early_stopping_patience': 10,
    'save_best_only': True
}

DEFECT_CLASSES = [
    'crazing', 'inclusion', 'patches',
    'pitted_surface', 'rolled-in_scale', 'scratches'
]

class SteelDefectDataset(Dataset):
    """Custom PyTorch Dataset for steel defect images."""
    
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

def get_transforms(is_training=True):
    """Get augmentation pipeline (FIXED - No warnings)."""
    if is_training:
        return A.Compose([
            A.Rotate(limit=30, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                scale=(0.8, 1.2),
                rotate=(-15, 15),
                p=0.5
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.Resize(CONFIG['image_size'], CONFIG['image_size']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(CONFIG['image_size'], CONFIG['image_size']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def create_model():
    """Create ConvNeXt V2 model with pre-trained weights."""
    print("\n" + "=" * 60)
    print("CREATING MODEL")
    print("=" * 60)
    
    model = timm.create_model(
        CONFIG['model_name'],
        pretrained=True,
        num_classes=CONFIG['num_classes']
    )
    
    print(f"\nModel: {CONFIG['model_name']}")
    print(f"Pre-trained: Yes (ImageNet-22K + ImageNet-1K)")
    print(f"Number of classes: {CONFIG['num_classes']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Device: {CONFIG['device']}")
    
    model = model.to(CONFIG['device'])
    
    return model

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{CONFIG["num_epochs"]} [Train]')
    
    for images, labels in pbar:
        images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
        
        optimizer.zero_grad()
        
        if CONFIG['mixed_precision'] and CONFIG['device'] == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/(pbar.n+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, epoch):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{CONFIG["num_epochs"]} [Val]  ')
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def train_model():
    """Main training function."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + "  MODEL TRAINING - ConvNeXt V2  ".center(60) + "║")
    print("╚" + "=" * 58 + "╝")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = LOGS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(CONFIG, f, indent=4)
    
    writer = SummaryWriter(log_dir=str(run_dir))
    
    print("\n" + "=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)
    
    train_dataset = SteelDefectDataset(
        PROCESSED_DATA_DIR / 'train_split.csv',
        transform=get_transforms(is_training=True)
    )
    val_dataset = SteelDefectDataset(
        PROCESSED_DATA_DIR / 'val_split.csv',
        transform=get_transforms(is_training=False)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    print(f"✓ Batch size: {CONFIG['batch_size']}")
    print(f"✓ Training batches per epoch: {len(train_loader)}")
    
    model = create_model()
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['num_epochs']
    )
    
    scaler = torch.cuda.amp.GradScaler() if CONFIG['mixed_precision'] and CONFIG['device'] == 'cuda' else None
    
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    print(f"\nOptimizer: AdamW (lr={CONFIG['learning_rate']}, weight_decay={CONFIG['weight_decay']})")
    print(f"Scheduler: CosineAnnealingLR")
    print(f"Loss Function: CrossEntropyLoss")
    print(f"Mixed Precision: {CONFIG['mixed_precision'] and CONFIG['device'] == 'cuda'}\n")
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(CONFIG['num_epochs']):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch
        )
        
        val_loss, val_acc = validate(model, val_loader, criterion, epoch)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        writer.add_scalar('Learning Rate', current_lr, epoch)
        
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}\n")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': CONFIG
            }
            
            torch.save(checkpoint, MODELS_DIR / f'best_model_{timestamp}.pth')
            print(f"✓ New best model saved! Val Acc: {val_acc:.2f}%\n")
        else:
            patience_counter += 1
        
        if patience_counter >= CONFIG['early_stopping_patience']:
            print(f"\nEarly stopping triggered! No improvement for {CONFIG['early_stopping_patience']} epochs.")
            break
    
    writer.close()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved: {MODELS_DIR / f'best_model_{timestamp}.pth'}")
    print(f"TensorBoard logs: {run_dir}")
    print("\nTo view training curves, run:")
    print(f"  tensorboard --logdir={LOGS_DIR}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    train_model()