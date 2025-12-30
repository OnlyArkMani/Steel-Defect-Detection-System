"""
Steel Defect Detection - Step 2: Data Preprocessing & Augmentation (FIXED VERSION)

This script handles:
1. Data organization and splitting (train/val/test)
2. Image preprocessing (resize, normalize)
3. Data augmentation using Albumentations
4. Creating PyTorch datasets
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil
import random
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# Configuration
PROJECT_ROOT = Path(r"C:\Projects\CV_SDT")
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Image settings
IMAGE_SIZE = 224
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Dataset split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Defect classes
DEFECT_CLASSES = [
    'crazing',
    'inclusion',
    'patches',
    'pitted_surface',
    'rolled-in_scale',
    'scratches'
]

def find_and_organize_images():
    """Find all images and organize them by class."""
    print("=" * 60)
    print("STEP 1: FINDING AND ORGANIZING IMAGES")
    print("=" * 60)
    
    class_images = {cls: [] for cls in DEFECT_CLASSES}
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for img_path in RAW_DATA_DIR.rglob('*'):
        if img_path.suffix.lower() in image_extensions:
            parent_folder = img_path.parent.name.lower()
            filename = img_path.stem.lower()
            
            for cls in DEFECT_CLASSES:
                if cls in parent_folder or cls in filename:
                    class_images[cls].append(img_path)
                    break
    
    print("\nDataset Statistics:")
    print("-" * 60)
    total_images = 0
    for cls in DEFECT_CLASSES:
        count = len(class_images[cls])
        total_images += count
        print(f"  {cls.replace('_', ' ').title():<25} : {count:>4} images")
    print("-" * 60)
    print(f"  Total{'':<25} : {total_images:>4} images\n")
    
    return class_images

def split_dataset(class_images):
    """Split dataset into train, validation, and test sets."""
    print("=" * 60)
    print("STEP 2: SPLITTING DATASET")
    print("=" * 60)
    
    train_data = []
    val_data = []
    test_data = []
    
    for cls_idx, (cls, images) in enumerate(class_images.items()):
        train_val_imgs, test_imgs = train_test_split(
            images, 
            test_size=TEST_RATIO, 
            random_state=SEED
        )
        
        train_imgs, val_imgs = train_test_split(
            train_val_imgs,
            test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
            random_state=SEED
        )
        
        train_data.extend([(img, cls_idx) for img in train_imgs])
        val_data.extend([(img, cls_idx) for img in val_imgs])
        test_data.extend([(img, cls_idx) for img in test_imgs])
        
        print(f"\n{cls.replace('_', ' ').title()}:")
        print(f"  Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")
    
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    print("\n" + "-" * 60)
    print(f"Total - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print("-" * 60 + "\n")
    
    return train_data, val_data, test_data

def get_augmentation_pipeline(is_training=True):
    """Create augmentation pipeline using Albumentations (FIXED - No warnings)."""
    if is_training:
        return A.Compose([
            # Geometric transformations (FIXED: Using Affine instead of ShiftScaleRotate)
            A.Rotate(limit=30, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                scale=(0.8, 1.2),
                rotate=(-15, 15),
                p=0.5
            ),
            
            # Image quality variations (FIXED: Removed var_limit from GaussNoise)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            
            # Normalize to ImageNet stats
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

def save_processed_data(train_data, val_data, test_data):
    """Save processed data information to CSV."""
    print("=" * 60)
    print("STEP 3: SAVING DATASET INFORMATION")
    print("=" * 60)
    
    def save_split(data, split_name):
        df = pd.DataFrame(data, columns=['image_path', 'label'])
        df['class_name'] = df['label'].map(lambda x: DEFECT_CLASSES[x])
        
        output_path = PROCESSED_DATA_DIR / f"{split_name}_split.csv"
        df.to_csv(output_path, index=False)
        print(f"✓ Saved {split_name} split: {output_path}")
    
    save_split(train_data, 'train')
    save_split(val_data, 'val')
    save_split(test_data, 'test')
    
    class_mapping = pd.DataFrame({
        'class_idx': range(len(DEFECT_CLASSES)),
        'class_name': DEFECT_CLASSES
    })
    class_mapping.to_csv(PROCESSED_DATA_DIR / 'class_mapping.csv', index=False)
    print(f"✓ Saved class mapping\n")

def visualize_augmentations(train_data):
    """Visualize augmentation effects on sample images."""
    print("=" * 60)
    print("STEP 4: VISUALIZING AUGMENTATIONS")
    print("=" * 60)
    
    sample_images = []
    for cls_idx in range(len(DEFECT_CLASSES)):
        for img_path, label in train_data:
            if label == cls_idx:
                sample_images.append((img_path, label))
                break
    
    aug_pipeline = get_augmentation_pipeline(is_training=True)
    
    fig, axes = plt.subplots(len(DEFECT_CLASSES), 4, figsize=(16, len(DEFECT_CLASSES) * 3))
    fig.suptitle('Augmentation Examples (Original + 3 Augmented Versions)', fontsize=16)
    
    for idx, (img_path, label) in enumerate(sample_images):
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f'{DEFECT_CLASSES[label]} (Original)')
        axes[idx, 0].axis('off')
        
        for aug_idx in range(1, 4):
            augmented = aug_pipeline(image=image)['image']
            aug_img = augmented.permute(1, 2, 0).numpy()
            aug_img = aug_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            aug_img = np.clip(aug_img, 0, 1)
            
            axes[idx, aug_idx].imshow(aug_img)
            axes[idx, aug_idx].set_title(f'Augmented {aug_idx}')
            axes[idx, aug_idx].axis('off')
    
    plt.tight_layout()
    output_path = PROJECT_ROOT / "results" / "augmentation_examples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved augmentation visualization: {output_path}\n")
    plt.close()

def main():
    """Main execution function."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + "  DATA PREPROCESSING & AUGMENTATION  ".center(60) + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    class_images = find_and_organize_images()
    train_data, val_data, test_data = split_dataset(class_images)
    save_processed_data(train_data, val_data, test_data)
    visualize_augmentations(train_data)
    
    print("=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print("\nWhat we did:")
    print("  ✓ Organized images by defect class")
    print("  ✓ Split into train/val/test sets (70/15/15)")
    print("  ✓ Created augmentation pipeline")
    print("  ✓ Saved dataset splits to CSV")
    print("  ✓ Generated augmentation examples")
    print("\nNext Step: Model Training with ConvNeXt V2")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()