import os
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Projects\CV_SDT")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
AUGMENTED_DATA_DIR = DATA_DIR / "augmented"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"


def create_directory_structure():
    directories = [
        PROJECT_ROOT,
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        AUGMENTED_DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        LOGS_DIR
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def download_dataset():
    try:
        import kaggle

        dataset_name = "kaustubhdikshit/neu-surface-defect-database"
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(RAW_DATA_DIR),
            unzip=True
        )
        return True

    except ImportError:
        print("Kaggle API not installed. Run: pip install kaggle")
        return False

    except Exception as e:
        print(f"Dataset download failed: {e}")
        return False


def verify_dataset():
    expected_classes = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches"
    ]

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = []

    for ext in image_extensions:
        all_images.extend(RAW_DATA_DIR.rglob(f"*{ext}"))

    if not all_images:
        print("No images found.")
        return False

    class_counts = {}

    for img in all_images:
        parent = img.parent.name.lower()
        name = img.stem.lower()

        for cls in expected_classes:
            if cls in parent or cls in name:
                class_counts[cls] = class_counts.get(cls, 0) + 1
                break

    for cls in expected_classes:
        print(f"{cls}: {class_counts.get(cls, 0)}")

    print(f"Total images: {sum(class_counts.values())}")
    return True


def main():
    create_directory_structure()

    proceed = 'y'
    if proceed == 'y':
        if download_dataset():
            verify_dataset()


if __name__ == "__main__":
    main()
