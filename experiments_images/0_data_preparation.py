"""Prepare the PlantVillage dataset.

Downloads the PlantVillage dataset, extracts selected tomato classes, splits
into train/val/test, and prints dataset statistics.
"""

import sys
import os
sys.path.append(os.getcwd())

import shutil
import zipfile
from pathlib import Path
import urllib.request
from tqdm import tqdm
import random

from utils import DATA_DIR, RAW_DIR, PROCESSED_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE

# CLASSES WE WANT 3 TOMATO DISEASES
TARGET_CLASSES: list[str] = [
    "Tomato_healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Late_blight",
]

# SPLIT RATIOS
TRAIN_RATIO: float = 0.7
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def format_table(headers: list[str], rows: list[list[object]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))

    header_line = "| " + " | ".join(
        f"{headers[i]:<{widths[i]}}" for i in range(len(headers))
    ) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"

    lines = [header_line, sep_line]
    for row in rows:
        lines.append("| " + " | ".join(
            f"{str(row[i]):<{widths[i]}}" for i in range(len(headers))
        ) + " |")
    return "\n".join(lines)

def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        1
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTS
    )

def collect_split_counts() -> dict[str, dict[str, int]]:
    split_dirs = {
        "train": TRAIN_DIR,
        "val": VAL_DIR,
        "test": TEST_DIR,
    }
    counts: dict[str, dict[str, int]] = {}
    for split, split_dir in split_dirs.items():
        split_counts: dict[str, int] = {}
        for class_name in TARGET_CLASSES:
            split_counts[class_name] = count_images(split_dir / class_name)
        counts[split] = split_counts
    return counts

def print_dataset_stats() -> None:
    if not PROCESSED_DIR.exists():
        print(f"No prepared dataset found at {PROCESSED_DIR}")
        return

    split_counts = collect_split_counts()
    total_by_split = {
        split: sum(class_counts.values())
        for split, class_counts in split_counts.items()
    }
    total = sum(total_by_split.values())
    features = f"{IMG_SIZE[0]}x{IMG_SIZE[1]}x3"

    dataset_rows = [[
        "PlantVillage",
        "classification",
        total,
        features,
        len(TARGET_CLASSES),
    ]]
    split_rows = [
        [
            "PlantVillage",
            split,
            "classification",
            total_by_split.get(split, 0),
            features,
            len(TARGET_CLASSES),
        ]
        for split in ["train", "val", "test"]
    ]
    class_rows: list[list[object]] = []
    for split in ["train", "val", "test"]:
        for class_name in TARGET_CLASSES:
            class_rows.append([
                "PlantVillage",
                split,
                class_name,
                split_counts.get(split, {}).get(class_name, 0),
            ])

    print("=== Dataset Stats ===\n")
    print(format_table(
        ["Dataset", "Task", "Size", "Features", "Classes"],
        dataset_rows,
    ))
    print()
    print(format_table(
        ["Dataset", "Split", "Task", "Size", "Features", "Classes"],
        split_rows,
    ))
    print()
    print(format_table(
        ["Dataset", "Split", "Class", "Size"],
        class_rows,
    ))
    print()

def dataset_ready() -> bool:
    if not PROCESSED_DIR.exists():
        return False
    if not TRAIN_DIR.exists() or not VAL_DIR.exists() or not TEST_DIR.exists():
        return False
    for class_name in TARGET_CLASSES:
        for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
            if count_images(split_dir / class_name) > 0:
                return True
    return False

def download_dataset() -> Path:
    """Download the PlantVillage dataset from Kaggle.

    Returns:
        Path to the downloaded zip file.
    """
    print("Downloading PlantVillage dataset...")
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "plantdisease.zip"
    
    url = "https://www.kaggle.com/api/v1/datasets/download/emmarex/plantdisease"
    
    # DOWNLOAD WITH PROGRESS BAR
    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in iter(lambda: response.read(8192), b''):
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"Downloaded to {zip_path}")
    return zip_path


def extract_dataset(zip_path: Path) -> None:
    """Extract a zip file into the raw data directory.

    Args:
        zip_path: Path to the downloaded zip file.
    """
    print(f"Extracting to {RAW_DIR}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(RAW_DIR)
    
    # CLEANUP ZIP
    zip_path.unlink()
    print("Extraction complete, zip removed")


def organize_dataset() -> None:
    """Organize the selected classes into train/val/test splits."""
    print("Organizing dataset into train/val/test...")
    
    # CREATE TARGET DIRECTORIES
    for split in ['train', 'val', 'test']:
        for class_name in TARGET_CLASSES:
            (PROCESSED_DIR / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # FIND RAW CLASS FOLDERS
    # Assuming structure: plantvillage_raw/PlantVillage/{class_name}/
    plantvillage_root = RAW_DIR / "PlantVillage"
    
    if not plantvillage_root.exists():
        # TRY DIRECT STRUCTURE
        plantvillage_root = RAW_DIR
    
    # PROCESS EACH TARGET CLASS
    for class_name in TARGET_CLASSES:
        raw_class_dir = plantvillage_root / class_name
        
        if not raw_class_dir.exists():
            print(f"Warning: {raw_class_dir} not found, skipping...")
            continue
        
        # GET ALL IMAGES
        images = list(raw_class_dir.glob("*.jpg")) + list(raw_class_dir.glob("*.JPG"))
        print(f"Found {len(images)} images in {class_name}")
        
        # SHUFFLE AND SPLIT
        random.seed(42)
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        # COPY TO SPLITS
        for img in tqdm(train_imgs, desc=f"  train/{class_name}"):
            shutil.copy(img, PROCESSED_DIR / "train" / class_name / img.name)
        
        for img in tqdm(val_imgs, desc=f"  val/{class_name}"):
            shutil.copy(img, PROCESSED_DIR / "val" / class_name / img.name)
        
        for img in tqdm(test_imgs, desc=f"  test/{class_name}"):
            shutil.copy(img, PROCESSED_DIR / "test" / class_name / img.name)
        
        print(f"  {class_name}: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")
    
    # CLEANUP RAW DATA
    if RAW_DIR.exists():
        print(f"Removing raw data from {RAW_DIR}...")
        shutil.rmtree(RAW_DIR)


def main() -> None:
    """Run the end-to-end dataset preparation pipeline."""
    print("========== PlantVillage Dataset Preparation ==========")

    if dataset_ready():
        print(f"Dataset already prepared at: {PROCESSED_DIR}")
        print_dataset_stats()
        return
    
    # DOWNLOAD
    zip_path = download_dataset()
    
    # EXTRACT
    extract_dataset(zip_path)
    
    # ORGANIZE
    organize_dataset()
    
    print(f"Dataset preparation complete. Data ready at: {PROCESSED_DIR}")
    print_dataset_stats()


if __name__ == "__main__":
    main()
