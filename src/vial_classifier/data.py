
"""Data access: pull from the object store, load, split and batch the images."""
 
import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
 
import boto3
import numpy as np
import pandas as pd
import torch
from botocore import UNSIGNED
from botocore.config import Config
from PIL import Image
from torch.utils.data import DataLoader, Dataset
 
from vial_classifier import config
from vial_classifier.features import define_transforms
 
 
def prepare_directories():
    """Create the working directories the pipeline writes to."""
    for directory in (
        config.ARTIFACT_DIR,
        config.MLFLOW_STATE_DIR,
        config.MLFLOW_ARTIFACT_LOCATION,
        config.IMAGE_DATA_DIR,
    ):
        os.makedirs(directory, exist_ok=True)
    print("Created artifacts directory")
 
 
# --- Pulling data -------------------------------------------------------------
 
def _setup_s3_client():
    """Create a boto3 S3 client for the public (unsigned) Hetzner bucket."""
    return boto3.client(
        "s3",
        endpoint_url=config.S3_ENDPOINT,
        aws_access_key_id="",
        aws_secret_access_key="",
        config=Config(signature_version=UNSIGNED),
    )
 
 
def _download_file(s3_client, key, output_path):
    """Download one file. Returns True only if it was actually downloaded."""
    output_path = Path(output_path)
    if output_path.exists():
        return False
    try:
        s3_client.download_file(config.S3_BUCKET, key, str(output_path))
        return True
    except Exception as e:
        print(f"Failed to download {key}: {e}")
        return False
 
 
def download_files_from_metadata(metadata_path, output_dir, prefix=""):
    """Download all files listed in a metadata csv from the object store."""
    s3_client = _setup_s3_client()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
 
    with open(metadata_path) as f:
        rows = list(csv.DictReader(f))
 
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
 
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for row in rows:
            filename = row["filename"]
            output_path = Path(output_dir) / filename
            future = executor.submit(_download_file, s3_client, f"{prefix}{filename}", output_path)
            futures[future] = output_path
 
        for future in as_completed(futures):
            output_path = futures[future]
            if future.result():
                downloaded_count += 1
            elif output_path.exists():
                skipped_count += 1
            else:
                failed_count += 1
 
    print(f"Downloaded {downloaded_count} images, skipped {skipped_count} existing, failed {failed_count}")
 
 
def pull_data_from_dvc():
    """Fetch the DVC-tracked metadata csvs, then download the images they list."""
    subprocess.run(["dvc", "pull"], check=True)
    download_files_from_metadata(config.METADATA_PATH, config.IMAGE_DATA_DIR, "images/raw/")
    download_files_from_metadata(config.TEST_METADATA_PATH, config.TEST_DATA_DIR, "test/")
 
 
# --- Loading and exploring ----------------------------------------------------
 
def load_image_dataset():
    """Load (image_path, label) pairs from the raw images and the metadata."""
    print(f"Loading image dataset from {config.IMAGE_DATA_DIR}")
    print(f"Using metadata from {config.METADATA_PATH}")
 
    if not Path(config.METADATA_PATH).exists():
        raise FileNotFoundError(f"Metadata file not found at {config.METADATA_PATH}")
 
    metadata = pd.read_csv(config.METADATA_PATH)
    print(f"Metadata loaded. Total entries: {len(metadata)}")
 
    image_dataset = []
    missing_files = []
    for _, row in metadata.iterrows():
        filename = row["filename"]
        label = int(row["class"])
        image_path = Path(config.IMAGE_DATA_DIR) / filename
        if image_path.exists():
            image_dataset.append((image_path, label))
        else:
            missing_files.append(filename)
 
    print(f"Loaded {len(image_dataset)} valid images with labels")
    if missing_files:
        print(f"Warning: {len(missing_files)} image files not found")
 
    return image_dataset
 
 
def print_class_distribution(image_dataset):
    """Print the class balance of the dataset."""
    labels = [label for _, label in image_dataset]
    unique, counts = np.unique(labels, return_counts=True)
    print("Class distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} images ({count / len(labels) * 100:.1f}%)")
 
    metadata_df = pd.read_csv(config.METADATA_PATH)
    print("\nImages per class name:")
    print(metadata_df["class_name"].value_counts())
 
 
# --- Dataset, split and loaders -----------------------------------------------
 
class GlassVialDataset(Dataset):
    """Loads vial images and their labels and applies the transform."""
 
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
 
    def __len__(self):
        return len(self.image_paths)
 
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
 
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            image = img.copy()
 
        if self.transform:
            image = self.transform(image)
 
        return image, label
 
 
def split_dataset(image_dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """Stratified split so both classes keep the same balance in every set."""
    print(f"\nSplitting dataset into train ({train_ratio}), val ({val_ratio}), test ({test_ratio})...")
 
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
 
    class_0 = [(path, label) for path, label in image_dataset if label == 0]
    class_1 = [(path, label) for path, label in image_dataset if label == 1]
    print(f"Class distribution before split: Class 0: {len(class_0)}, Class 1: {len(class_1)}")
 
    np.random.shuffle(class_0)
    np.random.shuffle(class_1)
 
    def split_class(data):
        total = len(data)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        return (
            data[:train_size],
            data[train_size:train_size + val_size],
            data[train_size + val_size:],
        )
 
    train_0, val_0, test_0 = split_class(class_0)
    train_1, val_1, test_1 = split_class(class_1)
 
    train_dataset = train_0 + train_1
    val_dataset = val_0 + val_1
    test_dataset = test_0 + test_1
 
    np.random.shuffle(train_dataset)
    np.random.shuffle(val_dataset)
    np.random.shuffle(test_dataset)
 
    print("Split complete:")
    print(f"  Training set: {len(train_dataset)} images")
    print(f"  Validation set: {len(val_dataset)} images")
    print(f"  Test set: {len(test_dataset)} images")
 
    train_labels = [label for _, label in train_dataset]
    val_labels = [label for _, label in val_dataset]
    test_labels = [label for _, label in test_dataset]
    print(f"  Train - Class 0: {train_labels.count(0)}, Class 1: {train_labels.count(1)}")
    print(f"  Val   - Class 0: {val_labels.count(0)}, Class 1: {val_labels.count(1)}")
    print(f"  Test  - Class 0: {test_labels.count(0)}, Class 1: {test_labels.count(1)}")
 
    return train_dataset, val_dataset, test_dataset
 
 
def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=config.BATCH_SIZE):
    print(f"\nCreating DataLoaders with batch_size={batch_size}...")
 
    train_transforms, val_transforms = define_transforms()
 
    train_paths, train_labels = zip(*train_dataset) if train_dataset else ([], [])
    val_paths, val_labels = zip(*val_dataset) if val_dataset else ([], [])
    test_paths, test_labels = zip(*test_dataset) if test_dataset else ([], [])
 
    train_data = GlassVialDataset(list(train_paths), list(train_labels), train_transforms)
    val_data = GlassVialDataset(list(val_paths), list(val_labels), val_transforms)
    test_data = GlassVialDataset(list(test_paths), list(test_labels), val_transforms)
 
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
 
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")
 
    return train_loader, val_loader, test_loader