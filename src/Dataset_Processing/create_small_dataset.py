"""
Create a small subset of the HED dataset with 200 images and their corresponding edges.
This script samples from the processed_HED_v2 dataset and creates HED_Small folder.
Distributes 200 images across train (140), val (30), and test (30) splits.
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Configuration
BASE_DIR = Path(r"E:\Edge Detection\datasets")
SOURCE_DIR = BASE_DIR / "processed_HED_v2"
OUTPUT_DIR = BASE_DIR / "HED_Small"
SAMPLE_SIZE = 200

# Distribution ratio: train / val / test
SPLIT_DISTRIBUTION = {
    'train': 140,  # 70%
    'val': 30,     # 15%
    'test': 30     # 15%
}

def create_small_dataset():
    """
    Create a small dataset by sampling 200 images from processed_HED_v2.
    Properly distributes across train/val/test splits.
    """
    
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / 'edges').mkdir(parents=True, exist_ok=True)
    
    print(f"Created output directory structure at {OUTPUT_DIR}")
    
    # Get all image files from source dataset (train split only since val/test are empty)
    all_images = []
    source_split_dir = SOURCE_DIR / 'train' / 'images'
    if source_split_dir.exists():
        images = list(source_split_dir.glob('*'))
        all_images = images
    
    print(f"Found {len(all_images)} total images in source dataset")
    
    if len(all_images) < SAMPLE_SIZE:
        print(f"Warning: Found only {len(all_images)} images, but requested {SAMPLE_SIZE}")
        print(f"Using all available images")
        sample_size = len(all_images)
    else:
        sample_size = SAMPLE_SIZE
    
    # Randomly sample all images
    sampled = random.sample(all_images, sample_size)
    print(f"Sampled {sample_size} images")
    
    # Distribute sampled images across splits
    train_samples = sampled[:SPLIT_DISTRIBUTION['train']]
    val_samples = sampled[SPLIT_DISTRIBUTION['train']:SPLIT_DISTRIBUTION['train'] + SPLIT_DISTRIBUTION['val']]
    test_samples = sampled[SPLIT_DISTRIBUTION['train'] + SPLIT_DISTRIBUTION['val']:]
    
    distribution = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }
    
    # Copy images and edges for each split
    total_copied = 0
    total_missing = 0
    
    for split_name, img_list in distribution.items():
        print(f"\nProcessing {split_name} split ({len(img_list)} images)...")
        copy_count = 0
        missing_count = 0
        
        for img_path in tqdm(img_list, desc=f"Copying {split_name}"):
            img_name = img_path.name
            # Source edge is always from train split since val/test edges don't exist
            edge_path = SOURCE_DIR / 'train' / 'edges' / img_name
            
            # Construct output paths
            output_img_path = OUTPUT_DIR / split_name / 'images' / img_name
            output_edge_path = OUTPUT_DIR / split_name / 'edges' / img_name
            
            # Copy image
            try:
                shutil.copy2(img_path, output_img_path)
                copy_count += 1
            except Exception as e:
                print(f"Error copying image {img_name}: {e}")
                continue
            
            # Copy corresponding edge map
            if edge_path.exists():
                try:
                    shutil.copy2(edge_path, output_edge_path)
                except Exception as e:
                    print(f"Error copying edge map {img_name}: {e}")
                    missing_count += 1
            else:
                print(f"Warning: Edge map not found for {img_name}")
                missing_count += 1
        
        print(f"✓ {split_name.upper()}: Copied {copy_count}, Missing {missing_count}")
        total_copied += copy_count
        total_missing += missing_count
    
    # Print summary
    print("\n" + "="*60)
    print("Dataset Creation Summary")
    print("="*60)
    print(f"Successfully copied: {total_copied} image pairs")
    print(f"Missing edge maps: {total_missing}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nSplit distribution:")
    print(f"  Train: {SPLIT_DISTRIBUTION['train']} images")
    print(f"  Val:   {SPLIT_DISTRIBUTION['val']} images")
    print(f"  Test:  {SPLIT_DISTRIBUTION['test']} images")
    print(f"  Total: {sum(SPLIT_DISTRIBUTION.values())} images")
    print("="*60)


if __name__ == "__main__":
    print("Creating HED_Small dataset...")
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Sample size: {SAMPLE_SIZE} images")
    print(f"Distribution: train={SPLIT_DISTRIBUTION['train']}, val={SPLIT_DISTRIBUTION['val']}, test={SPLIT_DISTRIBUTION['test']}")
    print()
    
    create_small_dataset()
    print("\nDone!")
