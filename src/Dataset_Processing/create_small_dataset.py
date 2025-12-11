"""
Create a small subset of the HED dataset with 200 images and their corresponding edges.
This script samples from the processed_HED_v2 dataset and creates HED_Small folder.
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

def create_small_dataset():
    """
    Create a small dataset by sampling 200 images from processed_HED_v2.
    """
    
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / 'edges').mkdir(parents=True, exist_ok=True)
    
    print(f"Created output directory structure at {OUTPUT_DIR}")
    
    # Get all image files from source dataset
    all_images = []
    for split in ['train', 'val', 'test']:
        split_dir = SOURCE_DIR / split / 'images'
        if split_dir.exists():
            images = list(split_dir.glob('*'))
            all_images.extend([(img, split) for img in images])
    
    print(f"Found {len(all_images)} total images in source dataset")
    
    if len(all_images) < SAMPLE_SIZE:
        print(f"Warning: Found only {len(all_images)} images, but requested {SAMPLE_SIZE}")
        print(f"Using all available images")
        sample_size = len(all_images)
    else:
        sample_size = SAMPLE_SIZE
    
    # Randomly sample images
    sampled = random.sample(all_images, sample_size)
    print(f"Sampled {sample_size} images")
    
    # Copy sampled images and their corresponding edges
    copy_count = 0
    missing_count = 0
    
    for img_path, split in tqdm(sampled, desc="Copying images and edges"):
        img_name = img_path.name
        edge_path = SOURCE_DIR / split / 'edges' / img_name
        
        # Construct output paths
        output_img_path = OUTPUT_DIR / 'train' / 'images' / img_name
        output_edge_path = OUTPUT_DIR / 'train' / 'edges' / img_name
        
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
    
    # Print summary
    print("\n" + "="*50)
    print("Dataset Creation Summary")
    print("="*50)
    print(f"Successfully copied: {copy_count} image pairs")
    print(f"Missing edge maps: {missing_count}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Train split: {OUTPUT_DIR / 'train' / 'images'}")
    print("="*50)


if __name__ == "__main__":
    print("Creating HED_Small dataset...")
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Sample size: {SAMPLE_SIZE} images")
    print()
    
    create_small_dataset()
    print("\nDone!")
