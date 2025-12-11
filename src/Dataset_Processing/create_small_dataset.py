"""
Create a tiny subset of the HED dataset with 50 images and their corresponding edges.
This script samples from the processed_HED_v2 dataset and creates HED_Tiny folder.
Distributes 50 images across train (35), val (8), and test (7) splits.
All images are resized to GLOBAL_SIZE (512, 512) with proper aspect ratio preservation.
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageOps
import numpy as np
import cv2

# Configuration
BASE_DIR = Path(r"E:\Edge Detection\datasets")
SOURCE_DIR = BASE_DIR / "processed_HED_v2"
OUTPUT_DIR = BASE_DIR / "HED_Small"
SAMPLE_SIZE = 50

# Global image size (width, height) - must match training configuration
GLOBAL_SIZE = (512, 512)

# Distribution ratio: train / val / test
SPLIT_DISTRIBUTION = {
    'train': 35,  # 70%
    'val': 8,     # 16%
    'test': 7     # 14%
}

def resize_image_v2(img, size, is_edge_map=False):
    """
    Research-grade, aspect-ratio preserving letterbox resize.
    - Preserves geometry (no distortion)
    - Uses interpolation based on scaling direction
    - Protects edge-label integrity
    - Produces pixel-aligned RGB + edge maps
    - Guaranteed output size
    """
    target_w, target_h = size
    w, h = img.size

    # Compute scale while preserving aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if is_edge_map:
        # Convert to float array in [0,1]
        arr = np.array(img).astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

        # Choose interpolation based on scaling direction
        interp = cv2.INTER_NEAREST if scale >= 1.0 else cv2.INTER_LINEAR
        arr_resized = cv2.resize(arr, (new_w, new_h), interpolation=interp)

        # Pad to target size with zeros
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        arr_padded = np.pad(
            arr_resized,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0.0,
        )

        out = (np.clip(arr_padded, 0.0, 1.0) * 255.0).astype(np.uint8)
        return Image.fromarray(out, mode='L')
    else:
        # High-quality downscale for RGB
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        # Pad with black to reach target size
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        img_padded = ImageOps.expand(img_resized, border=pad, fill=(0, 0, 0))
        return img_padded

def cleanup_output_dir():
    """
    Clean up the output directory by removing all existing files.
    This ensures we start with a fresh dataset.
    """
    if OUTPUT_DIR.exists():
        print(f"Cleaning up existing directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
        print("✓ Cleanup complete")
    else:
        print(f"Output directory doesn't exist yet: {OUTPUT_DIR}")

def create_small_dataset():
    """
    Create a tiny dataset by sampling 50 images from processed_HED_v2.
    Properly distributes across train/val/test splits and resizes all images to GLOBAL_SIZE.
    """
    
    # Clean up any existing output directory first
    cleanup_output_dir()
    
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
            
            # Process and resize image
            try:
                with Image.open(img_path) as img:
                    # Resize image to global size
                    img_resized = resize_image_v2(img, GLOBAL_SIZE, is_edge_map=False)
                    img_resized.save(output_img_path, format='JPEG', quality=95)
                    copy_count += 1
            except Exception as e:
                print(f"Error processing image {img_name}: {e}")
                continue
            
            # Process and resize corresponding edge map
            if edge_path.exists():
                try:
                    with Image.open(edge_path) as edge:
                        # Resize edge map to global size
                        edge_resized = resize_image_v2(edge, GLOBAL_SIZE, is_edge_map=True)
                        edge_resized.save(output_edge_path, format='PNG')
                except Exception as e:
                    print(f"Error processing edge map {img_name}: {e}")
                    missing_count += 1
            else:
                print(f"Warning: Edge map not found for {img_name}")
                missing_count += 1
        
        print(f"✓ {split_name.upper()}: Copied {copy_count}, Missing {missing_count}")
        total_copied += copy_count
        total_missing += missing_count
    
    # Print summary
    print("\n" + "="*60)
    print("HED_Tiny Dataset Creation Summary")
    print("="*60)
    print(f"Successfully processed: {total_copied} image pairs")
    print(f"Missing edge maps: {total_missing}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Global image size: {GLOBAL_SIZE[0]}x{GLOBAL_SIZE[1]}")
    print(f"\nSplit distribution:")
    print(f"  Train: {SPLIT_DISTRIBUTION['train']} images")
    print(f"  Val:   {SPLIT_DISTRIBUTION['val']} images")
    print(f"  Test:  {SPLIT_DISTRIBUTION['test']} images")
    print(f"  Total: {sum(SPLIT_DISTRIBUTION.values())} images")
    print("="*60)


if __name__ == "__main__":
    print("Creating HED_Tiny dataset...")
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Sample size: {SAMPLE_SIZE} images")
    print(f"Global size: {GLOBAL_SIZE}")
    print(f"Distribution: train={SPLIT_DISTRIBUTION['train']}, val={SPLIT_DISTRIBUTION['val']}, test={SPLIT_DISTRIBUTION['test']}")
    print()
    
    create_small_dataset()
    print("\nDone!")
