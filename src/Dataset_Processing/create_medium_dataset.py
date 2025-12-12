"""
Create a medium-sized dataset with 30% of the original HED dataset.
Source: processed_HED_v2 (9600 images total)
Output: HED_Medium with ~2880 images (30%)

Distribution:
- Train: 2016 images (70%)
- Val: 432 images (15%)
- Test: 432 images (15%)

All images are resized to 512x512 with aspect ratio preservation.
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
OUTPUT_DIR = BASE_DIR / "HED_Medium"

# 30% of 9600 images
TOTAL_IMAGES = 2880

# Distribution ratio: train / val / test (70% / 15% / 15%)
SPLIT_DISTRIBUTION = {
    'train': 2016,  # 70%
    'val': 432,     # 15%
    'test': 432     # 15%
}

# Global image size (width, height)
GLOBAL_SIZE = (512, 512)


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


def create_medium_dataset():
    """
    Create a medium-sized dataset with 30% of the original images.
    Properly distributes across train/val/test splits and resizes all images to GLOBAL_SIZE.
    """
    
    # Clean up any existing output directory first
    cleanup_output_dir()
    
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / 'edges').mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created output directory structure at {OUTPUT_DIR}")
    
    # Get all image files from source dataset (train split only since val/test are empty)
    all_images = []
    source_split_dir = SOURCE_DIR / 'train' / 'images'
    
    if not source_split_dir.exists():
        print(f"❌ Source directory not found: {source_split_dir}")
        return
    
    print(f"📂 Scanning source directory: {source_split_dir}")
    all_images = sorted(list(source_split_dir.glob('*.png')))
    
    if not all_images:
        all_images = sorted(list(source_split_dir.glob('*')))
    
    print(f"📊 Found {len(all_images)} total images in source dataset")
    
    if len(all_images) < TOTAL_IMAGES:
        print(f"⚠️  Warning: Found only {len(all_images)} images, but requested {TOTAL_IMAGES}")
        print(f"Using all available images")
        sample_size = len(all_images)
    else:
        sample_size = TOTAL_IMAGES
    
    # Randomly sample images
    print(f"\n🎲 Randomly sampling {sample_size} images (30% of dataset)...")
    random.seed(42)  # For reproducibility
    sampled = random.sample(all_images, sample_size)
    print(f"✓ Sampled {sample_size} images")
    
    # Distribute sampled images across splits
    print(f"\n📑 Distributing across splits:")
    print(f"   Train: {SPLIT_DISTRIBUTION['train']} (70%)")
    print(f"   Val:   {SPLIT_DISTRIBUTION['val']} (15%)")
    print(f"   Test:  {SPLIT_DISTRIBUTION['test']} (15%)")
    
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
    
    print(f"\n🔄 Processing and resizing images to {GLOBAL_SIZE[0]}x{GLOBAL_SIZE[1]}...\n")
    
    for split_name, img_list in distribution.items():
        print(f"Processing {split_name.upper()} split ({len(img_list)} images)...")
        copy_count = 0
        missing_count = 0
        resize_failed = 0
        
        pbar = tqdm(img_list, desc=f"  {split_name.upper():6}", unit="img")
        
        for img_path in pbar:
            img_name = img_path.name
            # Source edge is always from train split since val/test edges don't exist
            edge_path = SOURCE_DIR / 'train' / 'edges' / img_name
            
            # Construct output paths
            output_img_path = OUTPUT_DIR / split_name / 'images' / img_name
            output_edge_path = OUTPUT_DIR / split_name / 'edges' / img_name
            
            # Process and resize image
            try:
                with Image.open(img_path) as img:
                    img_resized = resize_image_v2(img, GLOBAL_SIZE, is_edge_map=False)
                    img_resized.save(output_img_path)
                    copy_count += 1
            except Exception as e:
                pbar.write(f"❌ Error processing image {img_name}: {e}")
                resize_failed += 1
                continue
            
            # Process and resize corresponding edge map
            if edge_path.exists():
                try:
                    with Image.open(edge_path) as edge:
                        edge_resized = resize_image_v2(edge, GLOBAL_SIZE, is_edge_map=True)
                        edge_resized.save(output_edge_path)
                except Exception as e:
                    pbar.write(f"❌ Error processing edge map {img_name}: {e}")
                    missing_count += 1
            else:
                missing_count += 1
        
        pbar.close()
        print(f"   ✓ Copied: {copy_count} | Missing edges: {missing_count} | Resize failed: {resize_failed}")
        total_copied += copy_count
        total_missing += missing_count
    
    # Verify the created dataset
    print(f"\n✅ Verifying created dataset...")
    train_count = len(list((OUTPUT_DIR / 'train' / 'images').glob('*')))
    val_count = len(list((OUTPUT_DIR / 'val' / 'images').glob('*')))
    test_count = len(list((OUTPUT_DIR / 'test' / 'images').glob('*')))
    
    # Print summary
    print("\n" + "="*70)
    print("HED_MEDIUM Dataset Creation Summary")
    print("="*70)
    print(f"✓ Successfully processed: {total_copied} image pairs")
    print(f"⚠ Missing edge maps: {total_missing}")
    print(f"\n📊 Output Statistics:")
    print(f"   Train images: {train_count}")
    print(f"   Val images:   {val_count}")
    print(f"   Test images:  {test_count}")
    print(f"   Total:        {train_count + val_count + test_count}")
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print(f"🖼 Image size: {GLOBAL_SIZE[0]}x{GLOBAL_SIZE[1]}")
    print(f"📈 Dataset scale: 30% of original (2880 / 9600 images)")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Creating HED_MEDIUM Dataset (30% of original)")
    print("="*70)
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Total images: {TOTAL_IMAGES} (30% of 9600)")
    print(f"Image size: {GLOBAL_SIZE}")
    print(f"Distribution: train={SPLIT_DISTRIBUTION['train']}, "
          f"val={SPLIT_DISTRIBUTION['val']}, test={SPLIT_DISTRIBUTION['test']}")
    print("="*70 + "\n")
    
    create_medium_dataset()
    print("\n✨ Done!\n")
