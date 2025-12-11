"""
Create a thickened edge detection dataset from processed_HED_v2.

This script:
1. Loads images and edge maps from processed_HED_v2
2. Thickens edges using dilation morphological operations
3. Creates a new 'dataset_hed_thick' folder with a subset of 200 images
4. Preserves the train/val/test split structure

Purpose: Test if thicker edges improve model training performance
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import random

# Configuration
BASE_DIR = Path(r"E:\Edge Detection\datasets")
SOURCE_DIR = BASE_DIR / "processed_HED_v2"  # Source dataset
OUTPUT_DIR = BASE_DIR / "HED_Thick"  # New thick edges dataset

# Number of images per split to create  
IMAGES_PER_SPLIT = {
    'train': 140,  # 70% (140 images)
    'val': 30,     # 15% (30 images)
    'test': 30     # 15% (30 images)
}  # Total = 200 images

# Morphological operation parameters for edge thickening
KERNEL_SIZE = 3
THICKENING_ITERATIONS = 3  # Number of morphological thickening iterations


def create_output_dirs():
    """Create the output directory structure."""
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / 'edges').mkdir(parents=True, exist_ok=True)
    print(f"Created output directories in {OUTPUT_DIR}")


def thick_edges(edge_map):
    """
    Thicken edge map using dilation morphological operations.
    
    Parameters:
    -----------
    edge_map : PIL Image or np.ndarray
        Input edge map (should be binary or grayscale)
    
    Returns:
    --------
    PIL Image
        Thickened edge map
    """
    # Convert to numpy array if needed
    if isinstance(edge_map, Image.Image):
        arr = np.array(edge_map, dtype=np.uint8)
    else:
        arr = edge_map.astype(np.uint8)
    
    # Create a kernel for thickening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
    
    # Apply morphological dilation to thicken edges
    # Work directly with grayscale values (don't threshold)
    thickened = arr.copy()

    for _ in range(THICKENING_ITERATIONS):
        thickened = cv2.dilate(thickened, kernel, iterations=1)
    
    return Image.fromarray(thickened, mode='L')


def thick_edges_v2(edge_map):
    """
    Thicken edges using dilation.
    Produces thicker (~10-15px) edges using basic morphological operations.
    
    Parameters:
    -----------
    edge_map : PIL Image or np.ndarray
        Input edge map
    
    Returns:
    --------
    PIL Image
        Thickened edge map
    """
    if isinstance(edge_map, Image.Image):
        arr = np.array(edge_map, dtype=np.uint8)
    else:
        arr = edge_map.astype(np.uint8)
    
    # Create elliptical kernel for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Thicken edges using morphological dilation
    # Work directly with grayscale values
    thickened = cv2.dilate(arr, kernel, iterations=2)
    
    return Image.fromarray(thickened, mode='L')


def thick_edges_v3(edge_map, target_thickness=5):
    """
    Smart edge thickening using dilation operations.
    
    Parameters:
    -----------
    edge_map : PIL Image or np.ndarray
        Input edge map
    target_thickness : int
        Number of dilation iterations (more iterations = thicker edges)
    
    Returns:
    --------
    PIL Image
        Thickened edge map
    """
    if isinstance(edge_map, Image.Image):
        arr = np.array(edge_map, dtype=np.uint8)
    else:
        arr = edge_map.astype(np.uint8)
    
    # Don't threshold - work with grayscale values directly
    # Create kernel for thickening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Apply morphological dilation to thicken edges
    # Each iteration makes edges thicker
    thickened = cv2.dilate(arr, kernel, iterations=target_thickness)
    
    return Image.fromarray(thickened, mode='L')


def process_dataset():
    """Process the dataset with thicker edges."""
    print("\n" + "="*70)
    print("CREATING THICK EDGES DATASET (HED_Thick)")
    print("="*70)
    
    create_output_dirs()
    
    total_processed = 0
    
    # Get all images from train split (since val/test may be empty or have different data)
    train_imgs = list((SOURCE_DIR / 'train' / 'images').glob('*.png'))
    print(f"\nTotal source images available in 'train': {len(train_imgs)}")
    print(f"Sampling {sum(IMAGES_PER_SPLIT.values())} images with splits: train={IMAGES_PER_SPLIT['train']}, val={IMAGES_PER_SPLIT['val']}, test={IMAGES_PER_SPLIT['test']}\n")
    
    # Shuffle and split
    random.shuffle(train_imgs)
    train_count = IMAGES_PER_SPLIT['train']
    val_count = IMAGES_PER_SPLIT['val']
    test_count = IMAGES_PER_SPLIT['test']
    
    train_split = train_imgs[:train_count]
    val_split = train_imgs[train_count:train_count + val_count]
    test_split = train_imgs[train_count + val_count:train_count + val_count + test_count]
    
    print(f"Allocated: train={len(train_split)}, val={len(val_split)}, test={len(test_split)}\n")
    
    splits_data = {
        'train': train_split,
        'val': val_split,
        'test': test_split,
    }
    
    # Process each split
    for split in ['train', 'val', 'test']:
        img_files = splits_data.get(split, [])
        
        if not img_files:
            print(f"\n{split.upper()}: No images to process")
            continue
        
        print(f"\n{split.upper()}: Processing {len(img_files)} images...")
        print(f"(Creating a copy with THICKENED edges, original dataset unchanged)")
        
        count = 0
        for img_path in tqdm(img_files, desc=f"Thickening {split} edges"):
            try:
                # Load image and edge map from source (READ ONLY)
                img = Image.open(img_path).convert('RGB')
                
                # Determine which split the image is from in source
                source_split = None
                for check_split in ['train', 'val', 'test']:
                    if (SOURCE_DIR / check_split / 'images' / img_path.name).exists():
                        source_split = check_split
                        break
                
                if source_split is None:
                    source_split = 'train'  # Default to train if not found
                
                edge_path = SOURCE_DIR / source_split / 'edges' / img_path.name
                
                if not edge_path.exists():
                    print(f"Edge map not found for {img_path.name}")
                    continue
                
                edge = Image.open(edge_path).convert('L')
                
                # Thicken the edges
                thickened_edge = thick_edges_v3(edge, target_thickness=1)
                
                # Save to NEW dataset
                output_name = f"hed_thick_{split}_{count:06d}.png"
                
                # Ensure output directories exist
                (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
                (OUTPUT_DIR / split / 'edges').mkdir(parents=True, exist_ok=True)
                
                # Save image (copy of original, unchanged)
                img.save(OUTPUT_DIR / split / 'images' / output_name, 'PNG')
                # Save thickened edge map (modified)
                thickened_edge.save(OUTPUT_DIR / split / 'edges' / output_name, 'PNG')
                
                count += 1
                total_processed += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        print(f"[OK] Processed {count} {split} images with THICKENED edges")
    
    print_statistics()
    
    print("\n" + "="*70)
    print("DATASET CREATION COMPLETE!")
    print("="*70)
    print(f"Total images processed: {total_processed}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Edges have been THICKENED for better ground truth")
    print(f"Ready for training!")
    print("="*70)


def print_statistics():
    """Print statistics of the new dataset."""
    print("\n" + "-"*70)
    print("THICK EDGES DATASET STATISTICS")
    print("-"*70)
    
    total_images = 0
    total_edges = 0
    
    for split in ['train', 'val', 'test']:
        img_dir = OUTPUT_DIR / split / 'images'
        edge_dir = OUTPUT_DIR / split / 'edges'
        
        if img_dir.exists():
            n_images = len(list(img_dir.glob("*.png")))
            n_edges = len(list(edge_dir.glob("*.png")))
            print(f"{split.upper():5s}: {n_images:3d} images, {n_edges:3d} edge maps")
            total_images += n_images
            total_edges += n_edges
    
    print("-"*70)
    print(f"TOTAL:  {total_images:3d} images, {total_edges:3d} edge maps")
    print("-"*70)


def compare_edge_thicknesses():
    """
    Optional: Compare edge thickness before and after thickening.
    Useful for validation.
    """
    print("\nComparing edge thicknesses...")
    
    # Get a sample image from source
    sample_edge = SOURCE_DIR / "train" / "edges"
    sample_files = list(sample_edge.glob("*.png"))
    
    if sample_files:
        sample_path = sample_files[0]
        original_edge = Image.open(sample_path).convert('L')
        thickened_edge = thick_edges_v3(original_edge, target_thickness=1)
        
        orig_arr = np.array(original_edge)
        thick_arr = np.array(thickened_edge)
        
        orig_pixels = np.count_nonzero(orig_arr)
        thick_pixels = np.count_nonzero(thick_arr)
        
        print(f"\nSample comparison (from {sample_path.name}):")
        print(f"Original edge pixels:  {orig_pixels}")
        print(f"Thickened edge pixels: {thick_pixels}")
        print(f"Increase: {((thick_pixels/orig_pixels - 1)*100):.1f}%")


def main():
    """Main function."""
    print("="*70)
    print("THICK EDGES DATASET CREATOR")
    print("="*70)
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target subset size: {sum(IMAGES_PER_SPLIT.values())} images")
    print(f"Operation: THICKENING edges using dilation")
    print("="*70)
    
    # Check if source directory exists
    if not SOURCE_DIR.exists():
        print(f"\nERROR: Source directory not found: {SOURCE_DIR}")
        print("Please ensure processed_HED_v2 exists.")
        return
    
    # Process the dataset
    process_dataset()
    
    # Optional: Compare edge thicknesses
    compare_edge_thicknesses()


if __name__ == "__main__":
    main()
