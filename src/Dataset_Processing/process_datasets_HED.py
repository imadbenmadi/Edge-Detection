"""
Process and combine multiple edge detection datasets into a unified format.
Creates a 'processed' folder with train/val/test splits, all images resized to a global size.

Datasets processed:
1. BIPED - Barcelona Images for Perceptual Edge Detection
   - Train: edges/imgs/train/rgbr/real/*.jpg + edges/edge_maps/train/rgbr/real/*.png
   - Test: edges/imgs/test/rgbr/*.jpg + edges/edge_maps/test/rgbr/*.png
   
2. BIPEDv2 - BIPED version 2
   - Train: BIPED/edges/imgs/train/rgbr/real/*.jpg + BIPED/edges/edge_maps/train/rgbr/real/*.png
   - Test: BIPED/edges/imgs/test/rgbr/*.jpg + BIPED/edges/edge_maps/test/rgbr/*.png

3. HED-BSDS - HED Berkeley Segmentation Dataset
   - Train: HED-BSDS/train/aug_data/{rotation}/*.jpg + HED-BSDS/train/aug_gt/{rotation}/*.png
   - Test: HED-BSDS/test/*.jpg (ground truth from Kaggle .mat files)

4. Kaggle - Kaggle edge detection dataset (contains .mat files)
   - images/{train,val,test}/*.jpg
   - ground_truth/{train,val,test}/*.mat
"""

import os
import shutil
import glob
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import scipy.io as sio
import cv2

# Configuration
BASE_DIR = Path(r"E:\Edge Detection\datasets")
OUTPUT_DIR = BASE_DIR / "processed_HED"

# Global image size (width, height) - commonly used sizes for edge detection
GLOBAL_SIZE = (512, 512)  # You can adjust this

# Counter for unique naming across all datasets
global_counters = {'train': 0, 'val': 0, 'test': 0}

# Create output directory structure
def create_output_dirs():
    """Create the output directory structure."""
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / 'edges').mkdir(parents=True, exist_ok=True)
    print(f"Created output directories in {OUTPUT_DIR}")


def resize_image(img, size, is_edge_map=False):
    """
    Aspect-ratio preserving letterbox resize:
    - Scale the image so that it fits within target size without distortion.
    - Pad the remaining area to reach exact target size.
    
    For edge maps (grayscale, potentially soft labels):
    - Downscale with bilinear (preserves fractional labels)
    - Upscale with nearest (avoids creating spurious values)
    - Keep float in [0,1] until padding; convert to uint8 only at the end
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


def load_mat_file(mat_path):
    """
    Load edge map from .mat file.
    BSDS500 .mat files typically contain 'groundTruth' cell array.
    """
    try:
        mat = sio.loadmat(mat_path)
        
        # Try different possible keys for ground truth
        if 'groundTruth' in mat:
            # groundTruth is typically a cell array with multiple annotations
            gt = mat['groundTruth']
            # Average all annotations
            edges = []
            for i in range(gt.shape[1]):
                boundary = gt[0, i]['Boundaries'][0, 0]
                edges.append(boundary.astype(np.float32))
            edge_map = np.mean(edges, axis=0)
        elif 'edge' in mat:
            edge_map = mat['edge'].astype(np.float32)
        elif 'groundtruth' in mat:
            edge_map = mat['groundtruth'].astype(np.float32)
        else:
            # Try to find any array in the mat file
            for key in mat.keys():
                if not key.startswith('__'):
                    data = mat[key]
                    if isinstance(data, np.ndarray) and len(data.shape) == 2:
                        edge_map = data.astype(np.float32)
                        break
            else:
                print(f"Could not find edge map in {mat_path}")
                return None
        
        # Normalize to 0-255 range
        if edge_map.max() <= 1.0:
            edge_map = (edge_map * 255).astype(np.uint8)
        else:
            edge_map = edge_map.astype(np.uint8)
            
        return Image.fromarray(edge_map, mode='L')
    
    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return None


def save_image(img, output_path, is_edge_map=False):
    """Save image as PNG with proper format."""
    if is_edge_map:
        # Ensure edge map is in grayscale mode
        if img.mode != 'L':
            img = img.convert('L')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, 'PNG')


def process_biped():
    """Process BIPED dataset."""
    print("\n" + "="*60)
    print("Processing BIPED dataset...")
    print("="*60)
    
    biped_dir = BASE_DIR / "BIPED" / "edges"
    
    # Process training data - actual path: imgs/train/rgbr/real/*.jpg
    train_img_dir = biped_dir / "imgs" / "train" / "rgbr" / "real"
    train_edge_dir = biped_dir / "edge_maps" / "train" / "rgbr" / "real"
    
    if train_img_dir.exists():
        img_files = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))
        
        count = 0
        for img_path in tqdm(img_files, desc="BIPED train"):
            # Find corresponding edge map (same name but .png extension)
            edge_path = train_edge_dir / (img_path.stem + ".png")
            
            if edge_path.exists():
                try:
                    img = Image.open(img_path).convert('RGB')
                    edge = Image.open(edge_path).convert('L')
                    
                    img = resize_image(img, GLOBAL_SIZE)
                    edge = resize_image(edge, GLOBAL_SIZE, is_edge_map=True)
                    
                    output_name = f"biped_train_{count:06d}.png"
                    save_image(img, OUTPUT_DIR / "train" / "images" / output_name)
                    save_image(edge, OUTPUT_DIR / "train" / "edges" / output_name, is_edge_map=True)
                    count += 1
                    global_counters['train'] += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        print(f"Processed {count} BIPED training images")
    else:
        print(f"BIPED train directory not found: {train_img_dir}")
    
    # Process test data - actual path: imgs/test/rgbr/*.jpg
    test_img_dir = biped_dir / "imgs" / "test" / "rgbr"
    test_edge_dir = biped_dir / "edge_maps" / "test" / "rgbr"
    
    if test_img_dir.exists():
        img_files = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
        
        count = 0
        for img_path in tqdm(img_files, desc="BIPED test"):
            edge_path = test_edge_dir / (img_path.stem + ".png")
            
            if edge_path.exists():
                try:
                    img = Image.open(img_path).convert('RGB')
                    edge = Image.open(edge_path).convert('L')
                    
                    img = resize_image(img, GLOBAL_SIZE)
                    edge = resize_image(edge, GLOBAL_SIZE, is_edge_map=True)
                    
                    output_name = f"biped_test_{count:06d}.png"
                    save_image(img, OUTPUT_DIR / "test" / "images" / output_name)
                    save_image(edge, OUTPUT_DIR / "test" / "edges" / output_name, is_edge_map=True)
                    count += 1
                    global_counters['test'] += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        print(f"Processed {count} BIPED test images")
    else:
        print(f"BIPED test directory not found: {test_img_dir}")


def process_bipedv2():
    """Process BIPEDv2 dataset."""
    print("\n" + "="*60)
    print("Processing BIPEDv2 dataset...")
    print("="*60)
    
    bipedv2_dir = BASE_DIR / "BIPEDv2" / "BIPED" / "edges"
    
    # Process training data - actual path: imgs/train/rgbr/real/*.jpg
    train_img_dir = bipedv2_dir / "imgs" / "train" / "rgbr" / "real"
    train_edge_dir = bipedv2_dir / "edge_maps" / "train" / "rgbr" / "real"
    
    if train_img_dir.exists():
        img_files = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))
        
        count = 0
        for img_path in tqdm(img_files, desc="BIPEDv2 train"):
            edge_path = train_edge_dir / (img_path.stem + ".png")
            
            if edge_path.exists():
                try:
                    img = Image.open(img_path).convert('RGB')
                    edge = Image.open(edge_path).convert('L')
                    
                    img = resize_image(img, GLOBAL_SIZE)
                    edge = resize_image(edge, GLOBAL_SIZE, is_edge_map=True)
                    
                    output_name = f"bipedv2_train_{count:06d}.png"
                    save_image(img, OUTPUT_DIR / "train" / "images" / output_name)
                    save_image(edge, OUTPUT_DIR / "train" / "edges" / output_name, is_edge_map=True)
                    count += 1
                    global_counters['train'] += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        print(f"Processed {count} BIPEDv2 training images")
    else:
        print(f"BIPEDv2 train directory not found: {train_img_dir}")
    
    # Process test data - actual path: imgs/test/rgbr/*.jpg
    test_img_dir = bipedv2_dir / "imgs" / "test" / "rgbr"
    test_edge_dir = bipedv2_dir / "edge_maps" / "test" / "rgbr"
    
    if test_img_dir.exists():
        img_files = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
        
        count = 0
        for img_path in tqdm(img_files, desc="BIPEDv2 test"):
            edge_path = test_edge_dir / (img_path.stem + ".png")
            
            if edge_path.exists():
                try:
                    img = Image.open(img_path).convert('RGB')
                    edge = Image.open(edge_path).convert('L')
                    
                    img = resize_image(img, GLOBAL_SIZE)
                    edge = resize_image(edge, GLOBAL_SIZE, is_edge_map=True)
                    
                    output_name = f"bipedv2_test_{count:06d}.png"
                    save_image(img, OUTPUT_DIR / "test" / "images" / output_name)
                    save_image(edge, OUTPUT_DIR / "test" / "edges" / output_name, is_edge_map=True)
                    count += 1
                    global_counters['test'] += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        print(f"Processed {count} BIPEDv2 test images")
    else:
        print(f"BIPEDv2 test directory not found: {test_img_dir}")


def process_hed_bsds():
    """Process HED-BSDS dataset."""
    print("\n" + "="*60)
    print("Processing HED-BSDS dataset...")
    print("="*60)
    
    bsds_dir = BASE_DIR / "HED-BSDS" / "HED-BSDS"
    
    # Process training data - iterate through rotation/augmentation folders
    # Structure: train/aug_data/{rotation}/*.jpg and train/aug_gt/{rotation}/*.png
    train_data_dir = bsds_dir / "train" / "aug_data"
    train_gt_dir = bsds_dir / "train" / "aug_gt"
    
    if train_data_dir.exists():
        count = 0
        # Get all rotation/augmentation folders
        aug_folders = [f for f in train_data_dir.iterdir() if f.is_dir()]
        
        for aug_folder in tqdm(aug_folders, desc="HED-BSDS train folders"):
            gt_folder = train_gt_dir / aug_folder.name
            
            if not gt_folder.exists():
                continue
            
            img_files = list(aug_folder.glob("*.jpg")) + list(aug_folder.glob("*.png"))
            
            for img_path in img_files:
                edge_path = gt_folder / (img_path.stem + ".png")
                
                if edge_path.exists():
                    try:
                        img = Image.open(img_path).convert('RGB')
                        edge = Image.open(edge_path).convert('L')
                        
                        img = resize_image(img, GLOBAL_SIZE)
                        edge = resize_image(edge, GLOBAL_SIZE, is_edge_map=True)
                        
                        output_name = f"hedbsds_train_{count:06d}.png"
                        save_image(img, OUTPUT_DIR / "train" / "images" / output_name)
                        save_image(edge, OUTPUT_DIR / "train" / "edges" / output_name, is_edge_map=True)
                        count += 1
                        global_counters['train'] += 1
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
        
        print(f"Processed {count} HED-BSDS training images (with augmentations)")
    else:
        print(f"HED-BSDS train directory not found: {train_data_dir}")
    
    # Process test data - test/*.jpg with ground truth from Kaggle
    test_dir = bsds_dir / "test"
    kaggle_gt_test = BASE_DIR / "kaggle" / "ground_truth" / "test"
    
    if test_dir.exists():
        img_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        
        count = 0
        for img_path in tqdm(img_files, desc="HED-BSDS test"):
            try:
                img = Image.open(img_path).convert('RGB')
                img = resize_image(img, GLOBAL_SIZE)
                
                # Try to find corresponding ground truth from kaggle .mat files
                mat_path = kaggle_gt_test / f"{img_path.stem}.mat"
                
                if mat_path.exists():
                    edge = load_mat_file(mat_path)
                    if edge:
                        edge = resize_image(edge, GLOBAL_SIZE, is_edge_map=True)
                        
                        output_name = f"hedbsds_test_{count:06d}.png"
                        save_image(img, OUTPUT_DIR / "test" / "images" / output_name)
                        save_image(edge, OUTPUT_DIR / "test" / "edges" / output_name, is_edge_map=True)
                        count += 1
                        global_counters['test'] += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        print(f"Processed {count} HED-BSDS test images (with Kaggle ground truth)")
    else:
        print(f"HED-BSDS test directory not found: {test_dir}")


def process_kaggle():
    """Process Kaggle dataset with .mat files."""
    print("\n" + "="*60)
    print("Processing Kaggle dataset...")
    print("="*60)
    
    kaggle_dir = BASE_DIR / "kaggle"
    
    for split in ['train', 'val', 'test']:
        img_dir = kaggle_dir / "images" / split
        gt_dir = kaggle_dir / "ground_truth" / split
        
        if not img_dir.exists():
            print(f"Kaggle {split} images directory not found: {img_dir}")
            continue
        
        if not gt_dir.exists():
            print(f"Kaggle {split} ground_truth directory not found: {gt_dir}")
            continue
        
        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        # Filter out Thumbs.db and other non-image files
        img_files = [f for f in img_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        count = 0
        for img_path in tqdm(img_files, desc=f"Kaggle {split}"):
            # Find corresponding .mat ground truth
            mat_path = gt_dir / f"{img_path.stem}.mat"
            
            if mat_path.exists():
                try:
                    img = Image.open(img_path).convert('RGB')
                    edge = load_mat_file(mat_path)
                    
                    if edge:
                        img = resize_image(img, GLOBAL_SIZE)
                        edge = resize_image(edge, GLOBAL_SIZE, is_edge_map=True)
                        
                        output_name = f"kaggle_{split}_{count:06d}.png"
                        save_image(img, OUTPUT_DIR / split / "images" / output_name)
                        save_image(edge, OUTPUT_DIR / split / "edges" / output_name, is_edge_map=True)
                        count += 1
                        global_counters[split] += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        print(f"Processed {count} Kaggle {split} images")


def print_statistics():
    """Print statistics of the processed dataset."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    total_images = 0
    total_edges = 0
    
    for split in ['train', 'val', 'test']:
        img_dir = OUTPUT_DIR / split / "images"
        edge_dir = OUTPUT_DIR / split / "edges"
        
        if img_dir.exists():
            n_images = len(list(img_dir.glob("*.png")))
            n_edges = len(list(edge_dir.glob("*.png")))
            print(f"{split.upper():5s}: {n_images:6d} images, {n_edges:6d} edge maps")
            total_images += n_images
            total_edges += n_edges
    
    print("-"*60)
    print(f"TOTAL: {total_images:6d} images, {total_edges:6d} edge maps")
    print("="*60)
    print(f"Global image size: {GLOBAL_SIZE[0]}x{GLOBAL_SIZE[1]}")
    print(f"Output directory: {OUTPUT_DIR}")


def main():
    """Main function to process all datasets."""
    print("="*60)
    print("EDGE DETECTION DATASET PROCESSOR")
    print("="*60)
    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Global size: {GLOBAL_SIZE}")
    print("="*60)
    
    # Create output directories
    create_output_dirs()
    
    # Process each dataset
    # process_biped()
    # process_bipedv2()
    process_hed_bsds()
    # process_kaggle()
    
    # Print final statistics
    print_statistics()
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
