"""
Ground Truth Edge Thickening Script

This script processes ground truth edge detection images by:
1. Copying them to a new output directory
2. Applying edge thickening using morphological dilation (5px kernel)
3. Supporting configurable input/output paths

The thickened edges can improve training results by making ground truth
annotations more consistent and easier for the model to learn.

Usage:
    python thicken_ground_truth.py [--input INPUT_DIR] [--output OUTPUT_DIR] [--thickness THICKNESS]
    
Example:
    python thicken_ground_truth.py --input data/gt/original --output data/gt/thickened --thickness 5
"""

import os
import glob
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

# ============================================================================
# CONFIGURATION - Modify these variables to change behavior
# ============================================================================

# Input directory containing ground truth edge images
INPUT_GT_DIR = "data/ground_truth/original"

# Output directory for thickened ground truth images
OUTPUT_GT_DIR = "data/ground_truth/thickened"

# Edge thickness in pixels (kernel size for morphological dilation)
EDGE_THICKNESS_PX = 5

# Supported image file extensions
SUPPORTED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']

# ============================================================================
# FUNCTIONS
# ============================================================================

def create_output_dir(output_dir):
    """
    Create the output directory if it doesn't exist.
    
    Args:
        output_dir (str or Path): Path to the output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path.absolute()}")


def get_image_files(input_dir, extensions):
    """
    Get all image files from the input directory with supported extensions.
    
    Args:
        input_dir (str or Path): Path to the input directory
        extensions (list): List of supported file extensions
    
    Returns:
        list: List of Path objects for found image files
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise ValueError(
            f"Input directory does not exist: {input_path.absolute()}\n"
            f"Please verify the path and create the directory if needed."
        )
    
    image_files = []
    for ext in extensions:
        # Case-insensitive search
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    
    return image_files


def thicken_edges(image, thickness_px):
    """
    Thicken edges in a grayscale edge map using morphological dilation.
    
    Args:
        image (numpy.ndarray): Grayscale edge map (values 0-255)
        thickness_px (int): Edge thickness in pixels (kernel size)
    
    Returns:
        numpy.ndarray: Thickened edge map
    """
    # Create an elliptical kernel for natural-looking edge thickening
    # When thickness_px creates a square size, this produces a circular shape
    # Better for edge detection tasks than rectangular kernels
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (thickness_px, thickness_px)
    )
    
    # Apply morphological dilation to thicken the edges
    thickened = cv2.dilate(image, kernel, iterations=1)
    
    return thickened


def process_image(input_path, output_path, thickness_px):
    """
    Process a single ground truth image by thickening its edges.
    
    Args:
        input_path (Path): Path to input image
        output_path (Path): Path to save output image
        thickness_px (int): Edge thickness in pixels
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load image and convert to grayscale
        image = Image.open(input_path).convert('L')
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.uint8)
        
        # Thicken the edges
        thickened_array = thicken_edges(img_array, thickness_px)
        
        # Convert back to PIL Image
        thickened_image = Image.fromarray(thickened_array, mode='L')
        
        # Save the result (always as PNG for lossless quality)
        output_path_png = output_path.with_suffix('.png')
        thickened_image.save(output_path_png, 'PNG')
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_path.name}: {e}")
        return False


def print_statistics(total_files, successful_files, failed_files):
    """
    Print processing statistics.
    
    Args:
        total_files (int): Total number of files found
        successful_files (int): Number of successfully processed files
        failed_files (int): Number of failed files
    """
    print("\n" + "="*60)
    print("PROCESSING STATISTICS")
    print("="*60)
    print(f"Total files found:       {total_files}")
    print(f"Successfully processed:  {successful_files}")
    print(f"Failed:                  {failed_files}")
    print("="*60)


def main():
    """
    Main function to process all ground truth images.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Thicken edges in ground truth images using morphological dilation.'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=INPUT_GT_DIR,
        help=f'Input directory containing ground truth images (default: {INPUT_GT_DIR})'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=OUTPUT_GT_DIR,
        help=f'Output directory for thickened images (default: {OUTPUT_GT_DIR})'
    )
    parser.add_argument(
        '--thickness', '-t',
        type=int,
        default=EDGE_THICKNESS_PX,
        help=f'Edge thickness in pixels (default: {EDGE_THICKNESS_PX})'
    )
    
    args = parser.parse_args()
    
    # Use command-line arguments or defaults
    input_dir = args.input
    output_dir = args.output
    thickness = args.thickness
    
    print("="*60)
    print("GROUND TRUTH EDGE THICKENING SCRIPT")
    print("="*60)
    print(f"Input directory:   {Path(input_dir).absolute()}")
    print(f"Output directory:  {Path(output_dir).absolute()}")
    print(f"Edge thickness:    {thickness}px")
    print("="*60)
    
    # Create output directory
    create_output_dir(output_dir)
    
    # Get all image files from input directory
    try:
        image_files = get_image_files(input_dir, SUPPORTED_EXTENSIONS)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    if not image_files:
        print(f"Warning: No image files found in {input_dir}")
        print(f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}")
        return
    
    print(f"\nFound {len(image_files)} image file(s) to process\n")
    
    # Process each image
    successful = 0
    failed = 0
    
    for img_path in tqdm(image_files, desc="Processing images"):
        # Preserve the original filename base (extension will be .png)
        output_path = Path(output_dir) / img_path.name
        
        if process_image(img_path, output_path, thickness):
            successful += 1
        else:
            failed += 1
    
    # Print statistics
    print_statistics(len(image_files), successful, failed)
    
    if successful > 0:
        print(f"\n✓ Processing complete! Thickened images saved to:")
        print(f"  {Path(output_dir).absolute()}")
    else:
        print("\n✗ No images were successfully processed.")


if __name__ == "__main__":
    main()
