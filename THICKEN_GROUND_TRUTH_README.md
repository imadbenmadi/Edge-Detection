# Ground Truth Edge Thickening Script

## Overview

The `thicken_ground_truth.py` script processes ground truth edge detection images by applying morphological dilation to thicken the edges. This can improve training results by making ground truth annotations more consistent and easier for the model to learn.

## Features

- ✓ Copies ground truth images to a new directory
- ✓ Applies edge thickening using morphological dilation (5px kernel by default)
- ✓ Configurable input/output paths and edge thickness
- ✓ Supports common image formats (PNG, JPG, JPEG, BMP, TIFF)
- ✓ Progress tracking with progress bar
- ✓ Error handling and statistics reporting
- ✓ Preserves original filenames (converts all to PNG for quality)

## Requirements

The script requires the following Python packages (already in `requirements.txt`):
- Pillow
- numpy
- opencv-python

Additional dependency:
- tqdm (for progress bar)

Install dependencies:
```bash
pip install Pillow numpy opencv-python tqdm
```

## Configuration

Edit the following variables at the top of `thicken_ground_truth.py`:

```python
# Input directory containing ground truth edge images
INPUT_GT_DIR = "data/ground_truth/original"

# Output directory for thickened ground truth images
OUTPUT_GT_DIR = "data/ground_truth/thickened"

# Edge thickness in pixels (kernel size for morphological dilation)
EDGE_THICKNESS_PX = 5
```

## Usage

### Basic Usage

1. Update the `INPUT_GT_DIR` variable to point to your ground truth images
2. Update the `OUTPUT_GT_DIR` variable to specify where to save thickened images
3. Optionally adjust `EDGE_THICKNESS_PX` (default: 5 pixels)
4. Run the script:

```bash
python thicken_ground_truth.py
```

### Example

```bash
# Default configuration
python thicken_ground_truth.py
```

Output:
```
============================================================
GROUND TRUTH EDGE THICKENING SCRIPT
============================================================
Input directory:   /path/to/data/ground_truth/original
Output directory:  /path/to/data/ground_truth/thickened
Edge thickness:    5px
============================================================
Output directory: /path/to/data/ground_truth/thickened

Found 100 image file(s) to process

Processing images: 100%|████████████████| 100/100 [00:02<00:00, 45.3it/s]

============================================================
PROCESSING STATISTICS
============================================================
Total files found:       100
Successfully processed:  100
Failed:                  0
============================================================

✓ Processing complete! Thickened images saved to:
  /path/to/data/ground_truth/thickened
```

## How It Works

1. **Image Loading**: Loads ground truth images and converts them to grayscale
2. **Edge Thickening**: Applies morphological dilation using an elliptical kernel
3. **Saving**: Saves processed images as PNG files (lossless quality)

### Morphological Dilation

The script uses OpenCV's `cv2.dilate()` function with an elliptical structuring element:
- **Kernel Type**: `MORPH_ELLIPSE` (circular/elliptical shape)
- **Kernel Size**: Configurable (default: 5x5 pixels)
- **Iterations**: 1

This approach:
- Expands edge pixels uniformly in all directions
- Creates smooth, natural-looking thickened edges
- Maintains edge topology and structure
- Is commonly used in edge detection preprocessing

## Example Use Cases

### 1. Preparing Training Data
```python
INPUT_GT_DIR = "datasets/BSDS500/ground_truth/original"
OUTPUT_GT_DIR = "datasets/BSDS500/ground_truth/thickened_5px"
EDGE_THICKNESS_PX = 5
```

### 2. Creating Multiple Thickness Variants
Edit the script and run multiple times with different settings:
```python
# Run 1: Thin edges
EDGE_THICKNESS_PX = 3
OUTPUT_GT_DIR = "data/gt_3px"

# Run 2: Medium edges
EDGE_THICKNESS_PX = 5
OUTPUT_GT_DIR = "data/gt_5px"

# Run 3: Thick edges
EDGE_THICKNESS_PX = 7
OUTPUT_GT_DIR = "data/gt_7px"
```

### 3. Processing Dataset Splits
Process different dataset splits separately:
```python
# Training set
INPUT_GT_DIR = "data/train/edges"
OUTPUT_GT_DIR = "data/train/edges_thickened"

# Validation set
INPUT_GT_DIR = "data/val/edges"
OUTPUT_GT_DIR = "data/val/edges_thickened"

# Test set
INPUT_GT_DIR = "data/test/edges"
OUTPUT_GT_DIR = "data/test/edges_thickened"
```

## Output Format

- All output images are saved as **PNG** files for lossless quality
- Original filenames are preserved (with .png extension)
- Edge maps remain grayscale (single channel)
- Pixel values: 0-255 (0 = background, 255 = edge)

## Tips

- **Start with default settings**: The 5px thickness is a good starting point
- **Experiment**: Try different thickness values (3-7px) to find optimal results
- **Preserve originals**: The script creates copies, so your original data is safe
- **Check results**: Visually inspect a few thickened images before training
- **Batch processing**: The script processes entire directories automatically

## Troubleshooting

### Issue: "Input directory does not exist"
**Solution**: Check that `INPUT_GT_DIR` points to an existing directory with the correct path.

### Issue: "No image files found"
**Solution**: Ensure your ground truth images have supported extensions (.png, .jpg, .jpeg, .bmp, .tiff, .tif).

### Issue: "Error processing [filename]"
**Solution**: Check if the image file is corrupted or has unusual properties. The script will continue with other images.

## Technical Details

### Kernel Size vs Edge Thickness

The `EDGE_THICKNESS_PX` parameter controls the kernel size used for dilation:
- **3px**: Minimal thickening, preserves fine details
- **5px**: Moderate thickening, good balance (recommended)
- **7px**: Heavy thickening, may merge close edges
- **9+px**: Very thick edges, risk of over-smoothing

### Memory Usage

The script processes images one at a time, so memory usage is minimal. Even for large datasets (1000+ images), memory consumption stays low.

## License

This script is part of the Edge-Detection project and follows the same license.
