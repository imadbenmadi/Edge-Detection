# BPED - Bio-inspired Pyramid Edge Detection

A deep learning model for edge detection combining bio-inspired visual processing with multi-scale pyramid architecture.

## Overview

BPED is designed to mimic the hierarchical processing in the visual cortex:

- **V1-like orientation selectivity**: Multiple oriented filters capture edges at different angles
- **Multi-scale pyramid processing**: Captures edges at multiple scales simultaneously
- **Center-surround edge enhancement**: Bio-inspired lateral inhibition for edge sharpening
- **Progressive refinement**: Hierarchical decoder with skip connections

## Architecture

### Encoder (4 Stages)

1. **Stage 1** (64 ch): Oriented convolutions (4 orientations) + Pyramid block
2. **Stage 2** (128 ch): Oriented convolutions + Pyramid + Edge enhancement
3. **Stage 3** (256 ch): Standard convolutions + Pyramid + Edge enhancement
4. **Stage 4** (512 ch): Standard convolutions + Pyramid

### Decoder (Progressive Refinement)

- **4 decoding stages** with skip connections from encoder
- **Deep supervision**: Edge predictions at multiple scales (4 levels)
- **Multi-scale fusion**: Combines predictions from all scales

### Key Components

#### 1. Oriented Convolution

```python
class OrientedConv(nn.Module):
    """V1-like orientation-selective convolution"""
```

- Mimics V1 simple cells with 4 orientation channels
- Each orientation processed independently then concatenated

#### 2. Pyramid Block

```python
class PyramidBlock(nn.Module):
    """Multi-scale pyramid block"""
```

- Processes input at 3 scales: 3×3, 5×5, 7×7
- Fuses multi-scale features with residual connection

#### 3. Edge Enhancement

```python
class EdgeEnhancement(nn.Module):
    """Bio-inspired edge enhancement (center-surround)"""
```

- Center-surround antagonism: `center - 0.3 × surround`
- Attention modulation for adaptive enhancement

## Model Statistics

- **Parameters**: ~7.8M
- **Input size**: 320×320 RGB images
- **Output**: Single-channel edge probability map
- **Training**: Balanced BCE loss with deep supervision

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --dataset_path ../datasets/HED_Small --epochs 10 --batch_size 4
```

**Arguments**:

- `--dataset_path`: Path to dataset (default: `../datasets/HED_Small`)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 1e-4)
- `--image_size`: Input image size (default: 320)
- `--save_dir`: Directory to save checkpoints (default: `checkpoints`)

### Evaluation

Open and run `evaluate.ipynb` in Jupyter:

```bash
jupyter notebook evaluate.ipynb
```

The notebook will:

1. Load trained model weights
2. Run inference on test images
3. Compute ODS, OIS, and AP metrics
4. Visualize results and save plots

### Inference (Python)

```python
import torch
from PIL import Image
from torchvision import transforms
from model import BPED

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BPED().to(device)
model.load_state_dict(torch.load('checkpoints/best_model.pth')['model_state_dict'])
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open('test.jpg').convert('RGB')
img_tensor = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    edge_map = model(img_tensor)

# Post-process
edge_map = edge_map.squeeze().cpu().numpy()
```

## Training Details

### Loss Function

- **Balanced BCE Loss**: Addresses class imbalance between edge and non-edge pixels
    ```
    w_pos = n_neg / n_total
    w_neg = n_pos / n_total
    loss = -w_pos * y * log(p) - w_neg * (1-y) * log(1-p)
    ```

### Deep Supervision

- Edge predictions at 5 scales (4 intermediate + 1 final)
- Loss averaged across all scales
- Helps gradient flow and multi-scale feature learning

### Optimization

- **Optimizer**: Adam (lr=1e-4)
- **Scheduler**: CosineAnnealingLR
- **Training time**: ~10 epochs for convergence

## Performance Metrics

Evaluated on HED_Small dataset:

| Metric  | Value | Description                                    |
| ------- | ----- | ---------------------------------------------- |
| **ODS** | TBD   | Optimal Dataset Scale (best fixed threshold)   |
| **OIS** | TBD   | Optimal Image Scale (best per-image threshold) |
| **AP**  | TBD   | Average Precision                              |

_Metrics computed after training_

## File Structure

```
BPED/
├── model.py              # BPED architecture
├── train.py              # Training script
├── evaluate.ipynb        # Evaluation notebook
├── requirements.txt      # Dependencies
├── README.md            # This file
└── checkpoints/         # Saved model weights
    └── bped_YYYYMMDD_HHMMSS/
        ├── best_model.pth
        └── epoch_*.pth
```

## Bio-Inspired Features

1. **Orientation Selectivity** (V1 simple cells)
    - 4 oriented filters capture edges at 0°, 45°, 90°, 135°
    - Mimics orientation columns in primary visual cortex

2. **Multi-scale Processing** (V1/V2 receptive fields)
    - Pyramid blocks with 3×3, 5×5, 7×7 kernels
    - Models different receptive field sizes

3. **Center-Surround Antagonism** (Retinal ganglion cells)
    - Edge enhancement through lateral inhibition
    - Suppresses uniform regions, enhances discontinuities

4. **Attention Modulation** (Top-down feedback)
    - Adaptive weighting of edge features
    - Models cortical feedback mechanisms

5. **Hierarchical Processing** (Visual cortex hierarchy)
    - Low → mid → high level features
    - Progressive refinement of edge representation

## References

This implementation combines concepts from:

- Bio-inspired visual processing (orientation selectivity, center-surround)
- Multi-scale feature extraction (feature pyramids)
- Deep supervision (holistically-nested edge detection)
- Encoder-decoder architectures (U-Net style)

## Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- OpenCV
- NumPy
- Pillow
- matplotlib
- tqdm

## Citation

If you use this code, please cite:

```
@misc{bped2024,
  title={BPED: Bio-inspired Pyramid Edge Detection},
  author={Your Name},
  year={2024}
}
```

## License

MIT License
