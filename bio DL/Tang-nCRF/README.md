# Tang et al. - nCRF for Contour Detection

Implementation of **"Learning Nonclassical Receptive Field Modulation for Contour Detection"** by Qiling Tang, Nong Sang, and Haihua Liu (IEEE Transactions on Image Processing, 2019).

## Overview

This model implements bio-inspired **nonclassical receptive field (nCRF)** modulation mechanisms for contour detection. The approach mimics contextual modulation observed in the primary visual cortex (V1).

### Key Biological Inspiration

- **Center-Surround Organization**: Neurons in V1 have receptive fields with distinct center and surround regions
- **Contextual Modulation**: Neural responses are modulated by stimuli in the surround region
- **Normalization**: Feature normalization before modulation mimics divisive normalization in biological vision

## Architecture

### nCRF Module

```
Input Features
    ↓
┌─────────────────────────┐
│   Center Pathway (3×3)  │
│   Surround Pathway (7×7)│
└─────────────────────────┘
    ↓
Normalization
    ↓
Modulation (Concatenate + 1×1 Conv)
    ↓
Output Features
```

### Full Network

- **Stage 1**: 64 channels with nCRF modulation
- **Stage 2**: 128 channels with pooling and nCRF
- **Stage 3**: 256 channels with pooling and nCRF
- **Output**: Single-channel contour map

**Parameters**: ~1.5M

## Features

✅ Bio-inspired center-surround modulation  
✅ Feature normalization mechanism  
✅ Multi-scale feature extraction  
✅ Balanced loss for edge detection  
✅ Comprehensive evaluation metrics (ODS, OIS, AP)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --dataset ../datasets/HED_Small \
                --batch-size 4 \
                --epochs 50 \
                --lr 1e-4 \
                --size 320
```

**Arguments:**

- `--dataset`: Path to dataset directory
- `--batch-size`: Training batch size (default: 4)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--size`: Input image size (default: 320)
- `--checkpoint-dir`: Directory to save checkpoints (default: checkpoints)

### Evaluation

Use the `evaluate.ipynb` notebook for comprehensive evaluation:

- Load trained model
- Run inference on test set
- Calculate metrics (ODS, OIS, AP)
- Visualize results

Or run programmatically:

```python
from model import TangNet
import torch

# Load model
model = TangNet()
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    output = model(input_tensor)
```

## Dataset Structure

```
dataset/
├── train/
│   ├── images/
│   └── edges/
└── test/
    ├── images/
    └── edges/
```

## Results

Evaluation metrics on BSDS500 (as reported in paper):

| Metric | Score |
| ------ | ----- |
| ODS    | 0.806 |
| OIS    | 0.823 |
| AP     | 0.847 |

_Note: Results may vary based on training configuration and dataset._

## Biological Inspiration Details

### Nonclassical Receptive Field (nCRF)

In neuroscience, the classical receptive field (CRF) of a visual neuron is the region where visual stimuli can directly drive its response. The nonclassical receptive field (nCRF) refers to the surrounding region that doesn't directly activate the neuron but modulates its response to stimuli in the CRF.

**Key Mechanisms:**

1. **Surround Suppression**: Stimuli in the surround can suppress responses to center stimuli
2. **Contextual Enhancement**: Surround can also enhance responses based on stimulus configuration
3. **Normalization**: Divisive normalization balances responses across neurons

### Implementation in Deep Learning

This model translates these biological mechanisms into learnable operations:

- **Center Pathway**: 3×3 convolution captures local features (CRF)
- **Surround Pathway**: 7×7 depthwise convolution captures contextual information (nCRF)
- **Normalization**: L2 normalization mimics divisive normalization
- **Modulation**: Learned combination of normalized features

## File Structure

```
Tang-nCRF/
├── model.py              # nCRF model implementation
├── train.py              # Training script
├── evaluate.ipynb        # Evaluation notebook
├── requirements.txt      # Dependencies
├── README.md            # This file
├── checkpoints/         # Saved model weights
│   └── tang_ncrf_*/
│       ├── best_model.pth
│       ├── final_model.pth
│       └── history.json
└── outputs/             # Generated predictions
    └── *.png
```

## Citation

```bibtex
@article{tang2019learning,
  title={Learning Nonclassical Receptive Field Modulation for Contour Detection},
  author={Tang, Qiling and Sang, Nong and Liu, Haihua},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={1192--1203},
  year={2019},
  publisher={IEEE}
}
```

## References

- Original Paper: [IEEE Xplore](https://doi.org/10.1109/TIP.2019.2940690)
- Biological Inspiration: Contextual modulation in primary visual cortex (V1)
- Related Work: Classical and nonclassical receptive field models in neuroscience

## License

This implementation is for research and educational purposes.
