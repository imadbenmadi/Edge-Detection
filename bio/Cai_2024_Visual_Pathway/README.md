# Cai et al. (2024) - Visual Pathway Information Transfer Network

Implementation of **"Image Contour Detection Based on Visual Pathway Information Transfer Mechanism"** by Pingping Cai et al., published in Neural Processing Letters, 2024.

**DOI**: [10.1007/s11063-024-11486-3](https://doi.org/10.1007/s11063-024-11486-3)

## 📄 Paper Overview

This paper proposes a bio-inspired contour detection method based on visual pathway information transfer mechanisms with:

1. **Double Receptive Fields**: Weighted combination of center-surround receptive fields
2. **Double Stream Fusion**: Magno/Parvo pathway integration
3. **Adaptive Response Adjustment**: Context-aware modulation
4. **Visual Hierarchy**: LGN → V1 → V2 → V4 processing

## 🏗️ Architecture

```
Input Image (RGB)
    ↓
LGN: Double Receptive Fields
    ├─ Center RF (3×3)
    └─ Surround RF (7×7)
    ↓
V1: Primary Visual Cortex
    ├─ 4 Orientation Filters
    └─ Adaptive Response Module
    ↓
V2: Secondary Visual Cortex
    ├─ Magno Stream (5×5)
    ├─ Parvo Stream (3×3)
    └─ Double Stream Fusion
    ↓
V4: Higher-level Processing
    └─ Shape Integration
    ↓
Decoder (Multi-scale Fusion)
    ↓
Edge Probability Map
```

## 🔬 Key Components

### 1. Double Receptive Field (LGN)

```python
class DoubleReceptiveField:
    - Center RF: Small, detailed processing
    - Surround RF: Large, contextual processing
    - Adaptive weighting based on input statistics
```

### 2. Double Stream Module (V2)

```python
class DoubleStreamModule:
    - Magno Stream: Fast, motion-sensitive (5×5)
    - Parvo Stream: Detailed, color-sensitive (3×3)
    - Attention-based fusion
```

### 3. Adaptive Response Module

```python
class AdaptiveResponseModule:
    - Global context pooling
    - Channel-wise modulation
    - Sigmoid gating mechanism
```

### 4. Visual Cortex Blocks

- **V1Block**: 4-orientation filters + adaptive response
- **V2Block**: Double stream + adaptive response
- **V4Block**: High-level shape integration + adaptive response

## 📊 Model Details

- **Parameters**: ~1.8M parameters (lightweight)
- **Input**: RGB images (320×320)
- **Output**: Edge probability map (0-1)
- **Deep Supervision**: 4 side outputs during training
- **Loss**: Weighted Binary Cross-Entropy

## 🚀 Usage

### Training

```bash
python train.py --data_root ../datasets/HED_Small \
                --epochs 20 \
                --batch_size 4 \
                --lr 1e-3 \
                --save_dir checkpoints
```

**Arguments**:

- `--data_root`: Path to dataset (default: `../datasets/HED_Small`)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 1e-3)
- `--save_dir`: Checkpoint save directory (default: `checkpoints`)
- `--device`: Device (default: auto-detect CUDA/CPU)

### Evaluation

Run the Jupyter notebook:

```bash
jupyter notebook evaluate.ipynb
```

Or use the model directly:

```python
from model import VisualPathwayNet
import torch

model = VisualPathwayNet(in_channels=3)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    edge_map = model(image_tensor)
```

## 📁 Directory Structure

```
Cai_2024_Visual_Pathway/
├── model.py              # Network architecture
├── train.py              # Training script
├── evaluate.ipynb        # Evaluation notebook
├── README.md             # This file
├── checkpoints/          # Saved model weights
│   └── best_model.pth
└── outputs/              # Results and visualizations
    ├── cai_2024_results.json
    └── visualization.png
```

## 🎯 Performance

Evaluated on **HED_Small** test set (20 images):

| Metric  | Score                           |
| ------- | ------------------------------- |
| **ODS** | To be determined after training |
| **OIS** | To be determined after training |
| **AP**  | To be determined after training |

_Note: Train the model first using `train.py`, then run `evaluate.ipynb`_

## 🔧 Requirements

```bash
pip install torch torchvision opencv-python numpy tqdm scikit-learn matplotlib
```

Or install from project root:

```bash
pip install -r requirements.txt
```

## 🧠 Bio-Inspired Features

### LGN (Lateral Geniculate Nucleus)

- **Biological**: Center-surround receptive fields in retinal ganglion cells
- **Implementation**: Dual convolutional paths (3×3 and 7×7)
- **Adaptive**: Weight generation based on global statistics

### V1 (Primary Visual Cortex)

- **Biological**: Simple cells with orientation selectivity
- **Implementation**: 4 orientation filters (0°, 45°, 90°, 135°)
- **Modulation**: Adaptive response based on local context

### V2 (Secondary Visual Cortex)

- **Biological**: Curvature detection and motion processing
- **Implementation**: Dual stream (Magno/Parvo pathways)
- **Fusion**: Attention-weighted combination

### V4 (Visual Area 4)

- **Biological**: Complex shape and contour integration
- **Implementation**: Higher-level feature extraction
- **Refinement**: Adaptive response modulation

### Multi-level Processing

- **Biological**: Hierarchical information flow in visual cortex
- **Implementation**: Skip connections + multi-scale decoder
- **Supervision**: Deep supervision with side outputs

## 📝 Citation

```bibtex
@article{cai2024image,
  title={Image Contour Detection Based on Visual Pathway Information Transfer Mechanism},
  author={Cai, Pingping and others},
  journal={Neural Processing Letters},
  year={2024},
  doi={10.1007/s11063-024-11486-3}
}
```

## 🔗 Related Work

- **XYW-Net**: X/Y/W retinal pathways
- **LVP-Net**: Lateral visual pathway network
- **Bio-IFN**: Interactive feedback network

See `../bio DL/` for other bio-inspired deep learning models.

## 📄 License

This implementation is for research purposes. Please cite the original paper if used.

---

**Keywords**: Contour detection, Adaptive adjustment, Double receptive fields, Double stream fusion, Visual pathway, Bio-inspired, Deep learning
