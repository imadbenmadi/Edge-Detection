# XYW-Net: Bio-inspired XYW Parallel Pathway Edge Detection Network

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Core Architecture](#core-architecture)
4. [Network Components](#network-components)
5. [Data Loading](#data-loading)
6. [Training Pipeline](#training-pipeline)
7. [Testing & Inference](#testing--inference)
8. [Advanced Techniques](#advanced-techniques)
9. [Datasets](#datasets)
10. [Configuration](#configuration)
11. [References](#references)

---

## Overview

**XYW-Net** is a bio-inspired edge detection neural network that mimics parallel visual pathways inspired by the receptive fields found in the human visual cortex. The network is designed to detect edges in images using three parallel processing pathways: **X (horizontal)**, **Y (vertical)**, and **W (wavelet/texture)** channels.

### Key Features:

-   **Bio-inspired Architecture**: Implements parallel pathways based on visual cortex mechanisms
-   **Multi-scale Processing**: Processes features at multiple scales using hierarchical encoding
-   **Advanced Convolutions**: Uses Pixel Difference Convolutions (PDC) for efficient edge detection
-   **Spatial Attention**: Implements Compact Spatial Attention Mechanism (CSAM)
-   **Dilated Convolutions**: Uses Compact Dilated Convolution Modules (CDCM) for expanded receptive fields
-   **Multi-dataset Support**: Compatible with BSDS500, PASCAL-VOC, PASCAL-Context, and NYUD datasets

---

## Project Structure

```
XYW-Net/
├── model.py                    # Core neural network architecture
├── data.py                     # Data loading and preprocessing
├── train.py                    # Training pipeline
├── test.py                     # Testing/inference pipeline
├── utils.py                    # Loss functions and utilities
├── transforms.py               # Image transformations
├── cfgs.yaml                   # Configuration file
├── README.md                   # Original README
└── Matlab_results/             # Evaluation results from Matlab tools
    ├── BIPED/
    ├── BSDS500/
    ├── Muticue/
    └── NYUD-RGBHHA/
```

---

## Core Architecture

### 1. Main Network Structure (Net Class)

```python
class Net(nn.Module):
    def __init__(self):
        self.encode = encode()      # Encoder: 4 encoding stages (s1, s2, s3, s4)
        self.decode = decode_rcf()  # Decoder: RCF-style progressive refinement

    def forward(self, x):
        end_point = self.encode(x)  # Returns [s1, s2, s3, s4]
        x = self.decode(end_point)  # Progressive upsampling and fusion
        return x
```

### 2. Encoder (4-Stage Hierarchical Processing)

The encoder progressively downsamples features through 4 stages using XYW parallel pathways:

#### Stage 1 (s1) - Initial Feature Extraction

-   Input: RGB image (3 channels)
-   Output: 30-channel feature map
-   **Process**:
    -   Conv2d: 3→30 channels, kernel=7, dilation=2
    -   XYW_S: Initial split into X, Y, W pathways
    -   XYW: Processing with parallel pathways
    -   XYW_E: End of stage, fusion of pathways
    -   Skip connection: original features added back

#### Stage 2 (s2) - First Downsampling

-   Input: 30-channel features from s1
-   Output: 60-channel feature map at 1/2 resolution
-   **Process**:
    -   MaxPool2d: 1/2 spatial resolution
    -   XYW_S: Split into pathways with stride
    -   XYW: Pathway processing
    -   XYW_E: Pathway fusion
    -   Skip connection: shortcut with Conv2d (30→60)

#### Stage 3 (s3) - Second Downsampling

-   Input: 60-channel features from s2
-   Output: 120-channel feature map at 1/4 resolution
-   **Process**:
    -   MaxPool2d: Further 1/2 reduction
    -   XYW_S: Split into pathways
    -   XYW: Pathway processing
    -   XYW_E: Fusion
    -   Skip connection: shortcut (60→120 channels)

#### Stage 4 (s4) - Third Downsampling

-   Input: 120-channel features from s3
-   Output: 120-channel feature map at 1/8 resolution
-   **Process**:
    -   MaxPool2d: Final 1/2 reduction
    -   XYW_S → XYW → XYW_E: Parallel pathway processing
    -   Skip connection: Identity mapping

### 3. Decoder (RCF-Style Progressive Refinement)

```python
class decode_rcf(nn.Module):
    def forward(self, x):  # x = [s1, s2, s3, s4]
        s3 = self.f43(x[2], x[3])  # Fuse s3 and s4, upsample 2x
        s2 = self.f32(x[1], s3)    # Fuse s2 and s3, upsample 2x
        s1 = self.f21(x[0], s2)    # Fuse s1 and s2, upsample 2x
        x = self.f(s1)             # Final 1x1 conv to single channel
        return x.sigmoid()         # Output: probability map [0, 1]
```

The decoder progressively upsamples and fuses features using `Refine_block2_1` modules which combine:

-   **Adaptive convolutions** (adap_conv) for feature refinement
-   **Bilinear upsampling** for smooth spatial interpolation

---

## Network Components

### 1. XYW Pathway System

The core innovation of XYW-Net is the parallel processing of three distinct pathways inspired by visual cortex mechanisms:

#### XYW_S (Start) - Initial Split

```python
class XYW_S(nn.Module):
    def forward(self, x):
        xc = self.x_c(x)   # X pathway: horizontal differences
        yc = self.y_c(x)   # Y pathway: vertical differences
        w = self.w(x)      # W pathway: wavelet/texture
        return xc, yc, w
```

#### XYW (Middle) - Parallel Processing

```python
class XYW(nn.Module):
    def forward(self, xc, yc, w):
        xc = self.x_c(xc)  # Further X processing
        yc = self.y_c(yc)  # Further Y processing
        w = self.w(w)      # Further W processing
        return xc, yc, w
```

#### XYW_E (End) - Fusion

```python
class XYW_E(nn.Module):
    def forward(self, xc, yc, w):
        return xc + yc + w  # Sum fusion of all pathways
```

### 2. X Pathway (Horizontal Receptive Field)

```python
class Xc1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.Xcenter = Conv2d(1x1)          # Center response
        self.Xsurround = Conv2d(3x1)        # Surround response

    def forward(self, input):
        xcenter = self.Xcenter(input)
        xsurround = self.Xsurround(input)
        return xsurround - xcenter          # Center-surround difference
```

-   **Purpose**: Detects horizontal edges
-   **Mechanism**: Difference between local surround (3×1 kernel) and center (1×1 kernel)
-   **Biological Inspiration**: Mimics X-cells in the retina

### 3. Y Pathway (Vertical Receptive Field)

```python
class Yc1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.Ycenter = Conv2d(1x1)          # Center response
        self.Ysurround = Conv2d(5x5, dilation=2)  # Larger surround

    def forward(self, input):
        ycenter = self.Ycenter(input)
        ysurround = self.Ysurround(input)
        return ysurround - ycenter          # Center-surround difference
```

-   **Purpose**: Detects vertical and multi-scale edges
-   **Mechanism**: Larger receptive field (5×5 with dilation=2)
-   **Biological Inspiration**: Mimics Y-cells in the retina with larger receptive fields

### 4. W Pathway (Wavelet/Texture)

```python
class W(nn.Module):
    def forward(self, x):
        h = self.relu(self.h(x))  # Horizontal convolution: (1, 3)
        h = self.convh_1(h)

        v = self.relu(self.v(h))  # Vertical convolution: (3, 1)
        v = self.convv_1(v)

        return v
```

-   **Purpose**: Captures texture and wavelet-like patterns
-   **Mechanism**: Sequential application of horizontal and vertical 1D convolutions
-   **Effect**: Detects higher-order patterns and textures

### 5. Pixel Difference Convolution (PDC)

PDC is an advanced convolution technique that replaces standard convolution with difference-based operations:

```python
def createPDCFunc(PDC_type):
    if PDC_type == 'cd':  # Center Difference
        # y = ∑(wi * xi) - Xj * ∑(wi)  [Formula 7]
        y = F.conv2d(x, weights, ...) - F.conv2d(x, weights_c, ...)

    elif PDC_type == 'sd':  # Surround Difference
        # Center position gets: w5 - 2*(w1+w2+w3+w4+w6+w7+w8+w9)
        buffer[:, :, [4]] = weights[:, :, [4]] - 2 * weights[:, :, other_positions]

    elif PDC_type == 'ad':  # Angular Difference
        # Clockwise difference: (w1 - w4)*x1 + (w2 - w1)*x2 + ...
        weights_conv = (weights - weights_rotated)

    # ... other PDC types: rd, p2d, 2sd, 2cd
```

**Benefits**:

-   More efficient edge detection (differencing amplifies edges)
-   Better boundary preservation
-   Reduced computational cost through derivative-like operations

### 6. Compact Spatial Attention Mechanism (CSAM)

```python
class CSAM(nn.Module):
    def forward(self, x):
        y = self.relu(x)              # ReLU activation
        y = self.conv1_1(y)           # 1×1 conv: C → 4
        y = self.conv3_3(y)           # 3×3 conv: 4 → 1
        y = self.sigmoid(y)           # Spatial attention map [0, 1]
        return x * y                  # Element-wise multiplication
```

**Purpose**:

-   Generates spatial attention weights for each location
-   Emphasizes important edge regions
-   Suppresses background/irrelevant areas
-   Computationally efficient (compact version)

### 7. Compact Dilated Convolution Module (CDCM)

```python
class CDCM(nn.Module):
    def forward(self, x):
        x = self.relu(x)
        x = self.conv1(x)           # 1×1: reduce channels
        x1 = self.conv2_1(x)        # 3×3, dilation=5, padding=5
        x2 = self.conv2_2(x)        # 3×3, dilation=7, padding=7
        x3 = self.conv2_3(x)        # 3×3, dilation=9, padding=9
        x4 = self.conv2_4(x)        # 3×3, dilation=11, padding=11
        return x1 + x2 + x3 + x4    # Sum of different receptive fields
```

**Purpose**:

-   Expands receptive field without downsampling
-   Multi-scale feature extraction in parallel
-   Different dilation rates capture features at different scales
-   Output size preserved

### 8. Refine Block (Refinement Module)

```python
class Refine_block2_1(nn.Module):
    def forward(self, *input):
        x1 = self.pre_conv1(input[0])   # Process first feature map
        x2 = self.pre_conv2(input[1])   # Process second feature map

        # Upsample x2 using deconvolution with bilinear weights
        x2 = F.conv_transpose2d(x2, self.deconv_weight, stride=factor, ...)

        return x1 + x2                   # Skip connection fusion
```

---

## Data Loading

### 1. Dataset Classes

#### BSDS_500 Dataset

```python
class BSDS_500(Dataset):
    # Supports training and testing modes
    # Training: Loads image-label pairs with optional PASCAL-VOC augmentation
    # Testing: Loads test images

    def __getitem__(self, item):
        image = Image.open(self.im_list[item])
        if self.flag == 'train':
            label = np.array(Image.open(self.gt_list[item]).convert('L'))
            label = Image.fromarray(label.astype(np.float32) / np.max(label))
        return {'images': image, 'labels': label}
```

#### PASCAL_VOC12 Dataset

```python
class PASCAL_VOC12(Dataset):
    # Loads PASCAL VOC 2012 boundary detection dataset
    # Supports train/test splits with automatic normalization
```

#### PASCAL_Context Dataset

```python
class PASCAL_Context(Dataset):
    # Loads PASCAL Context dataset for contour detection
    # More challenging with contextual boundaries
```

#### NYUD Dataset

```python
class NYUD(Dataset):
    # RGB-D edge detection on NYUD dataset
    # Supports RGB and HHA modalities
```

### 2. Data Augmentation (transforms.py)

```python
class Compose:        # Chain multiple transforms
class ToTensor:       # Convert PIL Image to tensor
class RandomScale:    # Random scaling augmentation
class RandomCrop:     # Random cropping
class RandomRotationCrop:  # Rotation with crop
class Normalize:       # ImageNet normalization
```

**Example Transform Pipeline**:

```python
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

---

## Training Pipeline

### 1. Training Setup (train.py)

```python
# Load configuration
cfgs = yaml.load('./cfgs.yaml')

# Create dataset and dataloader
dataset = BSDS_500(root=cfgs['dataset'], VOC=True, transform=trans)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=cfgs['batch_size'],
    shuffle=True,
    num_workers=6
)

# Initialize network and loss
net = model.Net().train()
criterion = utils.Cross_Entropy()

# Setup optimizer (Adam or SGD)
if cfgs['method'] == 'Adam':
    optimizer = torch.optim.Adam(
        [{'params': net.parameters()},
         {'params': criterion.parameters()}],
        lr=cfgs['lr']
    )

# Move to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
criterion.to(device)
```

### 2. Loss Functions (utils.py)

#### Cross Entropy Loss

```python
class Cross_Entropy(nn.Module):
    def forward(self, pred, labels, side_output=None):
        # Main loss: cross-entropy per image
        total_loss = cross_entropy_per_image(pred, labels) + \
                     0.00 * 0.1 * dice_loss_per_image(pred, labels)

        # Optional side outputs for multi-scale supervision
        if side_output is not None:
            for s in side_output:
                total_loss += cross_entropy_per_image(s, labels) / len(side_output)

        return total_loss, (1-pred_pos).abs(), pred_neg
```

#### Cross Entropy with Weighted Loss

```python
def cross_entropy_with_weight(logits, labels):
    eps = 1e-6
    pred_pos = logits[labels > 0].clamp(eps, 1.0-eps)
    pred_neg = logits[labels == 0].clamp(eps, 1.0-eps)
    w_annotation = labels[labels > 0]

    # Weighted positive + unweighted negative
    cross_entropy = (-pred_pos.log() * w_annotation).mean() + \
                    (-(1.0 - pred_neg).log()).mean()
    return cross_entropy
```

#### Dice Loss (Alternative)

```python
def dice_loss_per_image(logits, labels):
    # Dice coefficient: 2*|A∩B| / (|A| + |B|)
    # Useful for imbalanced edge/background pixels
```

### 3. Training Loop

```python
for epoch in range(cfgs['max_epoch']):
    for i, data in enumerate(dataloader):
        # Forward pass
        images = data['images'].to(device)
        labels = data['labels'].to(device)
        prediction = net(images)

        # Compute loss
        loss, dp, dn = criterion(prediction, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Learning rate decay
        if epoch == 5:
            optimizer = torch.optim.Adam(..., lr=cfgs['lr']*0.1)

        # Validation visualization every 100 batches
        if i % 100 == 99:
            prediction_np = prediction.cpu().detach().numpy()
            cv2.imwrite('./validation/' + str(j) + '.png', prediction_np[j] * 255)

            # Histogram of positive/negative predictions
            ax1 = plt.subplot(1, 2, 1)
            ax1.hist(dp.cpu().detach().numpy(), bins=100)
            ax2 = plt.subplot(1, 2, 2)
            ax2.hist(dn.cpu().detach().numpy(), bins=100)
            plt.savefig('./validation/test' + str(epoch) + '.png')
```

### 4. Key Training Parameters (cfgs.yaml)

```yaml
batch_size: 1 # Small batch for memory efficiency
max_epoch: 6 # Total training epochs
decay_rate: 0.1 # LR decay rate
decay_steps: 8 # Decay every N epochs

method: Adam # Optimizer choice
lr: 1.0e-3 # Learning rate
momentum: 0.9 # SGD momentum
weight_decay: 2.0e-4 # L2 regularization

save_name: _XYW_ep7_lrd5.pth # Model checkpoint name
```

---

## Testing & Inference

### 1. Testing Pipeline (test.py)

```python
# Load trained model
net = model.Net().eval()
net.load_state_dict(torch.load('./xyw.pth', map_location='cuda:0'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# Prepare dataset
dataset = BSDS_500(root=cfgs['dataset'], flag='test', VOC=False, transform=trans)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Inference on each image
t_time = 0
for i, data in enumerate(dataloader):
    with torch.no_grad():
        images = data['images'].to(device)

        # Forward pass
        start_time = time.time()
        prediction = net(images)[0].cpu().detach().numpy().squeeze()
        duration = time.time() - start_time

        # Save output
        cv2.imwrite('./test/1X/' + name_list[i] + '.png', prediction * 255)

        # Track timing
        t_time += duration
        print(f'Processed image {i}/{length}')

print(f'Average time: {t_time/length:.3f}s, Average FPS: {length/t_time:.3f}')
```

### 2. Output Format

-   **Output**: Single-channel probability map (0-1 range)
-   **Saved as**: PNG with values scaled to 0-255
-   **Metrics**: FPS and average inference time

---

## Advanced Techniques

### 1. Multi-Scale Supervision

The network can be trained with side outputs at multiple scales:

-   Feature maps from intermediate stages (s1, s2, s3, s4)
-   Each scale supervises with the same ground truth
-   Weighted contribution to total loss
-   Improves gradient flow and multi-scale edge detection

### 2. Bilinear Upsampling Weights

```python
def bilinear_upsample_weights(factor, number_of_classes):
    # Pre-computed bilinear interpolation kernels
    # Used in transposed convolution for smooth upsampling
    # Avoids checkerboard artifacts in upsampling
```

### 3. Learnable Weight Parameters

```python
class adap_conv(nn.Module):
    def forward(self, x):
        x = self.conv(x) * self.weight.sigmoid()
        # Learnable weight gating for each pathway
        # Allows network to learn importance of each component
```

### 4. Center-Surround Mechanism

The X, Y, W pathways implement biological center-surround receptive fields:

-   **Center**: Small kernel (1×1) - local response
-   **Surround**: Larger kernel (3×1, 5×5, etc.) - contextual response
-   **Output**: Difference amplifies edges and suppresses flat regions

### 5. Group Convolutions

Used in X, Y, W pathways to reduce parameters:

```python
nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, groups=in_channels)
# Depthwise convolution: each channel processed independently
```

---

## Datasets

### Supported Datasets

1. **BSDS500** (Berkeley Segmentation Dataset)

    - 200 training + 100 test images
    - Multiple ground truth annotations per image
    - Standard benchmark for edge detection
    - Download: `http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz`

2. **PASCAL VOC** (Visual Object Classes)

    - 10,000+ training images with boundary annotations
    - Real-world scene complexity
    - Download: `http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz`

3. **PASCAL Context**

    - Contextual boundary detection
    - More challenging than PASCAL VOC
    - Fine-grained boundary annotations

4. **NYUD** (NYU Depth Dataset)

    - RGB-D edge detection task
    - Supports both RGB and HHA (Horizontal disparity, Height, Angle) modalities
    - Indoor scene dataset
    - Download: `http://mftp.mmcheng.net/liuyun/rcf/data/NYUD.tar.gz`

5. **Multicue Dataset**

    - Multi-cue edge detection
    - High-quality annotations
    - Download: `https://drive.google.com/file/d/1-tyt_KyzlYc9APafdh5mHJzh2K_F2hM8/view?usp=sharing`

6. **BIPED** (Boundary Detection Paired Edge Dataset)
    - Paired high-quality edges
    - Download: `https://drive.google.com/drive/folders/1lZuvJxL4dvhVGgiITmZsjUJPBBrFI_bM`

### Evaluation Tools

1. **Edge Detection Evaluation** (pdollar/edges)

    - Computes ODS (Optimal Dataset Scale) F-measure
    - Computes OIS (Optimal Image Scale) F-measure
    - Repository: `https://github.com/pdollar/edges`

2. **PR Curve Plotting** (MCG-NKU)
    - Generates Precision-Recall curves
    - Visualization of boundary detection performance
    - Repository: `https://github.com/MCG-NKU/plot-edge-pr-curves`

---

## Configuration

### Key Configuration Parameters (cfgs.yaml)

| Parameter      | Value              | Description                          |
| -------------- | ------------------ | ------------------------------------ |
| `batch_size`   | 1                  | Training batch size (memory-limited) |
| `max_epoch`    | 6                  | Total training epochs                |
| `decay_rate`   | 0.1                | Learning rate decay multiplier       |
| `decay_steps`  | 8                  | Epochs between decay                 |
| `method`       | Adam               | Optimizer (Adam or SGD)              |
| `lr`           | 1.0e-3             | Initial learning rate                |
| `momentum`     | 0.9                | SGD momentum (if applicable)         |
| `weight_decay` | 2.0e-4             | L2 regularization factor             |
| `save_name`    | \_XYW_ep7_lrd5.pth | Checkpoint filename                  |

### Dataset Configuration

```yaml
dataset:
    BSDS: F:\matlab_tool\Dataset\ori_data\HED-BSDS
    BSDS-VOC: F:\matlab_tool\Dataset\new_data\BSDS500
    PASCAL-VOC12: D:\DataSet\VOCdevkit\VOC2012_trainval
    PASCAL-Context: D:\DataSet\PASCAL_Context
```

---

## References

The XYW-Net implementation references:

1. **LMGCN** (Local Multi-scale Guided CNN)

    - Repository: `https://github.com/cimerainbow/LMGCN`
    - Multi-scale feature learning for edge detection

2. **PidiNet** (Pixel Difference Network)

    - Repository: `https://github.com/zhuoinoulu/pidinet`
    - Efficient edge detection with pixel difference convolutions
    - Inspired PDC implementations in XYW-Net

3. **RCF** (Richer Convolutional Features)

    - Progressive feature refinement with skip connections
    - Multi-scale prediction strategy

4. **Biological Vision Research**
    - X and Y cells in retina (different receptive field sizes)
    - Center-surround receptive field mechanisms
    - Parallel pathways in visual cortex

### Related Concepts

-   **Pixel Difference Convolution (PDC)**: Replaces standard convolution with difference-based operations
-   **Center-Surround Receptive Fields**: Biological mechanism for edge and texture detection
-   **Multi-scale Processing**: Hierarchical feature extraction at different resolutions
-   **Spatial Attention Mechanisms**: Adaptive weighting of spatial regions
-   **Dilated Convolutions**: Expand receptive field without downsampling

---

## Usage Quick Start

### Training

```bash
python train.py
```

-   Loads configuration from `cfgs.yaml`
-   Uses BSDS500+PASCAL-VOC dataset
-   Saves checkpoints to `./checkpoint/`
-   Validation outputs to `./validation/`

### Testing

```bash
python test.py
```

-   Loads trained model from `./xyw.pth`
-   Evaluates on BSDS500 test set
-   Saves predictions to `./test/1X/`
-   Reports average inference time and FPS

### Inference on Custom Image

```python
import torch
import model
from PIL import Image
import transforms

# Load model
net = model.Net().eval()
net.load_state_dict(torch.load('./checkpoint/model.pth'))

# Prepare image
image = Image.open('image.jpg')
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
sample = {'images': image, 'labels': image}
sample = trans(sample)

# Inference
with torch.no_grad():
    pred = net(sample['images'].unsqueeze(0))
```

---

## Summary

XYW-Net is a sophisticated edge detection network that combines:

-   **Bio-inspired parallel pathways** (X, Y, W) mimicking visual cortex
-   **Advanced convolutions** (PDC, dilated, group convolutions)
-   **Attention mechanisms** (CSAM) for spatial refinement
-   **Multi-scale processing** (hierarchical encoding-decoding)
-   **Robust training** with weighted cross-entropy loss
-   **Comprehensive evaluation** on multiple benchmark datasets

The architecture demonstrates how biological principles can be effectively translated into deep learning models for computer vision tasks, achieving state-of-the-art edge detection performance.
