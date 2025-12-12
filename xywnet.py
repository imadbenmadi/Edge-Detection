"""
XYW-Net: Edge Detection Model
Complete implementation with training and evaluation pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from sklearn.metrics import precision_recall_curve, average_precision_score
import os

# ============================================================
# DEVICE SETUP
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# DATASET VERIFICATION
# ============================================================
def verify_edges(root_dir, split="train"):
    """Verify dataset integrity"""
    root_dir = Path(root_dir)
    img_dir = root_dir / split / "images"
    edge_dir = root_dir / split / "edges"

    image_files = sorted(list(img_dir.glob("*.png")))

    missing_edges = []
    unreadable_edges = []
    empty_edges = []
    size_mismatch = []

    print(f"\nVerifying split: {split}")
    print("=" * 60)

    for img_path in tqdm(image_files, desc=f"Checking {split}", unit="img"):
        edge_path = edge_dir / img_path.name

        if not edge_path.exists():
            missing_edges.append(img_path.name)
            continue

        edge = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
        if edge is None:
            unreadable_edges.append(img_path.name)
            continue

        if np.sum(edge) == 0:
            empty_edges.append(img_path.name)

        img = cv2.imread(str(img_path))
        if img is None or edge.shape != img.shape[:2]:
            size_mismatch.append(img_path.name)

    # Print verification report
    print(f"\nVerification Report for split: {split}")
    print("=" * 60)
    print(f"Total images checked: {len(image_files)}")
    print(f"Missing edge files: {len(missing_edges)}")
    print(f"Unreadable edge files: {len(unreadable_edges)}")
    print(f"Empty edge maps: {len(empty_edges)}")
    print(f"Size mismatches: {len(size_mismatch)}")

    if missing_edges:
        print("\nMissing edge files (first 10):")
        print(missing_edges[:10])

    if unreadable_edges:
        print("\nUnreadable edge files (first 10):")
        print(unreadable_edges[:10])

    if empty_edges:
        print("\nEmpty edge maps (first 10):")
        print(empty_edges[:10])

    if size_mismatch:
        print("\nSize mismatches (first 10):")
        print(size_mismatch[:10])

    if not (missing_edges or unreadable_edges or empty_edges or size_mismatch):
        print("\nAll images have valid, non-empty, correctly-sized edge maps.")

# ============================================================
# DATASET LOADER
# ============================================================
class ProcessedDataset(Dataset):
    """Load processed PNG images and edge maps"""

    def __init__(self, root_dir, split='train'):
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_dir = self.root_dir / split / 'images'
        self.edge_dir = self.root_dir / split / 'edges'
        self.samples = sorted(list(self.img_dir.glob('*.png')))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        edge_path = self.edge_dir / img_path.name

        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        # Load edge map
        edge = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
        edge = edge.astype(np.float32) / 255.0

        # Convert to tensors
        img = torch.from_numpy(img).permute(2, 0, 1)
        edge = torch.from_numpy(edge).unsqueeze(0)

        return {'images': img, 'labels': edge, 'filename': img_path.stem}

# ============================================================
# PDC CONVOLUTION (Pixel Difference Convolution)
# ============================================================
def createPDCFunc(PDC_type):
    assert PDC_type in ['cv', '2sd']
    
    if PDC_type == 'cv':
        return F.conv2d
    
    if PDC_type == '2sd':
        # Pixel difference convolution (paper uses this for ELC)
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert weights.size(2) == 3 and weights.size(3) == 3
            shape = weights.shape
            offset = weights[:, :, 
                             [1,1,1,0,0,0,2,2,2],
                             [0,1,2,0,1,2,0,1,2]
                             ].view(shape)
            
            diff_weights = weights - offset
            return F.conv2d(x, diff_weights, bias, stride, padding, dilation, groups)
        
        return func

class Conv2d(nn.Module):
    """PDC-enabled conv layer"""
    def __init__(self, pdc_func='cv', in_channels=1, out_channels=1, kernel_size=3,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.pdc = createPDCFunc(pdc_func)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.pdc(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

# ============================================================
# XYW COMPONENTS (ENCODER)
# ============================================================
class Xc1x1(nn.Module):
    """X: small RF center-surround"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.center = nn.Conv2d(in_channels, out_channels, 1)
        self.surround = nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        c = self.relu(self.center(x))
        s = self.relu(self.surround(x))
        s = self.proj(s)
        return s - c

class Yc1x1(nn.Module):
    """Y: large RF dilated center-surround"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.center = nn.Conv2d(in_channels, out_channels, 1)
        self.surround = nn.Conv2d(in_channels, out_channels, 5, padding=4, dilation=2, groups=in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        c = self.relu(self.center(x))
        s = self.relu(self.surround(x))
        s = self.proj(s)
        return s - c

class W(nn.Module):
    """W: directional horizontal + vertical pathway"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.h = nn.Conv2d(in_ch, in_ch, (1,3), padding=(0,1), groups=in_ch)
        self.v = nn.Conv2d(in_ch, in_ch, (3,1), padding=(1,0), groups=in_ch)
        self.proj1 = nn.Conv2d(in_ch, in_ch, 1)
        self.proj2 = nn.Conv2d(in_ch, out_ch, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.relu(self.h(x))
        h = self.proj1(h)
        v = self.relu(self.v(h))
        return self.proj2(v)

class XYW_S(nn.Module):
    """Start block"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.x = Xc1x1(in_ch, out_ch)
        self.y = Yc1x1(in_ch, out_ch)
        self.w = W(in_ch, out_ch)

    def forward(self, x):
        return self.x(x), self.y(x), self.w(x)

class XYW(nn.Module):
    """Middle block"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.x = Xc1x1(in_ch, out_ch)
        self.y = Yc1x1(in_ch, out_ch)
        self.w = W(in_ch, out_ch)

    def forward(self, xc, yc, w):
        return self.x(xc), self.y(yc), self.w(w)

class XYW_E(nn.Module):
    """End block: fusion of X+Y+W"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.x = Xc1x1(in_ch, out_ch)
        self.y = Yc1x1(in_ch, out_ch)
        self.w = W(in_ch, out_ch)

    def forward(self, xc, yc, w):
        return self.x(xc) + self.y(yc) + self.w(w)

# ============================================================
# ENCODER STAGES (4 RESOLUTION LEVELS)
# ============================================================
class s1(nn.Module):
    def __init__(self, ch=30):
        super().__init__()
        self.stem = nn.Conv2d(3, ch, 7, padding=6, dilation=2)
        self.relu = nn.ReLU(inplace=True)
        self.b1 = XYW_S(ch, ch)
        self.b2 = XYW(ch, ch)
        self.b3 = XYW_E(ch, ch)

    def forward(self, x):
        t = self.relu(self.stem(x))
        xc, yc, w = self.b1(t)
        xc, yc, w = self.b2(xc, yc, w)
        out = self.b3(xc, yc, w)
        return out + t

class s2(nn.Module):
    def __init__(self, ch=60):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.b1 = XYW_S(ch//2, ch)
        self.b2 = XYW(ch, ch)
        self.b3 = XYW_E(ch, ch)
        self.short = nn.Conv2d(ch//2, ch, 1)

    def forward(self, x):
        x = self.pool(x)
        xc, yc, w = self.b1(x)
        xc, yc, w = self.b2(xc, yc, w)
        out = self.b3(xc, yc, w)
        return out + self.short(x)

class s3(nn.Module):
    def __init__(self, ch=120):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.b1 = XYW_S(ch//2, ch)
        self.b2 = XYW(ch, ch)
        self.b3 = XYW_E(ch, ch)
        self.short = nn.Conv2d(ch//2, ch, 1)

    def forward(self, x):
        x = self.pool(x)
        sc = self.short(x)
        xc, yc, w = self.b1(x)
        xc, yc, w = self.b2(xc, yc, w)
        return self.b3(xc, yc, w) + sc

class s4(nn.Module):
    def __init__(self, ch=120):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.b1 = XYW_S(ch, ch)
        self.b2 = XYW(ch, ch)
        self.b3 = XYW_E(ch, ch)
        self.short = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        x = self.pool(x)
        sc = self.short(x)
        xc, yc, w = self.b1(x)
        xc, yc, w = self.b2(xc, yc, w)
        return self.b3(xc, yc, w) + sc

class encode(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = s1()
        self.s2 = s2()
        self.s3 = s3()
        self.s4 = s4()

    def forward(self, x):
        s1_out = self.s1(x)
        s2_out = self.s2(s1_out)
        s3_out = self.s3(s2_out)
        s4_out = self.s4(s3_out)
        return s1_out, s2_out, s3_out, s4_out

# ============================================================
# UPSAMPLE HELPERS 
# ============================================================
def upsample_filt(size):
    factor = (size + 1) // 2
    center = factor - 1 if size % 2 == 1 else factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

def bilinear_upsample_weights(factor, C):
    fs = 2 * factor - factor % 2
    w = np.zeros((C, C, fs, fs), dtype=np.float32)
    kern = upsample_filt(fs)
    for i in range(C):
        w[i, i] = kern
    return torch.Tensor(w)

class Refine_block2_1(nn.Module):
    """Refinement block used in ITM"""
    def __init__(self, in_ch, out_ch, factor):
        super().__init__()
        self.pre1 = Conv2d('2sd', in_ch[0], out_ch, 3, padding=1)
        self.pre2 = Conv2d('2sd', in_ch[1], out_ch, 3, padding=1)
        self.deconv_w = nn.Parameter(bilinear_upsample_weights(factor, out_ch), requires_grad=False)
        self.factor = factor

    def forward(self, x_high, x_low):
        h = self.pre1(x_high)
        l = self.pre2(x_low)
        l = F.conv_transpose2d(l, self.deconv_w, stride=self.factor,
                               padding=int(self.factor/2),
                               output_padding=(h.size(2) - l.size(2)*self.factor,
                                               h.size(3) - l.size(3)*self.factor))
        return h + l

# ============================================================
# ELC BLOCK (Edge Localization Convolution)
# ============================================================
class ELCBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.pdc = Conv2d('2sd', ch, ch, 3, padding=1)
        self.norm = nn.InstanceNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        x = self.pdc(x)
        x = self.norm(x)
        x = self.relu(x)
        return torch.sigmoid(self.out(x))

# ============================================================
# ITM + ELC DECODER (XYW-Net)
# ============================================================
class decode_xyw(nn.Module):
    def __init__(self):
        super().__init__()
        self.f43 = Refine_block2_1((120,120), 64, 2)
        self.f32 = Refine_block2_1((60,64),   48, 2)
        self.f21 = Refine_block2_1((30,48),   32, 2)
        self.elc = ELCBlock(32)

    def forward(self, endpoints):
        s1, s2, s3, s4 = endpoints
        x3 = self.f43(s3, s4)
        x2 = self.f32(s2, x3)
        x1 = self.f21(s1, x2)
        return self.elc(x1)

# ============================================================
# FULL XYW-NET (ENCODER + DECODER)
# ============================================================
class XYWNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encode()
        self.decoder = decode_xyw()

    def forward(self, x):
        endpoints = self.encoder(x)
        return self.decoder(endpoints)

    def forward_with_stages(self, x):
        """Forward pass returning intermediate stage outputs for visualization"""
        endpoints = self.encoder(x)
        final = self.decoder(endpoints)
        return final, endpoints

# ============================================================
# LOSS FUNCTION AND METRICS
# ============================================================
class EdgeLoss(nn.Module):
    """Weighted cross-entropy loss for edge detection"""
    def __init__(self):
        super(EdgeLoss, self).__init__()
    
    def forward(self, pred, label):
        pred_flat = pred.view(-1)
        label_flat = label.view(-1)
        eps = 1e-6
        
        # Positive and negative pixels
        pos_mask = label_flat > 0
        neg_mask = label_flat == 0
        
        pred_pos = pred_flat[pos_mask].clamp(eps, 1.0 - eps)
        pred_neg = pred_flat[neg_mask].clamp(eps, 1.0 - eps)
        
        # Weighted by annotation strength
        w_pos = label_flat[pos_mask]
        
        if len(pred_pos) > 0 and len(pred_neg) > 0:
            loss = (-pred_pos.log() * w_pos).mean() + (-(1.0 - pred_neg).log()).mean()
        elif len(pred_pos) > 0:
            loss = (-pred_pos.log() * w_pos).mean()
        else:
            loss = (-(1.0 - pred_neg).log()).mean()
        
        return loss

def compute_ods_ois_ap(preds, labels, thresholds=99):
    """
    Compute ODS (Optimal Dataset Scale), OIS (Optimal Image Scale), and AP.
    
    ODS: Best F-score using a single threshold for all images
    OIS: Average of best F-score per image
    AP: Average Precision
    """
    threshs = np.linspace(0.01, 0.99, thresholds)
    
    # For ODS (global)
    all_preds = []
    all_labels = []
    
    # For OIS (per-image best)
    ois_f1_scores = []
    
    for pred, label in zip(preds, labels):
        pred_np = pred.flatten()
        label_np = (label.flatten() > 0.5).astype(np.float32)
        
        all_preds.append(pred_np)
        all_labels.append(label_np)
        
        # Per-image best F1
        best_f1 = 0
        for t in threshs:
            pred_bin = (pred_np >= t).astype(np.float32)
            tp = np.sum(pred_bin * label_np)
            fp = np.sum(pred_bin * (1 - label_np))
            fn = np.sum((1 - pred_bin) * label_np)
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            best_f1 = max(best_f1, f1)
        
        ois_f1_scores.append(best_f1)
    
    # Concatenate all for global metrics
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # ODS: Best global threshold
    best_ods = 0
    for t in threshs:
        pred_bin = (all_preds >= t).astype(np.float32)
        tp = np.sum(pred_bin * all_labels)
        fp = np.sum(pred_bin * (1 - all_labels))
        fn = np.sum((1 - pred_bin) * all_labels)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        best_ods = max(best_ods, f1)
    
    # OIS: Average of per-image best
    ois = np.mean(ois_f1_scores)
    
    # AP: Average Precision
    try:
        ap = average_precision_score(all_labels, all_preds)
    except:
        ap = 0.0
    
    return best_ods, ois, ap

# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc='Evaluating'):
        images = batch['images'].to(device)
        labels = batch['labels']
        
        outputs = model(images)
        
        for i in range(outputs.shape[0]):
            all_preds.append(outputs[i, 0].cpu().numpy())
            all_labels.append(labels[i, 0].numpy())
    
    ods, ois, ap = compute_ods_ois_ap(all_preds, all_labels)
    return ods, ois, ap

def save_checkpoint(model, optimizer, scheduler, epoch, best_ods, path):
    """Save full training checkpoint"""
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_ods": best_ods
    }, path)

@torch.no_grad()
def visualize_predictions(model, dataset, device, num_samples=6, save_path='predictions.png'):
    """Visualize predictions on random samples"""
    model.eval()
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        img = sample['images'].unsqueeze(0).to(device)
        label = sample['labels'][0].numpy()
        
        pred = model(img)[0, 0].cpu().numpy()
        
        # Original image
        axes[i, 0].imshow(sample['images'].permute(1, 2, 0).numpy())
        axes[i, 0].set_title(f'Input: {sample["filename"]}')
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(label, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('XYW-Net Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Predictions saved to {save_path}")
    plt.close()

# ============================================================
# MAIN TRAINING SCRIPT
# ============================================================
def main():
    # Configuration
    DATA_ROOT = "./datasets/HED_Small"  
    BATCH_SIZE = 4
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    print("="*60)
    print("XYW-NET EDGE DETECTION TRAINING")
    print("="*60)
    print(f"Dataset: {DATA_ROOT}")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    
    # Load datasets
    train_dataset = ProcessedDataset(DATA_ROOT, split='train')
    val_dataset = ProcessedDataset(DATA_ROOT, split='val')
    test_dataset = ProcessedDataset(DATA_ROOT, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    # Verify sample
    sample = train_dataset[0]
    print(f"Image shape: {sample['images'].shape}")
    print(f"Edge shape: {sample['labels'].shape}")
    
    # Initialize model
    model = XYWNet().to(device)
    criterion = EdgeLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print("="*60)
    
    # Training loop
    history = {'train_loss': [], 'val_ods': [], 'val_ois': [], 'val_ap': []}
    best_ods = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        
        # Evaluate on validation
        val_ods, val_ois, val_ap = evaluate(model, val_loader, device)
        history['val_ods'].append(val_ods)
        history['val_ois'].append(val_ois)
        history['val_ap'].append(val_ap)
        
        scheduler.step()
        
        print(f"Loss: {train_loss:.4f} | ODS: {val_ods:.4f} | OIS: {val_ois:.4f} | AP: {val_ap:.4f}")
        
        # Save best model
        if val_ods > best_ods:
            best_ods = val_ods
            
            # Save state dict
            torch.save(model.state_dict(), f'models/{epoch+1}epoch.pth')
            torch.save(model.state_dict(), 'models/best_xyw_net.pth')
            
            # Save full checkpoint
            save_checkpoint(model, optimizer, scheduler, epoch, best_ods, "models/xywnet_full_checkpoint.pth")
            
            # Export deployment .pt model
            real_sample = train_dataset[0]
            example_input = real_sample['images'].unsqueeze(0).to(device)
            traced_model = torch.jit.trace(model, example_input)
            traced_model.save("models/xywnet_full_model.pt")
            
            print(f"  -> Saved best model (ODS: {best_ods:.4f})")
            print(f"  -> Saved epoch checkpoint: models/{epoch+1}epoch.pth")
            print(f"  -> Saved traced model: models/xywnet_full_model.pt")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Load best model for final test
    model.load_state_dict(torch.load('models/best_xyw_net.pth'))
    test_ods, test_ois, test_ap = evaluate(model, test_loader, device)
    
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"  ODS (Optimal Dataset Scale): {test_ods:.4f}")
    print(f"  OIS (Optimal Image Scale):   {test_ois:.4f}")
    print(f"  AP (Average Precision):      {test_ap:.4f}")
    print("="*60)
    
    # Generate visualization
    visualize_predictions(model, test_dataset, device, num_samples=6)
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Metrics
    axes[1].plot(history['val_ods'], 'g-', linewidth=2, label='ODS')
    axes[1].plot(history['val_ois'], 'b-', linewidth=2, label='OIS')
    axes[1].plot(history['val_ap'], 'r-', linewidth=2, label='AP')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Validation Metrics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("Training history saved to training_history.png")
    plt.close()
    
    print(f"\nTraining complete. Best ODS: {best_ods:.4f}")
    print("All files saved to models/ folder")

if __name__ == "__main__":
    main()