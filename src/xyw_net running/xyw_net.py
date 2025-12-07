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

# ==================== Setup ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

DATA_ROOT = r'e:\Edge Detection\datasets\processed'
BATCH_SIZE = 4

# ==================== Dataset ====================
class ProcessedDataset(Dataset):
    """Load processed PNG images and edge maps"""
    def __init__(self, root_dir, split='train'):
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_dir = self.root_dir / split / 'images'
        self.edge_dir = self.root_dir / split / 'edges'
        self.samples = sorted(list(self.img_dir.glob('*.png')))
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.img_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        edge_path = self.edge_dir / img_path.name

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        edge = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
        edge = edge.astype(np.float32) / 255.0

        img = torch.from_numpy(img).permute(2, 0, 1)  # (C, H, W)
        edge = torch.from_numpy(edge).unsqueeze(0)    # (1, H, W)

        return {'images': img, 'labels': edge, 'filename': img_path.stem}

print("Loading datasets...")
train_dataset = ProcessedDataset(DATA_ROOT, split='train')
val_dataset = ProcessedDataset(DATA_ROOT, split='val')
test_dataset = ProcessedDataset(DATA_ROOT, split='test')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
sample = train_dataset[0]
print(f"Sample shapes - Image: {sample['images'].shape}, Edge: {sample['labels'].shape}")

# ==================== Model ====================
def createPDCFunc(PDC_type):
    """Create Pixel Difference Convolution function"""
    assert PDC_type in ['cv', 'cd', 'ad', 'rd', 'sd', 'p2d', '2sd', '2cd']
    if PDC_type == 'cv':
        return F.conv2d
    if PDC_type == '2sd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert weights.size(2) == 3 and weights.size(3) == 3
            shape = weights.shape
            if groups == shape[0]:
                weights_conv = (weights - weights[:, :, [1, 1, 1, 0, 0, 0, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]].view(shape))
            else:
                weights_conv = (weights - weights[:, :, [1, 1, 1, 0, 0, 0, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]].view(shape).flip(0))
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    return F.conv2d

class Conv2d(nn.Module):
    """PDC-enabled Conv2d"""
    def __init__(self, pdc_func='cv', in_channels=1, out_channels=1, kernel_size=3,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__()
        self.pdc = createPDCFunc(pdc_func)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.pdc(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Xc1x1(nn.Module):
    """X pathway: Local contrast"""
    def __init__(self, in_channels, out_channels):
        super(Xc1x1, self).__init__()
        self.Xcenter = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.Xcenter_relu = nn.ReLU(inplace=True)
        self.Xsurround = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels)
        self.conv1_1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.Xsurround_relu = nn.ReLU(inplace=True)

    def forward(self, input):
        xcenter = self.Xcenter_relu(self.Xcenter(input))
        xsurround = self.Xsurround_relu(self.Xsurround(input))
        xsurround = self.conv1_1(xsurround)
        return xsurround - xcenter

class Yc1x1(nn.Module):
    """Y pathway: Large RF contrast"""
    def __init__(self, in_channels, out_channels):
        super(Yc1x1, self).__init__()
        self.Ycenter = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.Ycenter_relu = nn.ReLU(inplace=True)
        self.Ysurround = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=4, dilation=2, groups=in_channels)
        self.conv1_1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.Ysurround_relu = nn.ReLU(inplace=True)

    def forward(self, input):
        ycenter = self.Ycenter_relu(self.Ycenter(input))
        ysurround = self.Ysurround_relu(self.Ysurround(input))
        ysurround = self.conv1_1(ysurround)
        return ysurround - ycenter

class W(nn.Module):
    """W pathway: Directional (horizontal + vertical)"""
    def __init__(self, inchannel, outchannel):
        super(W, self).__init__()
        self.h = nn.Conv2d(inchannel, inchannel, kernel_size=(1, 3), padding=(0, 1), groups=inchannel)
        self.v = nn.Conv2d(inchannel, inchannel, kernel_size=(3, 1), padding=(1, 0), groups=inchannel)
        self.convh_1 = nn.Conv2d(inchannel, inchannel, kernel_size=1, bias=False)
        self.convv_1 = nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.h(x))
        h = self.convh_1(h)
        v = self.relu(self.v(h))
        v = self.convv_1(v)
        return v

class XYW_S(nn.Module):
    """XYW Start block"""
    def __init__(self, inchannel, outchannel, stride=1):
        super(XYW_S, self).__init__()
        self.y_c = Yc1x1(inchannel, outchannel)
        self.x_c = Xc1x1(inchannel, outchannel)
        self.w = W(inchannel, outchannel)

    def forward(self, x):
        return self.x_c(x), self.y_c(x), self.w(x)

class XYW(nn.Module):
    """XYW middle block"""
    def __init__(self, inchannel, outchannel, stride=1):
        super(XYW, self).__init__()
        self.y_c = Yc1x1(inchannel, outchannel)
        self.x_c = Xc1x1(inchannel, outchannel)
        self.w = W(inchannel, outchannel)

    def forward(self, xc, yc, w):
        return self.x_c(xc), self.y_c(yc), self.w(w)

class XYW_E(nn.Module):
    """XYW End block (combines X+Y+W)"""
    def __init__(self, inchannel, outchannel):
        super(XYW_E, self).__init__()
        self.y_c = Yc1x1(inchannel, outchannel)
        self.x_c = Xc1x1(inchannel, outchannel)
        self.w = W(inchannel, outchannel)

    def forward(self, xc, yc, w):
        return self.x_c(xc) + self.y_c(yc) + self.w(w)

class s1(nn.Module):
    def __init__(self, channel=30):
        super(s1, self).__init__()
        self.conv1 = nn.Conv2d(3, channel, kernel_size=7, padding=6, dilation=2)
        self.xyw1_1 = XYW_S(channel, channel)
        self.xyw1_2 = XYW(channel, channel)
        self.xyw1_3 = XYW_E(channel, channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        temp = self.relu(self.conv1(x))
        xc, yc, w = self.xyw1_1(temp)
        xc, yc, w = self.xyw1_2(xc, yc, w)
        xyw1_3 = self.xyw1_3(xc, yc, w)
        return xyw1_3 + temp

class s2(nn.Module):
    def __init__(self, channel=60):
        super(s2, self).__init__()
        self.xyw2_1 = XYW_S(channel//2, channel, stride=2)
        self.xyw2_2 = XYW(channel, channel)
        self.xyw2_3 = XYW_E(channel, channel)
        self.shortcut = nn.Conv2d(channel//2, channel, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        xc, yc, w = self.xyw2_1(x)
        xc, yc, w = self.xyw2_2(xc, yc, w)
        xyw2_3 = self.xyw2_3(xc, yc, w)
        return xyw2_3 + self.shortcut(x)

class s3(nn.Module):
    def __init__(self, channel=120):
        super(s3, self).__init__()
        self.xyw3_1 = XYW_S(channel//2, channel, stride=2)
        self.xyw3_2 = XYW(channel, channel)
        self.xyw3_3 = XYW_E(channel, channel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shortcut = nn.Conv2d(channel//2, channel, kernel_size=1)

    def forward(self, x):
        x = self.pool(x)
        shortcut = self.shortcut(x)
        xc, yc, w = self.xyw3_1(x)
        xc, yc, w = self.xyw3_2(xc, yc, w)
        xyw3_3 = self.xyw3_3(xc, yc, w)
        return xyw3_3 + shortcut

class s4(nn.Module):
    def __init__(self, channel=120):
        super(s4, self).__init__()
        self.xyw4_1 = XYW_S(channel, channel, stride=2)
        self.xyw4_2 = XYW(channel, channel)
        self.xyw4_3 = XYW_E(channel, channel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shortcut = nn.Conv2d(channel, channel, kernel_size=1)

    def forward(self, x):
        x = self.pool(x)
        shortcut = self.shortcut(x)
        xc, yc, w = self.xyw4_1(x)
        xc, yc, w = self.xyw4_2(xc, yc, w)
        xyw4_3 = self.xyw4_3(xc, yc, w)
        return xyw4_3 + shortcut

class encode(nn.Module):
    def __init__(self):
        super(encode, self).__init__()
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

def upsample_filt(size):
    factor = (size + 1) // 2
    center = factor - 1 if size % 2 == 1 else factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

def bilinear_upsample_weights(factor, num_classes):
    filter_size = 2 * factor - factor % 2
    weights = np.zeros((num_classes, num_classes, filter_size, filter_size), dtype=np.float32)
    upsample_kernel = upsample_filt(filter_size)
    for i in range(num_classes):
        weights[i, i, :, :] = upsample_kernel
    return torch.Tensor(weights)

class adap_conv(nn.Module):
    """Adaptive convolution with learnable weight"""
    def __init__(self, in_channels, out_channels, kz=3, pd=1):
        super(adap_conv, self).__init__()
        self.conv = nn.Sequential(
            Conv2d(pdc_func='2sd', in_channels=in_channels, out_channels=out_channels, kernel_size=kz, padding=pd),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.weight = nn.Parameter(torch.Tensor([0.]))

    def forward(self, x):
        return self.conv(x) * self.weight.sigmoid()

class Refine_block2_1(nn.Module):
    """Refinement block for decoder"""
    def __init__(self, in_channel, out_channel, factor, require_grad=False):
        super(Refine_block2_1, self).__init__()
        self.pre_conv1 = adap_conv(in_channel[0], out_channel, kz=3, pd=1)
        self.pre_conv2 = adap_conv(in_channel[1], out_channel, kz=3, pd=1)
        self.factor = factor
        self.deconv_weight = nn.Parameter(bilinear_upsample_weights(factor, out_channel), requires_grad=require_grad)

    def forward(self, *input):
        x1 = self.pre_conv1(input[0])
        x2 = self.pre_conv2(input[1])
        x2 = F.conv_transpose2d(
            x2, self.deconv_weight, stride=self.factor,
            padding=int(self.factor/2),
            output_padding=(x1.size(2) - x2.size(2)*self.factor,
                            x1.size(3) - x2.size(3)*self.factor)
        )
        return x1 + x2

class decode_rcf(nn.Module):
    def __init__(self):
        super(decode_rcf, self).__init__()
        self.f43 = Refine_block2_1(in_channel=(120, 120), out_channel=60, factor=2)
        self.f32 = Refine_block2_1(in_channel=(60, 60), out_channel=30, factor=2)
        self.f21 = Refine_block2_1(in_channel=(30, 30), out_channel=24, factor=2)
        self.f = nn.Conv2d(24, 1, kernel_size=1, padding=0)

    def forward(self, x):
        s3 = self.f43(x[2], x[3])
        s2 = self.f32(x[1], s3)
        s1 = self.f21(x[0], s2)
        out = self.f(s1)
        return out.sigmoid()

class XYWNet(nn.Module):
    def __init__(self):
        super(XYWNet, self).__init__()
        self.encode = encode()
        self.decode = decode_rcf()

    def forward(self, x):
        endpoints = self.encode(x)
        out = self.decode(endpoints)
        return out

    def forward_with_stages(self, x):
        s1, s2, s3, s4 = self.encode(x)
        final = self.decode((s1, s2, s3, s4))
        return final, (s1, s2, s3, s4)

model = XYWNet().to(device)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# ==================== Loss & Metrics ====================
class EdgeLoss(nn.Module):
    """Weighted cross-entropy-like loss for edge detection"""
    def __init__(self):
        super(EdgeLoss, self).__init__()

    def forward(self, pred, label):
        pred_flat = pred.view(-1)
        label_flat = label.view(-1)
        eps = 1e-6

        pos_mask = label_flat > 0
        neg_mask = label_flat == 0

        pred_pos = pred_flat[pos_mask].clamp(eps, 1.0 - eps)
        pred_neg = pred_flat[neg_mask].clamp(eps, 1.0 - eps)

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
    ODS: Best F-score using a single threshold for all images
    OIS: Average of best F-score per image
    AP: Average Precision
    """
    threshs = np.linspace(0.01, 0.99, thresholds)

    all_preds = []
    all_labels = []
    ois_f1_scores = []

    for pred, label in zip(preds, labels):
        pred_np = pred.flatten()
        label_np = (label.flatten() > 0.5).astype(np.float32)

        all_preds.append(pred_np)
        all_labels.append(label_np)

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

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

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

    ois = np.mean(ois_f1_scores)

    try:
        ap = average_precision_score(all_labels, all_preds)
    except Exception:
        ap = 0.0

    return best_ods, ois, ap

# ==================== Train & Eval ====================
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

# ==================== Training Loop ====================
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

model = XYWNet().to(device)
criterion = EdgeLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

history = {'train_loss': [], 'val_ods': [], 'val_ois': [], 'val_ap': []}
best_ods = 0

print(f"Training XYW-Net for {NUM_EPOCHS} epochs...")
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
print("="*60)

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    history['train_loss'].append(train_loss)

    val_ods, val_ois, val_ap = evaluate(model, val_loader, device)
    history['val_ods'].append(val_ods)
    history['val_ois'].append(val_ois)
    history['val_ap'].append(val_ap)

    scheduler.step()

    print(f"Loss: {train_loss:.4f} | ODS: {val_ods:.4f} | OIS: {val_ois:.4f} | AP: {val_ap:.4f}")

    if val_ods > best_ods:
        best_ods = val_ods
        torch.save(model.state_dict(), 'best_xyw_net.pth')
        print(f"  -> Saved best model (ODS: {best_ods:.4f})")

print("\n" + "="*60)
print(f"Training complete. Best ODS: {best_ods:.4f}")

# ==================== Plots & Test Eval ====================
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Training Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['val_ods'], 'g-', linewidth=2, label='ODS')
    axes[1].plot(history['val_ois'], 'b-', linewidth=2, label='OIS')
    axes[1].plot(history['val_ap'], 'r-', linewidth=2, label='AP')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Score'); axes[1].set_title('Validation Metrics')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()

plot_training_history(history)

# Load best and evaluate on test
model.load_state_dict(torch.load('best_xyw_net.pth', map_location=device))
print("Loaded best model weights.\n")
print("Evaluating on TEST set...")
test_ods, test_ois, test_ap = evaluate(model, test_loader, device)

print("\n" + "="*60)
print("FINAL TEST RESULTS")
print("="*60)
print(f"  ODS (Optimal Dataset Scale): {test_ods:.4f}")
print(f"  OIS (Optimal Image Scale):   {test_ois:.4f}")
print(f"  AP (Average Precision):      {test_ap:.4f}")
print("="*60)

@torch.no_grad()
def visualize_predictions(model, dataset, device, num_samples=6):
    model.eval()
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4*len(indices)))
    if len(indices) == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        img = sample['images'].unsqueeze(0).to(device)
        label = sample['labels'][0].numpy()
        pred = model(img)[0, 0].cpu().numpy()

        axes[i, 0].imshow(sample['images'].permute(1, 2, 0).numpy())
        axes[i, 0].set_title(f'Input: {sample["filename"]}'); axes[i, 0].axis('off')

        axes[i, 1].imshow(label, cmap='gray'); axes[i, 1].set_title('Ground Truth'); axes[i, 1].axis('off')
        axes[i, 2].imshow(pred, cmap='gray'); axes[i, 2].set_title('XYW-Net Prediction'); axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150)
    plt.show()

visualize_predictions(model, test_dataset, device, num_samples=6)

@torch.no_grad()
def visualize_stages(model, dataset, device, sample_idx=0):
    model.eval()
    sample = dataset[sample_idx]
    img = sample['images'].unsqueeze(0).to(device)

    final, (s1_, s2_, s3_, s4_) = model.forward_with_stages(img)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(sample['images'].permute(1, 2, 0).numpy()); axes[0, 0].set_title('Input'); axes[0, 0].axis('off')
    axes[0, 1].imshow(sample['labels'][0].numpy(), cmap='gray'); axes[0, 1].set_title('GT'); axes[0, 1].axis('off')
    axes[0, 2].imshow(final[0, 0].cpu().numpy(), cmap='gray'); axes[0, 2].set_title('Final'); axes[0, 2].axis('off')
    pred_binary = (final[0, 0].cpu().numpy() > 0.5).astype(float)
    axes[0, 3].imshow(pred_binary, cmap='gray'); axes[0, 3].set_title('Pred (0.5)'); axes[0, 3].axis('off')

    stages = [s1_, s2_, s3_, s4_]
    names = ['Stage 1 (30ch)', 'Stage 2 (60ch)', 'Stage 3 (120ch)', 'Stage 4 (120ch)']
    for i, (stage, name) in enumerate(zip(stages, names)):
        feat_mean = stage[0].mean(dim=0).cpu().numpy()
        feat_norm = (feat_mean - feat_mean.min()) / (feat_mean.max() - feat_mean.min() + 1e-8)
        axes[1, i].imshow(feat_norm, cmap='viridis')
        axes[1, i].set_title(f'{name}\nShape: {tuple(stage.shape[2:])}')
        axes[1, i].axis('off')

    plt.suptitle(f'XYW-Net Encoder Stage Outputs - {sample["filename"]}', fontsize=14)
    plt.tight_layout()
    plt.savefig('stage_outputs.png', dpi=150)
    plt.show()

for idx in [0, 1, 2]:
    if idx < len(test_dataset):
        visualize_stages(model, test_dataset, device, sample_idx=idx)

@torch.no_grad()
def plot_pr_curve(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    for batch in tqdm(loader, desc='Computing PR curve'):
        images = batch['images'].to(device)
        labels = batch['labels']
        outputs = model(images)
        for i in range(outputs.shape[0]):
            all_preds.append(outputs[i, 0].cpu().numpy().flatten())
            all_labels.append((labels[i, 0].numpy().flatten() > 0.5).astype(int))

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'XYW-Net (AP={ap:.4f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left'); plt.grid(True, alpha=0.3)
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.savefig('pr_curve.png', dpi=150)
    plt.show()
    return ap

ap = plot_pr_curve(model, test_loader, device)
print(f"Average Precision (PR curve): {ap:.4f}")

print("\n" + "="*70)
print("                    XYW-NET EVALUATION SUMMARY")
print("="*70)
print(f"  Dataset:          {DATA_ROOT}")
print(f"  Train samples:    {len(train_dataset)}")
print(f"  Val samples:      {len(val_dataset)}")
print(f"  Test samples:     {len(test_dataset)}")
print(f"")
print(f"  Training epochs:  {NUM_EPOCHS}")
print(f"  Learning rate:    {LEARNING_RATE}")
print(f"  Batch size:       {BATCH_SIZE}")
print(f"")
print("  " + "-"*50)
print(f"  TEST SET METRICS:")
print("  " + "-"*50)
print(f"  ODS (Optimal Dataset Scale):   {test_ods:.4f}")
print(f"  OIS (Optimal Image Scale):     {test_ois:.4f}")
print(f"  AP (Average Precision):        {test_ap:.4f}")
print("="*70)
print("Saved files:")
print("  - best_xyw_net.pth      (model weights)")
print("  - training_history.png  (loss/metrics plot)")
print("  - predictions.png       (sample predictions)")
print("  - stage_outputs.png     (encoder stages)")
print("  - pr_curve.png          (precision-recall curve)")