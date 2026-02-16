"""
Training script for Cai et al. 2024 Visual Pathway Network

Small version training on HED_Small dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from model import VisualPathwayNet


class EdgeDataset(Dataset):
    """Simple edge detection dataset"""
    def __init__(self, img_dir, gt_dir, transform=None):
        self.img_paths = sorted(list(Path(img_dir).glob('*.jpg')) + list(Path(img_dir).glob('*.png')))
        self.gt_dir = Path(gt_dir)
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load ground truth
        gt_path = self.gt_dir / img_path.name.replace('.jpg', '.png')
        if gt_path.exists():
            gt = cv2.imread(str(gt_path), 0)
        else:
            gt = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # Resize to fixed size for small training
        img = cv2.resize(img, (320, 320))
        gt = cv2.resize(gt, (320, 320))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        gt = gt.astype(np.float32) / 255.0
        
        # To tensor [C, H, W]
        img = torch.from_numpy(img.transpose(2, 0, 1))
        gt = torch.from_numpy(gt).unsqueeze(0)
        
        return img, gt


class WeightedBCELoss(nn.Module):
    """Weighted BCE loss for imbalanced edge detection"""
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, pred, target):
        # Calculate positive/negative weights
        pos_mask = (target > 0.5).float()
        neg_mask = (target <= 0.5).float()
        
        num_pos = pos_mask.sum() + 1e-8
        num_neg = neg_mask.sum() + 1e-8
        
        # Balance weights
        w_pos = num_neg / (num_pos + num_neg) * self.pos_weight
        w_neg = num_pos / (num_pos + num_neg)
        
        weights = pos_mask * w_pos + neg_mask * w_neg
        
        # BCE loss
        loss = -(weights * (target * torch.log(pred + 1e-8) + (1 - target) * torch.log(1 - pred + 1e-8)))
        return loss.mean()


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for img, gt in pbar:
        img = img.to(device)
        gt = gt.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(img)
        
        # Multi-scale loss (if training with side outputs)
        if isinstance(outputs, list):
            loss = sum([criterion(out, gt) for out in outputs]) / len(outputs)
        else:
            loss = criterion(outputs, gt)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for img, gt in tqdm(dataloader, desc='Validation'):
            img = img.to(device)
            gt = gt.to(device)
            
            # Forward pass
            output = model(img)
            if isinstance(output, list):
                output = output[0]  # Main output only
            
            loss = criterion(output, gt)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../datasets/HED_Small')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print(f"🚀 Training Cai et al. 2024 Visual Pathway Network")
    print(f"   Device: {device}")
    print(f"   Dataset: {args.data_root}")
    print(f"   Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    
    # Dataset (using train set for both training and validation in small dataset)
    train_dataset = EdgeDataset(
        Path(args.data_root) / 'train' / 'images',
        Path(args.data_root) / 'train' / 'edges'
    )
    val_dataset = EdgeDataset(
        Path(args.data_root) / 'test' / 'images',
        Path(args.data_root) / 'test' / 'edges'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"   Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Model
    model = VisualPathwayNet(in_channels=3).to(device)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = WeightedBCELoss(pos_weight=1.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n📅 Epoch {epoch}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_dir / 'best_model.pth')
            print(f"   ✅ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_dir / f'checkpoint_epoch{epoch}.pth')
    
    print(f"\n🎉 Training complete! Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
