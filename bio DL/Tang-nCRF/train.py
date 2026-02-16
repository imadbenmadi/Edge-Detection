"""
Training script for Tang et al. nCRF Contour Detection model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json
import argparse

from model import TangNet


class EdgeDataset(Dataset):
    """Dataset loader for edge detection."""
    
    def __init__(self, root, split='train', size=None):
        self.img_dir = root / split / 'images'
        self.gt_dir = root / split / 'edges'
        self.size = size
        
        # Load all image paths
        self.images = sorted(
            list(self.img_dir.glob('*.jpg')) + 
            list(self.img_dir.glob('*.png'))
        )
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {self.img_dir}")
        
        print(f"Loaded {len(self.images)} images from {split} split")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load ground truth edge map
        gt_path = self.gt_dir / img_path.name.replace('.jpg', '.png')
        if not gt_path.exists():
            gt_path = self.gt_dir / img_path.name
        
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        
        if gt is None:
            gt = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # Resize if specified
        if self.size is not None:
            img = cv2.resize(img, (self.size, self.size))
            gt = cv2.resize(gt, (self.size, self.size))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        gt = gt.astype(np.float32) / 255.0
        
        # Convert to tensors
        img = torch.from_numpy(img.transpose(2, 0, 1))  # HWC -> CHW
        gt = torch.from_numpy(gt).unsqueeze(0)  # Add channel dimension
        
        return img, gt


class EdgeLoss(nn.Module):
    """
    Balanced cross-entropy loss for edge detection.
    Handles class imbalance between edge and non-edge pixels.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # Calculate class weights
        pos = target == 1
        neg = target == 0
        
        n_pos = pos.sum().float()
        n_neg = neg.sum().float()
        
        if n_pos == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Balanced weights
        beta = n_neg / (n_pos + n_neg + 1e-8)
        
        # Binary cross entropy
        loss = -beta * (target * torch.log(pred + 1e-8)) - \
               (1 - beta) * ((1 - target) * torch.log(1 - pred + 1e-8))
        
        return loss.mean()


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for imgs, gts in pbar:
        imgs, gts = imgs.to(device), gts.to(device)
        
        # Forward pass
        preds = model(imgs)
        loss = criterion(preds, gts)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for imgs, gts in tqdm(loader, desc='Validating'):
            imgs, gts = imgs.to(device), gts.to(device)
            
            preds = model(imgs)
            loss = criterion(preds, gts)
            
            total_loss += loss.item()
    
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description='Train Tang nCRF model')
    parser.add_argument('--dataset', type=str, default='../datasets/HED_Small',
                        help='Path to dataset')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--size', type=int, default=320,
                        help='Image size')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = Path(args.checkpoint_dir) / f'tang_ncrf_{timestamp}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Create datasets
    dataset_root = Path(args.dataset)
    train_dataset = EdgeDataset(dataset_root, 'train', size=args.size)
    val_dataset = EdgeDataset(dataset_root, 'test', size=args.size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    model = TangNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = EdgeLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # TensorBoard
    writer = SummaryWriter(checkpoint_dir / 'runs')
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_dir / 'best_model.pth')
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, checkpoint_dir / 'final_model.pth')
    
    # Save training history
    with open(checkpoint_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    writer.close()
    print(f"\n✓ Training complete! Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
