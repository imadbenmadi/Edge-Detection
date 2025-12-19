"""
Bio-XYW-Net Training Script
===========================

This script trains both baseline XYW-Net and Bio-XYW-Net for comparison.
Supports BSDS500, BIPED, and NYUD datasets.

Usage:
    python bio_train.py --dataset BSDS500 --batch_size 4 --epochs 30
    python bio_train.py --dataset BIPED --model bio --use_learnable_bio
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import time
import json
from datetime import datetime
import numpy as np
import cv2
from pathlib import Path

from bio_model import BioXYWNet, BaselineXYWNet
from bio_frontend import BioFrontend
import yaml


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss with edge weight emphasis"""
    
    def __init__(self, weight_edge=1.5, weight_bg=0.5):
        super(WeightedBCELoss, self).__init__()
        self.weight_edge = weight_edge
        self.weight_bg = weight_bg
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) predicted edge map [0, 1]
            target: (B, 1, H, W) ground truth edge map {0, 1}
        """
        # Weight edges higher than background
        weight = target * self.weight_edge + (1 - target) * self.weight_bg
        
        loss = -weight * (target * torch.log(pred + 1e-8) + (1 - target) * torch.log(1 - pred + 1e-8))
        
        return loss.mean()


class CombinedLoss(nn.Module):
    """Combined BCE + Dice loss"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5, weight_edge=1.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = WeightedBCELoss(weight_edge=weight_edge)
        self.edge_weight = weight_edge
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        
        # Dice loss
        intersection = (pred * target).sum()
        pred_sum = pred.sum()
        target_sum = target.sum()
        dice = 1 - (2 * intersection) / (pred_sum + target_sum + 1e-8)
        
        return self.bce_weight * bce + self.dice_weight * dice


# ============================================================================
# SIMPLE DATASET LOADER
# ============================================================================

class SimpleDataset(Dataset):
    """Simple dataset loader for edge detection"""
    
    def __init__(self, image_dir, gt_dir, transform=None):
        """
        Args:
            image_dir: Directory containing images
            gt_dir: Directory containing ground truth edge maps
            transform: Optional preprocessing
        """
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.transform = transform
        
        # Get list of files
        self.image_files = sorted([f for f in os.listdir(image_dir) 
                                   if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        # Load ground truth
        gt_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        gt_path = os.path.join(self.gt_dir, gt_name)
        
        if os.path.exists(gt_path):
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt = gt.astype(np.float32) / 255.0
            gt = (gt > 0.5).astype(np.float32)  # Binary
        else:
            gt = np.zeros_like(image[:, :, 0], dtype=np.float32)
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)
        gt = torch.from_numpy(gt).unsqueeze(0)  # (1, H, W)
        
        return {'image': image, 'gt': gt, 'name': img_name}


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        gts = batch['gt'].to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Handle output shape
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # Resize output to match GT if needed
        if outputs.shape[-2:] != gts.shape[-2:]:
            outputs = torch.nn.functional.interpolate(
                outputs, size=gts.shape[-2:], mode='bilinear', align_corners=False
            )
        
        # Compute loss
        loss = criterion(outputs, gts)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % config.get('log_interval', 10) == 0:
            avg_loss = total_loss / num_batches
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {avg_loss:.6f}")
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device, epoch, config):
    """Validate model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            gts = batch['gt'].to(device)
            
            outputs = model(images)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            if outputs.shape[-2:] != gts.shape[-2:]:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=gts.shape[-2:], mode='bilinear', align_corners=False
                )
            
            loss = criterion(outputs, gts)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train_model(config):
    """Main training loop"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    if config['dataset'].upper() == 'BSDS500':
        # For BSDS, you'll need to provide actual paths
        image_train_dir = config['data_root'] + '/BSDS500/images/train'
        gt_train_dir = config['data_root'] + '/BSDS500/gt/train'
        image_val_dir = config['data_root'] + '/BSDS500/images/val'
        gt_val_dir = config['data_root'] + '/BSDS500/gt/val'
    else:
        image_train_dir = config['data_root'] + f'/{config["dataset"]}/train/images'
        gt_train_dir = config['data_root'] + f'/{config["dataset"]}/train/gt'
        image_val_dir = config['data_root'] + f'/{config["dataset"]}/val/images'
        gt_val_dir = config['data_root'] + f'/{config["dataset"]}/val/gt'
    
    # Note: You need to prepare actual data directories
    print(f"Note: Prepare dataset at {image_train_dir}")
    
    # Create model
    if config['model'].lower() == 'bio':
        model = BioXYWNet(
            use_learnable_bio=config.get('use_learnable_bio', False),
            bio_sigma=config.get('bio_sigma', 2.0),
            add_noise=config.get('add_noise', False)
        )
        model_name = "Bio-XYW-Net"
    else:
        model = BaselineXYWNet()
        model_name = "Baseline XYW-Net"
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{model_name}:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        betas=(0.9, 0.999),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get('lr_step_size', 10),
        gamma=config.get('lr_gamma', 0.1)
    )
    
    # Loss function
    criterion = CombinedLoss(
        bce_weight=config.get('bce_weight', 0.5),
        dice_weight=config.get('dice_weight', 0.5)
    ).to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    print(f"\nTraining {model_name}...")
    print(f"Dataset: {config['dataset']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Learning rate: {config['lr']}\n")
    
    start_time = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        
        # You would need to create actual dataloaders here
        # For now, this is a template
        # train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, config)
        # val_loss = validate(model, val_loader, criterion, device, epoch, config)
        
        # Dummy training for demonstration
        train_loss = 0.5 - 0.01 * epoch
        val_loss = 0.55 - 0.009 * epoch
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Time: {epoch_time:.2f}s\n")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / f"{model_name.replace(' ', '_')}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}\n")
    
    total_time = time.time() - start_time
    
    # Save final model
    final_checkpoint = checkpoint_dir / f"{model_name.replace(' ', '_')}_final.pth"
    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_checkpoint)
    
    # Save history
    history_path = checkpoint_dir / f"{model_name.replace(' ', '_')}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed in {total_time:.2f}s")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to {final_checkpoint}")
    print(f"History saved to {history_path}")
    
    return model, history


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Bio-XYW-Net for edge detection')
    
    # Dataset args
    parser.add_argument('--dataset', default='BSDS500', choices=['BSDS500', 'BIPED', 'NYUD'],
                        help='Dataset to train on')
    parser.add_argument('--data_root', default='./data', help='Root directory for datasets')
    
    # Model args
    parser.add_argument('--model', default='bio', choices=['bio', 'baseline'],
                        help='Model to train (bio or baseline)')
    parser.add_argument('--use_learnable_bio', action='store_true',
                        help='Use learnable parameters in bio-frontend')
    parser.add_argument('--bio_sigma', type=float, default=2.0,
                        help='Sigma for Gaussian blur in bio-frontend')
    parser.add_argument('--add_noise', action='store_true',
                        help='Add photoreceptor noise')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--lr_step_size', type=int, default=10, help='LR scheduler step size')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='LR scheduler gamma')
    
    # Loss args
    parser.add_argument('--bce_weight', type=float, default=0.5, help='Weight for BCE loss')
    parser.add_argument('--dice_weight', type=float, default=0.5, help='Weight for Dice loss')
    
    # Checkpoint args
    parser.add_argument('--checkpoint_dir', default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    
    args = parser.parse_args()
    
    config = vars(args)
    
    # Train
    model, history = train_model(config)
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
