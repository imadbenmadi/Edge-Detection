"""
Training script for BPED (Bio-inspired Pyramid Edge Detection)

Usage:
    python train.py --dataset_path datasets/HED_Small --epochs 10 --batch_size 4

Features:
    - Deep supervision loss
    - Class-balanced binary cross-entropy
    - Learning rate scheduling
    - Checkpoint saving
"""

import os
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from model import BPED


class EdgeDataset(Dataset):
    """Edge detection dataset loader"""
    def __init__(self, root_dir, split='train', image_size=320):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        
        # Get image and edge paths
        split_dir = os.path.join(root_dir, split)
        img_dir = os.path.join(split_dir, 'images')
        self.edge_dir = os.path.join(split_dir, 'edges')
        
        # Check multiple possible structures
        if os.path.exists(img_dir):
            # Structure: root/split/images/
            self.images = sorted([
                os.path.join(img_dir, f) for f in os.listdir(img_dir)
                if f.endswith(('.jpg', '.png', '.jpeg'))
            ])
        elif os.path.exists(split_dir):
            # Structure: root/split/
            self.images = sorted([
                os.path.join(split_dir, f) for f in os.listdir(split_dir)
                if f.endswith(('.jpg', '.png', '.jpeg')) and not f.startswith('.')
            ])
            self.edge_dir = split_dir
        else:
            # Fallback to root
            self.images = sorted([
                os.path.join(root_dir, f) for f in os.listdir(root_dir)
                if f.endswith(('.jpg', '.png', '.jpeg')) and not f.startswith('.')
            ])
            self.edge_dir = root_dir
        
        print(f"Found {len(self.images)} images in {split} set")
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.edge_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Try to load ground truth edge
        img_filename = os.path.basename(img_path)
        edge_filename = os.path.splitext(img_filename)[0] + '.png'
        edge_path = os.path.join(self.edge_dir, edge_filename)
        
        if not os.path.exists(edge_path):
            # Try same extension as image
            edge_path = os.path.join(self.edge_dir, img_filename)
        
        if not os.path.exists(edge_path):
            # Try _gt suffix
            base = os.path.splitext(img_filename)[0]
            edge_path = os.path.join(self.edge_dir, base + '_gt.png')
        
        if os.path.exists(edge_path):
            edge = Image.open(edge_path).convert('L')
        else:
            # Generate synthetic edge using Canny for demonstration
            import cv2
            img_np = np.array(image)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge = Image.fromarray(edges)
        
        # Transform
        image = self.transform(image)
        edge = self.edge_transform(edge)
        
        # Threshold edge to binary
        edge = (edge > 0.5).float()
        
        return image, edge


class BalancedBCELoss(nn.Module):
    """Class-balanced binary cross-entropy loss"""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        # Count positive and negative pixels
        n_pos = target.sum() + 1e-6
        n_neg = (1 - target).sum() + 1e-6
        n_total = target.numel()
        
        # Class weights
        w_pos = n_neg / n_total
        w_neg = n_pos / n_total
        
        # Balanced BCE
        loss = -w_pos * target * torch.log(pred + 1e-6) - \
               w_neg * (1 - target) * torch.log(1 - pred + 1e-6)
        
        return loss.mean()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for images, edges in pbar:
        images = images.to(device)
        edges = edges.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Deep supervision loss
        if isinstance(outputs, tuple):
            # Training mode: multiple outputs
            final_out, out1, out2, out3, out4 = outputs
            
            # Resize targets for different scales
            h, w = edges.shape[2:]
            
            loss = criterion(final_out, edges)
            loss += criterion(out1, edges)
            loss += criterion(out2, F.interpolate(edges, out2.shape[2:], mode='bilinear'))
            loss += criterion(out3, F.interpolate(edges, out3.shape[2:], mode='bilinear'))
            loss += criterion(out4, F.interpolate(edges, out4.shape[2:], mode='bilinear'))
            
            loss = loss / 5.0  # Average
        else:
            # Single output
            loss = criterion(outputs, edges)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, edges in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            edges = edges.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Use final output only
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = criterion(outputs, edges)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train BPED model')
    parser.add_argument('--dataset_path', type=str, default='../datasets/HED_Small',
                        help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--image_size', type=int, default=320,
                        help='Input image size')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.save_dir, f"bped_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    print(f"Checkpoints will be saved to: {save_path}")
    
    # Model
    model = BPED().to(device)
    print(f"Model created with {model.count_parameters():,} parameters")
    
    # Loss and optimizer
    criterion = BalancedBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Datasets
    try:
        train_dataset = EdgeDataset(args.dataset_path, 'train', args.image_size)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                   shuffle=True, num_workers=2)
    except:
        print("Warning: Could not load training data, using test data for demo")
        train_dataset = EdgeDataset(args.dataset_path, 'test', args.image_size)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=0)
    
    print(f"\nTraining with {len(train_dataset)} images")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
    print("="*60)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"  Train Loss: {train_loss:.4f}")
        
        # Save checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }
            save_file = os.path.join(save_path, 'best_model.pth')
            torch.save(checkpoint, save_file)
            print(f"  ✓ Saved best model: {train_loss:.4f}")
        
        # Save periodic checkpoint
        if epoch % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }
            save_file = os.path.join(save_path, f'epoch_{epoch}.pth')
            torch.save(checkpoint, save_file)
            print(f"  ✓ Saved checkpoint: epoch {epoch}")
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning rate: {current_lr:.6f}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {save_path}")


if __name__ == '__main__':
    import torch.nn.functional as F
    main()
