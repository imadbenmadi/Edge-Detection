#!/usr/bin/env python3
"""
Dataset Download Helper
=======================

Downloads and prepares edge detection datasets.

Supported datasets:
- BSDS500
- BIPED
- NYUD

Usage:
    python download_datasets.py --dataset BSDS500
    python download_datasets.py --all  # Download all datasets
"""

import os
import sys
import argparse
import urllib.request
import tarfile
from pathlib import Path
import shutil

class DatasetDownloader:
    """Dataset downloader"""
    
    def __init__(self, output_dir='./data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url, output_path, name=None):
        """Download file with progress"""
        if name is None:
            name = Path(url).name
        
        if Path(output_path).exists():
            print(f"  ✓ {name} already exists")
            return
        
        print(f"  ⬇ Downloading {name}...")
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"  ✓ Downloaded to {output_path}")
        except Exception as e:
            print(f"  ✗ Error downloading {name}: {e}")
            return False
        
        return True
    
    def extract_tar(self, tar_path, output_dir):
        """Extract tar archive"""
        print(f"  ⏳ Extracting {tar_path.name}...")
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(output_dir)
            print(f"  ✓ Extracted to {output_dir}")
            return True
        except Exception as e:
            print(f"  ✗ Error extracting: {e}")
            return False
    
    def download_bsds500(self):
        """Download BSDS500"""
        print("\n" + "="*60)
        print("BSDS500 Dataset")
        print("="*60)
        
        bsds_dir = self.output_dir / 'BSDS500'
        bsds_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nNote: BSDS500 dataset is large (~1.5 GB)")
        print("Source: http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz")
        
        url = "http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz"
        tar_path = bsds_dir / 'HED-BSDS.tar.gz'
        
        if self.download_file(url, tar_path, "HED-BSDS.tar.gz"):
            self.extract_tar(tar_path, bsds_dir)
            # Clean up tar
            tar_path.unlink()
        
        # Check structure
        if (bsds_dir / 'HED-BSDS' / 'train' / 'images').exists():
            print("  ✓ Dataset structure verified")
        else:
            print("  ⚠ Dataset structure may need adjustment")
    
    def download_biped(self):
        """Download BIPED"""
        print("\n" + "="*60)
        print("BIPED Dataset")
        print("="*60)
        
        print("\nBIPED dataset must be downloaded manually:")
        print("1. Visit: https://github.com/xavysp/BIPED")
        print("2. Follow the download instructions")
        print("3. Extract to: ./data/BIPED/")
        print("\nExpected structure:")
        print("  ./data/BIPED/")
        print("    ├── BSDS/")
        print("    ├── train/")
        print("    │   ├── images/")
        print("    │   └── edges/")
        print("    └── test/")
        print("        ├── images/")
        print("        └── edges/")
    
    def download_nyud(self):
        """Download NYUD"""
        print("\n" + "="*60)
        print("NYUD Dataset")
        print("="*60)
        
        nyud_dir = self.output_dir / 'NYUD'
        nyud_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nNote: NYUD dataset is large (~1.0 GB)")
        print("Source: http://mftp.mmcheng.net/liuyun/rcf/data/NYUD.tar.gz")
        
        url = "http://mftp.mmcheng.net/liuyun/rcf/data/NYUD.tar.gz"
        tar_path = nyud_dir / 'NYUD.tar.gz'
        
        if self.download_file(url, tar_path, "NYUD.tar.gz"):
            self.extract_tar(tar_path, nyud_dir)
            # Clean up tar
            tar_path.unlink()
        
        print("  ✓ NYUD dataset downloaded")
    
    def verify_datasets(self):
        """Verify downloaded datasets"""
        print("\n" + "="*60)
        print("Dataset Verification")
        print("="*60)
        
        datasets = {
            'BSDS500': self.output_dir / 'BSDS500',
            'BIPED': self.output_dir / 'BIPED',
            'NYUD': self.output_dir / 'NYUD'
        }
        
        for name, path in datasets.items():
            if path.exists():
                num_files = len(list(path.rglob('*')))
                print(f"✓ {name}: {num_files} files")
            else:
                print(f"✗ {name}: Not found")

def main():
    parser = argparse.ArgumentParser(
        description="Download edge detection datasets"
    )
    parser.add_argument('--dataset', type=str, 
                       choices=['BSDS500', 'BIPED', 'NYUD'],
                       help='Specific dataset to download')
    parser.add_argument('--all', action='store_true',
                       help='Download all datasets')
    parser.add_argument('--verify', action='store_true',
                       help='Verify downloaded datasets')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Bio-XYW-Net Dataset Downloader")
    print("="*60 + "\n")
    
    downloader = DatasetDownloader(args.output_dir)
    
    if args.verify:
        downloader.verify_datasets()
    elif args.all:
        print("Downloading all datasets (this may take a while)...\n")
        downloader.download_bsds500()
        downloader.download_nyud()
        downloader.download_biped()
        downloader.verify_datasets()
    elif args.dataset:
        if args.dataset == 'BSDS500':
            downloader.download_bsds500()
        elif args.dataset == 'BIPED':
            downloader.download_biped()
        elif args.dataset == 'NYUD':
            downloader.download_nyud()
    else:
        parser.print_help()
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
