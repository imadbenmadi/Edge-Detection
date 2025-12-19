import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
import model
import os
import time

def test_single_image(image_path, model_path='./xyw.pth'):
    """
    Test XYW-Net on a single image
    
    Args:
        image_path: Path to the input image
        model_path: Path to trained model checkpoint
    """
    
    # Load model
    print("Loading model...")
    net = model.Net().eval()
    
    # Check if GPU available - PRIORITY: CUDA first
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"xyw.pth Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠️ GPU not available, using CPU (slower)")
    
    net = net.to(device)
    
    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        net.load_state_dict(checkpoint)
        print(f"xyw.pth Model loaded from {model_path}")
    else:
        print(f"⚠️ Model not found at {model_path}")
        print("Using randomly initialized model (not trained!)")
    
    # Load and preprocess image
    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (W, H)
    print(f"Original image size: {original_size}")
    
    img_array = np.array(img, dtype=np.float32)
    print(f"Image array shape: {img_array.shape}")
    
    # Normalize to [0, 1]
    if img_array.max() > 1:
        img_array = img_array / 255.0
    
    # Convert to tensor (B, C, H, W)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    print(f"Tensor shape: {img_tensor.shape}")
    
    # Run inference
    print("Running inference...")
    start_time = time.time()
    with torch.no_grad():
        output = net(img_tensor)
    inference_time = time.time() - start_time
    
    # Process output
    output = output.cpu().numpy()[0, 0]  # Remove batch and channel dims
    output = (output * 255).astype(np.uint8)  # Convert to 0-255
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min()}, {output.max()}]")
    print(f"⏱️ Inference time: {inference_time:.3f}s")
    
    # Save result to MAIN FOLDER (c:\Users\imed\Desktop\XYW-Net\)
    # Get filename without extension
    filename_only = os.path.basename(image_path).split('.')[0]
    file_extension = os.path.splitext(image_path)[1]
    
    # Save in main project folder
    output_path = f'./{filename_only}_edge{file_extension}'
    
    cv2.imwrite(output_path, output)
    print(f"xyw.pth Edge map saved to: {output_path}")
    
    # Display
    print("\n" + "="*60)
    print("📊 RESULTS:")
    print("="*60)
    print(f"  Input image:   {image_path}")
    print(f"  Output image:  {output_path}")
    print(f"  Device used:   {device}")
    print(f"  Image size:    {original_size}")
    print(f"  Inference time: {inference_time:.3f}s")
    print("="*60)
    
    return output, output_path

if __name__ == '__main__':
    # Test on the image
    image_path = './test_image.jpg'  # ← Change this to the image path
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("\n" + "="*60)
        print("HOW TO USE:")
        print("="*60)
        print("1. Place the image in the project folder")
        print("2. Update 'image_path' variable with the image filename")
        print("3. Run: python run.py")
        print("\nExample:")
        print("   image_path = './my_photo.jpg'")
        print("   image_path = './picture.png'")
        print("="*60)
    else:
        output, output_path = test_single_image(image_path)
        print("\nxyw.pth Done! the edge detection result is ready!")