"""
Create a 2x2 comparison image with original + all 3 edge detection results
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_comparison_image():
    """Combine original image with 3 edge detection results into one 2x2 grid"""
    
    # Load images
    original = cv2.imread('./test_image.jpg')
    xyw = cv2.imread('./xyw_result.jpg', cv2.IMREAD_GRAYSCALE)
    pidinet = cv2.imread('./pidinet_result.jpg', cv2.IMREAD_GRAYSCALE)
    canny = cv2.imread('./canny_result.jpg', cv2.IMREAD_GRAYSCALE)
    
    if original is None or xyw is None or pidinet is None or canny is None:
        print("❌ Error: Missing one or more result images!")
        print("   Make sure these files exist:")
        print("   - test_image.jpg")
        print("   - xyw_result.jpg")
        print("   - pidinet_result.jpg")
        print("   - canny_result.jpg")
        return False
    
    # Get dimensions
    h, w = original.shape[:2]
    print(f"📐 Original image size: {w}×{h}")
    
    # Convert grayscale results to BGR for stacking
    xyw_bgr = cv2.cvtColor(xyw, cv2.COLOR_GRAY2BGR)
    pidinet_bgr = cv2.cvtColor(pidinet, cv2.COLOR_GRAY2BGR)
    canny_bgr = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    
    # Resize all to match original dimensions
    h_orig, w_orig = original.shape[:2]
    xyw_bgr = cv2.resize(xyw_bgr, (w_orig, h_orig))
    pidinet_bgr = cv2.resize(pidinet_bgr, (w_orig, h_orig))
    canny_bgr = cv2.resize(canny_bgr, (w_orig, h_orig))
    
    print(f"xyw.pth All images resized to: {w_orig}×{h_orig}")
    
    # Create top row (original + XYW)
    top_row = np.hstack([original, xyw_bgr])
    
    # Create bottom row (PiDiNet + Canny)
    bottom_row = np.hstack([pidinet_bgr, canny_bgr])
    
    # Create full 2x2 grid
    grid = np.vstack([top_row, bottom_row])
    
    # Add text labels using PIL (for better text rendering)
    grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(grid_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # Try to use a nice font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", int(h_orig * 0.04))
    except:
        font = ImageFont.load_default()
    
    # Add labels with background
    label_color = (255, 255, 255)  # White text
    bg_color = (0, 0, 0)  # Black background
    
    # Helper function to draw text with background
    def draw_label(x, y, text):
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle(bbox, fill=bg_color)
        draw.text((x, y), text, fill=label_color, font=font)
    
    # Add labels
    margin = int(w_orig * 0.02)
    draw_label(margin, margin, "📸 Original")
    draw_label(w_orig + margin, margin, "🔵 XYW-Net")
    draw_label(margin, h_orig + margin, "🔴 PiDiNet")
    draw_label(w_orig + margin, h_orig + margin, "🟡 Canny")
    
    # Convert back to OpenCV format
    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Save result
    output_path = './comparison_grid.jpg'
    cv2.imwrite(output_path, result)
    
    print(f"\nxyw.pth Comparison image created!")
    print(f"   Saved to: {output_path}")
    print(f"   Size: {result.shape[1]}×{result.shape[0]} pixels")
    
    return True

if __name__ == '__main__':
    print("="*70)
    print("🎨 Creating 2x2 Comparison Image")
    print("="*70)
    create_comparison_image()
