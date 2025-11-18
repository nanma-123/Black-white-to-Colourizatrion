# visualize.py
from PIL import Image
import matplotlib.pyplot as plt

def show_side_by_side(black_path, color_path, pred_path):
    b = Image.open(black_path).convert('L')
    c = Image.open(color_path).convert('RGB')
    p = Image.open(pred_path).convert('RGB')
    fig, axes = plt.subplots(1,3, figsize=(12,5))
    axes[0].imshow(b, cmap='gray'); axes[0].set_title('Input (B/W)'); axes[0].axis('off')
    axes[1].imshow(c); axes[1].set_title('Ground Truth'); axes[1].axis('off')
    axes[2].imshow(p); axes[2].set_title('Prediction'); axes[2].axis('off')
    plt.show()
