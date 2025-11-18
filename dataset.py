# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class PairedImageDataset(Dataset):
    """
    Expects:
    root/train_black, root/train_colour
    Filenames must match in both folders.
    """
    def __init__(self, root, split='train', image_size=256):
        self.root = root
        self.split = split
        self.image_size = image_size
        black_dir = os.path.join(root, f"{split}_black")
        color_dir = os.path.join(root, f"{split}_colour")
        if not os.path.isdir(black_dir) or not os.path.isdir(color_dir):
            raise ValueError(f"Missing folders: {black_dir} or {color_dir}")

        self.black_paths = sorted([os.path.join(black_dir, f) for f in os.listdir(black_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        self.color_paths = sorted([os.path.join(color_dir, f) for f in os.listdir(color_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])

        # match by filename - create a simple map
        black_map = {os.path.basename(p): p for p in self.black_paths}
        color_map = {os.path.basename(p): p for p in self.color_paths}
        self.pairs = []
        for name, bpath in black_map.items():
            if name in color_map:
                self.pairs.append((bpath, color_map[name]))
        if len(self.pairs) == 0:
            raise ValueError("No matching filenames found between black and colour folders.")

        # transforms
        self.transform_black = T.Compose([
            T.Resize((image_size, image_size)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),  # [0,1]
            T.Normalize((0.5,), (0.5,))  # [-1,1]
        ])
        self.transform_color = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        black_path, color_path = self.pairs[idx]
        black = Image.open(black_path).convert('L')  # grayscale
        color = Image.open(color_path).convert('RGB')
        return self.transform_black(black), self.transform_color(color)
