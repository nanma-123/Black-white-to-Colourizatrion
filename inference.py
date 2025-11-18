# inference.py
import torch
from PIL import Image
import torchvision.transforms as T
import os
from models import UNetGenerator

def load_image(path, image_size=256, device='cpu'):
    transform_in = T.Compose([
        T.Resize((image_size,image_size)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    img = Image.open(path).convert('L')
    return transform_in(img).unsqueeze(0).to(device)

def save_output(tensor, out_path):
    # denormalize
    t = tensor.detach().cpu() * 0.5 + 0.5
    # clamp and convert to PIL
    T.ToPILImage()(t.squeeze(0)).save(out_path)

def colorize_single(ckpt_path, input_path, out_path, image_size=256, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    G = UNetGenerator(in_channels=1, out_channels=3).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(ckpt['G_state'])
    G.eval()
    inp = load_image(input_path, image_size=image_size, device=device)
    with torch.no_grad():
        out = G(inp)
    save_output(out, out_path)
    print("Saved colorized:", out_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--size', type=int, default=256)
    args = parser.parse_args()
    colorize_single(args.ckpt, args.input, args.out, image_size=args.size)
