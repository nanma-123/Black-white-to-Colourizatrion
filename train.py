# train.py
import os
import random
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import save_image

from models import UNetGenerator, PatchDiscriminator
from colorizers.util import load_img

# ---- Dataset ----
class PairedColorDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=256, augment=True):
        self.gray_dir = os.path.join(root_dir, f"{split}_black")
        self.color_dir = os.path.join(root_dir, f"{split}_colour")
        self.files = sorted(os.listdir(self.color_dir))
        self.augment = augment and split=='train'
        self.img_size = img_size
        self.to_tensor = T.Compose([T.Resize((img_size,img_size)), T.ToTensor()])
        self.normalize = T.Normalize(mean=[0.5]*3, std=[0.5]*3)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        gray_path = os.path.join(self.gray_dir, fname)
        color_path = os.path.join(self.color_dir, fname)
        if not os.path.exists(gray_path):
            color_img = Image.open(color_path).convert('RGB')
            gray_img = color_img.convert('L').convert('RGB')
        else:
            gray_img = Image.open(gray_path).convert('RGB')
            color_img = Image.open(color_path).convert('RGB')

        if self.augment and random.random() < 0.5:
            gray_img = T.functional.hflip(gray_img)
            color_img = T.functional.hflip(color_img)

        gray_t = self.to_tensor(gray_img)
        color_t = self.to_tensor(color_img)

        gray_single = T.functional.rgb_to_grayscale(gray_t, num_output_channels=1)
        gray_rep = gray_single.repeat(3,1,1)

        gray_rep = self.normalize(gray_rep)
        color_t = self.normalize(color_t)

        return {'gray': gray_rep, 'color': color_t, 'fname': fname}

# ---- helpers ----
def tensor_to_np_img(tensor):
    img = tensor.detach().cpu().float()
    img = (img + 1) / 2.0
    img = img.numpy().transpose(1,2,0)
    img = np.clip(img*255.0, 0, 255).astype(np.uint8)
    return img

# ---- train loop ----
def train(data_dir, epochs=30, batch_size=8, img_size=256, lr=2e-4, out_dir='checkpoints'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = PairedColorDataset(data_dir, split='train', img_size=img_size, augment=True)
    val_ds = PairedColorDataset(data_dir, split='test', img_size=img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))

    adv_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    os.makedirs(out_dir, exist_ok=True)

    best_psnr = 0.0
    for epoch in range(1, epochs+1):
        G.train(); D.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in loop:
            gray = batch['gray'].to(device)
            real_color = batch['color'].to(device)

            # Train D
            opt_D.zero_grad()
            real_in = torch.cat([gray, real_color], dim=1)
            out_real = D(real_in)
            real_label = torch.ones_like(out_real).to(device)
            loss_D_real = adv_loss(out_real, real_label)

            fake_color = G(gray)
            fake_in = torch.cat([gray, fake_color.detach()], dim=1)
            out_fake = D(fake_in)
            fake_label = torch.zeros_like(out_fake).to(device)
            loss_D_fake = adv_loss(out_fake, fake_label)

            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            opt_D.step()

            # Train G
            opt_G.zero_grad()
            gen_in = torch.cat([gray, fake_color], dim=1)
            out_gen = D(gen_in)
            loss_G_adv = adv_loss(out_gen, real_label)
            loss_G_l1 = l1_loss(fake_color, real_color) * 100.0
            loss_G = loss_G_adv + loss_G_l1
            loss_G.backward()
            opt_G.step()

            loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())

        # save checkpoint
        torch.save(G.state_dict(), os.path.join(out_dir, f"g_epoch{epoch}.pth"))
        torch.save(D.state_dict(), os.path.join(out_dir, f"d_epoch{epoch}.pth"))

        # simple validation: save first val example outputs
        G.eval()
        with torch.no_grad():
            for i, b in enumerate(val_loader):
                gray = b['gray'].to(device)
                real = b['color'].to(device)
                fake = G(gray)
                save_image((fake+1)/2.0, os.path.join(out_dir, f"val_{epoch}_{i}_fake.png"))
                save_image((real+1)/2.0, os.path.join(out_dir, f"val_{epoch}_{i}_real.png"))
                break

    print("Training finished. Models saved to", out_dir)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--out_dir', default='checkpoints')
    args = p.parse_args()
    train(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, img_size=args.img_size, out_dir=args.out_dir)
