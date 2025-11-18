# train.py
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import PairedImageDataset
from models import UNetGenerator, PatchDiscriminator
from torchvision.utils import save_image
from tqdm import tqdm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train(root='Data', epochs=100, batch_size=8, image_size=256, lr=2e-4, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    dataset = PairedImageDataset(root, split='train', image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    G = UNetGenerator(in_channels=1, out_channels=3).to(device)
    D = PatchDiscriminator(in_channels=4).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    # Losses
    adversarial_loss = nn.BCEWithLogitsLoss().to(device)
    l1_loss = nn.L1Loss().to(device)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    sample_dir = 'samples'
    os.makedirs(sample_dir, exist_ok=True)
    ckpt_dir = 'checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, epochs+1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for i, (black, color) in enumerate(pbar):
            black = black.to(device)          # (B,1,H,W)
            color = color.to(device)          # (B,3,H,W)

            # Ground truths
            valid = torch.ones((black.size(0),1,30,30), device=device)  # patch size depends on image_size; 256 -> about 30
            fake_label = torch.zeros_like(valid)

            # -------------
            # Train Generator (G)
            # -------------
            opt_G.zero_grad()
            fake_color = G(black)
            # discriminator on concatenated input
            pred_fake = D(torch.cat([black, fake_color], dim=1))
            g_adv = adversarial_loss(pred_fake, valid)
            g_l1 = l1_loss(fake_color, color) * 100.0  # lambda L1
            g_loss = g_adv + g_l1
            g_loss.backward()
            opt_G.step()

            # -------------
            # Train Discriminator (D)
            # -------------
            opt_D.zero_grad()
            pred_real = D(torch.cat([black, color], dim=1))
            d_real_loss = adversarial_loss(pred_real, valid)
            pred_fake_detach = D(torch.cat([black, fake_color.detach()], dim=1))
            d_fake_loss = adversarial_loss(pred_fake_detach, fake_label)
            d_loss = 0.5*(d_real_loss + d_fake_loss)
            d_loss.backward()
            opt_D.step()

            pbar.set_postfix({'d_loss': d_loss.item(), 'g_loss': g_loss.item(), 'g_l1': g_l1.item()})

            # save some samples occasionally
            if i % 200 == 0:
                # combine black (repeat channel) + fake + real for visualization
                # denormalize to [0,1]
                def denorm(x):
                    return (x * 0.5) + 0.5
                b = denorm(black.repeat(1,3,1,1))
                f = denorm(fake_color)
                r = denorm(color)
                grid = torch.cat([b, f, r], dim=0)
                save_image(grid, os.path.join(sample_dir, f'epoch{epoch:03d}_iter{i:04d}.png'), nrow=black.size(0))

        # save checkpoints
        torch.save({'epoch': epoch, 'G_state': G.state_dict(), 'D_state': D.state_dict(),
                    'opt_G': opt_G.state_dict(), 'opt_D': opt_D.state_dict()},
                   os.path.join(ckpt_dir, f'ckpt_epoch_{epoch}.pt'))

if __name__ == "__main__":
    train(root='Data', epochs=50, batch_size=8, image_size=256)
