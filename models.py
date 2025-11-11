import torch
import numpy as np
import cv2
from PIL import Image

def preprocess_img(img_rgb, HW=(256,256)):
    img_resized = img_rgb.resize(HW)
    img_resized_np = np.array(img_resized)[:, :, ::-1]  # RGB→BGR
    lab_rs = cv2.cvtColor(img_resized_np, cv2.COLOR_BGR2Lab).astype(np.float32)
    L_rs = lab_rs[:, :, 0] / 50.0 - 1.0

    img_orig_np = np.array(img_rgb)[:, :, ::-1]
    lab_orig = cv2.cvtColor(img_orig_np, cv2.COLOR_BGR2Lab).astype(np.float32)
    L_orig = lab_orig[:, :, 0] / 50.0 - 1.0

    L_rs_t = torch.from_numpy(L_rs).unsqueeze(0).unsqueeze(0).float()
    L_orig_t = torch.from_numpy(L_orig).unsqueeze(0).unsqueeze(0).float()
    return L_orig_t, L_rs_t

def postprocess_tens(L, ab):
    L = (L + 1.0) * 50.0
    ab = ab * 110.0
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    out_imgs = []
    for i in range(Lab.shape[0]):
        lab = Lab[i].astype(np.float32)
        rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
        rgb = np.clip(rgb / 255.0, 0, 1)
        out_imgs.append(rgb)
    return out_imgs[0]
