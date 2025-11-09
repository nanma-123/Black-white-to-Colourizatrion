# demo_app.py
import streamlit as st
import torch
import numpy as np
from PIL import Image
import os

from models import UNetGenerator
from colorizers.util import preprocess_img, postprocess_tens

st.set_page_config(page_title="Historical Image Colorization", layout="wide")
st.title("AI-Powered Historical Image Colorization")

uploaded = st.file_uploader("Upload a black-and-white image", type=["jpg","png","jpeg"])
checkpoint = st.text_input("Path to generator checkpoint (.pth)", value="checkpoints/g_best.pth")
use_gpu = st.checkbox("Use GPU if available", value=False)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')
    st.write("Device:", device)

    if os.path.exists(checkpoint):
        G = UNetGenerator().to(device)
        G.load_state_dict(torch.load(checkpoint, map_location=device))
        G.eval()

        L_orig, L_rs = preprocess_img(img, HW=(256,256))
        if device.type == 'cuda':
            L_rs = L_rs.cuda()

        with torch.no_grad():
            ab_pred = G(L_rs)
            out = postprocess_tens(L_orig, ab_pred.cpu())
        st.image((out*255).astype('uint8'), caption="Colorized", use_column_width=True)

        # provide download
        out_img = Image.fromarray((out*255).astype('uint8'))
        buf = st.download_button("Download colorized image", data=out_img.tobytes(), file_name="colorized.png")
    else:
        st.warning(f"Checkpoint not found at {checkpoint}. Train the model or provide checkpoint path.")
else:
    st.info("Upload an image to colorize.")
