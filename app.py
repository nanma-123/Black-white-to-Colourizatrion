import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import urllib.request

from models import UNetGenerator
from colorizers.util import preprocess_img, postprocess_tens

# -------------------------------------------------------
# Streamlit Configuration
# -------------------------------------------------------
st.set_page_config(page_title="AI-Powered Historical Image Colorization", layout="wide")
st.title("🎨 AI-Powered Historical Image Colorization")
st.write("Upload a historical black-and-white photograph and watch it come to life in color!")

# -------------------------------------------------------
# Helper: Ensure checkpoint exists
# -------------------------------------------------------
def ensure_checkpoint():
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/g_best.pth"
    if not os.path.exists(ckpt_path):
        st.info("Downloading pretrained model (~100 MB)... Please wait.")
        url = "https://people.eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_release/colorization_release_v2-9b330a0b.pth"
        urllib.request.urlretrieve(url, ckpt_path)
        st.success("✅ Model downloaded successfully!")
    return ckpt_path

checkpoint = ensure_checkpoint()

# -------------------------------------------------------
# File Upload Section
# -------------------------------------------------------
uploaded = st.file_uploader("📷 Upload a black-and-white image", type=["jpg", "png", "jpeg"])
use_gpu = st.checkbox("Use GPU if available", value=False)

# -------------------------------------------------------
# Inference / Colorization
# -------------------------------------------------------
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

    device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')
    st.write(f"**Device:** {device}")

    # Load model
    G = UNetGenerator().to(device)
    G.load_state_dict(torch.load(checkpoint, map_location=device))
    G.eval()

    # Preprocess
    L_orig, L_rs = preprocess_img(img, HW=(256, 256))
    if device.type == 'cuda':
        L_rs = L_rs.cuda()

    # Colorize
    with torch.no_grad():
        ab_pred = G(L_rs)
        out = postprocess_tens(L_orig, ab_pred.cpu())

    # Display
    st.image((out * 255).astype("uint8"), caption="🌈 Colorized Result", use_column_width=True)

    # Download Button
    out_img = Image.fromarray((out * 255).astype("uint8"))
    st.download_button(
        label="⬇️ Download Colorized Image",
        data=out_img.tobytes(),
        file_name="colorized_image.png",
        mime="image/png"
    )

else:
    st.info("⬆️ Please upload a historical black-and-white image to begin.")
