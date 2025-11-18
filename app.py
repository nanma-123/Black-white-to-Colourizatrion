import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import numpy as np

# -----------------------------
# 1. U-Net Model Architecture
# -----------------------------
class UNetColorization(nn.Module):
    def __init__(self):
        super(UNetColorization, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(1, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.final(d1))

# -----------------------------
# 2. Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = UNetColorization()
    model.load_state_dict(torch.load("unet_colorization.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -----------------------------
# 3. Streamlit UI
# -----------------------------
st.title("🎨 AI-Powered Historical Image Colorization")
st.write("Upload a black-and-white photo to generate a colorized version.")

uploaded_file = st.file_uploader("Upload grayscale image", type=["jpg", "jpeg", "png"])

# -----------------------------
# 4. Process Image
# -----------------------------
if uploaded_file is not None:
    # Load Image
    img = Image.open(uploaded_file).convert("L")  # Force grayscale
    img_resized = img.resize((128, 128))

    # Transform to tensor
    to_tensor = T.ToTensor()
    gray_tensor = to_tensor(img_resized).unsqueeze(0)  # Shape: [1,1,128,128]

    # Model Prediction
    with torch.no_grad():
        output = model(gray_tensor).squeeze(0).permute(1,2,0).numpy()

    # Clip values
    output = np.clip(output, 0, 1)

    # Convert to displayable image
    colorized_img = Image.fromarray((output * 255).astype("uint8"))

    # -----------------------------
    # 5. Show Results
    # -----------------------------
    st.subheader("Input vs Output")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img_resized, caption="Grayscale Input", use_column_width=True)

    with col2:
        st.image(colorized_img, caption="Colorized Output", use_column_width=True)

    st.success("Colorization completed!")

