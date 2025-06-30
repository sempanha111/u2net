# Make sure you have opencv installed:
# pip install opencv-python opencv-contrib-python

from PIL import Image

import os.path as osp
import glob
import cv2
import numpy as np
import torch
import app.model.upscale.RRDBNet_arch as arch

class Upscale: # Corrected class name typo from "Upscale"
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f"Upscaler initialized on device: {self.device}")
        
        # Cache for loaded ESRGAN models to avoid reloading
        self.esrgan_models = {}

    def upscale_with_esrgan(self, image: Image.Image, scale: int = 4):
        if scale not in [2, 4]:
            raise ValueError("Scale must be 2 or 4 for the available RealESRGANplus models.")

        print(f"Loading RealESRGAN_x{scale}plus model...")
        model_path = f'saved_models/RealESRGAN/RRDB_ESRGAN_x4.pth'
        
        # Initialize the model architecture
        model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        model = model.to(self.device)
            

        # --- 2. Pre-process Image ---
        # Convert PIL Image to NumPy array, then to PyTorch Tensor
        img = np.array(image.convert("RGB"))
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float() # BGR for model
        img_lr = img.unsqueeze(0)
        img_lr = img_lr.to(self.device)

        # --- 3. Run Inference ---
        with torch.no_grad():
            output_tensor = model(img_lr).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        
        # --- 4. Post-process and Convert to PIL Image ---
        output_np = np.transpose(output_tensor[[2, 1, 0], :, :], (1, 2, 0)) # Back to RGB
        output_np = (output_np * 255.0).round().astype(np.uint8)
        
        output_image = Image.fromarray(output_np)

        print("ESRGAN upscaling complete.")
        return output_image