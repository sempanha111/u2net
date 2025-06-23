from PIL import Image
from app.model.upscale.realesrganer import RealESRGANer
from app.model.upscale.rrdbnet_arch import RRDBNet

import cv2
import numpy as np

class Upscale:
    def __init__(self):
        
        pass

    def enhance(self, image: Image.Image, scale=2):
        if scale not in [2, 4]:
            raise ValueError("Scale must be 2 or 4")
        return self.upscale_with_realesrgan(image, scale)

    def upscale_with_realesrgan(self, image: Image.Image, scale: int = 4):
        model_path = f'saved_models/RealESRGAN/RealESRGAN_x{scale}plus.pth'
        # Real-ESRGAN x2 or x4 uses RRDBNet
        model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
        # print(model)


        restorer = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False
        )

        img_np = np.array(image)[:, :, ::-1]  # RGB to BGR
        output, _ = restorer.enhance(img_np, outscale=scale)
        result_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        return result_image
