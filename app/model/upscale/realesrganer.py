import torch
import numpy as np
import cv2

class RealESRGANer:
    def __init__(self, scale, model_path, model, tile=0, tile_pad=10, pre_pad=0,
                 half=False, device=None):
        self.scale = scale
        self.model_path = model_path
        self.model = model
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.half = half
        self.device = device if device else torch.device('cpu')

        self._load_model()

    def _load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'params_ema' in checkpoint:
            state_dict = checkpoint['params_ema']
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.model = self.model.to(self.device)

        if self.half:
            self.model = self.model.half()

    def enhance(self, img, outscale=4):
        # img: numpy.ndarray BGR uint8 image

        img = img.astype(np.float32) / 255.
        # Pad image if needed
        if self.pre_pad != 0:
            img = np.pad(img, ((self.pre_pad, self.pre_pad),
                               (self.pre_pad, self.pre_pad), (0, 0)), 'reflect')

        # CHW, BGR to RGB
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)
        if self.half:
            img = img.half()

        with torch.no_grad():
            output = self.model(img).data.float().cpu().clamp_(0, 1).numpy()

        output = np.transpose(output[0], (1, 2, 0))  # HWC RGB
        output = output[:, :, [2, 1, 0]]  # RGB to BGR
        output = (output * 255.0).round().astype(np.uint8)

        # Remove padding
        if self.pre_pad != 0:
            output = output[self.pre_pad:-self.pre_pad, self.pre_pad:-self.pre_pad, :]

        # Resize if outscale != model scale
        if outscale != self.scale:
            output = cv2.resize(output, None, fx=outscale / self.scale,
                                fy=outscale / self.scale,
                                interpolation=cv2.INTER_LINEAR)

        return output, None
