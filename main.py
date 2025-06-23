import os
from model.u2net import U2NET
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def load_model(model_path):
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()
    return net

def preprocess(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor, image

def postprocess(mask_tensor, original_image):
    mask = mask_tensor.squeeze().detach().numpy()
    mask = (mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask).resize(original_image.size, resample=Image.BILINEAR)
    alpha = mask_img.point(lambda p: p > 128 and 255)
    result = original_image.convert("RGBA")
    result.putalpha(alpha)
    return result

def remove_background(image_path, model_path, output_path):
    model = load_model(model_path)
    input_tensor, original_image = preprocess(image_path)
    with torch.no_grad():
        d1, *_ = model(input_tensor)
    result_img = postprocess(d1, original_image)
    result_img.save(output_path)
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    input_img = "test_images/adorable-angry-animal-208984-thumbnail.jpg"
    model_file = "saved_models/u2net/u2net.pth"
    output_img = "results/adorable-angry-animal-208984-thumbnail_no_bg.png"

    if not os.path.exists("results"):
        os.makedirs("results")

    remove_background(input_img, model_file, output_img)
