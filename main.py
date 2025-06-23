from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from model.u2net import U2NET
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
import os
import io
import uuid
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "saved_models/u2net/u2net.pth"
model = U2NET(3, 1)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()


def preprocess(image: Image.Image):
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


@app.post("/remove-bg/")
async def remove_bg(file: UploadFile = File(...)):
    contents = await file.read()
    
    # ✅ Save the uploaded image before processing
    os.makedirs("before_images", exist_ok=True)
    original_filename = f"before_images/{uuid.uuid4().hex}_{file.filename}"
    # with open(original_filename, "wb") as f:
    #     f.write(contents)


    # Convert to image and preprocess
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor, original_image = preprocess(image)

    # Inference
    with torch.no_grad():
        d1, *_ = model(input_tensor)

    # Postprocess and save result
    result_img = postprocess(d1, original_image)
    # os.makedirs("results", exist_ok=True)
    # output_filename = f"results/{uuid.uuid4().hex}.png"
    # result_img.save(output_filename)


    # # ✅ Return path to the image, not the PIL image itself
    # return FileResponse(output_filename, media_type="image/png")


    buffer = io.BytesIO()
    result_img.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")
