from fastapi import FastAPI, File, UploadFile, Form, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse, HTMLResponse
from app.model.remove_background.u2net import U2NET
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
import os
import io
import uuid
from fastapi.middleware.cors import CORSMiddleware
from app.model.upscale.upscale import Upscale 
import requests
from bs4 import BeautifulSoup
import yt_dlp
import subprocess
from pytubefix import YouTube
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

upscale_model = Upscale()


def preprocess(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0) # type: ignore
    return tensor, image

def postprocess(mask_tensor, original_image):
    mask = mask_tensor.squeeze().detach().numpy()
    mask = (mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask).resize(original_image.size, resample=Image.BILINEAR) # type: ignore
    alpha = mask_img.point(lambda p: p > 128 and 255) # type: ignore
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

@app.post("/upscale")
async def upscale(file: UploadFile = File(...), scale: int = Form(...)):
    contents = await file.read()
    
    # ✅ Save the uploaded image before processing
    # os.makedirs("before_images", exist_ok=True)
    # original_filename = f"before_images/{uuid.uuid4().hex}_{file.filename}"
    # with open(original_filename, "wb") as f:
    #     f.write(contents)

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Run your upscaling model
    try:
        upscaled_image = upscale_model.upscale_with_esrgan(image, scale=scale)
    except Exception as e:
        print(e)
        return {"error": str(e)}

    # ✅ Save the upscaled result
    os.makedirs("results", exist_ok=True)
    output_filename = f"results/{uuid.uuid4().hex}.png"
    upscaled_image.save(output_filename)

    # print("Saving to:", output_filename)


    # ✅ Return the result image
    return FileResponse(output_filename, media_type="image/png")


@app.get("/tiktok-download")
def download_tiktok(url: str = Query(...)):
    headers = {
        "User-Agent": "Mozilla/5.0",
    }
    try:
        session = requests.Session()

        # Step 1: Get homepage to retrieve CSRF token
        res = session.get("https://musicaldown.com/", headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")

        # Find the form
        form = soup.find("form", {"id": "submit-form"})

        # Extract all input fields
        input_fields = form.find_all("input")

        # Step 2: Build form data
        form_data = {}

        for idx, input_field in enumerate(input_fields):
            name = input_field.get("name")
            if not name:
                continue

            # First input: replace value with TikTok URL
            if idx == 0:
                form_data[name] = url
            else:
                value = input_field.get("value", "")
                form_data[name] = value

        post_url = "https://musicaldown.com/download"
        post_res = session.post(post_url, data=form_data, headers=headers)
        final_post_soup = BeautifulSoup(post_res.text, "html.parser")


        # Find all download buttons
        download_links = final_post_soup.findAll("a", {"class": "download"})
        thumbnail_div = final_post_soup.find("div", {"class": "video-header"})

        # 1. Extract background image from `style`
        bg_url = ""
        style = thumbnail_div.get("style", "")
        if "background-image:url(" in style:
            start = style.find("url(") + 4
            end = style.find(")", start)
            bg_url = style[start:end]

        # 2. Extract inner <img> thumbnail (circle image)
        img_tag = thumbnail_div.find("img")
        circle_img_url = img_tag.get("src") if img_tag else ""

        # 3. Extract username
        username = thumbnail_div.find("h2", class_="video-author").get_text(strip=True)

        # 4. Extract caption/description
        caption = thumbnail_div.find("p", class_="video-desc").get_text(strip=True)

        links = {
            "mp4": None,
            "mp4_hd": None,
            "mp4_watermark": None
        }

        for link in download_links:
            event = link.get("data-event")
            href = link.get("href")

            if event == "mp4_download_click":
                links["mp4"] = href
            elif event == "hd_download_click":
                links["mp4_hd"] = href
            elif event == "watermark_download_click":
                links["mp4_watermark"] = href

        return JSONResponse(content={
            "success": True,
            "bg_url": bg_url,
            "circle_img_url": circle_img_url,
            "username": username,
            "caption": caption,
            "MP4": links["mp4"],
            "MP4_HD": links["mp4_hd"],
            "MP4_with_Watermark": links["mp4_watermark"],
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })



@app.get("/facebook-download")
def download_facebook(url: str = Query(...)):
    try:
        ydl_opts = {
            'quiet': True,
            'skip_download': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        return {
            "page_name": info.get("uploader") or None,
            "title": info.get("title"),
            "video_url": info.get("url"),
            "thumbnail": info.get("thumbnail"),
            "formats": [
                {"format": f.get("format"), "url": f.get("url")}
                for f in info.get("formats", [])
                if f.get("ext") in ["mp4", "m4a"]
            ]
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })
    
@app.get("/download-proxy")
def download_proxy(url: str):
    try:
        r = requests.get(url, stream=True, timeout=10)
        filename = url.split("/")[-1].split("?")[0] or "download.mp4"
        return StreamingResponse(
            r.iter_content(chunk_size=8192),
            media_type='application/octet-stream',
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    




@app.get("/youtube-download/options")
def get_youtube_options(url: str = Query(...)):
    try:
        yt = YouTube(url)

        video_streams = yt.streams.filter(adaptive=True, only_video=True).order_by('resolution').desc()
        options = []

        for i, stream in enumerate(video_streams):
            options.append({
                "itag": stream.itag,
                "resolution": stream.resolution,
                "fps": stream.fps,
                "video_codec": stream.video_codec,
                "container": stream.mime_type.split('/')[-1],
                "filesize_mb": round(stream.filesize / 1048576, 2) if stream.filesize else 'N/A'
            })

        return JSONResponse(status_code=200, content={
            "title": yt.title,
            "channel_name": yt.author,             # ✅ Channel name
            "thumbnail_url": yt.thumbnail_url,     # ✅ Thumbnail
            "video_options": options
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    


@app.get("/youtube-download/download")
def download_and_merge_youtube(
    url: str = Query(...),
    itag: int = Query(...),
    background_tasks: BackgroundTasks = None
):
    try:
        yt = YouTube(url)
        video_stream = yt.streams.get_by_itag(itag)
        if not video_stream:
            return JSONResponse(status_code=400, content={"error": "Invalid itag"})

        audio_stream = yt.streams.filter(only_audio=True).order_by("abr").desc().first()
        if not audio_stream:
            return JSONResponse(status_code=500, content={"error": "No audio stream found"})

        os.makedirs("downloads", exist_ok=True)

        # Generate a unique ID for temp filenames
        uid = uuid.uuid4().hex
        safe_title = "".join(c for c in yt.title if c.isalnum() or c in (" ", ".", "_")).rstrip()

        video_file = f"{uid}_video.{video_stream.subtype}"
        audio_file = f"{uid}_audio.{audio_stream.subtype}"
        output_file = f"{safe_title} - {video_stream.resolution}_{uid}.mp4"

        video_path = os.path.join("downloads", video_file)
        audio_path = os.path.join("downloads", audio_file)
        output_path = os.path.join("downloads", output_file)

        video_stream.download(filename=video_file, output_path="downloads")
        audio_stream.download(filename=audio_file, output_path="downloads")

        subprocess.run([
            "ffmpeg", "-i", video_path, "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac", "-y", output_path
        ], check=True)

        # Background task to delete temp files
        def cleanup():
            for path in [video_path, audio_path, output_path]:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass  # don't fail if already removed

        background_tasks.add_task(cleanup)

        return FileResponse(
            path=output_path,
            filename=f"{safe_title} - {video_stream.resolution}.mp4",
            media_type="video/mp4",
            background=background_tasks
        )

    except subprocess.CalledProcessError as e:
        return JSONResponse(status_code=500, content={"error": f"FFmpeg failed: {e}"} )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)} )