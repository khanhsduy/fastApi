

# ==== IMPORTS ====
import os
import io
from fastapi import FastAPI, File, UploadFile, Form, Response, Request
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ==== APP INIT ====
app = FastAPI()

# ==== ROUTES ====
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello World"}


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    """Nhận ảnh gửi lên và lưu vào storage."""
    contents = await file.read()
    storage_dir = "storage"
    os.makedirs(storage_dir, exist_ok=True)
    file_path = os.path.join(storage_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)
    return {"filename": file.filename, "size": len(contents), "saved_path": file_path}


@app.post("/upload-crop-image/")
async def upload_crop_image(
    file: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    width: int = Form(...),
    height: int = Form(...)
):
    """Nhận ảnh và bounding box, cắt ảnh, lưu và trả về ảnh đã cắt."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    cropped = image.crop((x, y, x + width, y + height))
    storage_dir = "storage"
    os.makedirs(storage_dir, exist_ok=True)
    base, ext = os.path.splitext(file.filename)
    cropped_filename = f"{base}_cropped{ext}"
    cropped_path = os.path.join(storage_dir, cropped_filename)
    cropped.save(cropped_path)
    img_byte_arr = io.BytesIO()
    cropped.save(img_byte_arr, format=image.format or 'PNG')
    img_byte_arr.seek(0)
    return Response(
        content=img_byte_arr.read(),
        media_type=file.content_type or 'image/png',
        headers={"Content-Disposition": f"attachment; filename={cropped_filename}"}
    )


@app.get("/list-images")
async def list_images(request: Request):
    """Trả về danh sách file và link ảnh trong storage."""
    storage_dir = "storage"
    if not os.path.exists(storage_dir):
        return {"images": []}
    files = [f for f in os.listdir(storage_dir) if os.path.isfile(os.path.join(storage_dir, f))]
    image_urls = [str(request.base_url) + f"images/{f}" for f in files]
    return {"images": files, "image_urls": image_urls}


# ==== STATIC FILES ====
app.mount("/images", StaticFiles(directory="storage"), name="images")


# ==== DEMO ROUTES ====
@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.get("/update/{name}")
async def say_update(name: str):
    return {"message": f"Update {name}"}



