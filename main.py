

# ==== IMPORTS ====
import io, os, math
from fastapi import FastAPI, File, UploadFile, Form, Response, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

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
    x: float = Form(...),
    y: float = Form(...),
    width: float = Form(...),
    height: float = Form(...),
):
    """Nhận ảnh + bbox (x,y,w,h theo ảnh gốc), cắt, lưu và trả ảnh đã cắt."""

    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="width/height phải > 0")

    contents = await file.read()

    # 1) Mở ảnh & xử lý xoay EXIF
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh")
    image = ImageOps.exif_transpose(image)

    W, H = image.size

    # 2) Clamp biên + dùng floor/ceil
    L = max(0, min(math.floor(x), W - 1))
    T = max(0, min(math.floor(y), H - 1))
    R = max(L + 1, min(math.ceil(x + width), W))
    B = max(T + 1, min(math.ceil(y + height), H))

    # 3) Crop
    cropped = image.crop((L, T, R, B))

    # 4) Lưu file (JPEG để nhẹ hơn)
    storage_dir = "storage"
    os.makedirs(storage_dir, exist_ok=True)
    base, _ = os.path.splitext(file.filename or "image")
    cropped_filename = f"{base}_cropped.jpg"
    cropped_path = os.path.join(storage_dir, cropped_filename)

    if cropped.mode != "RGB":
        cropped = cropped.convert("RGB")

    cropped.save(cropped_path, format="JPEG", quality=95)

    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=95)
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="image/jpeg",
        headers={"Content-Disposition": f'inline; filename="{cropped_filename}"'}
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



