
from fastapi import FastAPI, File, UploadFile

from fastapi import Form, Response
from PIL import Image
import io

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


# API nhận ảnh gửi lên từ app di động
import os

# API cắt ảnh theo bounding box gửi lên từ app di động
@app.post("/upload-crop-image/")
async def upload_crop_image(
    file: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    width: int = Form(...),
    height: int = Form(...)
):
    # Đọc nội dung file
    contents = await file.read()
    # Mở ảnh bằng Pillow
    image = Image.open(io.BytesIO(contents))
    # Cắt ảnh theo bounding box
    cropped = image.crop((x, y, x + width, y + height))
    # Tạo thư mục storage nếu chưa tồn tại
    storage_dir = "storage"
    os.makedirs(storage_dir, exist_ok=True)
    # Tạo tên file mới
    base, ext = os.path.splitext(file.filename)
    cropped_filename = f"{base}_cropped{ext}"
    cropped_path = os.path.join(storage_dir, cropped_filename)
    # Lưu ảnh đã cắt vào storage
    cropped.save(cropped_path)
    # Trả file ảnh mới về app
    img_byte_arr = io.BytesIO()
    cropped.save(img_byte_arr, format=image.format or 'PNG')
    img_byte_arr.seek(0)
    return Response(content=img_byte_arr.read(), media_type=file.content_type or 'image/png', headers={"Content-Disposition": f"attachment; filename={cropped_filename}"})


# API lấy danh sách ảnh trong thư mục storage
@app.get("/list-images")
async def list_images():
    storage_dir = "storage"
    if not os.path.exists(storage_dir):
        return {"images": []}
    files = [f for f in os.listdir(storage_dir) if os.path.isfile(os.path.join(storage_dir, f))]
    return {"images": files}

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Đọc nội dung file
    contents = await file.read()
    # Tạo thư mục storage nếu chưa tồn tại
    storage_dir = "storage"
    os.makedirs(storage_dir, exist_ok=True)
    # Đường dẫn lưu file
    file_path = os.path.join(storage_dir, file.filename)
    # Lưu file vào thư mục storage
    with open(file_path, "wb") as f:
        f.write(contents)
    # Trả về tên file và kích thước
    return {"filename": file.filename, "size": len(contents), "saved_path": file_path}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.get("/update/{name}")
async def say_hello(name: str):
    return {"message": f"Update {name}"}



