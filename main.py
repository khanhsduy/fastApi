# ==== IMPORTS ====
import io
import math
import os
import threading
from urllib.parse import quote

import bchlib
import numpy as np
import tensorflow as tf

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
    Response,
    Request,
    HTTPException,
)
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants


# ==== CONFIG ====
STORAGE_DIR = "storage"

# Đổi đường dẫn này thành thư mục model của anh
MODEL_PATH = os.getenv(
    "STEGASTAMP_MODEL_PATH",
    "saved_model/stegastamp_pretrained",
)

IMAGE_SIZE = 400
PACKET_BITS = 96

BCH_POLYNOMIAL = 137
BCH_BITS = 5


# Tạo storage trước khi mount StaticFiles
os.makedirs(STORAGE_DIR, exist_ok=True)


# ==== TENSORFLOW COMPATIBILITY ====
# Hỗ trợ TensorFlow 1.x và TensorFlow 2.x chạy chế độ compat.v1
TF1 = tf.compat.v1 if hasattr(tf, "compat") else tf

if hasattr(tf, "compat"):
    TF1.disable_eager_execution()


# TensorFlow 1.x StegaStamp có thể cần import module này
try:
    import tensorflow.contrib.image  # noqa: F401
except (ImportError, ModuleNotFoundError):
    pass


# ==== GLOBAL DECODER OBJECTS ====
decoder_graph = None
decoder_session = None
input_image_tensor = None
output_secret_tensor = None

# Tránh nhiều request gọi cùng Session đồng thời
decoder_lock = threading.Lock()


# ==== APP INIT ====
app = FastAPI(
    title="StegaStamp Decode API",
    version="1.0.0",
)


# ==== BCH INIT ====
def create_bch():
    """
    Hỗ trợ cả bchlib phiên bản cũ và mới.
    """

    try:
        # Cách dùng trong project StegaStamp cũ
        return bchlib.BCH(
            BCH_POLYNOMIAL,
            BCH_BITS,
        )
    except Exception:
        # Cách dùng của một số phiên bản bchlib mới
        return bchlib.BCH(
            BCH_BITS,
            poly=BCH_POLYNOMIAL,
        )


bch = create_bch()


# ==== LOAD MODEL ====
@app.on_event("startup")
def load_stegastamp_model():
    """
    Load SavedModel một lần khi FastAPI khởi động.
    Không load lại model trong từng request.
    """

    global decoder_graph
    global decoder_session
    global input_image_tensor
    global output_secret_tensor

    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(
            f"Không tìm thấy StegaStamp model: {MODEL_PATH}"
        )

    print(f"Đang load StegaStamp model: {MODEL_PATH}")

    decoder_graph = tf.Graph()
    decoder_session = TF1.Session(graph=decoder_graph)

    with decoder_graph.as_default():
        model = TF1.saved_model.loader.load(
            decoder_session,
            [tag_constants.SERVING],
            MODEL_PATH,
        )

        signature = model.signature_def[
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        ]

        input_image_name = signature.inputs["image"].name
        output_secret_name = signature.outputs["decoded"].name

        input_image_tensor = decoder_graph.get_tensor_by_name(
            input_image_name
        )

        output_secret_tensor = decoder_graph.get_tensor_by_name(
            output_secret_name
        )

    print("StegaStamp model đã sẵn sàng")
    print(f"Input tensor: {input_image_tensor.name}")
    print(f"Output tensor: {output_secret_tensor.name}")


@app.on_event("shutdown")
def close_stegastamp_model():
    global decoder_session

    if decoder_session is not None:
        decoder_session.close()
        decoder_session = None


# ==== DECODE FUNCTIONS ====
def prepare_image(contents: bytes):
    """
    Xử lý ảnh giống hệt decode_image.py:
    - Xoay ảnh theo EXIF
    - Chuyển RGB
    - ImageOps.fit về 400x400
    - Chuẩn hóa về [0, 1]
    """

    try:
        image = Image.open(io.BytesIO(contents))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Không đọc được ảnh: {exc}",
        )

    original_width, original_height = image.size

    # Giữ đúng xử lý của decode_image.py
    image = ImageOps.fit(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
    )

    image_array = np.array(
        image,
        dtype=np.float32,
    )

    image_array /= 255.0

    return image_array, original_width, original_height


def decode_bch_secret(secret):
    """
    Lấy 96 bit đầu và giải mã BCH thành chuỗi UTF-8.
    """

    if len(secret) < PACKET_BITS:
        return {
            "success": False,
            "decoded_text": None,
            "bitflips": -1,
            "message": (
                f"Model chỉ trả về {len(secret)} bit, "
                f"cần ít nhất {PACKET_BITS} bit"
            ),
        }

    # Giữ đúng cách chuyển bit trong decode_image.py
    packet_binary = "".join(
        str(int(bit))
        for bit in secret[:PACKET_BITS]
    )

    try:
        packet = bytearray(
            int(packet_binary[i:i + 8], 2)
            for i in range(0, len(packet_binary), 8)
        )
    except ValueError as exc:
        return {
            "success": False,
            "decoded_text": None,
            "bitflips": -1,
            "bit_string": packet_binary,
            "message": f"Không thể chuyển bit thành dữ liệu: {exc}",
        }

    data = packet[:-bch.ecc_bytes]
    ecc = packet[-bch.ecc_bytes:]

    try:
        if hasattr(bch, "decode_inplace"):
            # bchlib phiên bản cũ
            bitflips = bch.decode_inplace(data, ecc)
        else:
            # bchlib phiên bản mới
            bitflips = bch.decode(data, recv_ecc=ecc)

            if bitflips >= 0 and hasattr(bch, "correct"):
                bch.correct(data, ecc)

    except Exception as exc:
        return {
            "success": False,
            "decoded_text": None,
            "bitflips": -1,
            "bit_string": packet_binary,
            "message": f"Lỗi BCH: {exc}",
        }

    if bitflips == -1:
        return {
            "success": False,
            "decoded_text": None,
            "bitflips": -1,
            "bit_string": packet_binary,
            "message": "Không tìm thấy watermark hoặc BCH không sửa được lỗi",
        }

    try:
        decoded_text = bytes(data).decode("utf-8")
        decoded_text = decoded_text.rstrip("\x00")
    except UnicodeDecodeError:
        return {
            "success": False,
            "decoded_text": None,
            "bitflips": bitflips,
            "bit_string": packet_binary,
            "message": "Giải mã BCH thành công nhưng dữ liệu không phải UTF-8",
        }

    return {
        "success": True,
        "decoded_text": decoded_text,
        "bitflips": bitflips,
        "bit_string": packet_binary,
        "message": "Giải mã watermark thành công",
    }


# ==== ROUTES ====
@app.get("/")
async def root():
    return {
        "message": "StegaStamp API is running",
        "model_path": MODEL_PATH,
        "model_loaded": decoder_session is not None,
    }


@app.post("/decode-image/")
async def decode_image(file: UploadFile = File(...)):
    """
    Nhận một ảnh, chạy StegaStamp decoder và trả thông tin giải mã.
    """

    if decoder_session is None:
        raise HTTPException(
            status_code=503,
            detail="StegaStamp model chưa được load",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="File ảnh rỗng",
        )

    image_array, original_width, original_height = prepare_image(
        contents
    )

    feed_dict = {
        input_image_tensor: [image_array]
    }

    try:
        with decoder_lock:
            result = decoder_session.run(
                [output_secret_tensor],
                feed_dict=feed_dict,
            )

        secret = result[0][0]

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi chạy decoder: {exc}",
        )

    decoded_result = decode_bch_secret(secret)

    return {
        "filename": file.filename,
        "original_width": original_width,
        "original_height": original_height,
        "model_input_width": IMAGE_SIZE,
        "model_input_height": IMAGE_SIZE,
        "secret_size": len(secret),
        **decoded_result,
    }


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    """Nhận ảnh gửi lên và lưu vào storage."""

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="File ảnh rỗng",
        )

    safe_filename = os.path.basename(
        file.filename or "image.jpg"
    )

    file_path = os.path.join(
        STORAGE_DIR,
        safe_filename,
    )

    with open(file_path, "wb") as output_file:
        output_file.write(contents)

    return {
        "filename": safe_filename,
        "size": len(contents),
        "saved_path": file_path,
    }


@app.post("/upload-crop-image/")
async def upload_crop_image(
    file: UploadFile = File(...),
    x: float = Form(...),
    y: float = Form(...),
    width: float = Form(...),
    height: float = Form(...),
):
    """
    Nhận ảnh và bbox x, y, width, height theo ảnh gốc.
    Cắt, lưu và trả lại ảnh JPEG.
    """

    if width <= 0 or height <= 0:
        raise HTTPException(
            status_code=400,
            detail="width/height phải > 0",
        )

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents))
        image = ImageOps.exif_transpose(image)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Không đọc được ảnh",
        )

    image_width, image_height = image.size

    left = max(
        0,
        min(math.floor(x), image_width - 1),
    )

    top = max(
        0,
        min(math.floor(y), image_height - 1),
    )

    right = max(
        left + 1,
        min(math.ceil(x + width), image_width),
    )

    bottom = max(
        top + 1,
        min(math.ceil(y + height), image_height),
    )

    cropped = image.crop(
        (left, top, right, bottom)
    )

    if cropped.mode != "RGB":
        cropped = cropped.convert("RGB")

    safe_filename = os.path.basename(
        file.filename or "image"
    )

    base, _ = os.path.splitext(safe_filename)
    cropped_filename = f"{base}_cropped.jpg"

    cropped_path = os.path.join(
        STORAGE_DIR,
        cropped_filename,
    )

    cropped.save(
        cropped_path,
        format="JPEG",
        quality=95,
    )

    buffer = io.BytesIO()

    cropped.save(
        buffer,
        format="JPEG",
        quality=95,
    )

    return Response(
        content=buffer.getvalue(),
        media_type="image/jpeg",
        headers={
            "Content-Disposition":
                f'inline; filename="{cropped_filename}"'
        },
    )


@app.get("/list-images")
async def list_images(request: Request):
    files = [
        filename
        for filename in os.listdir(STORAGE_DIR)
        if os.path.isfile(
            os.path.join(STORAGE_DIR, filename)
        )
    ]

    image_urls = [
        f"{str(request.base_url)}images/{quote(filename)}"
        for filename in files
    ]

    return {
        "images": files,
        "image_urls": image_urls,
    }


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.get("/update/{name}")
async def say_update(name: str):
    return {"message": f"Update {name}"}


# ==== STATIC FILES ====
app.mount(
    "/images",
    StaticFiles(directory=STORAGE_DIR),
    name="images",
)