
import base64
import os
from io import BytesIO

import torch
import torchvision.transforms as transforms

from fastapi import FastAPI, HTTPException, UploadFile, File
from starlette.responses import StreamingResponse, HTMLResponse
from PIL import Image
from typing import Optional

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}




@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):

    contents = await file.read()
    # Lưu tạm thời nội dung ảnh vào một tệp
    with open("uploaded_image.jpg", "wb") as f:
        f.write(contents)

    image_path = os.path.abspath("uploaded_image.jpg")
    print(f"Image saved at {image_path}")

    return {"message": "Image uploaded successfully"}


def genData(data):
    newd = []
    for i in data:
        newd.append(format(ord(i), '08b'))
    return newd


def modPix(pix, data):
    datalist = genData(data)
    lendata = len(datalist)
    imdata = iter(pix)

    for i in range(lendata):
        pix = [value for value in next(imdata)[:3] +
               next(imdata)[:3] +
               next(imdata)[:3]]

        for j in range(0, 8):
            if datalist[i][j] == '0' and pix[j] % 2 != 0:
                pix[j] -= 1
            elif datalist[i][j] == '1' and pix[j] % 2 == 0:
                pix[j] = pix[j] - 1 if pix[j] != 0 else pix[j] + 1

        if i == lendata - 1:
            pix[-1] = pix[-1] - 1 if pix[-1] % 2 == 0 else pix[-1]
        else:
            pix[-1] = pix[-1] - 1 if pix[-1] % 2 != 0 else pix[-1]

        pix = tuple(pix)
        yield pix[0:3]
        yield pix[3:6]
        yield pix[6:9]


def encode_enc(newimg, data):
    w = newimg.size[0]
    (x, y) = (0, 0)
    for pixel in modPix(newimg.getdata(), data):
        newimg.putpixel((x, y), pixel)
        if x == w - 1:
            x = 0
            y += 1
        else:
            x += 1


@app.post("/encode/")
async def encode(file: UploadFile = File(...), data: str = None):
    try:
        image = Image.open(file.file, 'r')

        if data is None:
            raise ValueError('Data is empty')

        newimg = image.copy()
        encode_enc(newimg, data)

        return {"message": "Image encoded successfully!"}

    except Exception as e:
        return {"error": str(e)}


@app.post("/decode/")
async def decode(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file, 'r')

        data = ''
        imgdata = iter(image.getdata())

        while True:
            pixels = [value for value in next(imgdata)[:3] +
                      next(imgdata)[:3] +
                      next(imgdata)[:3]]

            binstr = ''.join(['0' if i % 2 == 0 else '1' for i in pixels[:8]])
            data += chr(int(binstr, 2))

            if pixels[-1] % 2 != 0:
                return {"message": "Data decoded successfully!", "data": data}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
