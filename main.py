
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



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
