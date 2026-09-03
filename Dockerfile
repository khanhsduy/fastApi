FROM ghcr.io/prefix-dev/pixi:latest

WORKDIR /app

COPY pixi.toml pixi.lock ./

RUN pixi install --locked

COPY . .

ENV STEGASTAMP_MODEL_PATH=/app/saved_model/stegastamp_pretrained
ENV STORAGE_DIR=/home/storage
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["pixi", "run", "start"]
