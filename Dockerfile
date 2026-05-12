FROM python:3.10.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libglib2.0-0 \
    libavcodec-dev \
    libswscale-dev \
    libgl1-mesa-glx \
    libavformat-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir git+https://github.com/openai/CLIP.git

COPY guard/ ./guard/

# CMD ["python3", "-m", "app.main"]

CMD ["tail", "-f", "/dev/null"]