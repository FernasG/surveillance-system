FROM python:3.10.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY guard/requirements.txt /app/

# Install CPU-only torch first: the default PyPI aarch64 wheels bundle ~3.5GB
# of CUDA/triton libraries that are useless on the Pi.
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        $(grep -E '^(torch|torchvision)==' requirements.txt) \
    && pip install --no-cache-dir -r requirements.txt

COPY guard/ ./guard/

EXPOSE 3000

CMD ["uvicorn", "guard.api.server:app", "--host", "0.0.0.0", "--port", "3000"]