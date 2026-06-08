FROM python:3.10.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y && rm -rf /var/lib/apt/lists/*

COPY guard/requirements.txt /app/

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY guard/ ./guard/

EXPOSE 3000

CMD ["uvicorn", "guard.api.server:app", "--reload", "--host", "0.0.0.0", "--port", "3000"]