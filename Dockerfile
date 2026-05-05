FROM python:3.10-slim
ENV TZ=Asia/Karachi

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-hf.txt .

RUN pip install --no-cache-dir -r requirements-hf.txt

COPY backend/ ./backend/
COPY src/ ./src/

EXPOSE 7860

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]