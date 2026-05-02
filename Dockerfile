FROM python:3.10-slim

WORKDIR /app

COPY requirements-hf.txt .

RUN pip install --no-cache-dir -r requirements-hf.txt

COPY backend/ ./backend/
COPY src/ ./src/

EXPOSE 7860

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]