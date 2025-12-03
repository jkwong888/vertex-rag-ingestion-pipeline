FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY build_index.py .

ENTRYPOINT ["python", "build_index.py"]
