FROM python:3.11-slim

# Hugging Face Spaces runs containers as uid 1000, and anything the app writes
# at runtime must live under that user's home.
RUN useradd -m -u 1000 user

# Create the upload directory with the right owner before any volume is mounted
# over it: Docker seeds a fresh named volume from the image, so the ownership
# set here is what the mounted volume inherits. Created as root and it is
# unwritable by uid 1000.
RUN mkdir -p /data/uploads && chown -R user:user /data

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PORT=7860

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . /app
USER user

# Download the embedding weights during the build. Left to runtime, the first
# visitor waits about twenty seconds for a 130MB download.
RUN python -c "from doc_agent.embeddings import model; model()"

EXPOSE 7860

CMD ["python", "app.py"]
