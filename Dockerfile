# CPU inference image (works on Hugging Face Spaces "Docker" SDK, Fly.io, Render, ...).
# Build after training:  docker build -t tinylm . && docker run -p 7860:7860 tinylm
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir einops numpy regex tqdm fastapi "uvicorn[standard]"
COPY tinylm/ tinylm/
COPY server/ server/
COPY artifacts/tokenizer-tinystories-10k/ artifacts/tokenizer-tinystories-10k/
COPY deploy/model.pt deploy/model.pt
ENV TINYLM_CKPT=deploy/model.pt \
    TINYLM_TOKENIZER=artifacts/tokenizer-tinystories-10k \
    TINYLM_DEVICE=cpu \
    PYTHONPATH=/app
EXPOSE 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
