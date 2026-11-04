FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 ARTIFACT_DIR=/app/artifacts

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu .

# Mount or copy the output of scripts/train_fusion.py here.
COPY artifacts ./artifacts

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "steam_rec.serve:app", "--host", "0.0.0.0", "--port", "8000"]
