FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY rag ./rag
RUN pip install .
COPY samples ./samples

RUN useradd --create-home --uid 1000 rag && mkdir -p /app/storage && chown -R rag /app
USER rag

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
