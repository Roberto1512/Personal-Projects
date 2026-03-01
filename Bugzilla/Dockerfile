FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# (Opzionale ma spesso necessario) toolchain per pacchetti con estensioni native
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Non-root user (best practice)
RUN useradd -m -u 1000 appuser
RUN mkdir -p /app/labels && chown -R appuser:appuser /app/labels && touch /app/labels/manual_labels.jsonl && chown appuser:appuser /app/labels/manual_labels.jsonl
USER appuser
ENV PATH="/home/appuser/.local/bin:$PATH"

# Copio solo il file requirements prima -> cache OK
COPY --chown=appuser:appuser requirements.docker.txt /app/requirements.docker.txt

# Install deps senza "-e ." (così non serve ancora il codice)
RUN python -m pip install -U pip && grep -vE '^-e[[:space:]]+\.' /app/requirements.docker.txt > /tmp/req.txt && pip install --no-cache-dir -r /tmp/req.txt

# Ora copio il progetto
COPY --chown=appuser:appuser . /app

# Install del progetto in editable (equivalente a "-e .")
RUN pip install --no-cache-dir -e .

EXPOSE 7860
CMD ["sh", "-c", "uvicorn naplace.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]

