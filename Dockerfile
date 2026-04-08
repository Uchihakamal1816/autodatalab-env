# AutoDataLab — Hugging Face Spaces (Docker SDK)
# Runs the FastAPI / OpenEnv server on port 7860.
#
# This is the SUBMISSION entrypoint — it exposes the OpenEnv HTTP API:
#   POST /reset    POST /step    GET /state    GET /health
#
# For the interactive Gradio demo, run locally:  python app.py
#
# Secrets (set in HF Space Settings → Repository secrets):
#   GROQ_API_KEY    optional — only needed if running inference from the Space
#   HF_TOKEN        optional — same as above
#
# Build context: repository root
FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY data_cleaning_env /app/data_cleaning_env
COPY inference.py validate_submission.py app.py /app/

# Install env package + extras (openai = LLM client + python-docx)
RUN pip install --no-cache-dir \
    -e "/app/data_cleaning_env[openai]" \
    gradio>=4.0.0 \
    matplotlib>=3.7.0

# Non-root user (HF Spaces requirement)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1
ENV PORT=7860

EXPOSE 7860

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Health check hits the FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:7860/health || exit 1

# Combined mode: Gradio UI at /ui + OpenEnv API at / (reset, step, state, health)
CMD ["python", "app.py", "--hf"]
