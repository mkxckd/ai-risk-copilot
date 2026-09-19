FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app app
COPY sql sql
COPY scripts scripts
# Train the detector and pre-seed history at build time so the container starts serving immediately.
ENV RISK_DB_PATH=/srv/data/risk.db RISK_MODEL_PATH=/srv/models/anomaly.joblib
RUN mkdir -p data models && python -m scripts.seed
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
