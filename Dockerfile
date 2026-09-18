FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rf_analyzer/ rf_analyzer/
COPY tests/ tests/
COPY .streamlit/ .streamlit/
RUN mkdir -p data/synthetic data/samples

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "rf_analyzer/gui/app.py", "--server.address=0.0.0.0"]
