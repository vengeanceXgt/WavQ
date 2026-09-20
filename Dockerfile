FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rf_analyzer/ rf_analyzer/

# Expose both FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Copy a startup script
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
