# Stage 1: Build React frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend_react/package*.json ./
RUN npm ci
COPY frontend_react/ ./
RUN npm run build

# Stage 2: Production Python API backend
FROM python:3.11-slim
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY rf_analyzer/ rf_analyzer/
COPY backend/ backend/
COPY data/ data/
COPY --from=frontend-builder /frontend/dist frontend_react/dist

# Copy start script
COPY start.sh .
RUN chmod +x start.sh

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["./start.sh"]
