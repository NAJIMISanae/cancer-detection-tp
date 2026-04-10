# Dockerfile — Cancer Detection App
# Base : Python 3.11 slim pour image légère
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Sanae Najimi"
LABEL description="Détection Cancer Pulmonaire — Pipeline IA Multimodal"
LABEL version="1.0"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# Répertoire de travail
WORKDIR /app

# Dépendances système minimales
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python en premier
# (couche Docker cachée — pas reconstruite si requirements.txt ne change pas)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le reste du projet
COPY app.py .
COPY models/ ./models/

# Port exposé (Render utilise $PORT dynamiquement)
EXPOSE 8501

# Script de démarrage
# Render injecte $PORT automatiquement
CMD streamlit run app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
