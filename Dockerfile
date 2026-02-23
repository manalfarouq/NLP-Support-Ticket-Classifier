# Dockerfile SIMPLE - NLP Ticket Classifier

FROM python:3.9-slim

# Répertoire de travail
WORKDIR /app

# Installer gcc et build-essential pour compiler certaines dépendances
RUN apt-get update && apt-get install -y gcc build-essential && rm -rf /var/lib/apt/lists/*

# Copier requirements et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY config/ ./config/
COPY src/ ./src/
COPY .env .env

# Créer les dossiers
RUN mkdir -p data/raw data/processed data/embeddings \
             data/chroma_db models/classifier \
             reports/evidently logs

# Lancer le pipeline
CMD ["python", "-m", "src.main"]
