FROM python:3.10-slim

WORKDIR /app

# Copier les requirements
COPY requirements.txt .

# Installer les dependances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le projet
COPY . .

# Commande par defaut
CMD ["python", "run_pipeline.py"]