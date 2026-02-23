# NLP Support Ticket Classification

Pipeline NLP automatise pour classifier les tickets de support IT.

## Structure du projet

```
NLP-SUPPORT-TICKET-CLASSIF/
├── .github/workflows/     # CI/CD GitHub Actions
├── config/                # Configuration Prometheus
├── data/
│   ├── raw/              # Dataset brut
│   ├── processed/        # Donnees nettoyees
│   └── embeddings/       # Embeddings et ChromaDB
├── models/               # Modeles entraines
├── notebooks/            # Notebooks Jupyter (optionnel)
├── src/
│   ├── preprocessing/    # Scripts de nettoyage
│   ├── embeddings/       # Generation embeddings
│   ├── models/           # Entrainement modeles
│   ├── monitoring/       # Evidently AI
│   └── pipeline/         # Pipeline scripts
├── requirements.txt      # Dependances Python
├── run_pipeline.py       # Script principal
├── Dockerfile           # Docker
└── docker-compose.yml   # Monitoring infrastructure
```

## Installation

### 1. Configurer les variables d'environnement

```bash
# Copier le fichier template
cp .env.example .env

# Modifier les valeurs selon tes besoins (optionnel)
# Les valeurs par defaut fonctionnent bien
```

Pour plus de details, consulte [ENV_GUIDE.md](ENV_GUIDE.md)

### 2. Creer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Installer les dependances

```bash
pip install -r requirements.txt
```

## Execution du pipeline

### Option 1: Executer tout le pipeline

```bash
python run_pipeline.py
```

### Option 2: Executer etape par etape

```bash
# Etape 1: Preprocessing
python src/preprocessing/preprocess.py

# Etape 2: Generation des embeddings
python src/embeddings/generate_embeddings.py

# Etape 3: Stockage dans ChromaDB
python src/embeddings/store_chromadb.py

# Etape 4: Entrainement du modele
python src/models/train_classifier.py

# Etape 5: Monitoring avec Evidently
python src/monitoring/evidently_monitoring.py
```

## Resultats

Apres execution, les fichiers suivants seront generes:

- `data/processed/preprocessed_data.csv` - Donnees nettoyees
- `data/embeddings/embeddings.npy` - Vecteurs embeddings
- `data/embeddings/chroma_db/` - Base vectorielle ChromaDB
- `models/classifier.pkl` - Modele de classification entraine
- `src/monitoring/reports/*.html` - Rapports Evidently AI

## Docker

### Build l'image

```bash
docker build -t nlp-ticket-classifier .
```

### Executer le container

```bash
docker run nlp-ticket-classifier
```

## Monitoring Infrastructure

### Lancer Prometheus + Grafana

```bash
docker-compose up -d
```

### Acces aux interfaces

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- cAdvisor: http://localhost:8080
- Node Exporter: http://localhost:9100

## Performances attendues

- Preprocessing: 2 minutes pour 20K tickets
- Embeddings: 10 minutes pour 5K tickets
- Classification: 3 minutes entrainement
- Accuracy: ~86%

## Support

Pour toute question, consulte la documentation ou contacte le support.