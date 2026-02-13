"""Application configuration settings"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Configuration du projet NLP Support Ticket Classification.
    
    Toutes les valeurs sont chargees depuis le fichier .env
    """
    
    # Configuration generale
    PROJECT_NAME: str = "NLP Support Ticket Classification"
    ENVIRONMENT: str = "development"
    
    # Chemins des donnees
    DATA_RAW_PATH: str = "data/raw"
    DATA_PROCESSED_PATH: str = "data/processed"
    DATA_EMBEDDINGS_PATH: str = "data/embeddings"
    MODELS_PATH: str = "models"
    REPORTS_PATH: str = "src/monitoring/reports"
    
    # Configuration Hugging Face
    HF_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Echantillonnage
    SAMPLE_SIZE: int = 5000
    
    # Configuration de la classification
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    N_ESTIMATORS: int = 100
    
    # Configuration ChromaDB
    CHROMA_PERSIST_DIR: str = "data/embeddings/chroma_db"
    CHROMA_COLLECTION_NAME: str = "support_tickets"
    
    # Monitoring Prometheus/Grafana
    PROMETHEUS_PORT: int = 9090
    GRAFANA_PORT: int = 3000
    CADVISOR_PORT: int = 8080
    NODE_EXPORTER_PORT: int = 9100
    
    # Grafana credentials
    GRAFANA_ADMIN_USER: str = "admin"
    GRAFANA_ADMIN_PASSWORD: str = "admin"
    
    # Docker configuration
    DOCKER_IMAGE_NAME: str = "nlp-ticket-classifier"
    DOCKER_IMAGE_TAG: str = "latest"
    
    # Kubernetes configuration
    K8S_JOB_NAME: str = "nlp-ticket-classifier-job"
    K8S_CRONJOB_NAME: str = "nlp-ticket-classifier-cronjob"
    K8S_MEMORY_LIMIT: str = "4Gi"
    K8S_MEMORY_REQUEST: str = "2Gi"
    K8S_CPU_LIMIT: str = "2"
    K8S_CPU_REQUEST: str = "1"
    
    # Logs
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


settings = Settings()