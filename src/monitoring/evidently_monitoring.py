"""
Monitoring ML avec Evidently AI
Detection de data drift et prediction drift
"""

import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
import joblib
import sys
from pathlib import Path

sys.path.append('.')
from config.config import settings

def main():
    print("="*50)
    print("MONITORING AVEC EVIDENTLY AI")
    print("="*50)
    
    # Charger les donnees depuis config
    print("\n1. Chargement des donnees...")
    embeddings = np.load(Path(settings.DATA_EMBEDDINGS_PATH) / 'embeddings.npy')
    df = pd.read_csv(Path(settings.DATA_EMBEDDINGS_PATH) / 'data_sample.csv')
    
    # Charger le modele depuis config
    model = joblib.load(Path(settings.MODELS_PATH) / 'classifier.pkl')
    
    # Creer des features pour Evidently
    print("\n2. Preparation des donnees...")
    
    # Convertir embeddings en DataFrame
    feature_names = [f'feature_{i}' for i in range(embeddings.shape[1])]
    df_features = pd.DataFrame(embeddings, columns=feature_names)
    
    # Ajouter les predictions
    predictions = model.predict(embeddings)
    df_features['prediction'] = predictions
    df_features['target'] = df['type'].values
    
    # Split reference (passe) et current (present)
    split_point = int(len(df_features) * 0.7)
    reference_data = df_features[:split_point]
    current_data = df_features[split_point:]
    
    print(f"   Reference: {len(reference_data)} samples")
    print(f"   Current: {len(current_data)} samples")
    
    # Creer le rapport Data Drift
    print("\n3. Generation du rapport Data Drift...")
    
    data_drift_report = Report(metrics=[
        DataDriftPreset()
    ])
    
    data_drift_report.run(
        reference_data=reference_data,
        current_data=current_data
    )
    
    # Sauvegarder depuis config
    reports_path = Path(settings.REPORTS_PATH)
    reports_path.mkdir(parents=True, exist_ok=True)
    
    drift_file = reports_path / 'data_drift_report.html'
    data_drift_report.save_html(str(drift_file))
    
    print(f"   Rapport sauvegarde: {drift_file}")
    
    # Creer le rapport Classification
    print("\n4. Generation du rapport Classification...")
    
    classification_report = Report(metrics=[
        ClassificationPreset()
    ])
    
    classification_report.run(
        reference_data=reference_data,
        current_data=current_data
    )
    
    class_file = reports_path / 'classification_report.html'
    classification_report.save_html(str(class_file))
    
    print(f"   Rapport sauvegarde: {class_file}")
    print("\nTermine!")
    print("\nOuvre les fichiers HTML dans un navigateur pour voir les rapports")

if __name__ == "__main__":
    main()