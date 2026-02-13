"""
Entrainement du modele de classification
Predit le type de ticket
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import sys
from pathlib import Path

sys.path.append('.')
from config.config import settings

def main():
    print("ENTRAINEMENT DU MODELE")
    
    # Charger les donnees depuis config
    print("\n1. Chargement des donnees...")
    embeddings = np.load(Path(settings.DATA_EMBEDDINGS_PATH) / 'embeddings.npy')
    df = pd.read_csv(Path(settings.DATA_EMBEDDINGS_PATH) / 'data_sample.csv')
    
    X = embeddings
    y = df['type'].values
    
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Classes: {set(y)}")
    
    # Split train/test depuis config
    print("\n2. Split train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE, stratify=y
    )
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Entrainement avec parametres depuis config
    print("\n3. Entrainement du modele...")
    print("   RandomForestClassifier (simple et efficace)")
    
    model = RandomForestClassifier(
        n_estimators=settings.N_ESTIMATORS,
        random_state=settings.RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("   Entrainement termine!")
    
    # Evaluation
    print("\n4. Evaluation...")
    y_pred = model.predict(X_test)
    
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\n   Matrice de confusion:")
    print(confusion_matrix(y_test, y_pred))
    
    # Sauvegarder le modele depuis config
    print("\n5. Sauvegarde du modele...")
    model_file = Path(settings.MODELS_PATH) / 'classifier.pkl'
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_file)
    
    print(f"   Modele sauvegarde: {model_file}")
    print("\nTermine!")

if __name__ == "__main__":
    main()