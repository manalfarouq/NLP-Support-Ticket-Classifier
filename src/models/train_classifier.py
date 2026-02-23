"""
ENTRAÎNEMENT DU MODÈLE - VERSION SIMPLIFIÉE
Pour débutants
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

sys.path.append('.')
from config.config import settings

print("="*60)
print("ENTRAÎNEMENT DU MODÈLE DE CLASSIFICATION")
print("="*60)

# ==========================================
# 1. CHARGER LES DONNÉES
# ==========================================
print("\n1. Chargement des données...")

# Chemins des fichiers
embeddings_file = Path(settings.DATA_EMBEDDINGS_PATH) / 'embeddings.npy'
data_file = Path(settings.DATA_EMBEDDINGS_PATH) / 'data_sample.csv'

# Charger les embeddings (vecteurs)
X = np.load(embeddings_file)
print(f"   ✓ Embeddings : {X.shape}")

# Charger les labels (types de tickets)
df = pd.read_csv(data_file)
y = df['type'].values
print(f"   ✓ Labels : {len(y)}")

# ==========================================
# 2. PRÉPARER LES DONNÉES
# ==========================================
print("\n2. Préparation des données...")

# Transformer les labels en nombres (0, 1, 2, 3)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"   Classes : {list(label_encoder.classes_)}")

# Afficher la distribution
unique, counts = np.unique(y_encoded, return_counts=True)
for cls, count in zip(label_encoder.classes_, counts):
    print(f"   {cls:12s} : {count:4d} tickets")

# ==========================================
# 3. SÉPARER TRAIN ET TEST (80/20)
# ==========================================
print("\n3. Séparation train/test (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,           # 20% pour le test
    random_state=42,         # Pour reproduire les résultats
    stratify=y_encoded       # Garder les proportions
)

print(f"   Train : {len(X_train)} tickets")
print(f"   Test  : {len(X_test)} tickets")

# ==========================================
# 4. ENTRAÎNER LES MODÈLES
# ==========================================
print("\n4. Entraînement des modèles...")

# Modèle 1 : Logistic Regression (simple et rapide)
print("\n   a) Logistic Regression...")
model_lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # Pour gérer le déséquilibre
    random_state=42
)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
score_lr = accuracy_score(y_test, y_pred_lr)
print(f"      Score : {score_lr:.2%}")

# Modèle 2 : Random Forest (plus puissant)
print("\n   b) Random Forest...")
model_rf = RandomForestClassifier(
    n_estimators=100,         # 100 arbres
    class_weight='balanced',
    random_state=42,
    n_jobs=-1                 # Utiliser tous les CPU
)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
score_rf = accuracy_score(y_test, y_pred_rf)
print(f"      Score : {score_rf:.2%}")

# ==========================================
# 5. CHOISIR LE MEILLEUR
# ==========================================
print("\n5. Sélection du meilleur modèle...")

if score_rf > score_lr:
    best_model = model_rf
    best_name = "RandomForest"
    best_predictions = y_pred_rf
    best_score = score_rf
else:
    best_model = model_lr
    best_name = "LogisticRegression"
    best_predictions = y_pred_lr
    best_score = score_lr

print(f"\n   ✓ Meilleur : {best_name}")
print(f"   ✓ Score : {best_score:.2%}")

# ==========================================
# 6. ÉVALUATION DÉTAILLÉE
# ==========================================
print("\n6. Évaluation détaillée...")

# Rapport de classification
print("\n" + "="*60)
report = classification_report(
    y_test, 
    best_predictions,
    target_names=label_encoder.classes_
)
print(report)

# Matrice de confusion
cm = confusion_matrix(y_test, best_predictions)
print("\nMatrice de confusion :")
print(cm)

# ==========================================
# 7. VISUALISATION
# ==========================================
print("\n7. Création de la visualisation...")

# Créer le dossier pour sauvegarder
models_dir = Path(settings.MODELS_PATH) / 'classifier'
models_dir.mkdir(parents=True, exist_ok=True)

# Matrice de confusion en image
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True,           # Afficher les nombres
    fmt='d',              # Format entier
    cmap='Blues',         # Couleur bleue
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.title(f'Matrice de Confusion - {best_name}', fontsize=14, fontweight='bold')
plt.ylabel('Vraie Classe')
plt.xlabel('Classe Prédite')
plt.tight_layout()
plt.savefig(models_dir / 'confusion_matrix.png', dpi=300)
print(f"   ✓ Image sauvegardée")
plt.close()

# ==========================================
# 8. SAUVEGARDER LE MODÈLE
# ==========================================
print("\n8. Sauvegarde du modèle...")

# Sauvegarder le modèle
joblib.dump(best_model, models_dir / 'model.pkl')
print(f"   ✓ Modèle : {models_dir / 'model.pkl'}")

# Sauvegarder l'encodeur
joblib.dump(label_encoder, models_dir / 'label_encoder.pkl')
print(f"   ✓ Encodeur : {models_dir / 'label_encoder.pkl'}")

# Sauvegarder les scores
metrics = {
    'model': best_name,
    'accuracy': float(best_score),
    'logistic_regression_score': float(score_lr),
    'random_forest_score': float(score_rf)
}
with open(models_dir / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"   ✓ Métriques : {models_dir / 'metrics.json'}")

# ==========================================
# RÉSUMÉ
# ==========================================
print("\n" + "="*60)
print("RÉSUMÉ")
print("="*60)
print(f"\nMeilleur modèle : {best_name}")
print(f"Précision : {best_score:.2%}")

if best_score > 0.75:
    print("✅ Objectif atteint ! (>75%)")
else:
    print("⚠️ Peut être amélioré")

print(f"\nFichiers créés dans : {models_dir}")
print("  • model.pkl")
print("  • label_encoder.pkl")
print("  • metrics.json")
print("  • confusion_matrix.png")

print("\n" + "="*60)
print("✓ TERMINÉ AVEC SUCCÈS")
print("="*60)