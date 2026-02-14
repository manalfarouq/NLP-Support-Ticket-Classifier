"""
MONITORING ML - VERSION SIMPLE (SANS EVIDENTLY)
Fonctionne avec n'importe quelle version
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('.')
from config.config import settings

print("="*60)
print("MONITORING ML - VERSION SIMPLE")
print("="*60)

# ==========================================
# 1. CHARGER LES DONNÉES
# ==========================================
print("\n1. Chargement des données...")

# Chemins
embeddings_file = Path(settings.DATA_EMBEDDINGS_PATH) / 'embeddings.npy'
data_file = Path(settings.DATA_EMBEDDINGS_PATH) / 'data_sample.csv'
model_file = Path(settings.MODELS_PATH) / 'classifier' / 'model.pkl'
encoder_file = Path(settings.MODELS_PATH) / 'classifier' / 'label_encoder.pkl'

# Charger
X = np.load(embeddings_file)
df = pd.read_csv(data_file)
model = joblib.load(model_file)
label_encoder = joblib.load(encoder_file)

print(f"   ✓ {len(X)} tickets chargés")

# ==========================================
# 2. GÉNÉRER LES PRÉDICTIONS
# ==========================================
print("\n2. Génération des prédictions...")

y_pred = model.predict(X)
y_pred_labels = label_encoder.inverse_transform(y_pred)
df['prediction'] = y_pred_labels

print(f"   ✓ Prédictions générées")

# ==========================================
# 3. SÉPARER EN 2 PÉRIODES
# ==========================================
print("\n3. Simulation de drift (70% référence / 30% actuel)...")

# Reference (70%) vs Current (30%)
split = int(len(df) * 0.7)
reference_df = df.iloc[:split].copy()
current_df = df.iloc[split:].copy()

print(f"   ✓ Reference : {len(reference_df)} tickets")
print(f"   ✓ Current   : {len(current_df)} tickets")

# ==========================================
# 4. MÉTRIQUES REFERENCE
# ==========================================
print("\n4. Analyse de la période de référence...")

ref_acc = accuracy_score(
    label_encoder.transform(reference_df['type']),
    label_encoder.transform(reference_df['prediction'])
)

print(f"\n   Accuracy : {ref_acc:.1%}")

# Distribution
print("\n   Distribution des classes :")
for cls in label_encoder.classes_:
    count = (reference_df['type'] == cls).sum()
    pct = (count / len(reference_df)) * 100
    print(f"   {cls:12s} : {count:5d} ({pct:5.1f}%)")

# ==========================================
# 5. MÉTRIQUES CURRENT
# ==========================================
print("\n5. Analyse de la période actuelle...")

curr_acc = accuracy_score(
    label_encoder.transform(current_df['type']),
    label_encoder.transform(current_df['prediction'])
)

print(f"\n   Accuracy : {curr_acc:.1%}")

# Distribution
print("\n   Distribution des classes :")
for cls in label_encoder.classes_:
    count = (current_df['type'] == cls).sum()
    pct = (count / len(current_df)) * 100
    print(f"   {cls:12s} : {count:5d} ({pct:5.1f}%)")

# ==========================================
# 6. COMPARAISON
# ==========================================
print("\n6. Comparaison et détection de drift...")

diff = curr_acc - ref_acc
print(f"\n   Différence d'accuracy : {diff:+.1%}")

if abs(diff) < 0.05:
    print(f"   ✅ Modèle stable (variation < 5%)")
    drift_status = "STABLE"
else:
    print(f"   ⚠️ Drift détecté (variation > 5%)")
    drift_status = "DRIFT DETECTE"

# ==========================================
# 7. RAPPORT TEXTE
# ==========================================
print("\n7. Génération du rapport...")

# Créer le dossier
reports_dir = Path(settings.REPORTS_PATH) / 'evidently'
reports_dir.mkdir(parents=True, exist_ok=True)

# Rapport texte
rapport_path = reports_dir / 'monitoring_report.txt'
with open(rapport_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("RAPPORT DE MONITORING ML\n")
    f.write("="*60 + "\n\n")
    
    f.write("PÉRIODE DE RÉFÉRENCE\n")
    f.write("-"*60 + "\n")
    f.write(f"Nombre de tickets : {len(reference_df)}\n")
    f.write(f"Accuracy : {ref_acc:.2%}\n\n")
    
    f.write("Distribution des classes :\n")
    for cls in label_encoder.classes_:
        count = (reference_df['type'] == cls).sum()
        pct = (count / len(reference_df)) * 100
        f.write(f"  {cls:12s} : {count:5d} ({pct:5.1f}%)\n")
    
    f.write("\n" + "="*60 + "\n\n")
    
    f.write("PÉRIODE ACTUELLE\n")
    f.write("-"*60 + "\n")
    f.write(f"Nombre de tickets : {len(current_df)}\n")
    f.write(f"Accuracy : {curr_acc:.2%}\n\n")
    
    f.write("Distribution des classes :\n")
    for cls in label_encoder.classes_:
        count = (current_df['type'] == cls).sum()
        pct = (count / len(current_df)) * 100
        f.write(f"  {cls:12s} : {count:5d} ({pct:5.1f}%)\n")
    
    f.write("\n" + "="*60 + "\n\n")
    
    f.write("ANALYSE DE DRIFT\n")
    f.write("-"*60 + "\n")
    f.write(f"Différence d'accuracy : {diff:+.2%}\n")
    f.write(f"Statut : {drift_status}\n")

print(f"   ✓ Rapport texte : {rapport_path}")

# ==========================================
# 8. GRAPHIQUE COMPARATIF
# ==========================================
print("\n8. Génération du graphique...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Graphique 1 : Distributions
classes = label_encoder.classes_
ref_counts = [( reference_df['type'] == cls).sum() for cls in classes]
curr_counts = [(current_df['type'] == cls).sum() for cls in classes]

x = np.arange(len(classes))
width = 0.35

axes[0].bar(x - width/2, ref_counts, width, label='Référence', alpha=0.8)
axes[0].bar(x + width/2, curr_counts, width, label='Actuel', alpha=0.8)
axes[0].set_xlabel('Classe')
axes[0].set_ylabel('Nombre de tickets')
axes[0].set_title('Distribution des Classes')
axes[0].set_xticks(x)
axes[0].set_xticklabels(classes, rotation=45)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Graphique 2 : Accuracy
axes[1].bar(['Référence', 'Actuel'], [ref_acc, curr_acc], 
            color=['#2ecc71', '#3498db'], alpha=0.8)
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Comparaison de Performance')
axes[1].set_ylim([0, 1])
axes[1].axhline(y=0.75, color='r', linestyle='--', label='Objectif (75%)')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

# Ajouter les valeurs sur les barres
for i, v in enumerate([ref_acc, curr_acc]):
    axes[1].text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
graph_path = reports_dir / 'monitoring_comparison.png'
plt.savefig(graph_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Graphique : {graph_path}")

# ==========================================
# 9. MÉTRIQUES JSON
# ==========================================
print("\n9. Sauvegarde des métriques...")

metrics = {
    'reference': {
        'n_samples': int(len(reference_df)),
        'accuracy': float(ref_acc),
        'distribution': {
            cls: int((reference_df['type'] == cls).sum())
            for cls in label_encoder.classes_
        }
    },
    'current': {
        'n_samples': int(len(current_df)),
        'accuracy': float(curr_acc),
        'distribution': {
            cls: int((current_df['type'] == cls).sum())
            for cls in label_encoder.classes_
        }
    },
    'drift_analysis': {
        'accuracy_difference': float(diff),
        'status': drift_status,
        'threshold': 0.05
    }
}

metrics_path = reports_dir / 'monitoring_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"   ✓ Métriques : {metrics_path}")

# ==========================================
# RÉSUMÉ
# ==========================================
print("\n" + "="*60)
print("RÉSUMÉ")
print("="*60)

print(f"\n✓ Rapports générés dans : {reports_dir}")
print(f"  • monitoring_report.txt")
print(f"  • monitoring_comparison.png")
print(f"  • monitoring_metrics.json")

print(f"\n✓ Performance :")
print(f"  Reference : {ref_acc:.1%}")
print(f"  Current   : {curr_acc:.1%}")
print(f"  Variation : {diff:+.1%}")

print(f"\n✓ Statut : {drift_status}")

print("\n" + "="*60)
print("✓ MONITORING TERMINÉ")
print("="*60)