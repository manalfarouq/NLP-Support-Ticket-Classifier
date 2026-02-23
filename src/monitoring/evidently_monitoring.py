"""
MONITORING ML - AVEC EVIDENTLY AI
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib
import json
from datetime import datetime

sys.path.append('.')
from config.config import settings

from evidently import DataDefinition, Dataset, MulticlassClassification, Report
from evidently.presets import ClassificationPreset, DataDriftPreset
from sklearn.metrics import accuracy_score, classification_report

print("="*60)
print("MONITORING ML - EVIDENTLY AI")
print("="*60)

# ==========================================
# 1. CHARGER LES DONNÉES
# ==========================================
print("\n1. Chargement des données...")
X             = np.load(Path(settings.DATA_EMBEDDINGS_PATH) / 'embeddings.npy')
df            = pd.read_csv(Path(settings.DATA_EMBEDDINGS_PATH) / 'data_sample.csv')
model         = joblib.load(Path(settings.MODELS_PATH) / 'classifier' / 'model.pkl')
label_encoder = joblib.load(Path(settings.MODELS_PATH) / 'classifier' / 'label_encoder.pkl')
print(f"   ✓ {len(X)} tickets chargés")

# ==========================================
# 2. PRÉDICTIONS
# ==========================================
print("\n2. Génération des prédictions...")
y_pred        = model.predict(X)
y_pred_labels = label_encoder.inverse_transform(y_pred)
df['target']      = df['type']
df['prediction']  = y_pred_labels
df['text_length'] = df['processed_text'].astype(str).apply(len)
print("   ✓ Prédictions ajoutées")

# ==========================================
# 3. SPLIT 70/30
# ==========================================
print("\n3. Séparation référence / actuel...")
split        = int(len(df) * 0.7)
reference_df = df.iloc[:split].copy().reset_index(drop=True)
current_df   = df.iloc[split:].copy().reset_index(drop=True)
print(f"   ✓ Référence : {len(reference_df)} tickets")
print(f"   ✓ Actuel    : {len(current_df)} tickets")

# ==========================================
# 4. LANCER EVIDENTLY (calcul)
# ==========================================
print("\n4. Exécution Evidently AI...")
cols = ['target', 'prediction', 'text_length']
data_definition = DataDefinition(
    classification=[MulticlassClassification(target='target', prediction_labels='prediction')]
)
ref_dataset = Dataset.from_pandas(reference_df[cols], data_definition=data_definition)
cur_dataset = Dataset.from_pandas(current_df[cols],   data_definition=data_definition)

report = Report([ClassificationPreset(), DataDriftPreset()])
report.run(reference_data=ref_dataset, current_data=cur_dataset)
print("   ✓ Rapport Evidently généré")

# ==========================================
# 5. CALCULER LES MÉTRIQUES MANUELLEMENT
# ==========================================
classes = list(label_encoder.classes_)

ref_acc  = accuracy_score(reference_df['target'], reference_df['prediction'])
curr_acc = accuracy_score(current_df['target'],   current_df['prediction'])
diff     = curr_acc - ref_acc
status   = "STABLE ✅" if abs(diff) < 0.05 else "DRIFT DÉTECTÉ ⚠️"

ref_report  = classification_report(reference_df['target'], reference_df['prediction'], output_dict=True)
curr_report = classification_report(current_df['target'],   current_df['prediction'],   output_dict=True)

def dist(dataframe):
    return {cls: int((dataframe['target'] == cls).sum()) for cls in classes}

ref_dist  = dist(reference_df)
curr_dist = dist(current_df)

# ==========================================
# 6. GÉNÉRER LE HTML
# ==========================================
print("\n5. Génération du rapport HTML...")

reports_dir = Path(settings.REPORTS_PATH) / 'evidently'
reports_dir.mkdir(parents=True, exist_ok=True)

def pct(n, total): return f"{n/total*100:.1f}%"

rows_ref  = "".join(f"<tr><td>{c}</td><td>{ref_dist[c]}</td><td>{pct(ref_dist[c],len(reference_df))}</td><td>{ref_report[c]['precision']:.2f}</td><td>{ref_report[c]['recall']:.2f}</td><td>{ref_report[c]['f1-score']:.2f}</td></tr>" for c in classes)
rows_curr = "".join(f"<tr><td>{c}</td><td>{curr_dist[c]}</td><td>{pct(curr_dist[c],len(current_df))}</td><td>{curr_report[c]['precision']:.2f}</td><td>{curr_report[c]['recall']:.2f}</td><td>{curr_report[c]['f1-score']:.2f}</td></tr>" for c in classes)

drift_color = "#27ae60" if abs(diff) < 0.05 else "#e74c3c"

html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Rapport Monitoring ML - Evidently AI</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f6fa; color: #2d3436; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #34495e; margin-top: 40px; }}
  .badge {{ display:inline-block; padding:6px 16px; border-radius:20px; font-weight:bold; color:white; background:{drift_color}; font-size:1.1em; }}
  .cards {{ display:flex; gap:20px; margin:20px 0; flex-wrap:wrap; }}
  .card {{ background:white; border-radius:10px; padding:20px 30px; box-shadow:0 2px 8px rgba(0,0,0,0.08); min-width:180px; text-align:center; }}
  .card .val {{ font-size:2em; font-weight:bold; color:#3498db; }}
  .card .label {{ color:#7f8c8d; font-size:0.9em; margin-top:4px; }}
  table {{ border-collapse:collapse; width:100%; background:white; border-radius:10px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.08); }}
  th {{ background:#3498db; color:white; padding:12px 16px; text-align:left; }}
  td {{ padding:10px 16px; border-bottom:1px solid #ecf0f1; }}
  tr:last-child td {{ border-bottom:none; }}
  tr:hover td {{ background:#f0f8ff; }}
  .section {{ background:white; border-radius:10px; padding:24px; margin:20px 0; box-shadow:0 2px 8px rgba(0,0,0,0.08); }}
  .diff {{ font-size:1.3em; font-weight:bold; color:{drift_color}; }}
  footer {{ text-align:center; color:#b2bec3; margin-top:40px; font-size:0.85em; }}
</style>
</head>
<body>

<h1>📊 Rapport Monitoring ML — Evidently AI</h1>
<p>Généré le : <strong>{datetime.now().strftime("%d/%m/%Y à %H:%M")}</strong></p>

<div class="section">
  <h2>🔍 Résultat du drift</h2>
  <div class="cards">
    <div class="card"><div class="val">{ref_acc:.1%}</div><div class="label">Accuracy — Référence</div></div>
    <div class="card"><div class="val">{curr_acc:.1%}</div><div class="label">Accuracy — Actuel</div></div>
    <div class="card"><div class="val diff">{diff:+.1%}</div><div class="label">Variation</div></div>
    <div class="card"><div class="val">{len(reference_df)}</div><div class="label">Tickets référence</div></div>
    <div class="card"><div class="val">{len(current_df)}</div><div class="label">Tickets actuels</div></div>
  </div>
  <p>Statut : <span class="badge">{status}</span></p>
  <p style="color:#7f8c8d">Seuil de drift : 5% de variation d'accuracy</p>
</div>

<h2>📋 Période de référence (70% — {len(reference_df)} tickets)</h2>
<table>
  <tr><th>Classe</th><th>Tickets</th><th>%</th><th>Précision</th><th>Rappel</th><th>F1</th></tr>
  {rows_ref}
  <tr style="background:#eaf4fb"><td><strong>Global</strong></td><td><strong>{len(reference_df)}</strong></td><td><strong>100%</strong></td><td><strong>{ref_report['weighted avg']['precision']:.2f}</strong></td><td><strong>{ref_report['weighted avg']['recall']:.2f}</strong></td><td><strong>{ref_report['weighted avg']['f1-score']:.2f}</strong></td></tr>
</table>

<h2>📋 Période actuelle (30% — {len(current_df)} tickets)</h2>
<table>
  <tr><th>Classe</th><th>Tickets</th><th>%</th><th>Précision</th><th>Rappel</th><th>F1</th></tr>
  {rows_curr}
  <tr style="background:#eaf4fb"><td><strong>Global</strong></td><td><strong>{len(current_df)}</strong></td><td><strong>100%</strong></td><td><strong>{curr_report['weighted avg']['precision']:.2f}</strong></td><td><strong>{curr_report['weighted avg']['recall']:.2f}</strong></td><td><strong>{curr_report['weighted avg']['f1-score']:.2f}</strong></td></tr>
</table>

<footer>Projet NLP Support Ticket Classifier — Monitoring avec Evidently AI</footer>
</body>
</html>"""

html_path = reports_dir / 'monitoring_report.html'
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"   ✓ Rapport HTML : {html_path}")

# JSON aussi
metrics = {
    'reference': {'n_samples': len(reference_df), 'accuracy': float(ref_acc), 'distribution': ref_dist},
    'current':   {'n_samples': len(current_df),   'accuracy': float(curr_acc),'distribution': curr_dist},
    'drift':     {'accuracy_difference': float(diff), 'status': status, 'threshold': 0.05}
}
with open(reports_dir / 'monitoring_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"   ✓ Métriques JSON : {reports_dir / 'monitoring_metrics.json'}")

print("\n" + "="*60)
print("✓ MONITORING TERMINÉ")
print(f"  → Ouvre dans ton navigateur : {html_path}")
print("="*60)