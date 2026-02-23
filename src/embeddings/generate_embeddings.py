"""
Generation des embeddings avec Hugging Face
Utilise sentence-transformers
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path

sys.path.append('.')
from config.config import settings

def main():
    print("="*50)
    print("GENERATION DES EMBEDDINGS")
    print("="*50)
    
    # Chemins depuis config
    input_file = Path(settings.DATA_PROCESSED_PATH) / 'preprocessed_data.csv'
    embeddings_file = Path(settings.DATA_EMBEDDINGS_PATH) / 'embeddings.npy'
    data_file = Path(settings.DATA_EMBEDDINGS_PATH) / 'data_sample.csv'
    
    # Charger le modele
    print("\n1. Chargement du modele Hugging Face...")
    print(f"   Modele: {settings.HF_MODEL_NAME}")
    model = SentenceTransformer(settings.HF_MODEL_NAME)
    
    # Charger les donnees
    print("\n2. Chargement des donnees...")
    df = pd.read_csv(input_file)
    
    # Prendre un echantillon depuis config
    df_sample = df.head(settings.SAMPLE_SIZE)
    print(f"   Utilisation de {len(df_sample)} tickets")
    
    # Generer les embeddings
    print("\n3. Generation des embeddings...")
    print("   Cela peut prendre quelques minutes...")
    
    texts = df_sample['processed_text'].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    
    print(f"\n   Shape: {embeddings.shape}")
    print(f"   Dimension: {embeddings.shape[1]}")
    
    # Sauvegarder
    print("\n4. Sauvegarde...")
    embeddings_file.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(embeddings_file, embeddings)
    df_sample.to_csv(data_file, index=False)
    
    print(f"\n   Embeddings: {embeddings_file}")
    print(f"   Donnees: {data_file}")
    print("\nTermine!")

if __name__ == "__main__":
    main()