"""
Script de preprocessing NLP
Nettoie et prepare les textes pour l'analyse
"""

import pandas as pd
import re
import sys
from pathlib import Path

sys.path.append('.')
from config.config import settings

def clean_text(text):
    """Nettoyer le texte"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_stopwords(text):
    """Supprimer les stopwords"""
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
        'can', 'could', 'may', 'might', 'must', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her',
        'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einen', 'einem'
    }
    
    words = text.split()
    words = [w for w in words if w not in stopwords and len(w) > 2]
    return ' '.join(words)

def main():
    print("="*50)
    print("PREPROCESSING NLP")
    print("="*50)
    
    # Chemins depuis config
    input_file = Path(settings.DATA_RAW_PATH) / 'dataset.csv'
    output_file = Path(settings.DATA_PROCESSED_PATH) / 'preprocessed_data.csv'
    
    # Charger
    print("\n1. Chargement des donnees...")
    df = pd.read_csv(input_file)
    print(f"   Nombre de tickets: {len(df)}")
    
    # Fusionner
    print("\n2. Fusion subject + body...")
    df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
    
    # Nettoyer
    print("\n3. Nettoyage du texte...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Stopwords
    print("\n4. Suppression des stopwords...")
    df['processed_text'] = df['cleaned_text'].apply(remove_stopwords)
    
    # Filtrer
    df = df[df['processed_text'].str.len() > 10]
    print(f"   Apres filtrage: {len(df)} tickets")
    
    # Sauvegarder
    print("\n5. Sauvegarde...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df[['text', 'processed_text', 'type']].to_csv(output_file, index=False)
    
    print(f"\n   Fichier sauvegarde: {output_file}")
    print("\nTermine!")

if __name__ == "__main__":
    main()