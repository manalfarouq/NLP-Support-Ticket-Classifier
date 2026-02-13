"""
Indexation des embeddings dans ChromaDB
Adapté à votre structure de projet
"""

import pandas as pd
import numpy as np
import chromadb
from chromadb.config import Settings
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append('.')
from config.config import settings

def main():
    print("="*50)
    print("INDEXATION CHROMADB")
    print("="*50)
    
    # Chemins depuis config
    embeddings_file = Path(settings.DATA_EMBEDDINGS_PATH) / 'embeddings.npy'
    data_file = Path(settings.DATA_EMBEDDINGS_PATH) / 'data_sample.csv'
    chroma_dir = Path(settings.CHROMA_PERSIST_DIR)
    
    # ========================================
    # 1. CHARGEMENT DES DONNÉES
    # ========================================
    print("\n1. Chargement des donnees...")
    
    # Charger les embeddings
    if not embeddings_file.exists():
        print(f"Fichier non trouvé : {embeddings_file}")
        print("   Exécutez d'abord : python -m src.embeddings.generate_embeddings")
        return
    
    embeddings = np.load(embeddings_file)
    print(f"   Embeddings: {embeddings.shape}")
    
    # Charger les métadonnées
    if not data_file.exists():
        print(f"Fichier non trouvé : {data_file}")
        return
    
    df = pd.read_csv(data_file)
    print(f"   Donnees: {len(df)} tickets")
    
    # Vérifier la cohérence
    assert len(embeddings) == len(df), "Nombre d'embeddings != nombre de tickets"
    
    # ========================================
    # 2. INITIALISATION DE CHROMADB
    # ========================================
    print("\n2. Initialisation de ChromaDB...")
    
    try:
        # Créer le dossier
        chroma_dir.mkdir(parents=True, exist_ok=True)
        
        # Client ChromaDB
        client = chromadb.PersistentClient(path=str(chroma_dir))
        print(f"   Client cree: {chroma_dir}")
        
        # Supprimer la collection si elle existe
        try:
            client.delete_collection(name=settings.CHROMA_COLLECTION_NAME)
            print(f"   Collection existante supprimee")
        except:
            pass
        
        # Créer la collection
        collection = client.create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"description": "Support tickets embeddings"}
        )
        print(f"   Collection creee: {settings.CHROMA_COLLECTION_NAME}")
        
    except Exception as e:
        print(f"Erreur ChromaDB : {e}")
        print("\nSolution :")
        print("   pip install chromadb")
        return
    
    # ========================================
    # 3. PRÉPARATION DES DONNÉES
    # ========================================
    print("\n3. Preparation des donnees...")
    
    # IDs (strings requis par ChromaDB)
    ids = [str(i) for i in range(len(df))]
    
    # Embeddings (liste de listes)
    embeddings_list = embeddings.tolist()
    
    # Métadonnées (vérifier les colonnes disponibles)
    metadatas = []
    for _, row in df.iterrows():
        metadata = {}
        
        # Colonnes communes
        if 'type' in df.columns:
            metadata['type'] = str(row['type'])
        if 'language' in df.columns:
            metadata['language'] = str(row['language'])
        if 'priority' in df.columns:
            metadata['priority'] = str(row['priority'])
        
        # Texte preview
        if 'processed_text' in df.columns:
            metadata['text_preview'] = str(row['processed_text'])[:200]
        elif 'text_cleaned' in df.columns:
            metadata['text_preview'] = str(row['text_cleaned'])[:200]
        
        metadatas.append(metadata)
    
    # Documents (textes pour affichage)
    if 'processed_text' in df.columns:
        documents = df['processed_text'].tolist()
    elif 'text_cleaned' in df.columns:
        documents = df['text_cleaned'].tolist()
    else:
        documents = [''] * len(df)
    
    print(f"   IDs: {len(ids)}")
    print(f"   Embeddings: {len(embeddings_list)}")
    print(f"   Metadatas: {len(metadatas)}")
    print(f"   Documents: {len(documents)}")
    
    # ========================================
    # 4. INDEXATION
    # ========================================
    print("\n4. Indexation en cours...")
    
    # Batch size pour optimiser
    BATCH_SIZE = 500
    num_batches = (len(ids) + BATCH_SIZE - 1) // BATCH_SIZE
    
    try:
        for i in tqdm(range(num_batches), desc="Indexation"):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(ids))
            
            collection.add(
                ids=ids[start_idx:end_idx],
                embeddings=embeddings_list[start_idx:end_idx],
                metadatas=metadatas[start_idx:end_idx],
                documents=documents[start_idx:end_idx]
            )
        
        print(f"\n   ✓ Indexation terminee")
        
    except Exception as e:
        print(f"\n❌ Erreur indexation : {e}")
        return
    
    # ========================================
    # 5. VÉRIFICATION
    # ========================================
    print("\n5. Verification...")
    
    count = collection.count()
    print(f"   Elements indexes: {count}")
    
    # ========================================
    # 6. TEST DE RECHERCHE
    # ========================================
    print("\n6. Test de recherche semantique...")
    
    # Prendre un ticket test
    test_idx = 5
    test_embedding = embeddings[test_idx].tolist()
    test_text = documents[test_idx]
    
    print(f"\n   Requete:")
    print(f"   Texte: {test_text[:100]}...")
    
    # Rechercher
    results = collection.query(
        query_embeddings=[test_embedding],
        n_results=6,  # 6 car le premier sera le ticket lui-même
        include=['metadatas', 'documents', 'distances']
    )
    
    print(f"\n   Top 5 tickets similaires:")
    # Ignorer le premier (lui-même)
    for i in range(1, 6):
        result_id = results['ids'][0][i]
        result_metadata = results['metadatas'][0][i]
        result_document = results['documents'][0][i]
        result_distance = results['distances'][0][i]
        similarity = 1 - result_distance
        
        print(f"\n   {i}. Similarite: {similarity:.4f}")
        print(f"      ID: {result_id}")
        if 'type' in result_metadata:
            print(f"      Type: {result_metadata['type']}")
        if 'language' in result_metadata:
            print(f"      Langue: {result_metadata['language']}")
        print(f"      Texte: {result_document[:80]}...")
    
    # ========================================
    # 7. RÉSUMÉ
    # ========================================
    print("\n" + "="*50)
    print("RÉSUMÉ")
    print("="*50)
    
    print(f"\n✓ Collection: {settings.CHROMA_COLLECTION_NAME}")
    print(f"✓ Tickets indexes: {count}")
    print(f"✓ Dimension: {len(embeddings_list[0])}")
    print(f"✓ Chemin: {chroma_dir}")
    
    print(f"\nProchaines etapes:")
    print(f"   1. ✓ Preprocessing termine")
    print(f"   2. ✓ Embeddings generes")
    print(f"   3. ✓ ChromaDB indexe")
    print(f"   4. → Entrainement du modele (Phase 3)")
    
    print("\nTermine!")

if __name__ == "__main__":
    main()