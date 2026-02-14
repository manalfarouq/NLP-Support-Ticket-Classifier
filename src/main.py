"""
Script principal pour executer tout le pipeline
"""

import sys
from pathlib import Path

def run_step(script_path, step_name):
    """Executer un script"""
    print("\n" + "="*60)
    print(f"EXECUTION: {step_name}")
    print("="*60)
    
    import subprocess
    result = subprocess.run([sys.executable, str(script_path)], capture_output=False)
    
    if result.returncode != 0:
        print(f"\nERREUR dans {step_name}")
        sys.exit(1)
    
    print(f"\n{step_name} termine avec succes!")

def main():
    print("\n" + "#"*60)
    print("# PIPELINE COMPLET NLP TICKET CLASSIFICATION")
    print("#"*60)
    
    steps = [
        (Path("src/preprocessing/preprocess.py"), "1. PREPROCESSING"),
        (Path("src/embeddings/generate_embeddings.py"), "2. EMBEDDINGS"),
        (Path("src/embeddings/index_chromadb.py"), "3. CHROMADB"),
        (Path("src/models/train_classifier.py"), "4. CLASSIFICATION"),
        (Path("src/monitoring/evidently_monitoring.py"), "5. MONITORING")
    ]
    
    for script, name in steps:
        run_step(script, name)
    
    print("\n" + "#"*60)
    print("# PIPELINE TERMINE AVEC SUCCES!")
    print("#"*60)
    print("\nConsulte les rapports dans src/monitoring/reports/")

if __name__ == "__main__":
    main()