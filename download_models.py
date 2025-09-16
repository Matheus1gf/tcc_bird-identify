#!/usr/bin/env python3
"""
Script para baixar modelos necessários para o sistema de identificação de pássaros.
Execute este script após clonar o repositório para baixar os modelos YOLO.
"""

import os
import urllib.request
from pathlib import Path

def download_file(url: str, filename: str) -> bool:
    """Baixa um arquivo da URL especificada"""
    try:
        print(f"📥 Baixando {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"✅ {filename} baixado com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro ao baixar {filename}: {e}")
        return False

def main():
    """Função principal para baixar todos os modelos necessários"""
    print("🚀 Iniciando download dos modelos necessários...")
    print("=" * 60)
    
    # Criar diretório se não existir
    os.makedirs(".", exist_ok=True)
    
    # URLs dos modelos YOLO (versões menores para desenvolvimento)
    models = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
    }
    
    # URLs alternativas caso as principais falhem
    alternative_models = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
    }
    
    downloaded_count = 0
    total_models = len(models)
    
    for filename, url in models.items():
        if os.path.exists(filename):
            print(f"⏭️  {filename} já existe, pulando...")
            downloaded_count += 1
            continue
            
        success = download_file(url, filename)
        if success:
            downloaded_count += 1
        else:
            # Tentar URL alternativa
            print(f"🔄 Tentando URL alternativa para {filename}...")
            alt_url = alternative_models.get(filename)
            if alt_url:
                success = download_file(alt_url, filename)
                if success:
                    downloaded_count += 1
    
    print("=" * 60)
    print(f"📊 Resumo: {downloaded_count}/{total_models} modelos baixados")
    
    if downloaded_count == total_models:
        print("🎉 Todos os modelos foram baixados com sucesso!")
        print("\n📝 Próximos passos:")
        print("1. Execute: python3 -m streamlit run main.py")
        print("2. Ou execute: python3 start_system.py")
    else:
        print("⚠️  Alguns modelos não foram baixados.")
        print("💡 Você pode baixar manualmente os modelos necessários:")
        print("   - yolov8n.pt (modelo nano - mais leve)")
        print("   - yolov8s.pt (modelo small - balanceado)")

if __name__ == "__main__":
    main()
