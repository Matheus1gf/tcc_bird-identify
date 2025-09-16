#!/usr/bin/env python3
"""
Simulação do que acontece na interface web
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.intuition import IntuitionEngine
from PIL import Image
import base64

def test_web_simulation():
    print("🌐 SIMULAÇÃO DA INTERFACE WEB")
    print("=" * 50)
    
    # Inicializar o sistema
    intuition_engine = IntuitionEngine("yolov8n.pt", "modelo_classificacao_passaros.keras")
    
    # Simular imagem de pássaro (rolinha)
    rolinha_data = base64.b64decode("""
    /9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcU
    FhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgo
    KCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIA
    AhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEB
    AQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX
    /9k=
    """)
    
    # Salvar imagem temporária
    with open("test_web.jpg", "wb") as f:
        f.write(rolinha_data)
    
    try:
        print("📊 Simulando análise da interface web...")
        
        # Simular exatamente o que acontece na web_app.py
        temp_path = "test_web.jpg"
        results = intuition_engine.analyze_image_intuition(temp_path)
        
        print(f"  - Resultados: {results is not None}")
        if results:
            print(f"  - Confiança: {results.get('confidence', 0):.2%}")
            print(f"  - Espécie: {results.get('species', 'N/A')}")
            print(f"  - Cor: {results.get('color', 'N/A')}")
            
            # Análise de intuição
            intuition_data = results.get('intuition_analysis', {})
            print(f"  - Candidatos encontrados: {intuition_data.get('candidates_found', 0)}")
            print(f"  - Recomendação: {intuition_data.get('recommendation', 'N/A')}")
            
            # Análise visual
            visual_analysis = intuition_data.get('visual_analysis', {})
            if visual_analysis:
                print(f"  - Score visual: {visual_analysis.get('bird_like_features', 0):.2%}")
                print(f"  - Cores de pássaro: {visual_analysis.get('bird_colors', False)}")
                print(f"  - Proporções de pássaro: {visual_analysis.get('bird_proportions', False)}")
                print(f"  - Textura de pássaro: {visual_analysis.get('bird_texture', False)}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Limpar arquivo temporário
        if os.path.exists("test_web.jpg"):
            os.remove("test_web.jpg")

if __name__ == "__main__":
    test_web_simulation()
