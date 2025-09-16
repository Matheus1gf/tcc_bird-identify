#!/usr/bin/env python3
"""
Teste de debug para verificar por que a intuição não está funcionando
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.intuition import IntuitionEngine
import base64

def test_intuition_debug():
    print("🔍 TESTE DE DEBUG DA INTUIÇÃO")
    print("=" * 50)
    
    # Inicializar o sistema
    intuition_engine = IntuitionEngine("yolov8n.pt", "modelo_classificacao_passaros.keras")
    
    # Imagem de teste simples (pixel branco)
    test_image_data = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
    
    # Salvar imagem temporária
    with open("test_debug.jpg", "wb") as f:
        f.write(test_image_data)
    
    try:
        print("📊 Testando análise visual...")
        visual_analysis = intuition_engine._analyze_visual_characteristics("test_debug.jpg")
        
        print(f"  - Score geral: {visual_analysis.get('bird_like_features', 0):.2%}")
        print(f"  - Cores de pássaro: {visual_analysis.get('bird_colors', False)}")
        print(f"  - Proporções de pássaro: {visual_analysis.get('bird_proportions', False)}")
        print(f"  - Textura de pássaro: {visual_analysis.get('bird_texture', False)}")
        
        detailed_scores = visual_analysis.get("detailed_scores", {})
        print(f"  - Scores detalhados:")
        print(f"    * colors: {detailed_scores.get('colors', 0):.2%}")
        print(f"    * proportions: {detailed_scores.get('proportions', 0):.2%}")
        print(f"    * texture: {detailed_scores.get('texture', 0):.2%}")
        print(f"    * contours: {detailed_scores.get('contours', 0):.2%}")
        
        print("\n📊 Testando análise completa...")
        full_analysis = intuition_engine.analyze_image_intuition("test_debug.jpg")
        
        print(f"  - Candidatos encontrados: {full_analysis.get('candidates_found', 0)}")
        print(f"  - Recomendação: {full_analysis.get('recommendation', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Limpar arquivo temporário
        if os.path.exists("test_debug.jpg"):
            os.remove("test_debug.jpg")

if __name__ == "__main__":
    test_intuition_debug()
