#!/usr/bin/env python3
"""
Teste específico para análise da rolinha-roxa
"""

import sys
import os
import numpy as np
from PIL import Image

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_rolinha_test_image():
    """Criar uma imagem de teste simulando uma rolinha-roxa"""
    # Criar imagem 224x224 com características de rolinha-roxa
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Fundo terroso/marrom claro
    image[:, :] = [180, 150, 120]  # Marrom claro
    
    # Corpo do pássaro (forma oval)
    center_x, center_y = 112, 120
    for y in range(224):
        for x in range(224):
            # Corpo principal (marrom-avermelhado)
            if ((x - center_x)**2 / 40**2 + (y - center_y)**2 / 60**2) <= 1:
                image[y, x] = [150, 80, 60]  # Marrom-avermelhado
            
            # Cabeça (menor, mais clara)
            if ((x - center_x)**2 / 25**2 + (y - (center_y - 30))**2 / 25**2) <= 1:
                image[y, x] = [200, 180, 160]  # Cabeça clara
            
            # Asas com manchas escuras
            if ((x - (center_x - 20))**2 / 30**2 + (y - center_y)**2 / 40**2) <= 1:
                if (x + y) % 8 < 3:  # Padrão de manchas
                    image[y, x] = [80, 40, 30]  # Manchas escuras
    
    return image

def test_rolinha_intuition():
    """Testar intuição com imagem de rolinha-roxa"""
    print("🔍 Testando intuição com rolinha-roxa...")
    
    try:
        from core.intuition import IntuitionEngine
        
        # Inicializar sistema
        from utils.debug_logger import DebugLogger
        debug_logger = DebugLogger()
        intuition_engine = IntuitionEngine("yolov8n.pt", "modelo_classificacao_passaros.keras", debug_logger)
        
        # Criar imagem de teste
        test_image = create_rolinha_test_image()
        pil_image = Image.fromarray(test_image)
        
        # Salvar imagem temporária
        temp_path = "test_rolinha.jpg"
        pil_image.save(temp_path)
        print(f"  💾 Imagem de rolinha salva em: {temp_path}")
        
        # Testar análise de intuição
        print("  🧠 Testando análise de intuição...")
        result = intuition_engine.analyze_image_intuition(temp_path)
        
        print("  📊 Resultados da análise:")
        print(f"    - Caminho da imagem: {result.get('image_path', 'N/A')}")
        
        # Análise YOLO
        yolo_analysis = result.get('yolo_analysis', {})
        print(f"    - Detecções YOLO: {yolo_analysis.get('total_detections', 0)}")
        print(f"    - Confiança média: {yolo_analysis.get('average_confidence', 0):.2%}")
        print(f"    - Tem partes de pássaro: {yolo_analysis.get('has_bird_parts', False)}")
        
        # Análise de intuição
        intuition_analysis = result.get('intuition_analysis', {})
        print(f"    - Candidatos encontrados: {intuition_analysis.get('candidates_found', 0)}")
        print(f"    - Nível de intuição: {intuition_analysis.get('intuition_level', 'N/A')}")
        print(f"    - Recomendação: {intuition_analysis.get('recommendation', 'N/A')}")
        
        # Análise visual detalhada
        visual_analysis = intuition_analysis.get('visual_analysis', {})
        if visual_analysis:
            print("    - Análise visual:")
            print(f"      * Score geral: {visual_analysis.get('bird_like_features', 0):.2%}")
            print(f"      * Cores de pássaro: {visual_analysis.get('bird_colors', False)}")
            print(f"      * Proporções de pássaro: {visual_analysis.get('bird_proportions', False)}")
            print(f"      * Textura de pássaro: {visual_analysis.get('bird_texture', False)}")
            print(f"      * Contornos de pássaro: {visual_analysis.get('bird_contours', False)}")
            
            detailed_scores = visual_analysis.get('detailed_scores', {})
            if detailed_scores:
                print("      * Scores detalhados:")
                for key, score in detailed_scores.items():
                    print(f"        - {key}: {score:.2%}")
        
        # Raciocínio
        reasoning = intuition_analysis.get('reasoning', [])
        if reasoning:
            print("    - Raciocínio da IA:")
            for i, reason in enumerate(reasoning, 1):
                print(f"      {i}. {reason}")
        
        # Limpar arquivo temporário
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print("  🗑️ Arquivo temporário removido")
        
        # Verificar se detectou corretamente
        bird_like_score = visual_analysis.get('bird_like_features', 0)
        if bird_like_score > 0.4:
            print("  ✅ SUCESSO: Sistema detectou características de pássaro!")
            return True
        else:
            print("  ❌ FALHA: Sistema não detectou características de pássaro")
            return False
        
    except Exception as e:
        print(f"  ❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visual_analysis_only():
    """Testar apenas a análise visual"""
    print("\n🔍 Testando análise visual isolada...")
    
    try:
        from core.intuition import IntuitionEngine
        
        from utils.debug_logger import DebugLogger
        debug_logger = DebugLogger()
        intuition_engine = IntuitionEngine("yolov8n.pt", "modelo_classificacao_passaros.keras", debug_logger)
        
        # Criar imagem de teste
        test_image = create_rolinha_test_image()
        pil_image = Image.fromarray(test_image)
        
        # Salvar imagem temporária
        temp_path = "test_visual.jpg"
        pil_image.save(temp_path)
        
        # Testar análise visual
        visual_analysis = intuition_engine._analyze_visual_characteristics(temp_path)
        
        print("  📊 Análise visual:")
        print(f"    - Score geral: {visual_analysis.get('bird_like_features', 0):.2%}")
        print(f"    - Cores de pássaro: {visual_analysis.get('bird_colors', False)}")
        print(f"    - Proporções de pássaro: {visual_analysis.get('bird_proportions', False)}")
        print(f"    - Textura de pássaro: {visual_analysis.get('bird_texture', False)}")
        print(f"    - Contornos de pássaro: {visual_analysis.get('bird_contours', False)}")
        
        detailed_scores = visual_analysis.get('detailed_scores', {})
        if detailed_scores:
            print("    - Scores detalhados:")
            for key, score in detailed_scores.items():
                print(f"      * {key}: {score:.2%}")
        
        # Limpar arquivo temporário
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return visual_analysis.get('bird_like_features', 0) > 0.3
        
    except Exception as e:
        print(f"  ❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Função principal"""
    print("🧪 TESTE DE ANÁLISE DA ROLINHA-ROXA")
    print("=" * 50)
    
    tests = [
        ("Análise Completa", test_rolinha_intuition),
        ("Análise Visual", test_visual_analysis_only)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Executando: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name}: PASSOU")
            else:
                print(f"❌ {test_name}: FALHOU")
        except Exception as e:
            print(f"❌ {test_name}: ERRO - {e}")
            results.append((test_name, False))
    
    # Resumo final
    print("\n" + "=" * 50)
    print("📊 RESUMO DOS TESTES")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{status}: {test_name}")
    
    print(f"\n📈 Taxa de Sucesso: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 TODOS OS TESTES PASSARAM! Sistema melhorado detecta rolinha-roxa.")
        return True
    else:
        print(f"\n⚠️ {total-passed} TESTES FALHARAM. Ajustes necessários.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
