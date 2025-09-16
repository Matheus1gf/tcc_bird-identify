#!/usr/bin/env python3
"""
Teste específico para o sistema de intuição
"""

import sys
import os
import numpy as np
from PIL import Image

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_intuition_analysis():
    """Testar análise de intuição"""
    print("🔍 Testando análise de intuição...")
    
    try:
        # Importar módulos necessários
        from core.intuition import IntuitionEngine
        
        print("  📦 IntuitionEngine importado")
        
        # Inicializar sistema
        print("  🧠 Inicializando IntuitionEngine...")
        intuition_engine = IntuitionEngine("yolov8n.pt", "modelo_classificacao_passaros.keras")
        print("  ✅ IntuitionEngine inicializado")
        
        # Criar imagem de teste
        print("  🖼️ Criando imagem de teste...")
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)
        
        # Salvar imagem temporária
        temp_path = "test_intuition.jpg"
        pil_image.save(temp_path)
        print(f"  💾 Imagem salva em: {temp_path}")
        
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
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_intuition_levels():
    """Testar diferentes níveis de intuição"""
    print("\n🔍 Testando diferentes níveis de intuição...")
    
    try:
        from core.intuition import IntuitionEngine
        
        intuition_engine = IntuitionEngine("yolov8n.pt", "modelo_classificacao_passaros.keras")
        
        # Testar com diferentes tipos de imagem
        test_cases = [
            ("Imagem aleatória", np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)),
            ("Imagem com padrões", np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)),
            ("Imagem escura", np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)),
        ]
        
        for name, image_array in test_cases:
            print(f"  🖼️ Testando: {name}")
            
            # Salvar imagem
            temp_path = f"test_{name.lower().replace(' ', '_')}.jpg"
            pil_image = Image.fromarray(image_array)
            pil_image.save(temp_path)
            
            # Analisar
            result = intuition_engine.analyze_image_intuition(temp_path)
            intuition_level = result.get('intuition_analysis', {}).get('intuition_level', 'Nenhuma')
            recommendation = result.get('intuition_analysis', {}).get('recommendation', 'N/A')
            
            print(f"    - Intuição: {intuition_level}")
            print(f"    - Recomendação: {recommendation}")
            
            # Limpar
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Função principal"""
    print("🧪 TESTE DO SISTEMA DE INTUIÇÃO")
    print("=" * 50)
    
    tests = [
        ("Análise de Intuição", test_intuition_analysis),
        ("Níveis de Intuição", test_intuition_levels)
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
        print("\n🎉 TODOS OS TESTES PASSARAM! Sistema de intuição funcionando.")
        return True
    else:
        print(f"\n⚠️ {total-passed} TESTES FALHARAM. Corrija os problemas.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
