#!/usr/bin/env python3
"""
Teste específico para imagens que NÃO são pássaros (iguana e tubarão)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.intuition import IntuitionEngine
from src.utils.logger import DebugLogger

def test_non_bird_images():
    """Testa imagens que definitivamente não são pássaros"""
    
    # Inicializar logger
    debug_logger = DebugLogger()
    
    # Inicializar motor de intuição
    try:
        engine = IntuitionEngine(
            keras_model_path="modelo_classificacao_passaros.keras",
            yolo_model_path="yolov8n.pt",
            debug_logger=debug_logger
        )
        print("✅ Motor de intuição inicializado com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao inicializar motor: {e}")
        return
    
    # Lista de imagens para testar
    test_images = [
        {
            "path": "test_iguana.jpg",
            "expected": "Não-Pássaro",
            "description": "Iguana verde (réptil)"
        },
        {
            "path": "test_shark.jpg",
            "expected": "Não-Pássaro", 
            "description": "Tubarão (peixe)"
        },
        {
            "path": "test_mammal.jpg", 
            "expected": "Não-Pássaro",
            "description": "Cachorro (mamífero)"
        }
    ]
    
    print("\n" + "="*80)
    print("🧪 TESTE DE IMAGENS NÃO-PÁSSAROS")
    print("="*80)
    
    results = []
    
    for i, test_case in enumerate(test_images, 1):
        image_path = test_case["path"]
        expected = test_case["expected"]
        description = test_case["description"]
        
        print(f"\n📸 TESTE {i}: {description}")
        print(f"📁 Arquivo: {image_path}")
        print(f"🎯 Esperado: {expected}")
        print("-" * 60)
        
        if not os.path.exists(image_path):
            print(f"❌ Arquivo não encontrado: {image_path}")
            continue
            
        try:
            # Analisar imagem
            result = engine.analyze_image_intuition(image_path)
            
            # Extrair resultados
            is_bird = result.get('is_bird', False)
            confidence = result.get('confidence', 0.0)
            species = result.get('species', 'Desconhecida')
            intuition_level = result.get('intuition_level', 'Baixa')
            needs_review = result.get('needs_manual_review', False)
            
            # Determinar resultado
            actual_result = "Pássaro" if is_bird else "Não-Pássaro"
            
            # Verificar se está correto
            is_correct = (actual_result == expected)
            
            print(f"🔍 Resultado: {actual_result}")
            print(f"🎯 Correto: {'✅ SIM' if is_correct else '❌ NÃO'}")
            print(f"📊 Confiança: {confidence:.2f}")
            print(f"🐦 Espécie: {species}")
            print(f"🧠 Nível de Intuição: {intuition_level}")
            print(f"👁️ Precisa Revisão: {'Sim' if needs_review else 'Não'}")
            
            # Mostrar características detectadas
            characteristics = result.get('characteristics_found', [])
            if characteristics:
                print(f"🔍 Características encontradas: {', '.join(characteristics)}")
            else:
                print("🔍 Nenhuma característica específica detectada")
            
            # Mostrar passos do raciocínio
            reasoning_steps = result.get('reasoning_steps', [])
            if reasoning_steps:
                print("🧠 Passos do raciocínio:")
                for step in reasoning_steps:
                    print(f"   • {step}")
            
            # Armazenar resultado
            results.append({
                'image': description,
                'expected': expected,
                'actual': actual_result,
                'correct': is_correct,
                'confidence': confidence,
                'species': species
            })
            
        except Exception as e:
            print(f"❌ Erro na análise: {e}")
            results.append({
                'image': description,
                'expected': expected,
                'actual': 'ERRO',
                'correct': False,
                'confidence': 0.0,
                'species': 'Erro'
            })
    
    # Resumo final
    print("\n" + "="*80)
    print("📊 RESUMO DOS TESTES")
    print("="*80)
    
    total_tests = len(results)
    correct_tests = sum(1 for r in results if r['correct'])
    accuracy = (correct_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"📈 Total de testes: {total_tests}")
    print(f"✅ Testes corretos: {correct_tests}")
    print(f"❌ Testes incorretos: {total_tests - correct_tests}")
    print(f"🎯 Precisão: {accuracy:.1f}%")
    
    print("\n📋 Detalhes:")
    for i, result in enumerate(results, 1):
        status = "✅" if result['correct'] else "❌"
        print(f"{i}. {status} {result['image']}: {result['actual']} (conf: {result['confidence']:.2f})")
    
    # Avaliação geral
    if accuracy >= 80:
        print(f"\n🎉 EXCELENTE! Sistema funcionando corretamente ({accuracy:.1f}% de precisão)")
    elif accuracy >= 60:
        print(f"\n⚠️ BOM, mas precisa melhorar ({accuracy:.1f}% de precisão)")
    else:
        print(f"\n❌ PROBLEMA! Sistema precisa de ajustes ({accuracy:.1f}% de precisão)")
    
    return results

if __name__ == "__main__":
    test_non_bird_images()
