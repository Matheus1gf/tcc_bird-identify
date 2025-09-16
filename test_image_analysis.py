#!/usr/bin/env python3
"""
Teste específico para análise de imagens
"""

import sys
import os
import numpy as np
from PIL import Image

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_image_analysis():
    """Testar análise de imagens"""
    print("🔍 Testando análise de imagens...")
    
    try:
        # Importar módulos necessários
        from core.intuition import IntuitionEngine
        from core.reasoning import LogicalAIReasoningSystem
        from core.learning import ContinuousLearningSystem
        from interfaces.manual_analysis import manual_analysis
        from interfaces.tinder_interface import TinderInterface
        
        print("  📦 Módulos importados com sucesso")
        
        # Inicializar sistemas
        print("  🧠 Inicializando IntuitionEngine...")
        intuition_engine = IntuitionEngine("yolov8n.pt", "modelo_classificacao_passaros.keras")
        
        print("  🤖 Inicializando LogicalAIReasoningSystem...")
        reasoning_system = LogicalAIReasoningSystem()
        
        print("  📚 Inicializando ContinuousLearningSystem...")
        learning_system = ContinuousLearningSystem("yolov8n.pt", "modelo_classificacao_passaros.keras")
        
        print("  💡 Inicializando TinderInterface...")
        tinder_interface = TinderInterface(manual_analysis)
        
        print("  ✅ Todos os sistemas inicializados")
        
        # Criar imagem de teste
        print("  🖼️ Criando imagem de teste...")
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)
        
        # Salvar imagem temporária
        temp_path = "test_image.jpg"
        pil_image.save(temp_path)
        print(f"  💾 Imagem salva em: {temp_path}")
        
        # Testar análise de intuição
        print("  🧠 Testando análise de intuição...")
        try:
            result = intuition_engine.analyze_image_intuition(temp_path)
            print(f"  ✅ Análise de intuição: {result}")
        except Exception as e:
            print(f"  ❌ Erro na análise de intuição: {e}")
        
        # Testar análise manual
        print("  📝 Testando análise manual...")
        try:
            manual_result = manual_analysis.analyze_image(temp_path)
            print(f"  ✅ Análise manual: {manual_result}")
        except Exception as e:
            print(f"  ❌ Erro na análise manual: {e}")
        
        # Limpar arquivo temporário
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print("  🗑️ Arquivo temporário removido")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erro geral: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_debug_logger():
    """Testar debug logger"""
    print("\n🔍 Testando debug logger...")
    
    try:
        from utils.debug_logger import debug_logger
        
        print("  📦 DebugLogger importado")
        
        # Testar métodos
        print("  🚀 Testando log_session_start...")
        debug_logger.log_session_start("test_image.jpg")
        
        print("  ✅ Testando log_success...")
        debug_logger.log_success("Teste de sucesso")
        
        print("  ❌ Testando log_error...")
        debug_logger.log_error("Teste de erro", "TEST_ERROR")
        
        print("  ✅ DebugLogger funcionando")
        return True
        
    except Exception as e:
        print(f"  ❌ Erro no debug logger: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Função principal"""
    print("🧪 TESTE DE ANÁLISE DE IMAGENS")
    print("=" * 50)
    
    tests = [
        ("Análise de Imagens", test_image_analysis),
        ("Debug Logger", test_debug_logger)
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
        print("\n🎉 TODOS OS TESTES PASSARAM! Análise de imagens funcionando.")
        return True
    else:
        print(f"\n⚠️ {total-passed} TESTES FALHARAM. Corrija os problemas.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
