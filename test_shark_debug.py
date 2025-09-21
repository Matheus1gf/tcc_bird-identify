#!/usr/bin/env python3
"""
Teste específico para debugar o problema do tubarão sendo identificado como pássaro
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.intuition import IntuitionEngine
from src.utils.logger import DebugLogger

def test_shark_detection():
    """Testa especificamente a detecção de tubarão"""
    
    print("🦈 TESTE ESPECÍFICO: DETECÇÃO DE TUBARÃO")
    print("="*60)
    
    # Inicializar logger
    debug_logger = DebugLogger()
    
    # Inicializar motor de intuição (mesma forma que na interface web)
    try:
        engine = IntuitionEngine("yolov8n.pt", "modelo_classificacao_passaros.keras", debug_logger)
        print("✅ Motor de intuição inicializado com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao inicializar motor: {e}")
        return
    
    # Testar com imagem de tubarão
    shark_image = "test_shark.jpg"
    
    if not os.path.exists(shark_image):
        print(f"❌ Arquivo não encontrado: {shark_image}")
        return
    
    print(f"\n🦈 Testando imagem: {shark_image}")
    print("-" * 40)
    
    try:
        # Analisar imagem
        result = engine.analyze_image_intuition(shark_image)
        
        # Extrair resultados
        is_bird = result.get('is_bird', False)
        confidence = result.get('confidence', 0.0)
        species = result.get('species', 'Desconhecida')
        intuition_level = result.get('intuition_level', 'Baixa')
        needs_review = result.get('needs_manual_review', False)
        
        # Determinar resultado
        actual_result = "Pássaro" if is_bird else "Não-Pássaro"
        
        print(f"🔍 Resultado: {actual_result}")
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
        
        # Mostrar análise visual detalhada
        visual_analysis = result.get('visual_analysis', {})
        if visual_analysis:
            print("\n🔬 Análise Visual Detalhada:")
            print(f"   • Score características de pássaro: {visual_analysis.get('bird_like_features', 0):.3f}")
            print(f"   • Score forma de pássaro: {visual_analysis.get('bird_shape_score', 0):.3f}")
            print(f"   • Score cores de pássaro: {visual_analysis.get('bird_color_score', 0):.3f}")
        
        # Mostrar características fundamentais
        fundamental_chars = result.get('fundamental_characteristics', {})
        if fundamental_chars:
            print("\n🧠 Características Fundamentais:")
            for char, value in fundamental_chars.items():
                print(f"   • {char}: {value}")
        
        # Verificar se está correto
        is_correct = (actual_result == "Não-Pássaro")
        
        print(f"\n🎯 Resultado: {'✅ CORRETO' if is_correct else '❌ INCORRETO'}")
        
        if not is_correct:
            print("\n⚠️ PROBLEMA DETECTADO!")
            print("O tubarão está sendo identificado incorretamente como pássaro.")
            print("Vamos investigar os métodos de detecção...")
            
            # Testar métodos individuais
            print("\n🔬 TESTE DOS MÉTODOS INDIVIDUAIS:")
            
            # Carregar imagem para testes manuais
            import cv2
            image = cv2.imread(shark_image)
            if image is not None:
                print(f"📏 Dimensões da imagem: {image.shape}")
                
                # Testar detecção de olhos
                has_eyes = engine._detect_simple_eyes(image)
                print(f"   • Olhos detectados: {has_eyes}")
                
                # Testar detecção de asas
                has_wings = engine._detect_simple_wings(image)
                print(f"   • Asas detectadas: {has_wings}")
                
                # Testar detecção de bico
                has_beak = engine._detect_simple_beak(image)
                print(f"   • Bico detectado: {has_beak}")
                
                # Testar detecção de penas
                has_feathers = engine._detect_simple_feathers(image)
                print(f"   • Penas detectadas: {has_feathers}")
                
                # Testar detecção de garras
                has_claws = engine._detect_simple_claws(image)
                print(f"   • Garras detectadas: {has_claws}")
                
                # Testar detecção de características de mamífero
                has_mammal = engine._detect_simple_mammal_features(image)
                print(f"   • Características de mamífero: {has_mammal}")
        
    except Exception as e:
        print(f"❌ Erro na análise: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_shark_detection()
