#!/usr/bin/env python3
"""
Teste Minucioso de Análise de Imagens
Analisa cada imagem detalhadamente para identificar problemas na detecção
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Adicionar o diretório src ao path
sys.path.append('src')

from utils.debug_logger import DebugLogger
from core.intuition import IntuitionEngine

def analyze_image_detailed(image_path: str, engine: IntuitionEngine, debug_logger: DebugLogger):
    """Análise detalhada de uma imagem"""
    print(f"\n{'='*80}")
    print(f"🔍 ANÁLISE DETALHADA: {os.path.basename(image_path)}")
    print(f"{'='*80}")
    
    try:
        # Carregar imagem
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Erro: Não foi possível carregar a imagem {image_path}")
            return None
            
        print(f"📏 Dimensões: {image.shape}")
        print(f"🎨 Formato: {image.dtype}")
        
        # Análise visual básica
        visual_analysis = engine._detect_visual_characteristics(image)
        print(f"\n🔬 ANÁLISE VISUAL:")
        print(f"  👁️  Olhos detectados: {visual_analysis.get('has_eyes', False)}")
        print(f"  🪶 Asas detectadas: {visual_analysis.get('has_wings', False)}")
        print(f"  🐦 Bico detectado: {visual_analysis.get('has_beak', False)}")
        print(f"  🪶 Penas detectadas: {visual_analysis.get('has_feathers', False)}")
        print(f"  🦅 Garras detectadas: {visual_analysis.get('has_claws', False)}")
        print(f"  🐕 Características de mamífero: {visual_analysis.get('has_mammal_features', False)}")
        print(f"  📊 Score características de pássaro: {visual_analysis.get('bird_like_features', 0):.3f}")
        print(f"  📐 Score forma de pássaro: {visual_analysis.get('bird_shape_score', 0):.3f}")
        print(f"  🎨 Score cores de pássaro: {visual_analysis.get('bird_color_score', 0):.3f}")
        
        # Análise de características fundamentais
        characteristics = engine._detect_fundamental_characteristics(image)
        print(f"\n🧠 CARACTERÍSTICAS FUNDAMENTAIS:")
        print(f"  🎯 Características encontradas: {characteristics.get('characteristics_found', [])}")
        print(f"  ❌ Características ausentes: {characteristics.get('missing_characteristics', [])}")
        print(f"  📈 Score geral: {characteristics.get('overall_score', 0):.3f}")
        
        # Análise de intuição completa
        result = engine.analyze_image_intuition(image_path)
        print(f"\n🎯 RESULTADO FINAL:")
        print(f"  🐦 É pássaro: {result.get('is_bird', False)}")
        print(f"  📊 Confiança: {result.get('confidence', 0):.3f}")
        print(f"  🐦 Espécie: {result.get('species', 'Desconhecida')}")
        print(f"  🧠 Nível de intuição: {result.get('intuition_level', 'Baixa')}")
        print(f"  👤 Precisa análise manual: {result.get('needs_manual_review', False)}")
        print(f"  📝 Passos do raciocínio:")
        for step in result.get('reasoning_steps', []):
            print(f"    • {step}")
        
        return result
        
    except Exception as e:
        print(f"❌ Erro na análise: {str(e)}")
        debug_logger.log_error(f"Erro na análise detalhada de {image_path}: {str(e)}")
        return None

def test_simple_detection_methods(image_path: str, engine: IntuitionEngine):
    """Testa métodos de detecção simples individualmente"""
    print(f"\n🔬 TESTE DOS MÉTODOS SIMPLES:")
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Erro: Não foi possível carregar a imagem {image_path}")
            return
            
        # Testar cada método individualmente
        print(f"  👁️  _detect_simple_eyes: {engine._detect_simple_eyes(image)}")
        print(f"  🪶 _detect_simple_wings: {engine._detect_simple_wings(image)}")
        print(f"  🐦 _detect_simple_beak: {engine._detect_simple_beak(image)}")
        print(f"  🪶 _detect_simple_feathers: {engine._detect_simple_feathers(image)}")
        print(f"  🦅 _detect_simple_claws: {engine._detect_simple_claws(image)}")
        print(f"  🐕 _detect_simple_mammal_features: {engine._detect_simple_mammal_features(image)}")
        
        # Análise de cores
        color_analysis = engine._analyze_colors(image)
        print(f"  🎨 Análise de cores: {color_analysis}")
        
        # Análise de formas
        shape_analysis = engine._analyze_shapes(image)
        print(f"  📐 Análise de formas: {shape_analysis}")
        
        # Análise de texturas
        texture_analysis = engine._analyze_textures(image)
        print(f"  🧵 Análise de texturas: {texture_analysis}")
        
    except Exception as e:
        print(f"❌ Erro no teste simples: {str(e)}")

def main():
    """Função principal de teste"""
    print("🚀 INICIANDO TESTES MINUCIOSOS DE ANÁLISE DE IMAGENS")
    print("="*80)
    
    # Inicializar componentes
    debug_logger = DebugLogger()
    
    # Tentar inicializar o engine
    try:
        engine = IntuitionEngine('yolov8n.pt', 'modelo_classificacao_passaros.keras', debug_logger)
        print("✅ Engine de intuição inicializado com sucesso")
    except Exception as e:
        print(f"❌ Erro ao inicializar engine: {str(e)}")
        return
    
    # Lista de imagens para testar (baseado nas descrições fornecidas)
    test_images = [
        {
            'path': '/Users/matheusferreira/Downloads/triste-cachorro.jpg',
            'expected': 'Não-Pássaro',
            'description': 'Filhote de cachorro Labrador Retriever'
        },
        {
            'path': 'test_bird.jpg',  # Assumindo que existe
            'expected': 'Pássaro',
            'description': 'Pássaro azul (Ultramarine Grosbeak)'
        },
        {
            'path': 'test_mammal.jpg',  # Assumindo que existe
            'expected': 'Não-Pássaro', 
            'description': 'Pássaro com crista vermelha'
        }
    ]
    
    # Verificar quais imagens existem
    available_images = []
    for img_info in test_images:
        if os.path.exists(img_info['path']):
            available_images.append(img_info)
        else:
            print(f"⚠️  Imagem não encontrada: {img_info['path']}")
    
    if not available_images:
        print("❌ Nenhuma imagem de teste encontrada!")
        return
    
    print(f"📸 Encontradas {len(available_images)} imagens para teste")
    
    # Executar testes
    results = []
    for img_info in available_images:
        print(f"\n{'='*60}")
        print(f"🖼️  TESTANDO: {img_info['description']}")
        print(f"📁 Arquivo: {img_info['path']}")
        print(f"🎯 Esperado: {img_info['expected']}")
        print(f"{'='*60}")
        
        # Análise detalhada
        result = analyze_image_detailed(img_info['path'], engine, debug_logger)
        
        # Teste dos métodos simples
        test_simple_detection_methods(img_info['path'], engine)
        
        if result:
            # Verificar se o resultado está correto
            is_correct = False
            if img_info['expected'] == 'Pássaro' and result.get('is_bird', False):
                is_correct = True
            elif img_info['expected'] == 'Não-Pássaro' and not result.get('is_bird', False):
                is_correct = True
            
            results.append({
                'image': img_info['description'],
                'expected': img_info['expected'],
                'actual': 'Pássaro' if result.get('is_bird', False) else 'Não-Pássaro',
                'correct': is_correct,
                'confidence': result.get('confidence', 0)
            })
            
            print(f"\n✅ RESULTADO: {'CORRETO' if is_correct else 'INCORRETO'}")
            print(f"   Esperado: {img_info['expected']}")
            print(f"   Obtido: {'Pássaro' if result.get('is_bird', False) else 'Não-Pássaro'}")
            print(f"   Confiança: {result.get('confidence', 0):.3f}")
    
    # Resumo final
    print(f"\n{'='*80}")
    print("📊 RESUMO DOS TESTES")
    print(f"{'='*80}")
    
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    
    print(f"✅ Testes corretos: {correct_count}/{total_count}")
    print(f"📊 Taxa de acerto: {(correct_count/total_count)*100:.1f}%")
    
    print(f"\n📋 DETALHES:")
    for result in results:
        status = "✅" if result['correct'] else "❌"
        print(f"  {status} {result['image']}: {result['expected']} → {result['actual']} (conf: {result['confidence']:.3f})")
    
    print(f"\n🎯 CONCLUSÃO:")
    if correct_count == total_count:
        print("🎉 TODOS OS TESTES PASSARAM! Sistema funcionando corretamente.")
    else:
        print(f"⚠️  {total_count - correct_count} teste(s) falharam. Necessário ajuste no sistema.")

if __name__ == "__main__":
    main()
