#!/usr/bin/env python3
"""
Debug específico para lógica de raciocínio
"""

import sys
import os
import cv2
import numpy as np

# Adicionar o diretório src ao path
sys.path.append('src')

from utils.debug_logger import DebugLogger
from core.intuition import IntuitionEngine

def debug_logical_reasoning(image_path: str):
    """Debug detalhado da lógica de raciocínio"""
    print(f"\n🧠 DEBUG LÓGICA DE RACIOCÍNIO: {os.path.basename(image_path)}")
    print("="*70)
    
    try:
        # Carregar imagem
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Erro: Não foi possível carregar a imagem {image_path}")
            return
            
        print(f"📏 Dimensões: {image.shape}")
        
        # Inicializar engine
        debug_logger = DebugLogger()
        engine = IntuitionEngine('yolov8n.pt', 'modelo_classificacao_passaros.keras', debug_logger)
        
        # Análise visual
        visual_analysis = engine._detect_visual_characteristics(image)
        print(f"\n🔬 ANÁLISE VISUAL:")
        print(f"  👁️  Olhos: {visual_analysis.get('has_eyes', False)}")
        print(f"  🪶 Asas: {visual_analysis.get('has_wings', False)}")
        print(f"  🐦 Bico: {visual_analysis.get('has_beak', False)}")
        print(f"  🪶 Penas: {visual_analysis.get('has_feathers', False)}")
        print(f"  🦅 Garras: {visual_analysis.get('has_claws', False)}")
        print(f"  🐕 Mamífero: {visual_analysis.get('has_mammal_features', False)}")
        print(f"  📊 Score características: {visual_analysis.get('bird_like_features', 0):.3f}")
        print(f"  📐 Score forma: {visual_analysis.get('bird_shape_score', 0):.3f}")
        print(f"  🎨 Score cores: {visual_analysis.get('bird_color_score', 0):.3f}")
        
        # Características fundamentais (simulando)
        characteristics = {
            'has_wings': visual_analysis.get('has_wings', False),
            'has_beak': visual_analysis.get('has_beak', False),
            'has_feathers': visual_analysis.get('has_feathers', False),
            'has_eyes': visual_analysis.get('has_eyes', False),
            'has_claws': visual_analysis.get('has_claws', False),
            'has_mammal_features': visual_analysis.get('has_mammal_features', False),
            'has_mammal_body': False,
            'has_fur_texture': False
        }
        
        print(f"\n🧠 CARACTERÍSTICAS FUNDAMENTAIS:")
        print(f"  🎯 Características: {characteristics}")
        
        # Contar características de pássaro
        bird_characteristics = [characteristics['has_wings'], characteristics['has_beak'], 
                              characteristics['has_feathers'], characteristics['has_eyes'], 
                              characteristics['has_claws']]
        bird_count = sum(bird_characteristics)
        print(f"  📊 Contagem de características de pássaro: {bird_count}")
        
        # Testar lógica de raciocínio passo a passo
        print(f"\n🔍 TESTE DA LÓGICA PASSO A PASSO:")
        
        # Regra 1: bird_count >= 3
        if bird_count >= 3:
            print(f"  ✅ REGRA 1 ATIVADA: {bird_count} características >= 3")
            print(f"    → Resultado: É pássaro (confiança: 0.9)")
            return
        
        # Regra 2: bird_count >= 2 and bird_like_features > 0.4
        bird_like_features = visual_analysis.get('bird_like_features', 0)
        if bird_count >= 2 and bird_like_features > 0.4:
            print(f"  ✅ REGRA 2 ATIVADA: {bird_count} características >= 2 e score {bird_like_features:.3f} > 0.4")
            print(f"    → Resultado: É pássaro (confiança: 0.8)")
            return
        
        # Regra 3: bird_count >= 1 and (bird_shape_score > 0.4 or bird_color_score > 0.4)
        bird_shape_score = visual_analysis.get('bird_shape_score', 0)
        bird_color_score = visual_analysis.get('bird_color_score', 0)
        if bird_count >= 1 and (bird_shape_score > 0.4 or bird_color_score > 0.4):
            print(f"  ✅ REGRA 3 ATIVADA: {bird_count} características >= 1 e (forma {bird_shape_score:.3f} > 0.4 ou cores {bird_color_score:.3f} > 0.4)")
            print(f"    → Resultado: É pássaro (confiança: 0.7)")
            return
        
        # Regra 4: bird_like_features > 0.5 and (has_eyes or has_wings)
        if bird_like_features > 0.5 and (characteristics['has_eyes'] or characteristics['has_wings']):
            print(f"  ✅ REGRA 4 ATIVADA: score {bird_like_features:.3f} > 0.5 e (olhos ou asas)")
            print(f"    → Resultado: É pássaro (confiança: 0.6)")
            return
        
        # Regra 5: bird_like_features > 0.4 and (bird_shape_score > 0.3 or bird_color_score > 0.3)
        if bird_like_features > 0.4 and (bird_shape_score > 0.3 or bird_color_score > 0.3):
            print(f"  ✅ REGRA 5 ATIVADA: score {bird_like_features:.3f} > 0.4 e (forma {bird_shape_score:.3f} > 0.3 ou cores {bird_color_score:.3f} > 0.3)")
            print(f"    → Resultado: É pássaro (confiança: 0.5)")
            return
        
        # Regra 6: bird_count >= 1 or bird_like_features > 0.3
        if bird_count >= 1 or bird_like_features > 0.3:
            print(f"  ✅ REGRA 6 ATIVADA: {bird_count} características >= 1 ou score {bird_like_features:.3f} > 0.3")
            print(f"    → Resultado: É pássaro (confiança: 0.4)")
            return
        
        # Regra 7: has_mammal_features and not (has_wings or has_beak or has_feathers)
        if characteristics['has_mammal_features'] and not (characteristics['has_wings'] or characteristics['has_beak'] or characteristics['has_feathers']):
            print(f"  ✅ REGRA 7 ATIVADA: características de mamífero e sem características de pássaro")
            print(f"    → Resultado: Não é pássaro (confiança: 0.9)")
            return
        
        # Regra 8: Default
        print(f"  ❌ NENHUMA REGRA ATIVADA - Default")
        print(f"    → Resultado: Não é pássaro (confiança: 0.2)")
        
    except Exception as e:
        print(f"❌ Erro no debug: {str(e)}")

def main():
    """Função principal"""
    print("🧠 DEBUG LÓGICA DE RACIOCÍNIO")
    print("="*70)
    
    # Testar com pássaro azul
    test_images = [
        'test_bird.jpg',  # Pássaro azul
        '/Users/matheusferreira/Downloads/triste-cachorro.jpg'  # Cachorro
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            debug_logical_reasoning(image_path)
        else:
            print(f"⚠️  Imagem não encontrada: {image_path}")

if __name__ == "__main__":
    main()
