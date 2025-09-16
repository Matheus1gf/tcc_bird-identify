#!/usr/bin/env python3
"""
Debug específico para detecção de mamíferos
"""

import cv2
import numpy as np
import sys
import os

# Adicionar o diretório src ao path
sys.path.append('src')

from utils.debug_logger import DebugLogger
from core.intuition import IntuitionEngine

def debug_mammal_detection(image_path: str):
    """Debug detalhado da detecção de mamíferos"""
    print(f"\n🔍 DEBUG DETECÇÃO DE MAMÍFEROS: {os.path.basename(image_path)}")
    print("="*60)
    
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
        
        # Testar detecção de mamíferos
        has_mammal_features = engine._detect_simple_mammal_features(image)
        print(f"🐕 Resultado final: {has_mammal_features}")
        
        # Debug detalhado
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"\n📊 ANÁLISE DETALHADA:")
        print(f"  🔢 Total de contornos: {len(contours)}")
        
        mammal_features = 0
        h, w = image.shape[:2]
        
        # Analisar cada contorno
        for i, contour in enumerate(contours):
            if len(contour) > 5:
                area = cv2.contourArea(contour)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                x, y, contour_w, contour_h = cv2.boundingRect(contour)
                aspect_ratio = contour_w / contour_h if contour_h > 0 else 1.0
                solidity = area / hull_area if hull_area > 0 else 0
                
                print(f"\n  🔍 Contorno {i+1}:")
                print(f"    📏 Área: {area:.1f}")
                print(f"    📐 Aspect ratio: {aspect_ratio:.2f}")
                print(f"    🔲 Solidity: {solidity:.3f}")
                print(f"    📍 Posição: ({x}, {y})")
                print(f"    📏 Tamanho: {contour_w}x{contour_h}")
                
                # Verificar critérios de orelha
                if len(contour) > 10 and area > 500:
                    if solidity > 0.95:
                        mammal_features += 1
                        print(f"    ✅ DETECTOU ORELHA! (solidity: {solidity:.3f})")
                    else:
                        print(f"    ❌ Não é orelha (solidity: {solidity:.3f} < 0.95)")
                else:
                    print(f"    ❌ Não é orelha (área: {area:.1f} < 500 ou pontos: {len(contour)} < 10)")
                
                # Verificar critérios de focinho
                if len(contour) > 10 and area > 300:
                    if aspect_ratio > 3.0 and y > h * 0.8:
                        mammal_features += 1
                        print(f"    ✅ DETECTOU FOCINHO! (aspect: {aspect_ratio:.2f}, y: {y} > {h*0.8:.1f})")
                    else:
                        print(f"    ❌ Não é focinho (aspect: {aspect_ratio:.2f} < 3.0 ou y: {y} < {h*0.8:.1f})")
                else:
                    print(f"    ❌ Não é focinho (área: {area:.1f} < 300 ou pontos: {len(contour)} < 10)")
                
                # Verificar critérios de nariz
                if len(contour) > 5 and area > 50:
                    if 0.8 < aspect_ratio < 1.2 and solidity > 0.9:
                        mammal_features += 1
                        print(f"    ✅ DETECTOU NARIZ! (aspect: {aspect_ratio:.2f}, solidity: {solidity:.3f})")
                    else:
                        print(f"    ❌ Não é nariz (aspect: {aspect_ratio:.2f}, solidity: {solidity:.3f})")
                else:
                    print(f"    ❌ Não é nariz (área: {area:.1f} < 50 ou pontos: {len(contour)} < 5)")
        
        print(f"\n🎯 RESULTADO FINAL:")
        print(f"  🐕 Características de mamífero encontradas: {mammal_features}")
        print(f"  📊 Threshold: >= 3")
        print(f"  ✅ É mamífero: {mammal_features >= 3}")
        
    except Exception as e:
        print(f"❌ Erro no debug: {str(e)}")

def main():
    """Função principal"""
    print("🔍 DEBUG DETECÇÃO DE MAMÍFEROS")
    print("="*60)
    
    # Testar com as imagens disponíveis
    test_images = [
        '/Users/matheusferreira/Downloads/triste-cachorro.jpg',
        'test_bird.jpg',
        'test_mammal.jpg'
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            debug_mammal_detection(image_path)
        else:
            print(f"⚠️  Imagem não encontrada: {image_path}")

if __name__ == "__main__":
    main()
