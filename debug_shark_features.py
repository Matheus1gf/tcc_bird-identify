#!/usr/bin/env python3
"""
Debug específico para entender por que o tubarão está sendo detectado como tendo bico e garras
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.intuition import IntuitionEngine
from src.utils.logger import DebugLogger
import cv2
import numpy as np

def debug_shark_features():
    """Debug específico das características do tubarão"""
    
    print("🦈 DEBUG: CARACTERÍSTICAS DO TUBARÃO")
    print("="*60)
    
    # Inicializar logger
    debug_logger = DebugLogger()
    
    # Inicializar motor de intuição
    try:
        engine = IntuitionEngine("yolov8n.pt", "modelo_classificacao_passaros.keras", debug_logger)
        print("✅ Motor de intuição inicializado com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao inicializar motor: {e}")
        return
    
    # Carregar imagem do tubarão
    shark_image_path = "temp_Images_%287%29tuba.jpg.png"
    
    if not os.path.exists(shark_image_path):
        print(f"❌ Arquivo não encontrado: {shark_image_path}")
        return
    
    print(f"\n📸 Analisando: {shark_image_path}")
    print("-" * 40)
    
    # Carregar imagem
    image = cv2.imread(shark_image_path)
    if image is None:
        print("❌ Erro ao carregar imagem")
        return
    
    print(f"📏 Dimensões: {image.shape}")
    
    # Testar cada método de detecção individualmente
    print("\n🔬 TESTE DETALHADO DOS MÉTODOS:")
    
    # 1. Detecção de olhos
    print("\n1️⃣ DETECÇÃO DE OLHOS:")
    has_eyes = engine._detect_simple_eyes(image)
    print(f"   • Resultado: {has_eyes}")
    
    # 2. Detecção de asas
    print("\n2️⃣ DETECÇÃO DE ASAS:")
    has_wings = engine._detect_simple_wings(image)
    print(f"   • Resultado: {has_wings}")
    
    # 3. Detecção de bico (PROBLEMA!)
    print("\n3️⃣ DETECÇÃO DE BICO (PROBLEMA!):")
    has_beak = engine._detect_simple_beak(image)
    print(f"   • Resultado: {has_beak}")
    
    if has_beak:
        print("   ⚠️ PROBLEMA: Tubarão detectado como tendo bico!")
        # Vamos investigar o que está sendo detectado como bico
        print("   🔍 Investigando detecção de bico...")
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar Canny
        edges = cv2.Canny(gray, 50, 150)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"   • Total de contornos encontrados: {len(contours)}")
        
        # Verificar contornos que podem ser detectados como bico
        beak_candidates = 0
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 80:  # Threshold do método _detect_simple_beak
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                if aspect_ratio > 2.5:  # Threshold do método
                    beak_candidates += 1
                    print(f"   • Contorno {i}: área={area:.1f}, aspect_ratio={aspect_ratio:.2f}, pos=({x},{y})")
        
        print(f"   • Candidatos a bico encontrados: {beak_candidates}")
    
    # 4. Detecção de penas
    print("\n4️⃣ DETECÇÃO DE PENAS:")
    has_feathers = engine._detect_simple_feathers(image)
    print(f"   • Resultado: {has_feathers}")
    
    # 5. Detecção de garras (PROBLEMA!)
    print("\n5️⃣ DETECÇÃO DE GARRAS (PROBLEMA!):")
    has_claws = engine._detect_simple_claws(image)
    print(f"   • Resultado: {has_claws}")
    
    if has_claws:
        print("   ⚠️ PROBLEMA: Tubarão detectado como tendo garras!")
        print("   🔍 Investigando detecção de garras...")
        
        # Verificar contornos que podem ser detectados como garras
        claw_candidates = 0
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 30:  # Threshold do método _detect_simple_claws
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                if aspect_ratio > 1.8:  # Threshold do método
                    claw_candidates += 1
                    print(f"   • Contorno {i}: área={area:.1f}, aspect_ratio={aspect_ratio:.2f}, pos=({x},{y})")
        
        print(f"   • Candidatos a garras encontrados: {claw_candidates}")
    
    # 6. Detecção de características de mamífero
    print("\n6️⃣ DETECÇÃO DE CARACTERÍSTICAS DE MAMÍFERO:")
    has_mammal = engine._detect_simple_mammal_features(image)
    print(f"   • Resultado: {has_mammal}")
    
    # Resumo
    print("\n📊 RESUMO DAS DETECÇÕES:")
    print(f"   • Olhos: {has_eyes}")
    print(f"   • Asas: {has_wings}")
    print(f"   • Bico: {has_beak} ⚠️")
    print(f"   • Penas: {has_feathers}")
    print(f"   • Garras: {has_claws} ⚠️")
    print(f"   • Mamífero: {has_mammal}")
    
    # Contar características de pássaro
    bird_characteristics = [has_eyes, has_wings, has_beak, has_feathers, has_claws]
    bird_count = sum(bird_characteristics)
    
    print(f"\n🐦 Total de características de pássaro: {bird_count}/5")
    
    if bird_count >= 2:
        print("   ⚠️ PROBLEMA: Tubarão tem muitas características de pássaro!")
        print("   🔧 Necessário ajustar thresholds de detecção")

if __name__ == "__main__":
    debug_shark_features()
