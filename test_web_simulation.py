#!/usr/bin/env python3
"""
Simula exatamente o que a interface web faz para processar imagens
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.intuition import IntuitionEngine
from src.utils.logger import DebugLogger
from PIL import Image
import cv2
import numpy as np

def simulate_web_processing():
    """Simula o processamento da interface web"""
    
    print("🌐 SIMULAÇÃO DO PROCESSAMENTO DA INTERFACE WEB")
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
    
    # Simular upload de imagem de tubarão
    original_image_path = "temp_Images_%287%29tuba.jpg.png"
    
    if not os.path.exists(original_image_path):
        print(f"❌ Arquivo não encontrado: {original_image_path}")
        return
    
    print(f"\n📸 Simulando upload: {original_image_path}")
    print("-" * 40)
    
    try:
        # Simular o que a interface web faz:
        # 1. Carregar imagem com PIL
        print("1️⃣ Carregando imagem com PIL...")
        image = Image.open(original_image_path)
        print(f"   • Formato original: {image.format}")
        print(f"   • Modo: {image.mode}")
        print(f"   • Tamanho: {image.size}")
        
        # 2. Salvar como PNG (como a interface web faz)
        print("\n2️⃣ Convertendo para PNG...")
        temp_path = f"temp_shark_web_simulation.png"
        image.save(temp_path)
        print(f"   • Salvo como: {temp_path}")
        
        # 3. Verificar diferenças entre as imagens
        print("\n3️⃣ Comparando imagens...")
        
        # Carregar imagem original
        original_cv = cv2.imread(original_image_path)
        temp_cv = cv2.imread(temp_path)
        
        if original_cv is not None and temp_cv is not None:
            print(f"   • Original CV2 shape: {original_cv.shape}")
            print(f"   • Temp CV2 shape: {temp_cv.shape}")
            
            # Verificar se são idênticas
            diff = cv2.absdiff(original_cv, temp_cv)
            diff_sum = np.sum(diff)
            print(f"   • Diferença total: {diff_sum}")
            
            if diff_sum == 0:
                print("   ✅ Imagens são idênticas")
            else:
                print("   ⚠️ Imagens são diferentes!")
        
        # 4. Analisar com imagem original
        print("\n4️⃣ Analisando imagem original...")
        original_result = engine.analyze_image_intuition(original_image_path)
        original_is_bird = original_result.get('is_bird', False)
        original_confidence = original_result.get('confidence', 0.0)
        original_species = original_result.get('species', 'Desconhecida')
        
        print(f"   • Resultado: {'Pássaro' if original_is_bird else 'Não-Pássaro'}")
        print(f"   • Confiança: {original_confidence:.2f}")
        print(f"   • Espécie: {original_species}")
        
        # 5. Analisar com imagem convertida (como a interface web faz)
        print("\n5️⃣ Analisando imagem convertida (como interface web)...")
        temp_result = engine.analyze_image_intuition(temp_path)
        temp_is_bird = temp_result.get('is_bird', False)
        temp_confidence = temp_result.get('confidence', 0.0)
        temp_species = temp_result.get('species', 'Desconhecida')
        
        print(f"   • Resultado: {'Pássaro' if temp_is_bird else 'Não-Pássaro'}")
        print(f"   • Confiança: {temp_confidence:.2f}")
        print(f"   • Espécie: {temp_species}")
        
        # 6. Comparar resultados
        print("\n6️⃣ Comparação dos resultados:")
        if original_is_bird == temp_is_bird:
            print("   ✅ Resultados são idênticos")
        else:
            print("   ❌ Resultados são diferentes!")
            print(f"   • Original: {'Pássaro' if original_is_bird else 'Não-Pássaro'}")
            print(f"   • Convertido: {'Pássaro' if temp_is_bird else 'Não-Pássaro'}")
        
        # 7. Mostrar características detectadas em ambos os casos
        print("\n7️⃣ Características detectadas:")
        
        original_chars = original_result.get('characteristics_found', [])
        temp_chars = temp_result.get('characteristics_found', [])
        
        print(f"   • Original: {original_chars}")
        print(f"   • Convertido: {temp_chars}")
        
        # 8. Mostrar raciocínio em ambos os casos
        print("\n8️⃣ Raciocínio:")
        
        original_reasoning = original_result.get('reasoning_steps', [])
        temp_reasoning = temp_result.get('reasoning_steps', [])
        
        print("   • Original:")
        for step in original_reasoning:
            print(f"     - {step}")
        
        print("   • Convertido:")
        for step in temp_reasoning:
            print(f"     - {step}")
        
        # Limpar arquivo temporário
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"\n🧹 Arquivo temporário removido: {temp_path}")
        
    except Exception as e:
        print(f"❌ Erro na simulação: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simulate_web_processing()