#!/usr/bin/env python3
"""
Teste específico para distinguir entre cachorros e pássaros
"""

import sys
import os
import numpy as np
from PIL import Image

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_dog_test_image():
    """Criar uma imagem de teste simulando um cachorro"""
    # Criar imagem 224x224 com características de cachorro
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Fundo verde (grama)
    image[:, :] = [50, 150, 50]  # Verde grama
    
    # Corpo do cachorro (forma alongada horizontal)
    center_x, center_y = 112, 140
    for y in range(224):
        for x in range(224):
            # Corpo principal (marrom)
            if ((x - center_x)**2 / 60**2 + (y - center_y)**2 / 30**2) <= 1:
                image[y, x] = [120, 80, 40]  # Marrom cachorro
            
            # Cabeça (maior, mais redonda)
            if ((x - center_x)**2 / 35**2 + (y - (center_y - 40))**2 / 35**2) <= 1:
                image[y, x] = [140, 100, 60]  # Cabeça marrom claro
            
            # Patas (formas alongadas verticais)
            if ((x - (center_x - 30))**2 / 8**2 + (y - (center_y + 20))**2 / 25**2) <= 1:
                image[y, x] = [100, 60, 30]  # Pata esquerda
            if ((x - (center_x + 30))**2 / 8**2 + (y - (center_y + 20))**2 / 25**2) <= 1:
                image[y, x] = [100, 60, 30]  # Pata direita
    
    return image

def create_bird_test_image():
    """Criar uma imagem de teste simulando um pássaro"""
    # Criar imagem 224x224 com características de pássaro
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Fundo terroso/marrom claro
    image[:, :] = [180, 150, 120]  # Marrom claro
    
    # Corpo do pássaro (forma oval vertical)
    center_x, center_y = 112, 120
    for y in range(224):
        for x in range(224):
            # Corpo principal (marrom-avermelhado)
            if ((x - center_x)**2 / 25**2 + (y - center_y)**2 / 50**2) <= 1:
                image[y, x] = [150, 80, 60]  # Marrom-avermelhado
            
            # Cabeça (menor, mais clara)
            if ((x - center_x)**2 / 20**2 + (y - (center_y - 30))**2 / 20**2) <= 1:
                image[y, x] = [200, 180, 160]  # Cabeça clara
            
            # Asas com manchas escuras
            if ((x - (center_x - 15))**2 / 20**2 + (y - center_y)**2 / 30**2) <= 1:
                if (x + y) % 6 < 2:  # Padrão de manchas
                    image[y, x] = [80, 40, 30]  # Manchas escuras
    
    return image

def test_dog_vs_bird_detection():
    """Testar detecção de cachorro vs pássaro"""
    print("🔍 Testando distinção cachorro vs pássaro...")
    
    try:
        from core.intuition import IntuitionEngine
        
        # Inicializar sistema
        intuition_engine = IntuitionEngine("yolov8n.pt", "modelo_classificacao_passaros.keras")
        
        # Teste 1: Cachorro
        print("\n🐕 Testando imagem de cachorro...")
        dog_image = create_dog_test_image()
        pil_dog = Image.fromarray(dog_image)
        dog_path = "test_dog.jpg"
        pil_dog.save(dog_path)
        
        dog_result = intuition_engine.analyze_image_intuition(dog_path)
        dog_intuition = dog_result.get('intuition_analysis', {})
        dog_visual = dog_intuition.get('visual_analysis', {})
        
        print(f"  📊 Análise do cachorro:")
        print(f"    - Score geral: {dog_visual.get('bird_like_features', 0):.2%}")
        print(f"    - Cores de pássaro: {dog_visual.get('bird_colors', False)}")
        print(f"    - Proporções de pássaro: {dog_visual.get('bird_proportions', False)}")
        print(f"    - Textura de pássaro: {dog_visual.get('bird_texture', False)}")
        print(f"    - Contornos de pássaro: {dog_visual.get('bird_contours', False)}")
        print(f"    - Candidatos encontrados: {dog_intuition.get('candidates_found', 0)}")
        
        # Teste 2: Pássaro
        print("\n🐦 Testando imagem de pássaro...")
        bird_image = create_bird_test_image()
        pil_bird = Image.fromarray(bird_image)
        bird_path = "test_bird.jpg"
        pil_bird.save(bird_path)
        
        bird_result = intuition_engine.analyze_image_intuition(bird_path)
        bird_intuition = bird_result.get('intuition_analysis', {})
        bird_visual = bird_intuition.get('visual_analysis', {})
        
        print(f"  📊 Análise do pássaro:")
        print(f"    - Score geral: {bird_visual.get('bird_like_features', 0):.2%}")
        print(f"    - Cores de pássaro: {bird_visual.get('bird_colors', False)}")
        print(f"    - Proporções de pássaro: {bird_visual.get('bird_proportions', False)}")
        print(f"    - Textura de pássaro: {bird_visual.get('bird_texture', False)}")
        print(f"    - Contornos de pássaro: {bird_visual.get('bird_contours', False)}")
        print(f"    - Candidatos encontrados: {bird_intuition.get('candidates_found', 0)}")
        
        # Limpar arquivos temporários
        for path in [dog_path, bird_path]:
            if os.path.exists(path):
                os.remove(path)
        
        # Avaliar resultados
        dog_score = dog_visual.get('bird_like_features', 0)
        bird_score = bird_visual.get('bird_like_features', 0)
        dog_candidates = dog_intuition.get('candidates_found', 0)
        bird_candidates = bird_intuition.get('candidates_found', 0)
        
        print(f"\n📈 Resultados:")
        print(f"  🐕 Cachorro: {dog_score:.2%} score, {dog_candidates} candidatos (não deve ser candidato)")
        print(f"  🐦 Pássaro: {bird_score:.2%} score, {bird_candidates} candidatos (deve ser candidato)")
        
        # Verificar se distinguiu corretamente
        # O importante é que o cachorro não seja detectado como candidato
        
        dog_correct = dog_candidates == 0  # Cachorro não deve ser candidato
        bird_correct = bird_candidates > 0  # Pássaro deve ser candidato
        
        if dog_correct and bird_correct:
            print("  ✅ SUCESSO: Sistema distinguiu corretamente!")
            return True
        else:
            print("  ❌ FALHA: Sistema não distinguiu corretamente")
            if not dog_correct:
                print("    - Cachorro foi classificado como pássaro")
            if not bird_correct:
                print("    - Pássaro não foi detectado como pássaro")
            return False
        
    except Exception as e:
        print(f"  ❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Função principal"""
    print("🧪 TESTE DE DISTINÇÃO CACHORRO VS PÁSSARO")
    print("=" * 50)
    
    success = test_dog_vs_bird_detection()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 TESTE PASSOU! Sistema distingue corretamente.")
    else:
        print("⚠️ TESTE FALHOU! Melhorias necessárias.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
