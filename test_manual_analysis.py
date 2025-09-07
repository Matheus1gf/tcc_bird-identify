#!/usr/bin/env python3
"""
Teste direto do sistema de análise manual
"""

import os
import cv2
import numpy as np
from manual_analysis_system import manual_analysis
from debug_logger import debug_logger

def test_manual_analysis():
    """Teste completo do sistema de análise manual"""
    
    print("=== TESTE DO SISTEMA DE ANÁLISE MANUAL ===")
    
    # 1. Criar imagem de teste
    print("1. Criando imagem de teste...")
    test_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.imwrite('test_bird.jpg', test_image)
    print("   ✓ Imagem criada: test_bird.jpg")
    
    # 2. Verificar se arquivo existe
    print("2. Verificando arquivo...")
    if os.path.exists('test_bird.jpg'):
        print("   ✓ Arquivo existe")
    else:
        print("   ✗ Arquivo não existe!")
        return
    
    # 3. Adicionar à análise manual
    print("3. Adicionando à análise manual...")
    detection_data = {
        'yolo_detections': [{'class': 'bird', 'confidence': 0.8684}],
        'confidence': 0.8684,
        'analysis_type': 'generic_bird',
        'timestamp': '2025-09-07T15:00:00'
    }
    
    try:
        pending_path = manual_analysis.add_image_for_analysis('test_bird.jpg', detection_data)
        print(f"   ✓ Imagem adicionada: {pending_path}")
    except Exception as e:
        print(f"   ✗ Erro ao adicionar: {e}")
        return
    
    # 4. Verificar se foi criada
    print("4. Verificando se foi criada...")
    if os.path.exists(pending_path):
        print("   ✓ Arquivo copiado com sucesso!")
    else:
        print("   ✗ Arquivo não foi copiado!")
        return
    
    # 5. Verificar pasta pending
    print("5. Verificando pasta pending...")
    pending_files = os.listdir('./manual_analysis/pending')
    print(f"   Arquivos na pasta pending: {len(pending_files)}")
    for file in pending_files:
        print(f"     - {file}")
    
    # 6. Testar get_pending_images
    print("6. Testando get_pending_images...")
    pending_images = manual_analysis.get_pending_images()
    print(f"   Imagens pendentes encontradas: {len(pending_images)}")
    
    for i, img_data in enumerate(pending_images):
        print(f"     Imagem {i+1}:")
        print(f"       - Filename: {img_data['filename']}")
        print(f"       - Path: {img_data['image_path']}")
        print(f"       - Exists: {os.path.exists(img_data['image_path'])}")
        print(f"       - Detection data: {img_data['detection_data']}")
    
    # 7. Limpeza
    print("7. Limpando arquivos de teste...")
    if os.path.exists('test_bird.jpg'):
        os.remove('test_bird.jpg')
        print("   ✓ Arquivo de teste removido")
    
    print("=== TESTE CONCLUÍDO ===")

if __name__ == "__main__":
    test_manual_analysis()
