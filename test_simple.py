#!/usr/bin/env python3
"""
Teste simples usando modelo YOLO pré-treinado
"""

from ultralytics import YOLO
import cv2
import os

def test_simple_detection():
    """Testa detecção usando modelo YOLO pré-treinado"""
    
    print("🔍 Testando detecção YOLO (modelo pré-treinado)...")
    
    # Usar modelo pré-treinado (mais estável)
    model = YOLO('yolov8n.pt')
    
    print(f"✅ Modelo carregado: yolov8n.pt")
    print(f"📋 Classes detectáveis: {len(model.names)} classes")
    print()
    
    # Testar com imagens do dataset_teste
    test_dir = './dataset_teste'
    
    if not os.path.exists(test_dir):
        print(f"❌ Diretório {test_dir} não encontrado")
        return
    
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(test_dir, filename)
            
            print(f"🖼️  Analisando: {filename}")
            
            # Carregar imagem
            image = cv2.imread(image_path)
            if image is None:
                print(f"   ❌ Erro ao carregar imagem")
                continue
            
            # Detectar objetos
            results = model(image, verbose=False)
            
            detected_objects = []
            for r in results:
                for box in r.boxes:
                    if box.conf > 0.5:  # Limiar de confiança
                        class_id = int(box.cls)
                        class_name = model.names[class_id]
                        confidence = float(box.conf)
                        detected_objects.append(f"{class_name} ({confidence:.2f})")
            
            if detected_objects:
                print(f"   ✅ Objetos detectados: {', '.join(detected_objects)}")
                
                # Verificar se há pássaros (bird)
                has_bird = any('bird' in obj.lower() for obj in detected_objects)
                if has_bird:
                    print(f"   🐦 CONCLUSÃO: Pássaro detectado!")
                else:
                    print(f"   ❓ CONCLUSÃO: Nenhum pássaro detectado")
            else:
                print(f"   ❌ Nenhum objeto detectado com confiança suficiente")
            
            print()

if __name__ == "__main__":
    test_simple_detection()
