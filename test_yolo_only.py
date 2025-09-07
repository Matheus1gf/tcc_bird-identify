#!/usr/bin/env python3
"""
Teste rápido usando apenas o modelo YOLO treinado
"""

from ultralytics import YOLO
import cv2
import os

def test_yolo_detection():
    """Testa detecção YOLO com imagens do dataset_teste"""
    
    print("🔍 Testando detecção YOLO...")
    
    # Carregar modelo YOLO treinado
    model_path = 'runs/detect/train/weights/best.pt'
    model = YOLO(model_path)
    
    print(f"✅ Modelo carregado: {model_path}")
    print(f"📋 Classes detectáveis: {list(model.names.values())}")
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
            
            # Detectar partes anatômicas
            results = model(image, verbose=False)
            
            detected_parts = []
            for r in results:
                for box in r.boxes:
                    if box.conf > 0.5:  # Limiar de confiança
                        class_id = int(box.cls)
                        class_name = model.names[class_id]
                        confidence = float(box.conf)
                        detected_parts.append(f"{class_name} ({confidence:.2f})")
            
            if detected_parts:
                print(f"   ✅ Partes detectadas: {', '.join(detected_parts)}")
                
                # Verificar se é pássaro
                is_bird = any('bico' in part or ('corpo' in part and 'asa' in part) 
                            for part in detected_parts)
                if is_bird:
                    print(f"   🐦 CONCLUSÃO: É um pássaro!")
                else:
                    print(f"   ❓ CONCLUSÃO: Não foi possível confirmar que é um pássaro")
            else:
                print(f"   ❌ Nenhuma parte detectada com confiança suficiente")
            
            print()

if __name__ == "__main__":
    test_yolo_detection()
