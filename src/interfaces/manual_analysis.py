#!/usr/bin/env python3
"""
Sistema de Análise Manual para Imagens Não Reconhecidas
Funciona independentemente das APIs externas (ChatGPT/Gemini)
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Any
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

class ManualAnalysisSystem:
    """Sistema para análise manual de imagens não reconhecidas"""
    
    def __init__(self, base_dir: str = "./manual_analysis"):
        self.base_dir = base_dir
        self.pending_dir = os.path.join(base_dir, "pending")
        self.approved_dir = os.path.join(base_dir, "approved")
        self.rejected_dir = os.path.join(base_dir, "rejected")
        self.annotations_dir = os.path.join(base_dir, "annotations")
        
        # Criar diretórios se não existirem
        for directory in [self.pending_dir, self.approved_dir, self.rejected_dir, self.annotations_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analisa uma imagem e retorna dados básicos"""
        try:
            if not os.path.exists(image_path):
                return {"error": f"Arquivo não encontrado: {image_path}"}
            
            # Carregar imagem
            image = Image.open(image_path)
            image_array = np.array(image)
            
            # Análise básica
            analysis = {
                "image_path": image_path,
                "filename": os.path.basename(image_path),
                "size": image.size,
                "mode": image.mode,
                "array_shape": image_array.shape,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            return {
                "error": f"Erro na análise: {str(e)}",
                "image_path": image_path,
                "status": "error"
            }
    
    def add_image_for_analysis(self, image_path: str, detection_data: Dict[str, Any]) -> str:
        """Adiciona uma imagem para análise manual"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"manual_{timestamp}_{os.path.basename(image_path)}"
        
        # Debug: verificar imagem original
        if os.path.exists(image_path):
            from PIL import Image
            import numpy as np
            
            original_image = Image.open(image_path)
            original_array = np.array(original_image)
        else:
            raise FileNotFoundError(f"Arquivo original não existe: {image_path}")
        
        # Copiar imagem para pending
        pending_path = os.path.join(self.pending_dir, filename)
        shutil.copy2(image_path, pending_path)
        
        # Debug: verificar imagem copiada
        if os.path.exists(pending_path):
            copied_image = Image.open(pending_path)
            copied_array = np.array(copied_image)
        else:
            raise FileNotFoundError(f"Arquivo copiado não foi criado: {pending_path}")
        
        # Salvar dados de detecção
        detection_file = os.path.join(self.pending_dir, f"{filename}.json")
        with open(detection_file, 'w', encoding='utf-8') as f:
            json.dump(detection_data, f, indent=2, ensure_ascii=False)
        
        return pending_path
    
    def get_pending_images(self) -> List[Dict[str, Any]]:
        """Retorna lista de imagens pendentes de análise"""
        pending_images = []
        
        for filename in os.listdir(self.pending_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(self.pending_dir, filename)
                json_path = os.path.join(self.pending_dir, f"{filename}.json")
                
                detection_data = {}
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        detection_data = json.load(f)
                
                # Extrair timestamp do filename de forma segura
                try:
                    if '_' in filename and len(filename.split('_')) >= 3:
                        timestamp = filename.split('_')[1] + '_' + filename.split('_')[2].split('.')[0]
                    else:
                        # Usar timestamp atual se não conseguir extrair
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                except:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                pending_images.append({
                    'filename': filename,
                    'image_path': image_path,
                    'detection_data': detection_data,
                    'timestamp': timestamp
                })
        
        return sorted(pending_images, key=lambda x: x['timestamp'], reverse=True)
    
    def approve_image(self, filename: str, species: str, confidence: float, notes: str = "", 
                     decision_reason: str = "", visual_characteristics: List[str] = None, 
                     additional_observations: str = ""):
        """Aprova uma imagem para treinamento com feedback detalhado"""
        
        # Mover arquivos para approved
        image_path = os.path.join(self.pending_dir, filename)
        json_path = os.path.join(self.pending_dir, f"{filename}.json")
        
        approved_image_path = os.path.join(self.approved_dir, filename)
        approved_json_path = os.path.join(self.approved_dir, f"{filename}.json")
        
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Arquivo de imagem não encontrado: {image_path}")
        
        # Mover imagem
        shutil.move(image_path, approved_image_path)
        
        # Mover JSON se existir
        if os.path.exists(json_path):
            shutil.move(json_path, approved_json_path)
        
        # Criar anotação YOLO
        try:
            annotation_path = self.create_yolo_annotation(approved_image_path, species, confidence, notes)
        except Exception as e:
            pass  # Anotação YOLO é opcional
        
        # Criar arquivo de feedback detalhado
        feedback_data = {
            "species": species,
            "confidence": confidence,
            "decision_reason": decision_reason,
            "visual_characteristics": visual_characteristics or [],
            "additional_observations": additional_observations,
            "notes": notes,
            "timestamp": datetime.now().isoformat(),
            "approval_type": "manual_with_feedback"
        }
        
        feedback_path = os.path.join(self.approved_dir, f"{filename}.feedback.json")
        with open(feedback_path, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, indent=2, ensure_ascii=False)
        
        # Adicionar ao cache de reconhecimento
        try:
            from .cache import image_cache
            
            # Preparar dados de análise para o cache
            analysis_data = {
                "yolo_analysis": {"detections": [], "status": "manual_approval"},
                "keras_analysis": {"species": species, "confidence": confidence},
                "manual_approval": True,
                "annotation_path": annotation_path,
                "feedback_data": feedback_data
            }
            
            # Adicionar ao cache
            image_cache.add_recognized_image(
                approved_image_path, 
                species, 
                confidence, 
                analysis_data, 
                notes
            )
        except Exception as e:
            pass  # Cache é opcional
        return approved_image_path
    
    def reject_image(self, filename: str, reason: str = "", decision_reason: str = "", 
                    visual_characteristics: List[str] = None, additional_observations: str = ""):
        """Rejeita uma imagem com feedback detalhado"""
        # Mover arquivos para rejected
        image_path = os.path.join(self.pending_dir, filename)
        json_path = os.path.join(self.pending_dir, f"{filename}.json")
        
        rejected_image_path = os.path.join(self.rejected_dir, filename)
        rejected_json_path = os.path.join(self.rejected_dir, f"{filename}.json")
        
        shutil.move(image_path, rejected_image_path)
        if os.path.exists(json_path):
            shutil.move(json_path, rejected_json_path)
        
        # Salvar motivo da rejeição com feedback detalhado
        rejection_data = {
            'filename': filename,
            'reason': reason,
            'decision_reason': decision_reason,
            'visual_characteristics': visual_characteristics or [],
            'additional_observations': additional_observations,
            'timestamp': datetime.now().isoformat(),
            'rejection_type': 'manual_with_feedback'
        }
        
        rejection_file = os.path.join(self.rejected_dir, f"{filename}_rejection.json")
        with open(rejection_file, 'w', encoding='utf-8') as f:
            json.dump(rejection_data, f, indent=2, ensure_ascii=False)
        
        return rejected_image_path
    
    def create_yolo_annotation(self, image_path: str, species: str, confidence: float, notes: str = ""):
        """Cria anotação YOLO para a imagem aprovada"""
        # Ler imagem para obter dimensões
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        # Criar bounding box que cobre toda a imagem (assumindo que é um pássaro)
        # Formato YOLO: class_id center_x center_y width height (normalizado)
        center_x = 0.5
        center_y = 0.5
        bbox_width = 0.8
        bbox_height = 0.8
        
        # Mapear espécie para class_id
        species_mapping = {
            'bem_te_vi': 0,
            'sabia': 1,
            'beija_flor': 2,
            'cardinal': 3,
            'painted_bunting': 4,
            'brown_pelican': 5,
            'generic_bird': 6
        }
        
        class_id = species_mapping.get(species.lower().replace(' ', '_'), 6)  # Default para generic_bird
        
        # Criar arquivo de anotação
        annotation_filename = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
        annotation_path = os.path.join(self.annotations_dir, annotation_filename)
        
        with open(annotation_path, 'w') as f:
            f.write(f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}\n")
        
        # Salvar metadados
        metadata = {
            'species': species,
            'confidence': confidence,
            'notes': notes,
            'annotation_path': annotation_path,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(self.annotations_dir, f"{annotation_filename}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return annotation_path
    
    def get_approved_images(self) -> List[Dict[str, Any]]:
        """Retorna lista de imagens aprovadas"""
        approved_images = []
        
        for filename in os.listdir(self.approved_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(self.approved_dir, filename)
                json_path = os.path.join(self.approved_dir, f"{filename}.json")
                
                detection_data = {}
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        detection_data = json.load(f)
                
                approved_images.append({
                    'filename': filename,
                    'image_path': image_path,
                    'detection_data': detection_data
                })
        
        return approved_images
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do sistema de análise manual"""
        pending_count = len([f for f in os.listdir(self.pending_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        approved_count = len([f for f in os.listdir(self.approved_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        rejected_count = len([f for f in os.listdir(self.rejected_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        annotation_count = len([f for f in os.listdir(self.annotations_dir) if f.endswith('.txt')])
        
        return {
            'pending': pending_count,
            'approved': approved_count,
            'rejected': rejected_count,
            'annotations': annotation_count,
            'total_processed': approved_count + rejected_count
        }

# Instância global do sistema
manual_analysis = ManualAnalysisSystem()
