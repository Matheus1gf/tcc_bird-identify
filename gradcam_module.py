import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)

class GradCAM:
    """
    Implementação de Grad-CAM para gerar mapas de calor e propostas de anotação.
    Usado no módulo de aprendizado contínuo para auto-anotação.
    """
    
    def __init__(self, model: Model, layer_name: str = None):
        """
        Inicializa o Grad-CAM
        
        Args:
            model: Modelo Keras treinado
            layer_name: Nome da camada convolucional para análise (se None, usa a última)
        """
        self.model = model
        self.layer_name = layer_name
        
        # Encontrar a camada alvo
        if layer_name is None:
            # Usar a última camada convolucional
            for layer in reversed(self.model.layers):
                if len(layer.output_shape) == 4:  # Camada convolucional
                    self.layer_name = layer.name
                    break
        
        if self.layer_name is None:
            raise ValueError("Não foi possível encontrar uma camada convolucional adequada")
        
        logging.info(f"Usando camada '{self.layer_name}' para Grad-CAM")
        
        # Criar modelo para Grad-CAM
        self.grad_model = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )
    
    def compute_gradcam(self, image: np.ndarray, class_idx: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computa o mapa Grad-CAM para uma imagem
        
        Args:
            image: Imagem de entrada (H, W, C)
            class_idx: Índice da classe (se None, usa a classe com maior predição)
            
        Returns:
            Tuple contendo (gradcam, predicao)
        """
        # Preparar imagem
        if len(image.shape) == 3:
            image_batch = np.expand_dims(image, axis=0)
        else:
            image_batch = image
        
        # Computar gradientes
        with tf.GradientTape() as tape:
            inputs = tf.cast(image_batch, tf.float32)
            tape.watch(inputs)
            
            # Forward pass
            conv_outputs, predictions = self.grad_model(inputs)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Gradiente da classe específica
            class_output = predictions[:, class_idx]
        
        # Computar gradientes
        grads = tape.gradient(class_output, conv_outputs)
        
        # Pooling global dos gradientes
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiplicar gradientes pelas ativações da camada
        conv_outputs = conv_outputs[0]
        gradcam = conv_outputs @ pooled_grads[..., tf.newaxis]
        gradcam = tf.squeeze(gradcam)
        
        # Normalizar para [0, 1]
        gradcam = tf.maximum(gradcam, 0)
        gradcam = gradcam / tf.reduce_max(gradcam)
        
        return gradcam.numpy(), predictions[0].numpy()
    
    def generate_heatmap(self, image: np.ndarray, class_idx: int = None, 
                        alpha: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera mapa de calor sobreposto na imagem original
        
        Args:
            image: Imagem original
            class_idx: Índice da classe
            alpha: Transparência do mapa de calor
            
        Returns:
            Tuple contendo (imagem_com_heatmap, gradcam)
        """
        gradcam, predictions = self.compute_gradcam(image, class_idx)
        
        # Redimensionar Grad-CAM para o tamanho da imagem original
        gradcam_resized = cv2.resize(gradcam, (image.shape[1], image.shape[0]))
        
        # Converter para colormap
        heatmap = cm.get_cmap('jet')(gradcam_resized)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Sobrepor na imagem original
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        overlaid = cv2.addWeighted(image_rgb, 1-alpha, heatmap, alpha, 0)
        
        return overlaid, gradcam_resized
    
    def propose_annotations(self, image: np.ndarray, class_idx: int = None,
                          threshold: float = 0.5) -> List[Dict]:
        """
        Propõe anotações baseadas no mapa Grad-CAM
        
        Args:
            image: Imagem de entrada
            class_idx: Índice da classe
            threshold: Limiar para considerar região relevante
            
        Returns:
            Lista de propostas de anotação
        """
        gradcam, predictions = self.compute_gradcam(image, class_idx)
        
        # Redimensionar para tamanho original
        gradcam_resized = cv2.resize(gradcam, (image.shape[1], image.shape[0]))
        
        # Aplicar threshold
        binary_mask = (gradcam_resized > threshold).astype(np.uint8)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        proposals = []
        for i, contour in enumerate(contours):
            # Calcular bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filtrar contornos muito pequenos
            if w < 20 or h < 20:
                continue
            
            # Calcular confiança baseada na intensidade do Grad-CAM
            roi_gradcam = gradcam_resized[y:y+h, x:x+w]
            confidence = np.mean(roi_gradcam)
            
            proposal = {
                "id": i,
                "bbox": [x, y, w, h],
                "confidence": float(confidence),
                "class_idx": int(class_idx) if class_idx is not None else int(np.argmax(predictions)),
                "class_confidence": float(predictions[class_idx] if class_idx is not None else np.max(predictions)),
                "area": int(cv2.contourArea(contour))
            }
            
            proposals.append(proposal)
        
        # Ordenar por confiança
        proposals.sort(key=lambda x: x["confidence"], reverse=True)
        
        return proposals

class AutoAnnotationSystem:
    """
    Sistema de auto-anotação que combina Grad-CAM com validação externa
    """
    
    def __init__(self, classification_model: Model, yolo_model, 
                 confidence_threshold: float = 0.6):
        """
        Inicializa o sistema de auto-anotação
        
        Args:
            classification_model: Modelo de classificação (MobileNetV2)
            yolo_model: Modelo YOLO para detecção de partes
            confidence_threshold: Limiar de confiança para aceitar anotações
        """
        self.classification_model = classification_model
        self.yolo_model = yolo_model
        self.confidence_threshold = confidence_threshold
        
        # Inicializar Grad-CAM
        self.gradcam = GradCAM(classification_model)
        
        # Mapeamento de classes (ajustar conforme seu dataset)
        self.class_names = [
            "Brown_Pelican", "Cardinal", "Painted_Bunting"
            # Adicionar outras classes conforme necessário
        ]
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analisa uma imagem e propõe anotações automáticas
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            Dicionário com análise completa
        """
        # Carregar imagem
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
        
        # Preparar imagem para classificação
        image_resized = cv2.resize(image, (224, 224))
        image_normalized = image_resized / 255.0
        
        # 1. Detecção YOLO (partes anatômicas)
        yolo_results = self.yolo_model(image, verbose=False)
        yolo_detections = []
        
        for r in yolo_results:
            for box in r.boxes:
                if box.conf > 0.5:  # Limiar de confiança YOLO
                    detection = {
                        "class": self.yolo_model.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist()
                    }
                    yolo_detections.append(detection)
        
        # 2. Classificação de espécie
        prediction = self.classification_model.predict(
            np.expand_dims(image_normalized, axis=0), verbose=0
        )
        class_idx = np.argmax(prediction[0])
        class_confidence = np.max(prediction[0])
        predicted_species = self.class_names[class_idx]
        
        # 3. Grad-CAM para regiões de interesse
        gradcam_proposals = self.gradcam.propose_annotations(
            image_normalized, class_idx, threshold=0.3
        )
        
        # 4. Análise combinada
        analysis = {
            "image_path": image_path,
            "yolo_detections": yolo_detections,
            "species_prediction": {
                "species": predicted_species,
                "confidence": float(class_confidence),
                "class_idx": int(class_idx)
            },
            "gradcam_proposals": gradcam_proposals,
            "needs_human_validation": class_confidence < self.confidence_threshold,
            "recommended_actions": []
        }
        
        # Determinar ações recomendadas
        if len(yolo_detections) == 0:
            analysis["recommended_actions"].append("YOLO não detectou partes - usar Grad-CAM")
        
        if class_confidence < self.confidence_threshold:
            analysis["recommended_actions"].append("Baixa confiança - validar com humano")
        
        if len(gradcam_proposals) > 0:
            analysis["recommended_actions"].append("Grad-CAM propôs anotações - revisar")
        
        return analysis
    
    def generate_annotation_suggestions(self, analysis: Dict) -> List[Dict]:
        """
        Gera sugestões de anotação baseadas na análise
        
        Args:
            analysis: Resultado da análise da imagem
            
        Returns:
            Lista de sugestões de anotação
        """
        suggestions = []
        
        # Sugestões baseadas em YOLO
        for detection in analysis["yolo_detections"]:
            suggestion = {
                "type": "yolo_detection",
                "class": detection["class"],
                "bbox": detection["bbox"],
                "confidence": detection["confidence"],
                "source": "YOLO",
                "action": "validate_detection"
            }
            suggestions.append(suggestion)
        
        # Sugestões baseadas em Grad-CAM
        for proposal in analysis["gradcam_proposals"]:
            if proposal["confidence"] > 0.4:  # Limiar para Grad-CAM
                suggestion = {
                    "type": "gradcam_proposal",
                    "class": analysis["species_prediction"]["species"],
                    "bbox": proposal["bbox"],
                    "confidence": proposal["confidence"],
                    "source": "Grad-CAM",
                    "action": "review_proposal"
                }
                suggestions.append(suggestion)
        
        return suggestions
    
    def save_analysis_report(self, analysis: Dict, output_path: str) -> None:
        """
        Salva relatório de análise em arquivo
        
        Args:
            analysis: Análise da imagem
            output_path: Caminho para salvar o relatório
        """
        import json
        
        # Preparar dados para serialização
        report_data = {
            "timestamp": str(np.datetime64('now')),
            "image_path": analysis["image_path"],
            "species_prediction": analysis["species_prediction"],
            "yolo_detections_count": len(analysis["yolo_detections"]),
            "gradcam_proposals_count": len(analysis["gradcam_proposals"]),
            "needs_human_validation": analysis["needs_human_validation"],
            "recommended_actions": analysis["recommended_actions"],
            "suggestions": self.generate_annotation_suggestions(analysis)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

# Exemplo de uso
if __name__ == "__main__":
    # Este exemplo seria usado após carregar os modelos treinados
    print("Módulo Grad-CAM implementado!")
    print("Para usar:")
    print("1. Carregue seus modelos treinados")
    print("2. Inicialize AutoAnnotationSystem")
    print("3. Use analyze_image() para analisar novas imagens")
    print("4. Use generate_annotation_suggestions() para propostas")
