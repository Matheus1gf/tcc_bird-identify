#!/usr/bin/env python3
"""
Anotador Automático - A "Invenção" da Anotação
Usa Grad-CAM para gerar anotações automáticas quando a IA tem intuição
mas não consegue detectar partes específicas.
"""

import cv2
import numpy as np
# Mock imports para evitar erros
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
except ImportError:
    class MockTF:
        def keras(self): return self
        def models(self): return self
        def load_model(self, path): return None
        def optimizers(self): return self
        def legacy(self): return self
        def Adam(self, *args, **kwargs): return None
    tf = MockTF()
    Model = object
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, List, Tuple, Optional
import logging
import os
import json
from dataclasses import dataclass
from datetime import datetime

from .intuition import LearningCandidate, LearningCandidateType

logging.basicConfig(level=logging.INFO)

@dataclass
class AutoAnnotation:
    """Anotação gerada automaticamente"""
    image_path: str
    annotation_file_path: str
    bbox: List[float]  # [x_center, y_center, width, height] (normalizado)
    class_name: str
    confidence: float
    grad_cam_strength: float
    generation_method: str
    timestamp: str = ""

class GradCAMAnnotator:
    """
    Anotador automático usando Grad-CAM para gerar bounding boxes
    """
    
    def __init__(self, keras_model_path: str, target_layer_name: str = None):
        """
        Inicializa o anotador automático
        
        Args:
            keras_model_path: Caminho para modelo Keras
            target_layer_name: Nome da camada para Grad-CAM
        """
        self.keras_model_path = keras_model_path
        self.target_layer_name = target_layer_name
        
        # Carregar modelo
        self._load_model()
        
        # Configurações
        self.grad_cam_threshold = 0.3  # Limiar para considerar Grad-CAM válido
        self.min_bbox_size = 0.05     # Tamanho mínimo do bounding box (5% da imagem)
        self.max_bbox_size = 0.95     # Tamanho máximo do bounding box (95% da imagem)
        
        # Histórico de anotações
        self.generated_annotations = []
    
    def _load_model(self):
        """Carrega modelo Keras e prepara para Grad-CAM"""
        try:
            logging.info("Carregando modelo Keras para Grad-CAM...")
            
            # Verificar se o arquivo existe
            if not os.path.exists(self.keras_model_path):
                logging.warning(f"⚠️ Modelo Keras não encontrado: {self.keras_model_path}")
                logging.info("📝 Grad-CAM não estará disponível até que o modelo Keras seja treinado")
                self.model = None
                return
            
            # Verificar se TensorFlow está funcionando
            if not hasattr(tf, 'keras'):
                logging.error("❌ TensorFlow não tem atributo 'keras'")
                self.model = None
                return
            
            # Verificar se tf.keras.models existe
            if not hasattr(tf.keras, 'models'):
                logging.error("❌ TensorFlow keras.models não disponível")
                self.model = None
                return
            
            self.model = tf.keras.models.load_model(
                self.keras_model_path,
                custom_objects={
                    'Adam': tf.keras.optimizers.legacy.Adam
                }
            )
            
            # Encontrar camada alvo para Grad-CAM
            if self.target_layer_name is None:
                # Usar a última camada convolucional
                for layer in reversed(self.model.layers):
                    if len(layer.output_shape) == 4:  # Camada convolucional
                        self.target_layer_name = layer.name
                        break
            
            if self.target_layer_name is None:
                raise ValueError("Não foi possível encontrar camada convolucional adequada")
            
            logging.info(f"✅ Modelo carregado. Camada Grad-CAM: {self.target_layer_name}")
            
            # Criar modelo para Grad-CAM
            self.grad_model = Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(self.target_layer_name).output, self.model.output]
            )
            
        except Exception as e:
            logging.error(f"❌ Erro ao carregar modelo: {e}")
            self.model = None
            self.grad_model = None
    
    def generate_auto_annotation(self, candidate: LearningCandidate) -> Optional[AutoAnnotation]:
        """
        Gera anotação automática para um candidato de aprendizado
        
        Args:
            candidate: Candidato identificado pelo módulo de intuição
            
        Returns:
            Anotação gerada ou None se não foi possível gerar
        """
        if self.model is None:
            logging.error("Modelo não disponível para geração de anotação")
            return None
        
        try:
            # Carregar imagem
            image = cv2.imread(candidate.image_path)
            if image is None:
                logging.error(f"Não foi possível carregar imagem: {candidate.image_path}")
                return None
            
            # Gerar Grad-CAM
            grad_cam, prediction = self._compute_gradcam(image, candidate.keras_prediction)
            
            # Verificar força do Grad-CAM
            grad_cam_strength = np.max(grad_cam)
            if grad_cam_strength < self.grad_cam_threshold:
                logging.warning(f"Grad-CAM muito fraco ({grad_cam_strength:.3f}). Não gerando anotação.")
                return None
            
            # Gerar bounding box a partir do Grad-CAM
            bbox = self._gradcam_to_bbox(grad_cam, image.shape)
            
            # Validar bounding box
            if not self._validate_bbox(bbox, image.shape):
                logging.warning("Bounding box inválido gerado. Não criando anotação.")
                return None
            
            # Determinar classe da anotação
            class_name = self._determine_annotation_class(candidate)
            
            # Criar arquivo de anotação
            annotation_file_path = self._create_annotation_file(
                candidate.image_path, bbox, class_name, candidate.keras_confidence
            )
            
            # Criar objeto de anotação
            annotation = AutoAnnotation(
                image_path=candidate.image_path,
                annotation_file_path=annotation_file_path,
                bbox=bbox,
                class_name=class_name,
                confidence=candidate.keras_confidence,
                grad_cam_strength=grad_cam_strength,
                generation_method="grad_cam_auto",
                timestamp=datetime.now().isoformat()
            )
            
            # Adicionar ao histórico
            self.generated_annotations.append(annotation)
            
            logging.info(f"✅ Anotação gerada: {annotation_file_path}")
            return annotation
            
        except Exception as e:
            logging.error(f"Erro ao gerar anotação automática: {e}")
            return None
    
    def _compute_gradcam(self, image: np.ndarray, target_class: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computa Grad-CAM para a imagem
        
        Args:
            image: Imagem de entrada
            target_class: Classe alvo para Grad-CAM
            
        Returns:
            Tuple (gradcam, prediction)
        """
        # Preparar imagem
        img_resized = cv2.resize(image, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0) / 255.0
        
        # Computar gradientes
        with tf.GradientTape() as tape:
            inputs = tf.cast(img_array, tf.float32)
            tape.watch(inputs)
            
            # Forward pass
            conv_outputs, predictions = self.grad_model(inputs)
            
            # Usar a classe com maior predição
            class_output = tf.reduce_max(predictions)
        
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
    
    def _gradcam_to_bbox(self, gradcam: np.ndarray, image_shape: Tuple[int, int, int]) -> List[float]:
        """
        Converte mapa Grad-CAM em bounding box
        
        Args:
            gradcam: Mapa de calor Grad-CAM
            image_shape: Formato da imagem original (height, width, channels)
            
        Returns:
            Bounding box normalizado [x_center, y_center, width, height]
        """
        # Redimensionar Grad-CAM para tamanho da imagem original
        gradcam_resized = cv2.resize(gradcam, (image_shape[1], image_shape[0]))
        
        # Aplicar threshold para criar máscara binária
        threshold = np.percentile(gradcam_resized, 70)  # Top 30% dos pixels
        binary_mask = (gradcam_resized > threshold).astype(np.uint8)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Se não há contornos, usar toda a área ativa
            y_indices, x_indices = np.where(gradcam_resized > threshold)
            if len(x_indices) == 0:
                # Fallback: usar centro da imagem
                return [0.5, 0.5, 0.3, 0.3]
            
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
        else:
            # Usar o maior contorno
            largest_contour = max(contours, key=cv2.contourArea)
            x_min, y_min, w, h = cv2.boundingRect(largest_contour)
            x_max = x_min + w
            y_max = y_min + h
        
        # Converter para formato YOLO normalizado
        img_height, img_width = image_shape[0], image_shape[1]
        
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        return [x_center, y_center, width, height]
    
    def _validate_bbox(self, bbox: List[float], image_shape: Tuple[int, int, int]) -> bool:
        """
        Valida se o bounding box é adequado
        
        Args:
            bbox: Bounding box normalizado
            image_shape: Formato da imagem
            
        Returns:
            True se válido, False caso contrário
        """
        x_center, y_center, width, height = bbox
        
        # Verificar se está dentro dos limites
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            return False
        
        # Verificar tamanho mínimo e máximo
        if (width < self.min_bbox_size or width > self.max_bbox_size or
            height < self.min_bbox_size or height > self.max_bbox_size):
            return False
        
        # Verificar se não está muito próximo das bordas
        if (x_center - width/2 < 0.01 or x_center + width/2 > 0.99 or
            y_center - height/2 < 0.01 or y_center + height/2 > 0.99):
            return False
        
        return True
    
    def _determine_annotation_class(self, candidate: LearningCandidate) -> str:
        """
        Determina a classe para a anotação baseada no tipo de candidato
        
        Args:
            candidate: Candidato de aprendizado
            
        Returns:
            Nome da classe para anotação
        """
        if candidate.candidate_type == LearningCandidateType.YOLO_FAILED_KERAS_MEDIUM:
            return "corpo"  # Classe genérica para casos de intuição mediana
        elif candidate.candidate_type == LearningCandidateType.YOLO_FAILED_KERAS_HIGH:
            return "corpo"  # Classe genérica para casos de alta confiança
        elif candidate.candidate_type == LearningCandidateType.NEW_SPECIES_DETECTED:
            return "corpo"  # Classe genérica para novas espécies
        else:
            return "corpo"  # Default para casos de conflito
    
    def _create_annotation_file(self, image_path: str, bbox: List[float], 
                              class_name: str, confidence: float) -> str:
        """
        Cria arquivo de anotação no formato YOLO
        
        Args:
            image_path: Caminho da imagem
            bbox: Bounding box normalizado
            class_name: Nome da classe
            confidence: Confiança da predição
            
        Returns:
            Caminho do arquivo de anotação criado
        """
        # Determinar caminho do arquivo de anotação
        image_dir = os.path.dirname(image_path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        annotation_file = os.path.join(image_dir, f"{image_name}.txt")
        
        # Mapear classe para ID (ajustar conforme seu dataset)
        class_to_id = {
            "olho": 0,
            "bico": 1,
            "asa": 2,
            "garra": 3,
            "cauda": 4,
            "corpo": 5
        }
        
        class_id = class_to_id.get(class_name, 5)  # Default para "corpo"
        
        # Criar conteúdo do arquivo YOLO
        x_center, y_center, width, height = bbox
        annotation_content = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        
        # Salvar arquivo
        with open(annotation_file, 'w') as f:
            f.write(annotation_content)
        
        # Criar arquivo de metadados
        metadata_file = annotation_file.replace('.txt', '_metadata.json')
        metadata = {
            "generation_method": "grad_cam_auto",
            "confidence": confidence,
            "class_name": class_name,
            "timestamp": datetime.now().isoformat(),
            "original_image": image_path
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return annotation_file
    
    def visualize_gradcam(self, image_path: str, candidate: LearningCandidate, 
                         output_path: str = None) -> str:
        """
        Visualiza Grad-CAM para debug e validação
        
        Args:
            image_path: Caminho da imagem
            candidate: Candidato de aprendizado
            output_path: Caminho para salvar visualização
            
        Returns:
            Caminho da imagem visualizada
        """
        if self.model is None:
            return None
        
        try:
            # Carregar imagem
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Gerar Grad-CAM
            gradcam, prediction = self._compute_gradcam(image, candidate.keras_prediction)
            
            # Redimensionar Grad-CAM
            gradcam_resized = cv2.resize(gradcam, (image.shape[1], image.shape[0]))
            
            # Criar visualização
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Imagem original
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Imagem Original')
            axes[0].axis('off')
            
            # Grad-CAM
            im = axes[1].imshow(gradcam_resized, cmap='jet')
            axes[1].set_title('Grad-CAM')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
            
            # Sobreposição
            heatmap = cm.get_cmap('jet')(gradcam_resized)[:, :, :3]
            heatmap = (heatmap * 255).astype(np.uint8)
            overlaid = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 0.6, heatmap, 0.4, 0)
            axes[2].imshow(overlaid)
            axes[2].set_title('Sobreposição')
            axes[2].axis('off')
            
            # Salvar visualização
            if output_path is None:
                output_path = image_path.replace('.jpg', '_gradcam.jpg').replace('.jpeg', '_gradcam.jpg')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logging.error(f"Erro ao visualizar Grad-CAM: {e}")
            return None
    
    def get_annotation_statistics(self) -> Dict:
        """Retorna estatísticas das anotações geradas"""
        if not self.generated_annotations:
            return {"total_annotations": 0}
        
        stats = {
            "total_annotations": len(self.generated_annotations),
            "by_class": {},
            "average_confidence": 0.0,
            "average_grad_cam_strength": 0.0,
            "high_quality_annotations": 0
        }
        
        total_confidence = 0
        total_grad_cam = 0
        
        for annotation in self.generated_annotations:
            # Contar por classe
            class_name = annotation.class_name
            stats["by_class"][class_name] = stats["by_class"].get(class_name, 0) + 1
            
            # Calcular médias
            total_confidence += annotation.confidence
            total_grad_cam += annotation.grad_cam_strength
            
            # Contar anotações de alta qualidade
            if (annotation.confidence > 0.6 and annotation.grad_cam_strength > 0.5):
                stats["high_quality_annotations"] += 1
        
        stats["average_confidence"] = total_confidence / len(self.generated_annotations)
        stats["average_grad_cam_strength"] = total_grad_cam / len(self.generated_annotations)
        
        return stats

# Exemplo de uso
if __name__ == "__main__":
    print("🎯 Anotador Automático - A Invenção da Anotação")
    print("=" * 50)
    print("Este módulo usa Grad-CAM para gerar anotações automáticas")
    print("quando a IA tem intuição mas não consegue detectar partes específicas.")
    print()
    print("Para usar:")
    print("1. Configure o caminho do modelo Keras")
    print("2. Use generate_auto_annotation() com candidatos de intuição")
    print("3. Visualize resultados com visualize_gradcam()")
    print()
    print("🚀 PRÓXIMO PASSO: Implementar Curador Híbrido com APIs de Visão")
