#!/usr/bin/env python3
"""
Módulo de Intuição - O "Santo Graal" da IA
Detecta quando o sistema encontra fronteiras do conhecimento atual
e marca candidatos para aprendizado automático.
"""

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import os
import json

logging.basicConfig(level=logging.INFO)

class LearningCandidateType(Enum):
    """Tipos de candidatos para aprendizado"""
    YOLO_FAILED_KERAS_MEDIUM = "yolo_failed_keras_medium"
    YOLO_FAILED_KERAS_HIGH = "yolo_failed_keras_high"
    YOLO_PARTIAL_KERAS_CONFLICT = "yolo_partial_keras_conflict"
    NEW_SPECIES_DETECTED = "new_species_detected"

@dataclass
class LearningCandidate:
    """Candidato para aprendizado automático"""
    image_path: str
    candidate_type: LearningCandidateType
    yolo_confidence: float
    keras_confidence: float
    keras_prediction: str
    yolo_detections: List[Dict]
    reasoning: str
    priority_score: float
    timestamp: str = ""

class IntuitionEngine:
    """
    Motor de Intuição - Detecta quando a IA encontra fronteiras do conhecimento
    """
    
    def __init__(self, yolo_model_path: str, keras_model_path: str):
        """
        Inicializa o motor de intuição
        
        Args:
            yolo_model_path: Caminho para modelo YOLO
            keras_model_path: Caminho para modelo Keras
        """
        self.yolo_model_path = yolo_model_path
        self.keras_model_path = keras_model_path
        
        # Carregar modelos
        self._load_models()
        
        # Configurações de intuição
        self.medium_confidence_range = (0.3, 0.7)  # Confiança mediana
        self.high_confidence_threshold = 0.7      # Confiança alta
        self.yolo_confidence_threshold = 0.5       # Limiar YOLO
        
        # Histórico de candidatos
        self.learning_candidates = []
        
    def _load_models(self):
        """Carrega modelos de detecção e classificação"""
        try:
            logging.info("Carregando modelo YOLO...")
            self.yolo_model = YOLO(self.yolo_model_path)
            logging.info("✅ Modelo YOLO carregado")
        except Exception as e:
            logging.error(f"❌ Erro ao carregar YOLO: {e}")
            self.yolo_model = None
            
        try:
            logging.info("Carregando modelo Keras...")
            self.keras_model = tf.keras.models.load_model(self.keras_model_path)
            logging.info("✅ Modelo Keras carregado")
        except Exception as e:
            logging.error(f"❌ Erro ao carregar Keras: {e}")
            self.keras_model = None
    
    def analyze_image_intuition(self, image_path: str) -> Dict:
        """
        Análise de intuição - Detecta candidatos para aprendizado
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            Análise completa com candidatos para aprendizado
        """
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"Não foi possível carregar a imagem: {image_path}"}
        
        # 1. Análise YOLO (Detector de Fatos)
        yolo_analysis = self._analyze_with_yolo(image)
        
        # 2. Análise Keras (Classificador de Espécies)
        keras_analysis = self._analyze_with_keras(image)
        
        # 3. Detecção de Intuição (O CORE da inovação)
        intuition_analysis = self._detect_intuition_candidates(
            yolo_analysis, keras_analysis, image_path
        )
        
        # 4. Compilar análise completa
        complete_analysis = {
            "image_path": image_path,
            "yolo_analysis": yolo_analysis,
            "keras_analysis": keras_analysis,
            "intuition_analysis": intuition_analysis,
            "learning_candidates": self._get_learning_candidates(intuition_analysis),
            "recommended_action": self._recommend_action(intuition_analysis)
        }
        
        return complete_analysis
    
    def _analyze_with_yolo(self, image: np.ndarray) -> Dict:
        """Análise usando YOLO (Detector de Fatos)"""
        if self.yolo_model is None:
            return {"error": "Modelo YOLO não disponível"}
        
        try:
            results = self.yolo_model(image, verbose=False)
            
            detections = []
            total_confidence = 0.0
            
            for r in results:
                for box in r.boxes:
                    if box.conf > self.yolo_confidence_threshold:
                        detection = {
                            "class": self.yolo_model.names[int(box.cls)],
                            "confidence": float(box.conf),
                            "bbox": box.xyxy[0].tolist()
                        }
                        detections.append(detection)
                        total_confidence += float(box.conf)
            
            avg_confidence = total_confidence / len(detections) if detections else 0.0
            
            return {
                "detections": detections,
                "total_detections": len(detections),
                "average_confidence": avg_confidence,
                "has_bird_parts": any('bird' in det['class'].lower() or 
                                    det['class'] in ['bico', 'asa', 'corpo', 'olho', 'garra', 'cauda'] 
                                    for det in detections),
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Erro na análise YOLO: {e}", "status": "failed"}
    
    def _analyze_with_keras(self, image: np.ndarray) -> Dict:
        """Análise usando Keras (Classificador de Espécies)"""
        if self.keras_model is None:
            return {"error": "Modelo Keras não disponível"}
        
        try:
            # Preparar imagem para Keras
            img_resized = cv2.resize(image, (224, 224))
            img_array = np.expand_dims(img_resized, axis=0) / 255.0
            
            # Predição
            prediction = self.keras_model.predict(img_array, verbose=0)
            class_id = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            # Mapear ID para nome da classe (ajustar conforme seu dataset)
            class_names = self._get_class_names()
            predicted_class = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            
            return {
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "class_id": int(class_id),
                "all_predictions": prediction[0].tolist(),
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Erro na análise Keras: {e}", "status": "failed"}
    
    def _get_class_names(self) -> List[str]:
        """Retorna nomes das classes (ajustar conforme seu dataset)"""
        # Por enquanto, usar classes genéricas
        return [
            "Brown_Pelican",
            "Cardinal", 
            "Painted_Bunting",
            "Pigeon_Guillemot",
            "Red_legged_Kittiwake"
        ]
    
    def _detect_intuition_candidates(self, yolo_analysis: Dict, 
                                   keras_analysis: Dict, 
                                   image_path: str) -> Dict:
        """
        CORE DA INOVAÇÃO: Detecta candidatos para aprendizado automático
        """
        candidates = []
        reasoning = []
        
        # CENÁRIO 1: YOLO falhou, mas Keras tem intuição mediana
        if (yolo_analysis.get("status") == "success" and 
            keras_analysis.get("status") == "success"):
            
            yolo_detections = yolo_analysis.get("total_detections", 0)
            keras_confidence = keras_analysis.get("confidence", 0)
            
            if (yolo_detections == 0 and 
                self.medium_confidence_range[0] <= keras_confidence <= self.medium_confidence_range[1]):
                
                candidate = LearningCandidate(
                    image_path=image_path,
                    candidate_type=LearningCandidateType.YOLO_FAILED_KERAS_MEDIUM,
                    yolo_confidence=0.0,
                    keras_confidence=keras_confidence,
                    keras_prediction=keras_analysis.get("predicted_class", ""),
                    yolo_detections=[],
                    reasoning="YOLO não detectou partes, mas Keras sugere espécie com confiança mediana",
                    priority_score=self._calculate_priority_score(keras_confidence, 0.0)
                )
                candidates.append(candidate)
                reasoning.append("🎯 CANDIDATO DETECTADO: YOLO falhou, Keras tem intuição mediana")
        
        # CENÁRIO 2: YOLO falhou, mas Keras tem alta confiança
        if (yolo_analysis.get("status") == "success" and 
            keras_analysis.get("status") == "success"):
            
            yolo_detections = yolo_analysis.get("total_detections", 0)
            keras_confidence = keras_analysis.get("confidence", 0)
            
            if (yolo_detections == 0 and keras_confidence > self.high_confidence_threshold):
                
                candidate = LearningCandidate(
                    image_path=image_path,
                    candidate_type=LearningCandidateType.YOLO_FAILED_KERAS_HIGH,
                    yolo_confidence=0.0,
                    keras_confidence=keras_confidence,
                    keras_prediction=keras_analysis.get("predicted_class", ""),
                    yolo_detections=[],
                    reasoning="YOLO não detectou partes, mas Keras tem alta confiança na espécie",
                    priority_score=self._calculate_priority_score(keras_confidence, 0.0)
                )
                candidates.append(candidate)
                reasoning.append("🚀 CANDIDATO PRIORITÁRIO: YOLO falhou, Keras tem alta confiança")
        
        # CENÁRIO 3: Conflito entre YOLO e Keras
        if (yolo_analysis.get("status") == "success" and 
            keras_analysis.get("status") == "success"):
            
            yolo_detections = yolo_analysis.get("total_detections", 0)
            keras_confidence = keras_analysis.get("confidence", 0)
            
            if (yolo_detections > 0 and keras_confidence < 0.3):
                
                candidate = LearningCandidate(
                    image_path=image_path,
                    candidate_type=LearningCandidateType.YOLO_PARTIAL_KERAS_CONFLICT,
                    yolo_confidence=yolo_analysis.get("average_confidence", 0),
                    keras_confidence=keras_confidence,
                    keras_prediction=keras_analysis.get("predicted_class", ""),
                    yolo_detections=yolo_analysis.get("detections", []),
                    reasoning="YOLO detectou partes, mas Keras tem baixa confiança na espécie",
                    priority_score=self._calculate_priority_score(keras_confidence, yolo_analysis.get("average_confidence", 0))
                )
                candidates.append(candidate)
                reasoning.append("⚠️ CONFLITO DETECTADO: YOLO vs Keras em desacordo")
        
        # CENÁRIO 4: Nova espécie potencial
        if (keras_analysis.get("status") == "success" and 
            keras_analysis.get("confidence", 0) > 0.8):
            
            predicted_class = keras_analysis.get("predicted_class", "")
            if predicted_class not in self._get_known_species():
                
                candidate = LearningCandidate(
                    image_path=image_path,
                    candidate_type=LearningCandidateType.NEW_SPECIES_DETECTED,
                    yolo_confidence=yolo_analysis.get("average_confidence", 0),
                    keras_confidence=keras_analysis.get("confidence", 0),
                    keras_prediction=predicted_class,
                    yolo_detections=yolo_analysis.get("detections", []),
                    reasoning=f"Nova espécie potencial detectada: {predicted_class}",
                    priority_score=1.0  # Máxima prioridade
                )
                candidates.append(candidate)
                reasoning.append("🌟 NOVA ESPÉCIE: Espécie desconhecida com alta confiança")
        
        # Adicionar candidatos ao histórico
        self.learning_candidates.extend(candidates)
        
        return {
            "candidates_found": len(candidates),
            "candidates": candidates,
            "reasoning": reasoning,
            "intuition_level": self._calculate_intuition_level(candidates),
            "recommendation": self._get_intuition_recommendation(candidates)
        }
    
    def _get_known_species(self) -> List[str]:
        """Retorna lista de espécies conhecidas"""
        return ["Brown_Pelican", "Cardinal", "Painted_Bunting"]
    
    def _calculate_priority_score(self, keras_conf: float, yolo_conf: float) -> float:
        """Calcula score de prioridade para aprendizado"""
        # Priorizar casos onde Keras tem confiança mas YOLO falhou
        if yolo_conf == 0 and keras_conf > 0.3:
            return keras_conf * 1.5  # Bonus por intuição
        elif yolo_conf > 0 and keras_conf < 0.3:
            return 0.8  # Conflito interessante
        else:
            return (keras_conf + yolo_conf) / 2
    
    def _calculate_intuition_level(self, candidates: List[LearningCandidate]) -> str:
        """Calcula nível de intuição detectado"""
        if not candidates:
            return "Nenhuma intuição detectada"
        
        high_priority = sum(1 for c in candidates if c.priority_score > 0.8)
        if high_priority > 0:
            return "Alta intuição - Candidatos prioritários"
        elif len(candidates) > 1:
            return "Média intuição - Múltiplos candidatos"
        else:
            return "Baixa intuição - Candidato único"
    
    def _get_intuition_recommendation(self, candidates: List[LearningCandidate]) -> str:
        """Gera recomendação baseada na intuição"""
        if not candidates:
            return "Prosseguir com análise normal"
        
        high_priority = [c for c in candidates if c.priority_score > 0.8]
        if high_priority:
            return f"🚀 ATIVAR APRENDIZADO AUTOMÁTICO: {len(high_priority)} candidato(s) prioritário(s)"
        
        medium_priority = [c for c in candidates if 0.5 <= c.priority_score <= 0.8]
        if medium_priority:
            return f"🎯 CONSIDERAR APRENDIZADO: {len(medium_priority)} candidato(s) interessante(s)"
        
        return "📝 REGISTRAR PARA ANÁLISE FUTURA"
    
    def _get_learning_candidates(self, intuition_analysis: Dict) -> List[Dict]:
        """Retorna candidatos formatados para aprendizado"""
        candidates = []
        for candidate in intuition_analysis.get("candidates", []):
            candidates.append({
                "image_path": candidate.image_path,
                "type": candidate.candidate_type.value,
                "priority_score": candidate.priority_score,
                "reasoning": candidate.reasoning,
                "keras_prediction": candidate.keras_prediction,
                "keras_confidence": candidate.keras_confidence
            })
        return candidates
    
    def _recommend_action(self, intuition_analysis: Dict) -> str:
        """Recomenda ação baseada na análise de intuição"""
        candidates_count = intuition_analysis.get("candidates_found", 0)
        
        if candidates_count == 0:
            return "PROCESSAR_NORMALMENTE"
        elif candidates_count == 1:
            return "ATIVAR_ANOTADOR_AUTOMATICO"
        else:
            return "ATIVAR_ANOTADOR_AUTOMATICO_PRIORITARIO"
    
    def get_learning_statistics(self) -> Dict:
        """Retorna estatísticas de aprendizado"""
        if not self.learning_candidates:
            return {"total_candidates": 0}
        
        stats = {
            "total_candidates": len(self.learning_candidates),
            "by_type": {},
            "average_priority": 0.0,
            "high_priority_count": 0
        }
        
        total_priority = 0
        for candidate in self.learning_candidates:
            # Contar por tipo
            type_name = candidate.candidate_type.value
            stats["by_type"][type_name] = stats["by_type"].get(type_name, 0) + 1
            
            # Calcular prioridade média
            total_priority += candidate.priority_score
            
            # Contar alta prioridade
            if candidate.priority_score > 0.8:
                stats["high_priority_count"] += 1
        
        stats["average_priority"] = total_priority / len(self.learning_candidates)
        
        return stats

# Exemplo de uso
if __name__ == "__main__":
    print("🧠 Módulo de Intuição - O Santo Graal da IA")
    print("=" * 50)
    print("Este módulo detecta quando a IA encontra fronteiras do conhecimento")
    print("e marca candidatos para aprendizado automático.")
    print()
    print("Para usar:")
    print("1. Configure os caminhos dos modelos")
    print("2. Use analyze_image_intuition() para analisar imagens")
    print("3. Verifique candidatos para aprendizado automático")
    print()
    print("🚀 PRÓXIMO PASSO: Implementar Anotador Automático com Grad-CAM")
