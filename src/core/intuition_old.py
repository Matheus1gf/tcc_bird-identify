#!/usr/bin/env python3
"""
Módulo de Intuição - O "Santo Graal" da IA
Detecta quando o sistema encontra fronteiras do conhecimento atual
e marca candidatos para aprendizado automático.
"""

import cv2
import numpy as np
# Mock imports para evitar erros
try:
    import tensorflow as tf
except ImportError:
    class MockTF:
        def keras(self): return self
        def models(self): return self
        def load_model(self, path): return None
        def optimizers(self): return self
        def legacy(self): return self
        def Adam(self, *args, **kwargs): return None
    tf = MockTF()

try:
    from ultralytics import YOLO
except ImportError:
    class MockYOLO:
        def __init__(self, *args, **kwargs): pass
        def predict(self, *args, **kwargs): return []
    YOLO = MockYOLO
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
    VISUAL_ANALYSIS = "visual_analysis"

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
            
            # Aplicar patch para resolver problema do PyTorch 2.6
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
            from patches import apply_yolo_patch
            apply_yolo_patch()
            
            # Tentar carregar modelo treinado primeiro
            if os.path.exists(self.yolo_model_path):
                try:
                    self.yolo_model = YOLO(self.yolo_model_path)
                    logging.info("✅ Modelo YOLO treinado carregado")
                except Exception as e:
                    logging.warning(f"⚠️ Erro ao carregar modelo treinado: {e}")
                    # Fallback para modelo pré-treinado
                    self.yolo_model = YOLO('yolov8n.pt')
                    logging.info("✅ Modelo YOLO pré-treinado carregado como fallback")
            else:
                # Usar modelo pré-treinado se não existe modelo treinado
                self.yolo_model = YOLO('yolov8n.pt')
                logging.info("✅ Modelo YOLO pré-treinado carregado")
            
        except Exception as e:
            logging.error(f"❌ Erro ao carregar YOLO: {e}")
            self.yolo_model = None
            
        try:
            logging.info("Carregando modelo Keras...")
            
            # Verificar se o arquivo existe
            if not os.path.exists(self.keras_model_path):
                logging.warning(f"⚠️ Modelo Keras não encontrado: {self.keras_model_path}")
                logging.info("📝 Sistema funcionará apenas com YOLO até que o modelo Keras seja treinado")
                self.keras_model = None
                return
            
            # Carregar modelo com otimizador legacy para compatibilidade
            self.keras_model = tf.keras.models.load_model(
                self.keras_model_path,
                custom_objects={
                    'Adam': tf.keras.optimizers.legacy.Adam
                }
            )
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
    
    def _analyze_visual_characteristics(self, image_path: str) -> Dict:
        """Análise visual melhorada para detectar características de pássaros"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"bird_like_features": 0, "bird_colors": False, "bird_proportions": False}
            
            # Converter para RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            
            # Análise de cores típicas de pássaros
            bird_colors = self._analyze_bird_colors(image_rgb)
            
            # Análise de forma e proporção
            bird_proportions = self._analyze_bird_proportions(image_rgb)
            
            # Análise de textura e padrões
            bird_texture = self._analyze_bird_texture(image_rgb)
            
            # Análise de contornos
            bird_contours = self._analyze_bird_contours(image_rgb)
            
            # Calcular score geral de características de pássaro
            # Dar mais peso às proporções e textura (mais discriminativas)
            bird_like_score = (
                bird_colors * 0.2 +      # Reduzido: cores podem confundir
                bird_proportions * 0.4 + # Aumentado: proporções são muito discriminativas
                bird_texture * 0.3 +     # Aumentado: textura distingue penas vs pelo
                bird_contours * 0.1      # Reduzido: contornos são menos confiáveis
            )
            
            return {
                "bird_like_features": bird_like_score,
                "bird_colors": bird_colors > 0.15,  # Threshold ainda mais permissivo
                "bird_proportions": bird_proportions > 0.15,  # Threshold ainda mais permissivo
                "bird_texture": bird_texture > 0.15,  # Threshold ainda mais permissivo
                "bird_contours": bird_contours > 0.15,  # Threshold ainda mais permissivo
                "detailed_scores": {
                    "colors": bird_colors,
                    "proportions": bird_proportions,
                    "texture": bird_texture,
                    "contours": bird_contours
                }
            }
            
        except Exception as e:
            logging.error(f"Erro na análise visual: {e}")
            return {"bird_like_features": 0, "bird_colors": False, "bird_proportions": False}
    
    def _analyze_bird_colors(self, image_rgb: np.ndarray) -> float:
        """Analisa cores típicas de pássaros com melhor distinção"""
        try:
            # Cores típicas de pássaros (mais específicas)
            bird_color_ranges = [
                # Marroms específicos de pássaros (mais avermelhados)
                ([80, 40, 20], [160, 100, 60]),   # Marrom avermelhado
                ([100, 60, 30], [180, 120, 80]),  # Marrom médio avermelhado
                
                # Cinzas específicos de pássaros
                ([90, 90, 90], [170, 170, 170]),  # Cinza médio
                ([110, 110, 110], [190, 190, 190]), # Cinza claro
                
                # Vermelhos e rosas específicos de pássaros
                ([160, 60, 60], [255, 140, 140]), # Vermelho/rosa
                ([180, 90, 90], [255, 170, 170]), # Rosa claro
                
                # Amarelos e dourados específicos de pássaros
                ([210, 160, 60], [255, 210, 140]), # Amarelo/dourado
                
                # Azuis específicos de pássaros
                ([60, 60, 160], [160, 160, 255]), # Azul
                
                # Verdes específicos de pássaros (alguns papagaios)
                ([40, 120, 40], [120, 200, 120]), # Verde
            ]
            
            # Cores que indicam NÃO-pássaro (cachorros, gatos, etc.)
            non_bird_color_ranges = [
                # Marroms muito escuros (típicos de cachorros)
                ([30, 20, 10], [80, 50, 30]),     # Marrom muito escuro
                ([40, 25, 15], [90, 60, 40]),     # Marrom escuro
                
                # Pretos (típicos de cachorros pretos)
                ([0, 0, 0], [50, 50, 50]),        # Preto
                
                # Brancos puros (típicos de cachorros brancos)
                ([200, 200, 200], [255, 255, 255]), # Branco
                
                # Tons de pele (cachorros com pele rosada)
                ([180, 140, 120], [255, 200, 180]), # Tons de pele
            ]
            
            total_pixels = image_rgb.shape[0] * image_rgb.shape[1]
            bird_pixels = 0
            non_bird_pixels = 0
            
            # Contar pixels de pássaros
            for lower, upper in bird_color_ranges:
                lower = np.array(lower)
                upper = np.array(upper)
                mask = cv2.inRange(image_rgb, lower, upper)
                bird_pixels += np.sum(mask > 0)
            
            # Contar pixels de não-pássaros
            for lower, upper in non_bird_color_ranges:
                lower = np.array(lower)
                upper = np.array(upper)
                mask = cv2.inRange(image_rgb, lower, upper)
                non_bird_pixels += np.sum(mask > 0)
            
            # Calcular proporções
            bird_color_ratio = bird_pixels / total_pixels
            non_bird_color_ratio = non_bird_pixels / total_pixels
            
            # Score baseado na diferença entre pássaro e não-pássaro
            if non_bird_color_ratio > 0.3:  # Se há muitas cores de não-pássaro
                return max(0.0, bird_color_ratio * 2 - non_bird_color_ratio * 3)
            else:
                return min(bird_color_ratio * 3, 1.0)
            
        except Exception as e:
            logging.error(f"Erro na análise de cores: {e}")
            return 0.0
    
    def _analyze_bird_proportions(self, image_rgb: np.ndarray) -> float:
        """Analisa proporções típicas de pássaros com melhor distinção"""
        try:
            h, w = image_rgb.shape[:2]
            
            # Converter para escala de cinza
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            
            # Detectar bordas
            edges = cv2.Canny(gray, 50, 150)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.0
            
            # Encontrar o maior contorno (provavelmente o animal)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calcular bounding box
            x, y, w_contour, h_contour = cv2.boundingRect(largest_contour)
            
            # Análise de proporções típicas de pássaros vs outros animais
            aspect_ratio = w_contour / h_contour if h_contour > 0 else 0
            
            # Pássaros: proporção mais vertical (0.6 a 1.2)
            # Cachorros: proporção mais horizontal (1.5 a 3.0)
            if 0.6 <= aspect_ratio <= 1.2:  # Proporção vertical (pássaro)
                proportion_score = 0.9
            elif 0.5 <= aspect_ratio <= 1.4:  # Proporção intermediária
                proportion_score = 0.6
            elif 1.5 <= aspect_ratio <= 3.0:  # Proporção horizontal (cachorro)
                proportion_score = 0.1
            else:
                proportion_score = 0.3
            
            # Análise de área ocupada
            contour_area = cv2.contourArea(largest_contour)
            image_area = h * w
            area_ratio = contour_area / image_area
            
            # Pássaros: área menor (5% a 40%)
            # Cachorros: área maior (30% a 70%)
            if 0.05 <= area_ratio <= 0.4:  # Área pequena (pássaro)
                area_score = 0.8
            elif 0.3 <= area_ratio <= 0.7:  # Área grande (cachorro)
                area_score = 0.2
            elif 0.02 <= area_ratio <= 0.6:  # Área intermediária
                area_score = 0.5
            else:
                area_score = 0.3
            
            # Análise de forma (pássaros são mais compactos)
            contour_perimeter = cv2.arcLength(largest_contour, True)
            if contour_perimeter > 0:
                compactness = (4 * np.pi * contour_area) / (contour_perimeter ** 2)
                # Pássaros têm formas mais compactas (0.3 a 0.8)
                # Cachorros têm formas menos compactas (0.1 a 0.4)
                if 0.3 <= compactness <= 0.8:  # Forma compacta (pássaro)
                    compactness_score = 0.8
                elif 0.1 <= compactness <= 0.4:  # Forma alongada (cachorro)
                    compactness_score = 0.2
                else:
                    compactness_score = 0.5
            else:
                compactness_score = 0.5
            
            return (proportion_score + area_score + compactness_score) / 3
            
        except Exception as e:
            logging.error(f"Erro na análise de proporções: {e}")
            return 0.0
    
    def _analyze_bird_texture(self, image_rgb: np.ndarray) -> float:
        """Analisa textura típica de penas vs pelo"""
        try:
            # Converter para escala de cinza
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            
            # Calcular gradientes (textura)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Magnitude do gradiente
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Calcular variância da textura
            texture_variance = np.var(gradient_magnitude)
            
            # Calcular uniformidade da textura (LBP simplificado)
            texture_uniformity = self._calculate_texture_uniformity(gray)
            
            # Pássaros (penas): textura mais uniforme e padrões regulares
            # Cachorros (pelo): textura mais irregular e variada
            
            # Análise de variância
            if 200 <= texture_variance <= 1500:  # Textura moderada (penas)
                variance_score = 0.8
            elif 100 <= texture_variance <= 2000:  # Textura intermediária
                variance_score = 0.6
            elif texture_variance > 2000:  # Textura muito variada (pelo)
                variance_score = 0.2
            else:  # Textura muito lisa
                variance_score = 0.3
            
            # Análise de uniformidade
            if texture_uniformity > 0.7:  # Textura uniforme (penas)
                uniformity_score = 0.8
            elif texture_uniformity > 0.5:  # Textura intermediária
                uniformity_score = 0.5
            else:  # Textura irregular (pelo)
                uniformity_score = 0.2
            
            return (variance_score + uniformity_score) / 2
                
        except Exception as e:
            logging.error(f"Erro na análise de textura: {e}")
            return 0.0
    
    def _calculate_texture_uniformity(self, gray_image: np.ndarray) -> float:
        """Calcula uniformidade da textura usando análise de padrões locais"""
        try:
            h, w = gray_image.shape
            uniformity_scores = []
            
            # Analisar padrões em janelas pequenas
            for y in range(1, h-1, 4):  # Amostrar a cada 4 pixels
                for x in range(1, w-1, 4):
                    # Janela 3x3
                    window = gray_image[y-1:y+2, x-1:x+2]
                    if window.shape == (3, 3):
                        # Calcular variação local
                        local_variance = np.var(window)
                        # Calcular diferenças entre pixels adjacentes (com proteção contra overflow)
                        diff_h = np.abs(int(window[1, 1]) - int(window[1, 0])) + np.abs(int(window[1, 1]) - int(window[1, 2]))
                        diff_v = np.abs(int(window[1, 1]) - int(window[0, 1])) + np.abs(int(window[1, 1]) - int(window[2, 1]))
                        local_uniformity = 1.0 / (1.0 + local_variance + (diff_h + diff_v) / 4.0)
                        uniformity_scores.append(local_uniformity)
            
            if uniformity_scores:
                return np.mean(uniformity_scores)
            else:
                return 0.5
                
        except Exception as e:
            logging.error(f"Erro no cálculo de uniformidade: {e}")
            return 0.5
    
    def _analyze_bird_contours(self, image_rgb: np.ndarray) -> float:
        """Analisa contornos típicos de pássaros"""
        try:
            # Converter para escala de cinza
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            
            # Aplicar blur para suavizar
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detectar bordas
            edges = cv2.Canny(blurred, 50, 150)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.0
            
            # Analisar o maior contorno
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calcular convexidade (pássaros têm formas relativamente convexas)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(largest_contour)
            
            if hull_area > 0:
                convexity = contour_area / hull_area
                if convexity > 0.7:  # Formas convexas são típicas de pássaros
                    return 0.8
                elif convexity > 0.5:
                    return 0.5
                else:
                    return 0.2
            
            return 0.0
            
        except Exception as e:
            logging.error(f"Erro na análise de contornos: {e}")
            return 0.0
    
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
                "has_bird_parts": any('bird' in det['class'].lower() for det in detections),
                "has_specific_parts": any(det['class'] in ['bico', 'asa', 'corpo', 'olho', 'garra', 'cauda'] 
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
        
        # ANÁLISE VISUAL MELHORADA
        visual_analysis = self._analyze_visual_characteristics(image_path)
        
        # CENÁRIO 0: Análise visual detecta pássaro independentemente de YOLO/Keras
        # Threshold mais alto para evitar falsos positivos
        bird_score = visual_analysis.get("bird_like_features", 0)
        bird_colors = visual_analysis.get("bird_colors", False)
        bird_proportions = visual_analysis.get("bird_proportions", False)
        bird_texture = visual_analysis.get("bird_texture", False)
        
        # Critérios simplificados e mais permissivos para detectar pássaros reais
        # Verificar scores detalhados para ser mais preciso
        detailed_scores = visual_analysis.get("detailed_scores", {})
        colors_score = detailed_scores.get("colors", 0)
        proportions_score = detailed_scores.get("proportions", 0)
        texture_score = detailed_scores.get("texture", 0)
        
        # Critérios rigorosos: detectar pássaros reais mas evitar falsos positivos
        # Score geral > 60% OU (score > 60% + múltiplas características) OU (score > 60% + todas as características)
        if (bird_score > 0.6) or (bird_score > 0.6 and (bird_proportions and bird_texture)) or (bird_score > 0.6 and bird_proportions and bird_texture and bird_colors):
            candidate = LearningCandidate(
                image_path=image_path,
                candidate_type=LearningCandidateType.VISUAL_ANALYSIS,
                yolo_confidence=0.0,
                keras_confidence=visual_analysis.get("bird_like_features", 0),
                keras_prediction="Análise Visual",
                yolo_detections=[],
                reasoning=f"Análise visual detectou características de pássaro (score: {visual_analysis.get('bird_like_features', 0):.2%})",
                priority_score=self._calculate_priority_score(visual_analysis.get("bird_like_features", 0), 0.0)
            )
            candidates.append(candidate)
            reasoning.append(f"🎯 CANDIDATO VISUAL: Análise visual sugere pássaro (score: {visual_analysis.get('bird_like_features', 0):.2%})")
        
        # CENÁRIO 1: YOLO falhou, mas Keras tem intuição mediana
        if (yolo_analysis.get("status") == "success" and 
            keras_analysis.get("status") == "success"):
            
            yolo_detections = yolo_analysis.get("total_detections", 0)
            keras_confidence = keras_analysis.get("confidence", 0)
            predicted_class = keras_analysis.get("predicted_class", "").lower()
            
            # Verificar se a predição do Keras é realmente de um pássaro
            bird_keywords = ['bird', 'pássaro', 'ave', 'passaro', 'beija', 'sabiá', 'bem-te-vi', 'canário', 'pardal', 'rolinha', 'pombo', 'dove', 'columbina']
            is_bird_prediction = any(keyword in predicted_class for keyword in bird_keywords)
            
            # Análise visual adicional
            has_bird_features = visual_analysis.get("bird_like_features", 0) > 0.3
            has_bird_colors = visual_analysis.get("bird_colors", False)
            has_bird_shape = visual_analysis.get("bird_proportions", False)
            
            # CENÁRIO ESPECIAL: Se análise visual sugere pássaro, mesmo sem YOLO/Keras
            if (visual_analysis.get("bird_like_features", 0) > 0.5 or 
                (visual_analysis.get("bird_colors", False) and visual_analysis.get("bird_like_features", 0) > 0.3)):
                
                candidate = LearningCandidate(
                    image_path=image_path,
                    candidate_type=LearningCandidateType.VISUAL_ANALYSIS,
                    yolo_confidence=0.0,
                    keras_confidence=visual_analysis.get("bird_like_features", 0),
                    keras_prediction="Análise Visual",
                    yolo_detections=[],
                    reasoning=f"Análise visual detectou características de pássaro (score: {visual_analysis.get('bird_like_features', 0):.2%})",
                    priority_score=self._calculate_priority_score(visual_analysis.get("bird_like_features", 0), 0.0)
                )
                candidates.append(candidate)
                reasoning.append(f"🎯 CANDIDATO VISUAL: Análise visual sugere pássaro (score: {visual_analysis.get('bird_like_features', 0):.2%})")
            
            if (yolo_detections == 0 and 
                self.medium_confidence_range[0] <= keras_confidence <= self.medium_confidence_range[1] and
                (is_bird_prediction or has_bird_features or has_bird_colors or has_bird_shape)):
                
                candidate = LearningCandidate(
                    image_path=image_path,
                    candidate_type=LearningCandidateType.YOLO_FAILED_KERAS_MEDIUM,
                    yolo_confidence=0.0,
                    keras_confidence=keras_confidence,
                    keras_prediction=keras_analysis.get("predicted_class", ""),
                    yolo_detections=[],
                    reasoning=f"YOLO não detectou partes, mas análise visual sugere pássaro (características: {visual_analysis.get('bird_like_features', 0):.2%})",
                    priority_score=self._calculate_priority_score(keras_confidence, visual_analysis.get("bird_like_features", 0))
                )
                candidates.append(candidate)
                reasoning.append(f"🎯 CANDIDATO DETECTADO: Análise visual sugere pássaro (confiança visual: {visual_analysis.get('bird_like_features', 0):.2%})")
        
        # CENÁRIO 2: YOLO falhou, mas Keras tem alta confiança
        if (yolo_analysis.get("status") == "success" and 
            keras_analysis.get("status") == "success"):
            
            yolo_detections = yolo_analysis.get("total_detections", 0)
            keras_confidence = keras_analysis.get("confidence", 0)
            predicted_class = keras_analysis.get("predicted_class", "").lower()
            
            # Verificar se a predição do Keras é realmente de um pássaro
            bird_keywords = ['bird', 'pássaro', 'ave', 'passaro', 'beija', 'sabiá', 'bem-te-vi', 'canário', 'pardal']
            is_bird_prediction = any(keyword in predicted_class for keyword in bird_keywords)
            
            if (yolo_detections == 0 and 
                keras_confidence > self.high_confidence_threshold and
                is_bird_prediction):
                
                candidate = LearningCandidate(
                    image_path=image_path,
                    candidate_type=LearningCandidateType.YOLO_FAILED_KERAS_HIGH,
                    yolo_confidence=0.0,
                    keras_confidence=keras_confidence,
                    keras_prediction=keras_analysis.get("predicted_class", ""),
                    yolo_detections=[],
                    reasoning="YOLO não detectou partes, mas Keras tem alta confiança em espécie de pássaro",
                    priority_score=self._calculate_priority_score(keras_confidence, 0.0)
                )
                candidates.append(candidate)
                reasoning.append("🚀 CANDIDATO PRIORITÁRIO: YOLO falhou, Keras tem alta confiança para pássaro")
        
        # CENÁRIO 3: Conflito entre YOLO e Keras
        if (yolo_analysis.get("status") == "success" and 
            keras_analysis.get("status") == "success"):
            
            yolo_detections = yolo_analysis.get("total_detections", 0)
            keras_confidence = keras_analysis.get("confidence", 0)
            predicted_class = keras_analysis.get("predicted_class", "").lower()
            
            # Verificar se a predição do Keras é realmente de um pássaro
            bird_keywords = ['bird', 'pássaro', 'ave', 'passaro', 'beija', 'sabiá', 'bem-te-vi', 'canário', 'pardal']
            is_bird_prediction = any(keyword in predicted_class for keyword in bird_keywords)
            
            if (yolo_detections > 0 and 
                keras_confidence < 0.3 and
                is_bird_prediction):
                
                candidate = LearningCandidate(
                    image_path=image_path,
                    candidate_type=LearningCandidateType.YOLO_PARTIAL_KERAS_CONFLICT,
                    yolo_confidence=yolo_analysis.get("average_confidence", 0),
                    keras_confidence=keras_confidence,
                    keras_prediction=keras_analysis.get("predicted_class", ""),
                    yolo_detections=yolo_analysis.get("detections", []),
                    reasoning="YOLO detectou partes, mas Keras tem baixa confiança na espécie de pássaro",
                    priority_score=self._calculate_priority_score(keras_confidence, yolo_analysis.get("average_confidence", 0))
                )
                candidates.append(candidate)
                reasoning.append("⚠️ CONFLITO DETECTADO: YOLO vs Keras em desacordo para pássaro")
        
        # CENÁRIO 4: Nova espécie potencial
        if (keras_analysis.get("status") == "success" and 
            keras_analysis.get("confidence", 0) > 0.8):
            
            predicted_class = keras_analysis.get("predicted_class", "").lower()
            
            # Verificar se a predição do Keras é realmente de um pássaro
            bird_keywords = ['bird', 'pássaro', 'ave', 'passaro', 'beija', 'sabiá', 'bem-te-vi', 'canário', 'pardal']
            is_bird_prediction = any(keyword in predicted_class for keyword in bird_keywords)
            
            if (is_bird_prediction and 
                predicted_class not in [s.lower() for s in self._get_known_species()]):
                
                candidate = LearningCandidate(
                    image_path=image_path,
                    candidate_type=LearningCandidateType.NEW_SPECIES_DETECTED,
                    yolo_confidence=yolo_analysis.get("average_confidence", 0),
                    keras_confidence=keras_analysis.get("confidence", 0),
                    keras_prediction=keras_analysis.get("predicted_class", ""),
                    yolo_detections=yolo_analysis.get("detections", []),
                    reasoning=f"Nova espécie de pássaro potencial detectada: {keras_analysis.get('predicted_class', '')}",
                    priority_score=1.0  # Máxima prioridade
                )
                candidates.append(candidate)
                reasoning.append("🌟 NOVA ESPÉCIE: Nova espécie de pássaro com alta confiança")
        
        # Adicionar candidatos ao histórico
        self.learning_candidates.extend(candidates)
        
        return {
            "candidates_found": len(candidates),
            "candidates": candidates,
            "reasoning": reasoning,
            "intuition_level": self._calculate_intuition_level(candidates),
            "recommendation": self._get_intuition_recommendation(candidates),
            "visual_analysis": visual_analysis
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
        """Recomenda ação baseada na análise de intuição melhorada"""
        candidates_count = intuition_analysis.get("candidates_found", 0)
        visual_analysis = intuition_analysis.get("visual_analysis", {})
        bird_like_score = visual_analysis.get("bird_like_features", 0)
        
        # Se há características visuais de pássaro, recomendar análise manual
        # Thresholds muito mais permissivos para detectar pássaros reais
        if bird_like_score > 0.4:
            return "ANALISAR_MANUALMENTE"
        elif bird_like_score > 0.2:
            return "PROCESSAR_COM_CUIDADO"
        elif candidates_count >= 2:
            return "ATIVAR_ANOTADOR_AUTOMATICO_PRIORITARIO"
        elif candidates_count == 1:
            return "ATIVAR_ANOTADOR_AUTOMATICO"
        else:
            return "PROCESSAR_NORMALMENTE"
    
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
