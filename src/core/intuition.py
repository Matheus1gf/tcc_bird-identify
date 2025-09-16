#!/usr/bin/env python3
"""
Sistema de Intuição Neuro-Simbólica Simplificado
Funciona como uma criança descobrindo características fundamentais de pássaros
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningCandidateType(Enum):
    """Tipos de candidatos para aprendizado"""
    VISUAL_ANALYSIS = "visual_analysis"
    SPECIES_UNKNOWN = "species_unknown"
    CHARACTERISTIC_LEARNING = "characteristic_learning"

@dataclass
class LearningCandidate:
    """Candidato para aprendizado contínuo"""
    type: LearningCandidateType
    confidence: float
    characteristics: Dict[str, Any]
    reasoning: str
    image_path: str
    metadata: Dict[str, Any]

class IntuitionEngine:
    """Motor de Intuição Neuro-Simbólica Simplificado para Pássaros"""
    
    def __init__(self, yolo_model_path: str, keras_model_path: str, debug_logger):
        self.yolo_model_path = yolo_model_path
        self.keras_model_path = keras_model_path
        self.debug_logger = debug_logger
        self.yolo_model = None
        self.keras_model = None
        
        # Conhecimento acumulado (como uma criança)
        self.learned_patterns = {
            'known_species': set(),
            'characteristic_patterns': {},
            'color_combinations': {},
            'shape_patterns': {}
        }
        
        self._load_models()
        
    def _load_models(self):
        """Carrega modelos híbridos com múltiplas bibliotecas de detecção"""
        self.detection_models = {}
        
        # 1. Tentar carregar YOLO (múltiplas tentativas)
        self._load_yolo_models()
        
        # 2. Carregar OpenCV DNN (alternativa robusta)
        self._load_opencv_dnn()
        
        # 3. Carregar MediaPipe (detecção de objetos)
        self._load_mediapipe()
        
        # 4. Carregar Keras
        self._load_keras()
        
        # 5. Log do status dos modelos
        self._log_model_status()
    
    def _load_yolo_models(self):
        """Carrega modelos YOLO com múltiplas versões e configurações avançadas"""
        try:
            from ultralytics import YOLO
            import torch
            import os
            
            # Configuração segura para PyTorch 2.6
            torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
            
            # Lista de modelos YOLO para tentar (em ordem de prioridade)
            yolo_models = [
                # Modelos customizados treinados
                (self.yolo_model_path, 'YOLO customizado (melhor)'),
                ('runs/detect/train/weights/best.pt', 'YOLO customizado (melhor)'),
                ('runs/detect/train/weights/last.pt', 'YOLO customizado (último)'),
                
                # Modelos YOLOv8 (diferentes tamanhos)
                ('yolov8n.pt', 'YOLOv8 Nano'),
                ('yolov8s.pt', 'YOLOv8 Small'),
                ('yolov8m.pt', 'YOLOv8 Medium'),
                ('yolov8l.pt', 'YOLOv8 Large'),
                ('yolov8x.pt', 'YOLOv8 Extra Large'),
                
                # Modelos YOLOv11 (mais recentes)
                ('yolo11n.pt', 'YOLOv11 Nano'),
                ('yolo11s.pt', 'YOLOv11 Small'),
                ('yolo11m.pt', 'YOLOv11 Medium'),
                
                # Modelos YOLOv10
                ('yolo10n.pt', 'YOLOv10 Nano'),
                ('yolo10s.pt', 'YOLOv10 Small'),
                
                # Modelos YOLOv9
                ('yolo9c.pt', 'YOLOv9 Compact'),
                ('yolo9e.pt', 'YOLOv9 Efficient'),
            ]
            
            # Tentar carregar cada modelo
            for model_path, model_name in yolo_models:
                try:
                    if os.path.exists(model_path) or model_path.startswith('yolo'):
                        # Configurações avançadas para melhor detecção
                        model = YOLO(model_path)
                        
                        # Configurar parâmetros avançados
                        model.overrides = {
                            'conf': 0.15,  # Confiança mínima mais baixa
                            'iou': 0.35,   # Intersection over Union mais rigoroso
                            'agnostic_nms': False,  # NMS não agnóstico
                            'max_det': 1000,  # Máximo de detecções
                            'half': False,  # Precisão completa
                            'dnn': False,   # Usar PyTorch
                            'device': 'cpu'  # Forçar CPU para compatibilidade
                        }
                        
                        self.detection_models['yolo_advanced'] = model
                        logger.info(f"✅ {model_name} carregado com configurações avançadas")
                        return
                        
                except Exception as e:
                    logger.warning(f"⚠️ {model_name} falhou: {e}")
                    continue
            
            # Tentativa final: YOLOv5 como fallback
            try:
                import torch.hub
                self.detection_models['yolo_v5'] = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                logger.info("✅ YOLOv5 fallback carregado")
            except Exception as e:
                logger.warning(f"⚠️ YOLOv5 fallback falhou: {e}")
                        
        except Exception as e:
            logger.warning(f"⚠️ Erro geral ao carregar YOLO: {e}")
    
    def _load_opencv_dnn(self):
        """Carrega modelos OpenCV DNN para detecção alternativa"""
        try:
            # Tentar carregar modelos pré-treinados do OpenCV
            # YOLOv4, YOLOv3, MobileNet-SSD, etc.
            
            # YOLOv4 (se disponível)
            try:
                yolo_config = "yolov4.cfg"
                yolo_weights = "yolov4.weights"
                if os.path.exists(yolo_config) and os.path.exists(yolo_weights):
                    net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
                    self.detection_models['opencv_yolo'] = net
                    logger.info("✅ OpenCV YOLO carregado")
            except Exception as e:
                logger.warning(f"⚠️ OpenCV YOLO não disponível: {e}")
            
            # MobileNet-SSD (mais leve e robusto)
            try:
                # Usar modelo pré-treinado do OpenCV
                prototxt = "MobileNetSSD_deploy.prototxt"
                model = "MobileNetSSD_deploy.caffemodel"
                
                # Se não existir, usar modelo padrão do OpenCV
                if not os.path.exists(prototxt):
                    # Usar modelo padrão do OpenCV
                    net = cv2.dnn.readNetFromTensorflow("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
                    self.detection_models['opencv_ssd'] = net
                    logger.info("✅ OpenCV SSD carregado")
            except Exception as e:
                logger.warning(f"⚠️ OpenCV SSD não disponível: {e}")
                
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar OpenCV DNN: {e}")
    
    def _load_mediapipe(self):
        """Carrega MediaPipe para detecção de objetos"""
        try:
            import mediapipe as mp
            
            # Detecção de objetos com MediaPipe
            self.detection_models['mediapipe'] = mp.solutions.objectron.Objectron(
                static_image_mode=True,
                max_num_objects=5,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("✅ MediaPipe carregado")
            
        except Exception as e:
            logger.warning(f"⚠️ MediaPipe não disponível: {e}")
    
    def _load_keras(self):
        """Carrega modelo Keras"""
        try:
            import tensorflow as tf
            import os
            
            # Tentar carregar como HDF5 primeiro
            if self.keras_model_path.endswith('.keras'):
                # Se é .keras, tentar como HDF5
                h5_path = self.keras_model_path.replace('.keras', '.h5')
                if os.path.exists(h5_path):
                    self.keras_model = tf.keras.models.load_model(h5_path)
                    logger.info("✅ Modelo Keras HDF5 carregado")
                else:
                    # Tentar carregar diretamente
                    self.keras_model = tf.keras.models.load_model(self.keras_model_path)
                    logger.info("✅ Modelo Keras carregado")
            else:
                self.keras_model = tf.keras.models.load_model(self.keras_model_path)
                logger.info("✅ Modelo Keras carregado")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar Keras: {e}")
            logger.info("🔄 Usando análise visual pura")
            self.keras_model = None
    
    def _log_model_status(self):
        """Log do status de todos os modelos carregados"""
        loaded_models = list(self.detection_models.keys())
        if loaded_models:
            logger.info(f"🎯 Modelos de detecção carregados: {', '.join(loaded_models)}")
        else:
            logger.warning("⚠️ Nenhum modelo de detecção carregado - usando apenas análise visual")
        
        if self.keras_model is not None:
            logger.info("✅ Modelo Keras disponível")
        else:
            logger.info("ℹ️ Modelo Keras não disponível")
    
    def analyze_image_intuition(self, image_path: str) -> Dict[str, Any]:
        """
        Análise principal de intuição - como uma criança descobrindo pássaros
        """
        try:
            # 1. Análise visual básica (como uma criança vê)
            visual_analysis = self._analyze_visual_characteristics(image_path)
            
            # 2. Detecção de características fundamentais
            fundamental_characteristics = self._detect_fundamental_characteristics(image_path)
            
            # 3. Raciocínio lógico (neuro-simbólico)
            logical_reasoning = self._logical_reasoning(visual_analysis, fundamental_characteristics)
            
            # 4. Detecção de candidatos para aprendizado
            learning_candidates = self._detect_learning_candidates(
                visual_analysis, fundamental_characteristics, logical_reasoning
            )
            
            # 5. Recomendação de ação
            recommendation = self._recommend_action(learning_candidates, logical_reasoning)
            
            return {
                'is_bird': logical_reasoning.get('is_bird', False),
                'confidence': logical_reasoning.get('confidence', 0.0),
                'species': logical_reasoning.get('species', 'Desconhecida'),
                'intuition_level': logical_reasoning.get('intuition_level', 'Baixa'),
                'needs_manual_review': logical_reasoning.get('needs_manual_review', False),
                'reasoning_steps': logical_reasoning.get('reasoning_steps', []),
                'characteristics_found': logical_reasoning.get('characteristics_found', []),
                'color': visual_analysis.get('dominant_color', 'Indefinida'),
                'intuition_analysis': {
                    'visual_analysis': visual_analysis,
                    'fundamental_characteristics': fundamental_characteristics,
                    'logical_reasoning': logical_reasoning,
                    'learning_candidates': learning_candidates,
                    'recommendation': recommendation,
                    'candidates_found': len(learning_candidates)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na análise de intuição: {e}")
            return {
                'confidence': 0.0,
                'species': 'Erro',
                'color': 'Erro',
                'intuition_analysis': {
                    'error': str(e),
                    'candidates_found': 0,
                    'recommendation': 'ERRO_ANALISE'
                }
            }
    
    def _analyze_visual_characteristics(self, image_path: str) -> Dict[str, Any]:
        """Análise visual básica - como uma criança vê cores e formas"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Imagem não carregada'}
            
            # Converter para HSV para análise de cores
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Análise de cores dominantes
            color_analysis = self._analyze_colors(hsv)
            
            # Análise de formas básicas
            shape_analysis = self._analyze_shapes(image)
            
            # Análise de texturas
            texture_analysis = self._analyze_textures(image)
            
            return {
                'dominant_color': color_analysis['dominant_color'],
                'color_distribution': color_analysis['distribution'],
                'shape_characteristics': shape_analysis,
                'texture_characteristics': texture_analysis,
                'bird_like_features': self._calculate_bird_like_score(
                    color_analysis, shape_analysis, texture_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na análise visual: {e}")
            return {'error': str(e)}
    
    def _analyze_colors(self, hsv_image: np.ndarray) -> Dict[str, Any]:
        """Analisa cores como uma criança reconheceria - mais sensível"""
        # Cores típicas de pássaros - ranges mais amplos
        bird_colors = {
            'brown': [(5, 30, 10), (25, 200, 255)],
            'black': [(0, 0, 0), (60, 60, 80)],
            'white': [(180, 180, 180), (255, 255, 255)],
            'red': [(0, 30, 50), (15, 200, 255)],
            'blue': [(90, 30, 0), (160, 200, 255)],
            'yellow': [(0, 80, 80), (60, 255, 255)],
            'green': [(40, 80, 0), (160, 255, 200)],
            'gray': [(0, 0, 50), (180, 30, 200)],
            'orange': [(5, 100, 100), (25, 255, 255)]
        }
        
        color_scores = {}
        
        for color_name, (lower, upper) in bird_colors.items():
            mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
            score = np.sum(mask) / (mask.shape[0] * mask.shape[1] * 255)
            color_scores[color_name] = score
        
        # Encontrar cor dominante
        dominant_color = max(color_scores, key=color_scores.get)
        
        # Calcular score de cores de pássaro - mais sensível
        bird_color_score = sum(color_scores.values()) / len(color_scores)
        
        # Bonus para cores muito comuns em pássaros
        common_bird_colors = ['brown', 'black', 'white', 'gray', 'green', 'blue']
        common_score = sum([color_scores[color] for color in common_bird_colors]) / len(common_bird_colors)
        
        # Penalty para cores raras em pássaros
        rare_bird_colors = ['purple', 'pink']
        rare_score = sum([color_scores.get(color, 0) for color in rare_bird_colors]) / len(rare_bird_colors)
        
        # Score final com bonus para cores comuns e penalty para cores raras
        final_score = (bird_color_score + common_score - rare_score * 0.3) / 2
        final_score = max(0, min(final_score, 1.0))  # Clamp entre 0 e 1
        
        # Análise de contraste e saturação
        contrast_score = self._analyze_color_contrast(hsv_image)
        saturation_score = self._analyze_color_saturation(hsv_image)
        
        return {
            'dominant_color': dominant_color,
            'distribution': color_scores,
            'bird_color_score': final_score,
            'contrast_score': contrast_score,
            'saturation_score': saturation_score,
            'color_complexity': len([c for c in color_scores.values() if c > 0.1])
        }
    
    def _analyze_color_contrast(self, hsv_image: np.ndarray) -> float:
        """Analisa o contraste de cores na imagem"""
        try:
            # Converter para escala de cinza para análise de contraste
            gray = cv2.cvtColor(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
            
            # Calcular desvio padrão como medida de contraste
            contrast = np.std(gray) / 255.0
            
            return min(contrast, 1.0)
        except Exception:
            return 0.0
    
    def _analyze_color_saturation(self, hsv_image: np.ndarray) -> float:
        """Analisa a saturação média das cores"""
        try:
            # Extrair canal de saturação
            saturation = hsv_image[:, :, 1]
            
            # Calcular saturação média
            avg_saturation = np.mean(saturation) / 255.0
            
            return min(avg_saturation, 1.0)
        except Exception:
            return 0.0
    
    def _analyze_shapes(self, image: np.ndarray) -> Dict[str, Any]:
        """Analisa formas básicas como uma criança reconheceria"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar contornos
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'error': 'Nenhum contorno encontrado'}
        
        # Encontrar maior contorno (assumindo que é o objeto principal)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Análise de proporções
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Análise de compactness
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        return {
            'aspect_ratio': aspect_ratio,
            'compactness': compactness,
            'area_ratio': area / (image.shape[0] * image.shape[1]) if (image.shape[0] * image.shape[1]) > 0 else 0,
            'bird_shape_score': self._calculate_shape_score(aspect_ratio, compactness)
        }
    
    def _analyze_textures(self, image: np.ndarray) -> Dict[str, Any]:
        """Analisa texturas como uma criança reconheceria"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Análise de textura usando gradientes
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calcular magnitude do gradiente
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Análise de uniformidade da textura
        texture_variance = np.var(magnitude)
        texture_uniformity = 1.0 / (1.0 + texture_variance) if texture_variance > 0 else 0.5
            
        return {
            'texture_variance': texture_variance,
            'texture_uniformity': texture_uniformity,
            'feather_like_score': self._calculate_feather_score(texture_variance, texture_uniformity)
        }
    
    def _calculate_bird_like_score(self, color_analysis: Dict, shape_analysis: Dict, texture_analysis: Dict) -> float:
        """Calcula score geral de características de pássaro - mais sensível"""
        color_score = color_analysis.get('bird_color_score', 0)
        shape_score = shape_analysis.get('bird_shape_score', 0)
        texture_score = texture_analysis.get('feather_like_score', 0)
        
        # Score baseado em características individuais
        base_score = (color_score * 0.3 + 
                     shape_score * 0.4 + 
                     texture_score * 0.3)
        
        # Bonus para múltiplas características positivas
        positive_features = sum([1 for score in [color_score, shape_score, texture_score] if score > 0.3])
        
        if positive_features >= 2:
            bonus = 0.2  # Bonus de 20% para múltiplas características
        elif positive_features >= 1:
            bonus = 0.1  # Bonus de 10% para pelo menos uma característica
        else:
            bonus = 0.0
        
        # Score final com bonus
        final_score = min(base_score + bonus, 1.0)
        
        return final_score
    
    def _calculate_shape_score(self, aspect_ratio: float, compactness: float) -> float:
        """Calcula score de forma baseado em características de pássaro - mais sensível"""
        # Pássaros típicos têm aspect_ratio entre 0.3 e 3.0 (mais amplo)
        if 0.3 <= aspect_ratio <= 3.0:
            aspect_score = 1.0
        elif 0.2 <= aspect_ratio <= 4.0:
            aspect_score = 0.8
        else:
            aspect_score = max(0, 1.0 - abs(aspect_ratio - 1.0) / 2.0)
        
        # Compactness típica de pássaros (formas arredondadas) - mais flexível
        if compactness > 0.1:  # Qualquer forma não muito alongada
            compactness_score = min(1.0, compactness * 2)  # Mais sensível
        else:
            compactness_score = compactness
        
        return (aspect_score + compactness_score) / 2
    
    def _calculate_feather_score(self, variance: float, uniformity: float) -> float:
        """Calcula score de textura de penas"""
        # Penas têm textura variada mas não muito uniforme
        variance_score = min(1.0, variance / 1000.0)  # Normalizar
        uniformity_score = uniformity
        
        return (variance_score + uniformity_score) / 2
    
    def _detect_fundamental_characteristics(self, image_path: str) -> Dict[str, Any]:
        """Detecção híbrida multi-biblioteca de características fundamentais de pássaros"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Imagem não carregada'}
            
            characteristics = {
                'has_wings': False,
                'has_beak': False,
                'has_claws': False,
                'has_feathers': False,
                'has_eyes': False,
                'bird_body_shape': False,
                'yolo_detection': False,
                'hybrid_confidence': 0.0,
                'mammal_score': 0.0,
                'bird_score': 0.0,
                'detection_votes': {},
                'model_consensus': 0.0
            }
            
            # 1. Detecção multi-biblioteca (YOLO, OpenCV, MediaPipe)
            detection_results = self._multi_library_detection(image)
            characteristics.update(detection_results)
            
            # 2. Análise visual híbrida (sempre executada - mantém funcionamento atual)
            visual_characteristics = self._detect_visual_characteristics(image)
            characteristics.update(visual_characteristics)
            
            # 3. Análise de contornos ultra-avançada (mantém funcionamento atual)
            contour_analysis = self._analyze_contours_ultra_advanced(image)
            characteristics.update(contour_analysis)
            
            # 4. Análise de textura ultra-híbrida (mantém funcionamento atual)
            texture_analysis = self._analyze_texture_ultra_hybrid(image)
            characteristics.update(texture_analysis)
            
            # 5. Análise de forma ultra-rigorosa (mantém funcionamento atual)
            shape_analysis = self._analyze_shape_ultra_rigorous(image)
            characteristics.update(shape_analysis)
            
            # 6. Análise de padrões biométricos (mantém funcionamento atual)
            biometric_analysis = self._analyze_biometric_patterns(image)
            characteristics.update(biometric_analysis)
            
            # 7. Sistema de votação ponderada entre todas as técnicas
            consensus_result = self._calculate_model_consensus(characteristics, detection_results)
            characteristics.update(consensus_result)
            
            # 8. Calcular scores de mamífero vs pássaro (melhorado com votação)
            mammal_score = self._calculate_mammal_score_enhanced(characteristics)
            bird_score = self._calculate_bird_score_enhanced(characteristics)
            
            characteristics['mammal_score'] = mammal_score
            characteristics['bird_score'] = bird_score
            
            # 9. Calcular confiança híbrida final
            hybrid_confidence = self._calculate_final_hybrid_confidence(
                characteristics, mammal_score, bird_score
            )
            characteristics['hybrid_confidence'] = hybrid_confidence
            
            return characteristics
            
        except Exception as e:
            self.debug_logger.log_error(f"❌ Erro na detecção de características: {str(e)}")
            return {'error': str(e)}
    
    def _detect_visual_characteristics(self, image: np.ndarray) -> Dict[str, Any]:
        """Detecção visual SIMPLIFICADA e EFICAZ"""
        try:
            # Análise básica de cores
            color_analysis = self._analyze_colors(image)
            
            # Análise básica de formas
            shape_analysis = self._analyze_shapes(image)
            
            # Análise básica de texturas
            texture_analysis = self._analyze_textures(image)
            
            # Detecção simples de características
            has_eyes = self._detect_simple_eyes(image)
            has_wings = self._detect_simple_wings(image)
            has_beak = self._detect_simple_beak(image)
            has_feathers = self._detect_simple_feathers(image)
            has_claws = self._detect_simple_claws(image)
            
            # Detecção simples de mamíferos
            has_mammal_features = self._detect_simple_mammal_features(image)
            
            # Calcular scores combinados
            bird_like_features = self._calculate_simple_bird_score(
                has_eyes, has_wings, has_beak, has_feathers, has_claws,
                color_analysis, shape_analysis, texture_analysis
            )
            
            bird_shape_score = self._calculate_simple_shape_score(shape_analysis)
            bird_color_score = color_analysis.get('bird_color_score', 0)
            
            return {
                'has_eyes': has_eyes,
                'has_wings': has_wings,
                'has_beak': has_beak,
                'has_feathers': has_feathers,
                'has_claws': has_claws,
                'has_mammal_features': has_mammal_features,
                'bird_like_features': bird_like_features,
                'bird_shape_score': bird_shape_score,
                'bird_color_score': bird_color_score,
                'color_analysis': color_analysis,
                'shape_analysis': shape_analysis,
                'texture_analysis': texture_analysis
            }
            
        except Exception as e:
            self.debug_logger.log_error(f"Erro na detecção visual: {str(e)}")
            return {
                'has_eyes': False,
                'has_wings': False,
                'has_beak': False,
                'has_feathers': False,
                'has_claws': False,
                'has_mammal_features': False,
                'bird_like_features': 0.0,
                'bird_shape_score': 0.0,
                'bird_color_score': 0.0,
                'error': str(e)
            }
    
    def _detect_simple_eyes(self, image: np.ndarray) -> bool:
        """Detecção RIGOROSA de olhos de pássaro"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Método 1: Detectar círculos pequenos e escuros (mais rigoroso)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 30, 
                                     param1=60, param2=40, minRadius=8, maxRadius=25)
            
            eye_count = 0
            if circles is not None:
                for circle in circles[0]:
                    x, y, r = circle
                    # Verificar se o círculo está em região escura (olho)
                    if y < image.shape[0] and x < image.shape[1]:
                        pixel_value = gray[int(y), int(x)]
                        if pixel_value < 80:  # Olhos são escuros
                            eye_count += 1
            
            # Método 2: Detectar regiões pequenas e escuras (mais rigoroso)
            _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            dark_regions = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 30 < area < 150:  # Tamanho mais restritivo
                    # Verificar se é circular
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.7:  # Mais circular
                            dark_regions += 1
            
            # Requer pelo menos 1 olho detectado por círculos OU 2 regiões escuras
            return eye_count >= 1 or dark_regions >= 2
            
        except:
            return False
    
    def _detect_simple_wings(self, image: np.ndarray) -> bool:
        """Detecção RIGOROSA de asas de pássaro"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Usar múltiplas técnicas de detecção de bordas
            edges1 = cv2.Canny(gray, 30, 100)
            edges2 = cv2.Canny(gray, 50, 150)
            combined_edges = cv2.bitwise_or(edges1, edges2)
            
            contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            wing_candidates = 0
            for contour in contours:
                if len(contour) > 10 and cv2.contourArea(contour) > 200:  # Mais rigoroso
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 1.0
                    
                    # Calcular características adicionais
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Asas são alongadas horizontalmente E têm forma específica
                    if (aspect_ratio > 2.0 and  # Mais alongadas
                        compactness > 0.1 and    # Não muito compactas
                        compactness < 0.6 and     # Mas não muito irregulares
                        area > 300):             # Área mínima maior
                        wing_candidates += 1
            
            # Requer pelo menos 1 candidato forte de asa
            return wing_candidates >= 1
            
        except:
            return False
    
    def _detect_simple_beak(self, image: np.ndarray) -> bool:
        """Detecção MELHORADA de bico de pássaro"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Método 1: Detecção por bordas (mais rigorosa)
            edges1 = cv2.Canny(gray, 30, 100)
            edges2 = cv2.Canny(gray, 50, 150)
            combined_edges = cv2.bitwise_or(edges1, edges2)
            
            contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            beak_candidates = 0
            for contour in contours:
                if len(contour) > 8 and cv2.contourArea(contour) > 80:  # Mais rigoroso
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 1.0
                    
                    # Calcular características adicionais
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Bicos são alongados, pontiagudos e têm forma específica
                    if (aspect_ratio > 2.5 and  # Mais alongados
                        compactness < 0.8 and  # Não muito compactos (pontiagudos)
                        area > 100 and        # Área mínima maior
                        area < 800):          # Não muito grandes
                        beak_candidates += 1
            
            # Método 2: Detecção por forma triangular (bicos têm formato triangular)
            for contour in contours:
                if len(contour) > 6 and cv2.contourArea(contour) > 60:
                    # Aproximar contorno por polígono
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Se tem 3 vértices, pode ser triangular (bico)
                    if len(approx) == 3:
                        beak_candidates += 1
            
            # Método 3: Detecção por posição (bicos estão na parte superior/frontal)
            h, w = image.shape[:2]
            for contour in contours:
                if len(contour) > 5 and cv2.contourArea(contour) > 50:
                    x, y, contour_w, contour_h = cv2.boundingRect(contour)
                    aspect_ratio = contour_w / contour_h if contour_h > 0 else 1.0
                    
                    # Bicos estão na parte superior da imagem
                    if (aspect_ratio > 2.0 and 
                        y < h * 0.3 and  # Parte superior
                        contour_w < w * 0.3):  # Não muito largos
                        beak_candidates += 1
            
            # Requer pelo menos 1 candidato forte de bico
            return beak_candidates >= 1
            
        except:
            return False
    
    def _detect_simple_feathers(self, image: np.ndarray) -> bool:
        """Detecção MELHORADA de penas para distinguir de pelo"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Método 1: Análise de textura usando gradientes (mais rigorosa)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            texture_variance = np.var(magnitude)
            texture_mean = np.mean(magnitude)
            
            # Método 2: Análise de padrões repetitivos (Fourier)
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Calcular energia em diferentes frequências
            h, w = magnitude_spectrum.shape
            center_y, center_x = h // 2, w // 2
            
            # Energia em frequências baixas (padrões grandes - pelo)
            low_freq_energy = np.sum(magnitude_spectrum[center_y-10:center_y+10, center_x-10:center_x+10])
            
            # Energia em frequências médias (padrões de penas)
            medium_freq_energy = np.sum(magnitude_spectrum[center_y-20:center_y+20, center_x-20:center_x+20]) - low_freq_energy
            
            # Energia total
            total_energy = np.sum(magnitude_spectrum)
            
            # Calcular ratios
            low_freq_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
            medium_freq_ratio = medium_freq_energy / total_energy if total_energy > 0 else 0
            
            # Método 3: Análise de densidade de bordas
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Método 4: Análise de contornos regulares (penas têm padrões mais regulares)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            regular_patterns = 0
            
            for contour in contours:
                if len(contour) > 10:
                    # Calcular regularidade do contorno
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    contour_area = cv2.contourArea(contour)
                    
                    if contour_area > 0:
                        solidity = contour_area / hull_area
                        # Penas têm contornos mais regulares que pelo
                        if 0.6 < solidity < 0.9:
                            regular_patterns += 1
            
            pattern_score = min(1.0, regular_patterns / 15.0)  # Normalizar
            
            # Critérios para distinguir penas de pelo
            feather_score = 0.0
            
            # Critério 1: Textura variável mas não muito (penas têm textura específica)
            if 150 < texture_variance < 500:  # Range mais específico
                feather_score += 0.3
            
            # Critério 2: Padrões de frequência média altos (penas)
            if medium_freq_ratio > 0.25:  # Mais rigoroso
                feather_score += 0.3
            
            # Critério 3: Densidade de bordas moderada (penas têm bordas definidas)
            if 0.08 < edge_density < 0.2:  # Range específico
                feather_score += 0.2
            
            # Critério 4: Padrões regulares (penas são mais regulares que pelo)
            if pattern_score > 0.3:  # Mais rigoroso
                feather_score += 0.2
            
            # Critério 5: Não deve ter muitos padrões de baixa frequência (pelo)
            if low_freq_ratio < 0.4:  # Evitar pelo
                feather_score += 0.1
            
            # Retornar True apenas se score for alto (penas) e não for pelo
            return feather_score > 0.6 and low_freq_ratio < 0.5
            
        except:
            return False
    
    def _detect_simple_claws(self, image: np.ndarray) -> bool:
        """Detecção MELHORADA de garras de pássaro"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Método 1: Detecção por bordas (mais rigorosa)
            edges1 = cv2.Canny(gray, 30, 100)
            edges2 = cv2.Canny(gray, 50, 150)
            combined_edges = cv2.bitwise_or(edges1, edges2)
            
            contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            claw_candidates = 0
            h, w = image.shape[:2]
            
            for contour in contours:
                if len(contour) > 6 and cv2.contourArea(contour) > 30:  # Mais rigoroso
                    x, y, contour_w, contour_h = cv2.boundingRect(contour)
                    aspect_ratio = contour_w / contour_h if contour_h > 0 else 1.0
                    area = cv2.contourArea(contour)
                    
                    # Garras são pequenas, alongadas e estão na parte inferior
                    if (aspect_ratio > 1.8 and  # Mais alongadas
                        area < 150 and         # Pequenas
                        area > 40 and          # Mas não muito pequenas
                        y > h * 0.6):          # Parte inferior da imagem
                        claw_candidates += 1
            
            # Método 2: Detecção por forma pontiaguda (garras são pontiagudas)
            for contour in contours:
                if len(contour) > 5 and cv2.contourArea(contour) > 25:
                    # Calcular convexidade
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    area = cv2.contourArea(contour)
                    
                    if hull_area > 0:
                        solidity = area / hull_area
                        
                        # Garras são menos convexas (mais pontiagudas)
                        if solidity < 0.7:  # Pontiagudas
                            x, y, contour_w, contour_h = cv2.boundingRect(contour)
                            aspect_ratio = contour_w / contour_h if contour_h > 0 else 1.0
                            
                            if (aspect_ratio > 1.5 and 
                                cv2.contourArea(contour) < 120 and
                                y > h * 0.5):  # Parte inferior
                                claw_candidates += 1
            
            # Método 3: Detecção por múltiplas pequenas estruturas (pés têm múltiplas garras)
            small_structures = 0
            for contour in contours:
                if len(contour) > 4 and cv2.contourArea(contour) > 20:
                    x, y, contour_w, contour_h = cv2.boundingRect(contour)
                    aspect_ratio = contour_w / contour_h if contour_h > 0 else 1.0
                    
                    if (aspect_ratio > 1.3 and 
                        cv2.contourArea(contour) < 80 and
                        y > h * 0.7):  # Parte muito inferior
                        small_structures += 1
            
            # Se há múltiplas pequenas estruturas na parte inferior, podem ser garras
            if small_structures >= 2:
                claw_candidates += 1
            
            # Requer pelo menos 1 candidato forte de garra
            return claw_candidates >= 1
            
        except:
            return False
    
    def _detect_simple_mammal_features(self, image: np.ndarray) -> bool:
        """Detecção ULTRA-RIGOROSA de características de mamíferos"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            mammal_features = 0
            
            # Detectar orelhas (formas arredondadas grandes) - EXTREMAMENTE RIGOROSO
            for contour in contours:
                if len(contour) > 15 and cv2.contourArea(contour) > 1000:  # EXTREMAMENTE maior
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    area = cv2.contourArea(contour)
                    
                    if hull_area > 0:
                        solidity = area / hull_area
                        
                        # Orelhas são EXTREMAMENTE convexas (solididade quase perfeita)
                        if solidity > 0.98:  # EXTREMAMENTE restritivo
                            mammal_features += 1
            
            # Detectar focinho (formas alongadas horizontalmente na parte inferior) - EXTREMAMENTE RIGOROSO
            h, w = image.shape[:2]
            for contour in contours:
                if len(contour) > 15 and cv2.contourArea(contour) > 500:  # EXTREMAMENTE maior
                    x, y, contour_w, contour_h = cv2.boundingRect(contour)
                    aspect_ratio = contour_w / contour_h if contour_h > 0 else 1.0
                    
                    # Focinho é EXTREMAMENTE alongado horizontalmente e está na parte inferior
                    if aspect_ratio > 4.0 and y > h * 0.85:  # EXTREMAMENTE restritivo
                        mammal_features += 1
            
            # Detectar nariz (pequenas formas circulares escuras) - EXTREMAMENTE RIGOROSO
            for contour in contours:
                if len(contour) > 10 and cv2.contourArea(contour) > 100:  # Mais rigoroso
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    area = cv2.contourArea(contour)
                    
                    if hull_area > 0:
                        solidity = area / hull_area
                        x, y, contour_w, contour_h = cv2.boundingRect(contour)
                        aspect_ratio = contour_w / contour_h if contour_h > 0 else 1.0
                        
                        # Nariz é EXTREMAMENTE circular e convexo
                        if 0.9 < aspect_ratio < 1.1 and solidity > 0.95:  # EXTREMAMENTE restritivo
                            mammal_features += 1
            
            # Retornar True apenas se encontrar características EXTREMAMENTE claras de mamífero
            return mammal_features >= 3  # Aumentado de 2 para 3
            
        except:
            return False
    
    def _calculate_simple_bird_score(self, has_eyes: bool, has_wings: bool, has_beak: bool, 
                                   has_feathers: bool, has_claws: bool,
                                   color_analysis: Dict, shape_analysis: Dict, texture_analysis: Dict) -> float:
        """Calcula score simples de características de pássaro"""
        score = 0.0
        
        # Contar características básicas
        characteristics = [has_eyes, has_wings, has_beak, has_feathers, has_claws]
        char_count = sum(characteristics)
        
        # Score baseado no número de características
        score += char_count * 0.2
        
        # Bonus por múltiplas características
        if char_count >= 3:
            score += 0.3
        elif char_count >= 2:
            score += 0.2
        
        # Adicionar scores de análise visual
        score += color_analysis.get('bird_color_score', 0) * 0.3
        score += shape_analysis.get('bird_shape_score', 0) * 0.3
        score += texture_analysis.get('bird_texture_score', 0) * 0.2
        
        return min(score, 1.0)
    
    def _calculate_simple_shape_score(self, shape_analysis: Dict) -> float:
        """Calcula score simples de forma"""
        return shape_analysis.get('bird_shape_score', 0)
    
    def _analyze_body_structure(self, image: np.ndarray) -> Dict[str, bool]:
        """Análise da estrutura corporal para distinguir pássaros de mamíferos"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar contornos principais
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'has_bird_body': False, 'has_mammal_body': False}
        
        # Encontrar maior contorno (corpo principal)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calcular características estruturais
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Compactness (circularidade)
        compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Convexidade
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Critérios para pássaros vs mamíferos
        has_bird_body = (
            0.3 < aspect_ratio < 2.5 and  # Proporções típicas de pássaros
            0.2 < compactness < 0.7 and   # Não muito circular, não muito alongado
            0.6 < solidity < 0.9          # Estrutura moderadamente convexa
        )
        
        has_mammal_body = (
            0.8 < aspect_ratio < 1.2 and  # Proporções típicas de mamíferos (mais quadrados)
            compactness > 0.6 and         # Mais circulares
            solidity > 0.85              # Muito convexos (formas arredondadas)
        )
        
        return {
            'has_bird_body': has_bird_body,
            'has_mammal_body': has_mammal_body,
            'body_aspect_ratio': aspect_ratio,
            'body_compactness': compactness,
            'body_solidity': solidity
        }
    
    def _analyze_advanced_texture(self, image: np.ndarray) -> Dict[str, bool]:
        """Análise avançada de textura para distinguir penas de pelo"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Análise de gradientes locais
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calcular estatísticas de textura
        texture_variance = np.var(magnitude)
        
        # 2. Análise de padrões repetitivos (Fourier)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Calcular energia em diferentes frequências
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # Energia em frequências baixas (padrões grandes - pelo)
        low_freq_energy = np.sum(magnitude_spectrum[center_y-10:center_y+10, center_x-10:center_x+10])
        
        # Energia em frequências médias (padrões de penas)
        medium_freq_energy = np.sum(magnitude_spectrum[center_y-20:center_y+20, center_x-20:center_x+20]) - low_freq_energy
        
        # Energia total
        total_energy = np.sum(magnitude_spectrum)
        
        # Calcular ratios
        low_freq_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
        medium_freq_ratio = medium_freq_energy / total_energy if total_energy > 0 else 0
        
        # 3. Análise de densidade de bordas
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Critérios para penas vs pelo
        has_feather_texture = (
            medium_freq_ratio > 0.3 and      # Padrões de penas dominantes
            edge_density > 0.1 and           # Alta densidade de bordas
            texture_variance > 100 and       # Alta variabilidade de textura
            low_freq_ratio < 0.4             # Poucos padrões grandes (pelo)
        )
        
        has_fur_texture = (
            low_freq_ratio > 0.5 and         # Padrões grandes dominantes (pelo)
            edge_density < 0.08 and          # Baixa densidade de bordas
            texture_variance < 200 and       # Baixa variabilidade de textura
            medium_freq_ratio < 0.2          # Poucos padrões de penas
        )
        
        return {
            'has_feather_texture': has_feather_texture,
            'has_fur_texture': has_fur_texture,
            'texture_variance': texture_variance,
            'edge_density': edge_density,
            'low_freq_ratio': low_freq_ratio,
            'medium_freq_ratio': medium_freq_ratio
        }
    
    def _analyze_body_proportions(self, image: np.ndarray) -> Dict[str, bool]:
        """Análise de proporções corporais para distinguir pássaros de mamíferos"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar contornos
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'has_bird_proportions': False, 'has_mammal_proportions': False}
        
        # Analisar todos os contornos significativos
        bird_features = 0
        mammal_features = 0
        
        for contour in contours:
            if cv2.contourArea(contour) < 100:  # Ignorar contornos muito pequenos
                continue
                
            # Calcular características do contorno
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 1.0
            
            # Compactness
            compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Critérios para pássaros (formas alongadas, moderadamente compactas)
            if 1.5 < aspect_ratio < 4.0 and 0.2 < compactness < 0.6:
                bird_features += 1
            
            # Critérios para mamíferos (formas mais arredondadas, muito compactas)
            elif 0.8 < aspect_ratio < 1.5 and compactness > 0.6:
                mammal_features += 1
        
        # Decisão baseada na contagem de características
        has_bird_proportions = bird_features > mammal_features and bird_features > 0
        has_mammal_proportions = mammal_features > bird_features and mammal_features > 0
        
        return {
            'has_bird_proportions': has_bird_proportions,
            'has_mammal_proportions': has_mammal_proportions,
            'bird_feature_count': bird_features,
            'mammal_feature_count': mammal_features
        }
    
    def _detect_bird_eyes(self, image: np.ndarray) -> bool:
        """Detecção específica de olhos de pássaros"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar círculos pequenos (olhos)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, 
                                 param1=50, param2=30, minRadius=3, maxRadius=25)
        
        if circles is not None:
            # Verificar se os círculos têm características de olhos de pássaro
            for circle in circles[0]:
                x, y, r = circle
                # Verificar intensidade (olhos são escuros)
                roi = gray[int(y-r):int(y+r), int(x-r):int(x+r)]
                if roi.size > 0:
                    mean_intensity = np.mean(roi)
                    if mean_intensity < 100:  # Olhos são escuros
                        return True
        
        return False
    
    def _detect_bird_wings(self, image: np.ndarray) -> bool:
        """Detecção específica de asas de pássaros"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        wing_count = 0
        for contour in contours:
            if len(contour) > 10 and cv2.contourArea(contour) > 200:
                # Ajustar elipse
                ellipse = cv2.fitEllipse(contour)
                (center, axes, angle) = ellipse
                major_axis, minor_axis = axes
                
                if major_axis > 0 and minor_axis > 0:
                    aspect_ratio = major_axis / minor_axis
                    # Asas são alongadas
                    if aspect_ratio > 2.0:
                        wing_count += 1
        
        return wing_count >= 1
    
    def _detect_bird_beak(self, image: np.ndarray) -> bool:
        """Detecção específica de bico de pássaros"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) > 5 and cv2.contourArea(contour) > 50:
                # Calcular convexidade
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                area = cv2.contourArea(contour)
                
                if hull_area > 0:
                    solidity = area / hull_area
                    
                    # Aspect ratio
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 1.0
                    
                    # Bicos são pontiagudos (baixa convexidade) e alongados
                    if solidity < 0.7 and aspect_ratio > 2.0:
                        return True
        
        return False
    
    def _detect_bird_feathers(self, image: np.ndarray) -> bool:
        """Detecção específica de penas de pássaros"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Análise de textura usando gradientes
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calcular variabilidade da textura
        texture_variance = np.var(magnitude)
        
        # Análise de padrões repetitivos
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Energia em frequências médias (padrões de penas)
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        medium_freq_energy = np.sum(magnitude_spectrum[center_y-15:center_y+15, center_x-15:center_x+15])
        total_energy = np.sum(magnitude_spectrum)
        medium_freq_ratio = medium_freq_energy / total_energy if total_energy > 0 else 0
        
        # Critérios para penas
        return texture_variance > 150 and medium_freq_ratio > 0.25
    
    def _detect_bird_claws(self, image: np.ndarray) -> bool:
        """Detecção específica de garras de pássaros"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        claw_count = 0
        for contour in contours:
            if len(contour) > 5 and cv2.contourArea(contour) > 20:
                # Calcular características de forma
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 1.0
                
                # Garras são pequenas, alongadas e pontiagudas
                if compactness < 0.4 and aspect_ratio > 1.5:
                    claw_count += 1
        
        return claw_count >= 1
    
    def _detect_mammal_features(self, image: np.ndarray) -> bool:
        """Detecção específica de características de mamíferos"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Detectar orelhas (formas triangulares ou arredondadas)
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ear_count = 0
        for contour in contours:
            if len(contour) > 5 and cv2.contourArea(contour) > 100:
                # Calcular convexidade
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                area = cv2.contourArea(contour)
                
                if hull_area > 0:
                    solidity = area / hull_area
                    
                    # Aspect ratio
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 1.0
                    
                    # Orelhas são convexas e aproximadamente circulares
                    if solidity > 0.8 and 0.7 < aspect_ratio < 1.3:
                        ear_count += 1
        
        # 2. Detectar focinho (forma alongada na parte inferior)
        # Análise da região inferior da imagem
        h, w = gray.shape
        bottom_region = gray[int(h*0.6):h, :]
        
        if bottom_region.size > 0:
            # Detectar bordas na região inferior
            bottom_edges = cv2.Canny(bottom_region, 30, 100)
            bottom_contours, _ = cv2.findContours(bottom_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            snout_count = 0
            for contour in bottom_contours:
                if len(contour) > 5 and cv2.contourArea(contour) > 50:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 1.0
                    
                    # Focinho é alongado horizontalmente
                    if aspect_ratio > 2.0:
                        snout_count += 1
            
            # Se encontrou orelhas ou focinho, provavelmente é mamífero
            return ear_count >= 1 or snout_count >= 1
        
        return ear_count >= 1
    
    def _detect_beak_shape(self, image: np.ndarray) -> bool:
        """Detecta formas pontiagudas que podem ser bicos"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar bordas
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        beak_like_shapes = 0
        for contour in contours:
            if len(contour) > 5:
                # Calcular convexidade
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                contour_area = cv2.contourArea(contour)
                
                if contour_area > 0:
                    solidity = contour_area / hull_area
                    # Formas pontiagudas têm baixa solididade
                    if solidity < 0.7:
                        # Verificar se tem forma alongada
                        ellipse = cv2.fitEllipse(contour)
                        (center, axes, orientation) = ellipse
                        major_axis = max(axes)
                        minor_axis = min(axes)
                        if minor_axis > 0 and major_axis / minor_axis > 2.0:
                            beak_like_shapes += 1
        
        return beak_like_shapes > 0
    
    def _detect_feather_texture(self, image: np.ndarray) -> bool:
        """Detecta textura de penas usando análise de padrões - mais rigoroso"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Método 1: Análise de Fourier para padrões repetitivos
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
        medium_freq_energy = np.sum(magnitude_spectrum[center_y-20:center_y+20, center_x-20:center_x+20])
        total_energy = np.sum(magnitude_spectrum)
        
        feather_score_fourier = medium_freq_energy / total_energy if total_energy > 0 else 0
        
        # Método 2: Análise de textura usando gradientes
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calcular variância da textura
        texture_variance = np.var(magnitude)
        texture_uniformity = 1.0 / (1.0 + texture_variance) if texture_variance > 0 else 0.5
        
        # Método 3: Análise de padrões repetitivos específicos de penas
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Método 4: Detectar padrões de penas específicos
        # Penas têm padrões mais regulares que pelo
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regular_patterns = 0
        
        for contour in contours:
            if len(contour) > 10:
                # Calcular regularidade do contorno
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                contour_area = cv2.contourArea(contour)
                
                if contour_area > 0:
                    solidity = contour_area / hull_area
                    # Penas têm contornos mais regulares que pelo
                    if 0.6 < solidity < 0.9:
                        regular_patterns += 1
        
        pattern_score = min(1.0, regular_patterns / 10.0)  # Normalizar
        
        # Combinar os métodos com pesos ajustados
        feather_score = (feather_score_fourier * 0.3 + 
                        texture_uniformity * 0.2 + 
                        edge_density * 0.2 +
                        pattern_score * 0.3)
        
        return feather_score > 0.15  # Threshold mais alto para evitar falsos positivos
    
    def _analyze_contours_advanced(self, image: np.ndarray) -> Dict[str, bool]:
        """Análise avançada de contornos para detectar características de pássaro"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Múltiplas técnicas de detecção de bordas
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges3 = cv2.Canny(gray, 100, 200)
        
        # Combinar bordas
        combined_edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
        
        # Encontrar contornos
        contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        characteristics = {
            'has_wings_advanced': False,
            'has_beak_advanced': False,
            'has_body_advanced': False,
            'contour_count': len(contours)
        }
        
        if not contours:
            return characteristics
        
        # Analisar cada contorno
        wing_like = 0
        beak_like = 0
        body_like = 0
        
        for contour in contours:
            if len(contour) < 5:
                continue
                
            # Calcular características do contorno
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area < 100:  # Ignorar contornos muito pequenos
                continue
            
            # Calcular aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 1.0
            
            # Calcular compactness
            compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Calcular convexidade
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Detectar asas (formas alongadas)
            if 2.0 < aspect_ratio < 6.0 and compactness > 0.1:
                wing_like += 1
            
            # Detectar bico (formas pontiagudas e alongadas)
            if aspect_ratio > 3.0 and solidity < 0.8:
                beak_like += 1
            
            # Detectar corpo (formas arredondadas)
            if 0.5 < aspect_ratio < 2.0 and compactness > 0.3:
                body_like += 1
        
        characteristics['has_wings_advanced'] = wing_like > 0
        characteristics['has_beak_advanced'] = beak_like > 0
        characteristics['has_body_advanced'] = body_like > 0
        
        return characteristics
    
    def _analyze_texture_hybrid(self, image: np.ndarray) -> Dict[str, bool]:
        """Análise híbrida de textura para distinguir penas de pelo"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        characteristics = {
            'has_feathers_hybrid': False,
            'has_fur_texture': False,
            'texture_regularity': 0.0
        }
        
        # Método 1: Análise de gradientes locais
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calcular regularidade da textura
        texture_variance = np.var(magnitude)
        texture_mean = np.mean(magnitude)
        
        # Método 2: Análise de padrões repetitivos
        # Usar transformada de Fourier para detectar padrões
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Calcular energia em diferentes frequências
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # Energia em frequências baixas (padrões grandes)
        low_freq_energy = np.sum(magnitude_spectrum[center_y-10:center_y+10, center_x-10:center_x+10])
        
        # Energia em frequências médias (padrões de penas)
        medium_freq_energy = np.sum(magnitude_spectrum[center_y-20:center_y+20, center_x-20:center_x+20]) - low_freq_energy
        
        # Energia total
        total_energy = np.sum(magnitude_spectrum)
        
        # Calcular scores
        low_freq_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
        medium_freq_ratio = medium_freq_energy / total_energy if total_energy > 0 else 0
        
        # Método 3: Análise de bordas locais
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Critérios para distinguir penas de pelo
        # Penas: padrões mais regulares, frequências médias altas
        # Pelo: padrões menos regulares, frequências baixas altas
        
        feather_score = (
            medium_freq_ratio * 0.4 +  # Padrões de penas
            edge_density * 0.3 +       # Densidade de bordas
            (1.0 - low_freq_ratio) * 0.3  # Menos padrões grandes
        )
        
        fur_score = (
            low_freq_ratio * 0.5 +     # Padrões grandes (pelo)
            (1.0 - medium_freq_ratio) * 0.3 +  # Menos padrões de penas
            (1.0 - edge_density) * 0.2  # Menos bordas
        )
        
        characteristics['has_feathers_hybrid'] = feather_score > 0.3
        characteristics['has_fur_texture'] = fur_score > 0.4
        characteristics['texture_regularity'] = feather_score
        
        return characteristics
    
    def _calculate_hybrid_confidence(self, characteristics: Dict, yolo_detections: int, visual_characteristics: Dict) -> float:
        """Calcula confiança híbrida baseada em múltiplas técnicas"""
        confidence_factors = []
        
        # Fator 1: Detecção YOLO
        if characteristics.get('yolo_detection', False):
            confidence_factors.append(0.8)  # Alta confiança se YOLO detectou
        
        # Fator 2: Características visuais básicas
        visual_count = sum([
            characteristics.get('has_wings', False),
            characteristics.get('has_beak', False),
            characteristics.get('has_feathers', False),
            characteristics.get('has_eyes', False)
        ])
        
        if visual_count >= 3:
            confidence_factors.append(0.9)
        elif visual_count >= 2:
            confidence_factors.append(0.7)
        elif visual_count >= 1:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.2)
        
        # Fator 3: Análise avançada de contornos
        advanced_count = sum([
            characteristics.get('has_wings_advanced', False),
            characteristics.get('has_beak_advanced', False),
            characteristics.get('has_body_advanced', False)
        ])
        
        if advanced_count >= 2:
            confidence_factors.append(0.8)
        elif advanced_count >= 1:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
        
        # Fator 4: Análise de textura
        if characteristics.get('has_feathers_hybrid', False):
            confidence_factors.append(0.7)
        elif characteristics.get('has_fur_texture', False):
            confidence_factors.append(0.1)  # Baixa confiança para pelo
        else:
            confidence_factors.append(0.4)
        
        # Calcular confiança média ponderada
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.0
    
    def _logical_reasoning(self, visual_analysis: Dict, characteristics: Dict) -> Dict[str, Any]:
        """Raciocínio lógico neuro-simbólico SIMPLIFICADO e EFICAZ"""
        reasoning = {
            'is_bird': False,
            'confidence': 0.0,
            'species': 'Desconhecida',
            'reasoning_steps': [],
            'characteristics_found': [],
            'missing_characteristics': [],
            'intuition_level': 'Baixa',
            'needs_manual_review': False
        }
        
        # Extrair dados básicos
        has_wings = characteristics.get('has_wings', False)
        has_beak = characteristics.get('has_beak', False)
        has_feathers = characteristics.get('has_feathers', False)
        has_eyes = characteristics.get('has_eyes', False)
        has_claws = characteristics.get('has_claws', False)
        
        bird_shape_score = visual_analysis.get('bird_shape_score', 0)
        bird_color_score = visual_analysis.get('bird_color_score', 0)
        bird_like_features = visual_analysis.get('bird_like_features', 0)
        
        # Detectar características de mamíferos
        has_mammal_features = characteristics.get('has_mammal_features', False)
        has_mammal_body = characteristics.get('has_mammal_body', False)
        has_fur_texture = characteristics.get('has_fur_texture', False)
        
        # Contar características de pássaro encontradas
        bird_characteristics = [has_wings, has_beak, has_feathers, has_eyes, has_claws]
        bird_count = sum(bird_characteristics)
        
        # Listar características encontradas
        if has_wings: reasoning['characteristics_found'].append('asas')
        if has_beak: reasoning['characteristics_found'].append('bico')
        if has_feathers: reasoning['characteristics_found'].append('penas')
        if has_eyes: reasoning['characteristics_found'].append('olhos')
        if has_claws: reasoning['characteristics_found'].append('garras')
        
        # LÓGICA SIMPLIFICADA E EFICAZ - PRIORIZANDO PÁSSAROS
        
        # 1. PRIMEIRO: Verificar características definitivas de pássaro (PRIORIDADE MÁXIMA)
        if bird_count >= 3:
            reasoning['is_bird'] = True
            reasoning['confidence'] = 0.9
            reasoning['intuition_level'] = 'Alta'
            reasoning['reasoning_steps'].append(f"✅ {bird_count} características definitivas de pássaro detectadas")
            
        # 2. SEGUNDO: Verificar características moderadas + análise visual
        elif bird_count >= 2 and bird_like_features > 0.4:
            reasoning['is_bird'] = True
            reasoning['confidence'] = 0.8
            reasoning['intuition_level'] = 'Alta'
            reasoning['reasoning_steps'].append(f"✅ {bird_count} características + análise visual positiva")
            
        # 3. TERCEIRO: Verificar características básicas + forma/cores adequadas
        elif bird_count >= 1 and (bird_shape_score > 0.4 or bird_color_score > 0.4):
            reasoning['is_bird'] = True
            reasoning['confidence'] = 0.7
            reasoning['intuition_level'] = 'Média'
            reasoning['reasoning_steps'].append("✅ Características básicas + forma/cores adequadas")
            
        # 4. QUARTO: Verificar análise visual muito positiva
        elif bird_like_features > 0.5 and (has_eyes or has_wings):
            reasoning['is_bird'] = True
            reasoning['confidence'] = 0.6
            reasoning['intuition_level'] = 'Média'
            reasoning['reasoning_steps'].append("✅ Análise visual muito positiva")
            
        # 5. QUINTO: Verificar análise visual moderada
        elif bird_like_features > 0.4 and (bird_shape_score > 0.3 or bird_color_score > 0.3):
            reasoning['is_bird'] = True
            reasoning['confidence'] = 0.5
            reasoning['intuition_level'] = 'Média'
            reasoning['reasoning_steps'].append("✅ Análise visual moderada")
            
        # 5.5. QUINTO E MEIO: Casos com forma perfeita de pássaro (prioridade máxima)
        elif bird_shape_score >= 1.0 and bird_color_score > 0.2:
            reasoning['is_bird'] = True
            reasoning['confidence'] = 0.5
            reasoning['intuition_level'] = 'Média'
            reasoning['needs_manual_review'] = True
            reasoning['reasoning_steps'].append("✅ Forma perfeita de pássaro detectada")
            
        # 6. SEXTO: Casos duvidosos - pode ser pássaro (MAIS RIGOROSO)
        elif bird_count >= 2 and bird_like_features > 0.4:
            reasoning['is_bird'] = True
            reasoning['confidence'] = 0.4
            reasoning['intuition_level'] = 'Baixa'
            reasoning['needs_manual_review'] = True
            reasoning['reasoning_steps'].append("❓ Caso duvidoso - recomenda análise manual")
            
        # 6.5. SEXTO E MEIO: Casos com 1 característica mas análise visual muito forte
        elif bird_count >= 1 and bird_like_features > 0.6 and (bird_shape_score > 0.7 or bird_color_score > 0.7):
            reasoning['is_bird'] = True
            reasoning['confidence'] = 0.4
            reasoning['intuition_level'] = 'Baixa'
            reasoning['needs_manual_review'] = True
            reasoning['reasoning_steps'].append("❓ Uma característica + análise visual muito forte")
            
        # 6.6. SEXTO E MEIO: Casos com análise visual extremamente forte (mesmo sem características específicas)
        elif bird_shape_score > 0.9 and bird_color_score > 0.2 and bird_like_features > 0.3:
            reasoning['is_bird'] = True
            reasoning['confidence'] = 0.4
            reasoning['intuition_level'] = 'Baixa'
            reasoning['needs_manual_review'] = True
            reasoning['reasoning_steps'].append("❓ Análise visual extremamente forte (forma perfeita)")
            
        # 7. SÉTIMO: Verificar se é definitivamente um mamífero (APENAS SE NÃO FOR PÁSSARO)
        elif has_mammal_features and not (has_wings or has_beak or has_feathers):
            reasoning['is_bird'] = False
            reasoning['confidence'] = 0.9
            reasoning['intuition_level'] = 'Alta'
            reasoning['reasoning_steps'].append("❌ Detectadas características específicas de mamíferos")
            
        # 8. OITAVO: Provavelmente não é pássaro
        else:
            reasoning['is_bird'] = False
            reasoning['confidence'] = 0.2
            reasoning['intuition_level'] = 'Baixa'
            reasoning['reasoning_steps'].append("❌ Poucas evidências de características de pássaro")
        
        # Determinar espécie (se for pássaro)
        if reasoning['is_bird']:
            species = self._determine_species(visual_analysis, characteristics)
            reasoning['species'] = species
            reasoning['reasoning_steps'].append(f"🐦 Espécie identificada: {species}")
        else:
            reasoning['species'] = 'Não-Pássaro'
            reasoning['reasoning_steps'].append("🚫 Não é um pássaro")
        
        # Calcular confiança geral
        reasoning['overall_confidence'] = reasoning['confidence']
        
        return reasoning
    
    def _advanced_cognitive_analysis(self, visual_analysis: Dict, characteristics: Dict, reasoning: Dict) -> Dict[str, float]:
        """Análise cognitiva avançada neuro-simbólica"""
        analysis = {
            'pattern_recognition': 0.0,
            'logical_inference': 0.0,
            'uncertainty_handling': 0.0,
            'adaptive_thinking': 0.0
        }
        
        try:
            # 1. Reconhecimento de Padrões
            pattern_score = 0.0
            
            # Padrões visuais
            if visual_analysis.get('bird_shape_score', 0) > 0.5:
                pattern_score += 0.3
            if visual_analysis.get('bird_color_score', 0) > 0.5:
                pattern_score += 0.3
            if visual_analysis.get('bird_like_features', 0) > 0.6:
                pattern_score += 0.4
            
            analysis['pattern_recognition'] = min(pattern_score, 1.0)
            
            # 2. Inferência Lógica
            logic_score = 0.0
            
            # Lógica baseada em características
            fundamental_count = len(reasoning.get('characteristics_found', []))
            if fundamental_count >= 3:
                logic_score += 0.4
            elif fundamental_count >= 2:
                logic_score += 0.3
            elif fundamental_count >= 1:
                logic_score += 0.2
            
            # Lógica baseada em confiança
            confidence = reasoning.get('confidence', 0)
            if confidence > 0.8:
                logic_score += 0.4
            elif confidence > 0.6:
                logic_score += 0.3
            elif confidence > 0.4:
                logic_score += 0.2
            
            analysis['logical_inference'] = min(logic_score, 1.0)
            
            # 3. Tratamento de Incerteza
            uncertainty_score = 0.0
            
            # Quanto mais características, menor a incerteza
            total_chars = len(reasoning.get('characteristics_found', [])) + len(reasoning.get('missing_characteristics', []))
            if total_chars > 0:
                certainty_ratio = len(reasoning.get('characteristics_found', [])) / total_chars
                uncertainty_score = 1.0 - abs(0.5 - certainty_ratio) * 2
            
            analysis['uncertainty_handling'] = uncertainty_score
            
            # 4. Pensamento Adaptativo
            adaptive_score = 0.0
            
            # Adaptabilidade baseada na análise visual
            if visual_analysis.get('bird_like_features', 0) > 0.7:
                adaptive_score += 0.3
            if reasoning.get('needs_manual_review', False):
                adaptive_score += 0.2  # Reconhece quando precisa de ajuda
            if reasoning.get('intuition_level') == 'Alta':
                adaptive_score += 0.3
            elif reasoning.get('intuition_level') == 'Média':
                adaptive_score += 0.2
            
            analysis['adaptive_thinking'] = min(adaptive_score, 1.0)
            
        except Exception as e:
            logger.warning(f"⚠️ Erro na análise cognitiva: {e}")
        
        return analysis
    
    def _calculate_neuro_symbolic_score(self, reasoning: Dict, cognitive_analysis: Dict) -> float:
        """Calcula score neuro-simbólico geral"""
        try:
            # Componentes do score
            confidence_component = reasoning.get('confidence', 0) * 0.3
            pattern_component = cognitive_analysis.get('pattern_recognition', 0) * 0.25
            logic_component = cognitive_analysis.get('logical_inference', 0) * 0.25
            uncertainty_component = cognitive_analysis.get('uncertainty_handling', 0) * 0.1
            adaptive_component = cognitive_analysis.get('adaptive_thinking', 0) * 0.1
            
            # Score final
            neuro_symbolic_score = (
                confidence_component + 
                pattern_component + 
                logic_component + 
                uncertainty_component + 
                adaptive_component
            )
            
            return min(neuro_symbolic_score, 1.0)
            
        except Exception as e:
            logger.warning(f"⚠️ Erro no cálculo do score neuro-simbólico: {e}")
            return 0.0
    
    def _assess_learning_potential(self, reasoning: Dict, characteristics: Dict) -> str:
        """Avalia o potencial de aprendizado da análise"""
        try:
            score = 0.0
            
            # Fatores que indicam potencial de aprendizado
            if reasoning.get('needs_manual_review', False):
                score += 0.3  # Casos que precisam de revisão são bons para aprender
            
            if reasoning.get('confidence', 0) < 0.6:
                score += 0.2  # Baixa confiança indica necessidade de aprendizado
            
            if len(reasoning.get('missing_characteristics', [])) > 0:
                score += 0.2  # Características faltantes indicam oportunidade de aprendizado
            
            if reasoning.get('intuition_level') == 'Baixa':
                score += 0.3  # Baixa intuição indica potencial de melhoria
            
            # Determinar nível
            if score >= 0.7:
                return 'Alto'
            elif score >= 0.4:
                return 'Médio'
            else:
                return 'Baixo'
                
        except Exception as e:
            logger.warning(f"⚠️ Erro na avaliação do potencial de aprendizado: {e}")
            return 'Baixo'
    
    def _assess_certainty_level(self, reasoning: Dict) -> str:
        """Avalia o nível de certeza da análise"""
        try:
            confidence = reasoning.get('confidence', 0)
            neuro_score = reasoning.get('neuro_symbolic_score', 0)
            
            # Calcular certeza combinada
            combined_certainty = (confidence + neuro_score) / 2
            
            if combined_certainty >= 0.8:
                return 'Alta Certeza'
            elif combined_certainty >= 0.6:
                return 'Certeza Moderada'
            elif combined_certainty >= 0.4:
                return 'Incerteza Moderada'
            else:
                return 'Alta Incerteza'
                
        except Exception as e:
            logger.warning(f"⚠️ Erro na avaliação do nível de certeza: {e}")
            return 'Incerteza'
    
    def _has_mammal_characteristics(self, visual_analysis: Dict, characteristics: Dict) -> bool:
        """Detecção híbrida de características típicas de mamíferos (cachorros, gatos, etc.)"""
        # Verificar características que indicam mamífero
        
        # 1. Ausência total de características de pássaro
        has_wings = characteristics.get('has_wings', False)
        has_beak = characteristics.get('has_beak', False)
        has_feathers = characteristics.get('has_feathers', False)
        has_wings_advanced = characteristics.get('has_wings_advanced', False)
        has_beak_advanced = characteristics.get('has_beak_advanced', False)
        
        bird_characteristics_count = sum([has_wings, has_beak, has_feathers, has_wings_advanced, has_beak_advanced])
        
        # 2. Análise de cores típicas de mamíferos
        dominant_color = visual_analysis.get('dominant_color', 'unknown')
        mammal_colors = ['brown', 'black', 'white', 'gray']
        
        # 3. Análise de forma - mamíferos têm formas mais arredondadas
        shape_score = visual_analysis.get('bird_shape_score', 0)
        
        # 4. Análise de textura híbrida - pelo vs penas
        has_feathers_hybrid = characteristics.get('has_feathers_hybrid', False)
        has_fur_texture = characteristics.get('has_fur_texture', False)
        texture_regularity = characteristics.get('texture_regularity', 0)
        
        # 5. Análise de contornos - mamíferos têm menos contornos complexos
        contour_count = characteristics.get('contour_count', 0)
        
        # Critérios rigorosos para identificar mamífero:
        mammal_score = 0
        
        # Critério 1: Nenhuma característica de pássaro detectada
        if bird_characteristics_count == 0:
            mammal_score += 2
        
        # Critério 2: Cor típica de mamífero
        if dominant_color in mammal_colors:
            mammal_score += 1
        
        # Critério 3: Forma muito arredondada (não típica de pássaro)
        if shape_score > 0.7:
            mammal_score += 1
        
        # Critério 4: Textura de pelo detectada
        if has_fur_texture:
            mammal_score += 2
        
        # Critério 5: Ausência de textura de penas
        if not has_feathers_hybrid:
            mammal_score += 1
        
        # Critério 6: Textura irregular (típica de pelo)
        if texture_regularity < 0.3:
            mammal_score += 1
        
        # Critério 7: Poucos contornos complexos (típico de mamíferos)
        if contour_count < 10:
            mammal_score += 1
        
        # Considerar mamífero se score >= 4 (de 7 possíveis)
        is_mammal = mammal_score >= 4
        
        return is_mammal
    
    def _analyze_contours_ultra_advanced(self, image: np.ndarray) -> Dict[str, Any]:
        """Análise ultra-avançada de contornos para detectar características de pássaro"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Múltiplas técnicas de detecção de bordas com diferentes parâmetros
        edges_configs = [
            (30, 100), (50, 150), (100, 200), (150, 300),
            (20, 80), (80, 160), (120, 240)
        ]
        
        all_edges = []
        for low, high in edges_configs:
            edges = cv2.Canny(gray, low, high)
            all_edges.append(edges)
        
        # Combinar todas as bordas
        combined_edges = all_edges[0]
        for edges in all_edges[1:]:
            combined_edges = cv2.bitwise_or(combined_edges, edges)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        characteristics = {
            'has_wings_ultra': False,
            'has_beak_ultra': False,
            'has_body_ultra': False,
            'contour_complexity': 0.0,
            'symmetry_score': 0.0,
            'edge_density': 0.0
        }
        
        if not contours:
            return characteristics
        
        # Calcular densidade de bordas
        edge_density = np.sum(combined_edges > 0) / (combined_edges.shape[0] * combined_edges.shape[1])
        characteristics['edge_density'] = edge_density
        
        # Analisar cada contorno com critérios ultra-rigorosos
        wing_like = 0
        beak_like = 0
        body_like = 0
        complexity_scores = []
        symmetry_scores = []
        
        for contour in contours:
            if len(contour) < 10:  # Ignorar contornos muito pequenos
                continue
                
            # Calcular características do contorno
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area < 200:  # Threshold mais alto para área mínima
                continue
            
            # Calcular aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 1.0
            
            # Calcular compactness
            compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Calcular convexidade
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Calcular complexidade (número de vértices vs área)
            complexity = len(contour) / area if area > 0 else 0
            complexity_scores.append(complexity)
            
            # Calcular simetria
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                
                # Verificar simetria horizontal
                left_points = [p for p in contour if p[0][0] < cx]
                right_points = [p for p in contour if p[0][0] > cx]
                
                if len(left_points) > 0 and len(right_points) > 0:
                    symmetry = min(len(left_points), len(right_points)) / max(len(left_points), len(right_points))
                    symmetry_scores.append(symmetry)
            
            # Critérios ultra-rigorosos para detectar características
            
            # Detectar asas (formas alongadas com complexidade moderada)
            if (2.5 < aspect_ratio < 8.0 and 
                0.1 < compactness < 0.4 and 
                0.3 < solidity < 0.8 and
                0.01 < complexity < 0.05):
                wing_like += 1
            
            # Detectar bico (formas pontiagudas e alongadas)
            if (aspect_ratio > 4.0 and 
                solidity < 0.7 and 
                compactness < 0.3 and
                complexity > 0.02):
                beak_like += 1
            
            # Detectar corpo (formas arredondadas com simetria)
            if (0.6 < aspect_ratio < 1.8 and 
                compactness > 0.4 and 
                solidity > 0.8 and
                complexity < 0.02):
                body_like += 1
        
        characteristics['has_wings_ultra'] = wing_like >= 2  # Pelo menos 2 formas de asa
        characteristics['has_beak_ultra'] = beak_like >= 1  # Pelo menos 1 forma de bico
        characteristics['has_body_ultra'] = body_like >= 1  # Pelo menos 1 forma de corpo
        
        # Calcular scores de complexidade e simetria
        if complexity_scores:
            characteristics['contour_complexity'] = np.mean(complexity_scores)
        if symmetry_scores:
            characteristics['symmetry_score'] = np.mean(symmetry_scores)
        
        return characteristics
    
    def _analyze_texture_ultra_hybrid(self, image: np.ndarray) -> Dict[str, Any]:
        """Análise ultra-híbrida de textura para distinguir penas de pelo"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        characteristics = {
            'has_feathers_ultra': False,
            'has_fur_ultra': False,
            'texture_regularity_ultra': 0.0,
            'pattern_frequency_score': 0.0,
            'texture_directionality': 0.0
        }
        
        # Método 1: Análise de gradientes locais avançada
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Calcular regularidade da textura
        texture_variance = np.var(magnitude)
        texture_mean = np.mean(magnitude)
        texture_std = np.std(magnitude)
        
        # Método 2: Análise de padrões repetitivos ultra-avançada
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Calcular energia em diferentes frequências
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # Energia em frequências baixas (padrões grandes - pelo)
        low_freq_energy = np.sum(magnitude_spectrum[center_y-15:center_y+15, center_x-15:center_x+15])
        
        # Energia em frequências médias (padrões de penas)
        medium_freq_energy = np.sum(magnitude_spectrum[center_y-30:center_y+30, center_x-30:center_x+30]) - low_freq_energy
        
        # Energia em frequências altas (detalhes finos)
        high_freq_energy = np.sum(magnitude_spectrum) - medium_freq_energy - low_freq_energy
        
        # Energia total
        total_energy = np.sum(magnitude_spectrum)
        
        # Calcular scores
        low_freq_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
        medium_freq_ratio = medium_freq_energy / total_energy if total_energy > 0 else 0
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # Método 3: Análise de direcionalidade da textura
        # Calcular histograma de direções
        hist, _ = np.histogram(direction.flatten(), bins=36, range=(-np.pi, np.pi))
        directionality = np.std(hist) / np.mean(hist) if np.mean(hist) > 0 else 0
        
        # Método 4: Análise de bordas locais ultra-avançada
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Método 5: Análise de padrões de repetição
        # Usar correlação para detectar padrões repetitivos
        correlation_scores = []
        for i in range(0, gray.shape[0] - 20, 20):
            for j in range(0, gray.shape[1] - 20, 20):
                patch = gray[i:i+20, j:j+20]
                if patch.size > 0:
                    # Calcular autocorrelação
                    corr = cv2.matchTemplate(patch, patch, cv2.TM_CCOEFF_NORMED)
                    correlation_scores.append(np.max(corr))
        
        pattern_frequency = np.mean(correlation_scores) if correlation_scores else 0
        
        # Critérios ultra-rigorosos para distinguir penas de pelo
        
        # Score para penas (padrões regulares, frequências médias altas)
        feather_score = (
            medium_freq_ratio * 0.3 +      # Padrões de penas
            edge_density * 0.2 +           # Densidade de bordas
            (1.0 - low_freq_ratio) * 0.2 + # Menos padrões grandes
            pattern_frequency * 0.2 +      # Padrões repetitivos
            directionality * 0.1           # Direcionalidade
        )
        
        # Score para pelo (padrões irregulares, frequências baixas altas)
        fur_score = (
            low_freq_ratio * 0.4 +         # Padrões grandes (pelo)
            (1.0 - medium_freq_ratio) * 0.3 + # Menos padrões de penas
            (1.0 - edge_density) * 0.2 +   # Menos bordas
            (1.0 - pattern_frequency) * 0.1 # Menos padrões repetitivos
        )
        
        characteristics['has_feathers_ultra'] = feather_score > 0.4  # Threshold mais alto
        characteristics['has_fur_ultra'] = fur_score > 0.5  # Threshold mais alto
        characteristics['texture_regularity_ultra'] = feather_score
        characteristics['pattern_frequency_score'] = pattern_frequency
        characteristics['texture_directionality'] = directionality
        
        return characteristics
    
    def _analyze_shape_ultra_rigorous(self, image: np.ndarray) -> Dict[str, Any]:
        """Análise ultra-rigorosa de formas para detectar características de pássaro"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar contornos principais
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        characteristics = {
            'bird_shape_score_ultra': 0.0,
            'mammal_shape_score_ultra': 0.0,
            'elongation_factor': 0.0,
            'roundness_factor': 0.0,
            'compactness_factor': 0.0
        }
        
        if not contours:
            return characteristics
        
        # Encontrar maior contorno (assumindo que é o objeto principal)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calcular características de forma
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Calcular aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Calcular compactness
        compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Calcular elongação
        elongation = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
        
        # Calcular redondez
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        roundness = area / hull_area if hull_area > 0 else 0
        
        characteristics['elongation_factor'] = elongation
        characteristics['roundness_factor'] = roundness
        characteristics['compactness_factor'] = compactness
        
        # Critérios ultra-rigorosos para pássaros vs mamíferos
        
        # Score para pássaros (formas alongadas, moderadamente compactas)
        bird_score = 0.0
        if 0.4 < aspect_ratio < 2.5:  # Aspect ratio típico de pássaros
            bird_score += 0.3
        if 0.2 < compactness < 0.6:  # Compactness típica de pássaros
            bird_score += 0.3
        if 0.7 < roundness < 0.95:  # Redondez típica de pássaros
            bird_score += 0.2
        if 1.2 < elongation < 3.0:  # Elongação típica de pássaros
            bird_score += 0.2
        
        # Score para mamíferos (formas muito arredondadas, muito compactas)
        mammal_score = 0.0
        if 0.8 < aspect_ratio < 1.2:  # Aspect ratio típico de mamíferos
            mammal_score += 0.4
        if compactness > 0.6:  # Compactness típica de mamíferos
            mammal_score += 0.3
        if roundness > 0.95:  # Redondez típica de mamíferos
            mammal_score += 0.3
        
        characteristics['bird_shape_score_ultra'] = bird_score
        characteristics['mammal_shape_score_ultra'] = mammal_score
        
        return characteristics
    
    def _analyze_biometric_patterns(self, image: np.ndarray) -> Dict[str, Any]:
        """Análise de padrões biométricos para distinguir pássaros de mamíferos"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        characteristics = {
            'biometric_bird_score': 0.0,
            'biometric_mammal_score': 0.0,
            'feature_density': 0.0,
            'structural_complexity': 0.0
        }
        
        # Método 1: Análise de densidade de características
        # Detectar cantos (Harris corner detection)
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        corner_count = np.sum(corners > 0.01 * corners.max())
        feature_density = corner_count / (gray.shape[0] * gray.shape[1])
        
        characteristics['feature_density'] = feature_density
        
        # Método 2: Análise de complexidade estrutural
        # Usar Laplaciano para detectar bordas
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        structural_complexity = np.var(laplacian)
        
        characteristics['structural_complexity'] = structural_complexity
        
        # Método 3: Análise de padrões de iluminação
        # Calcular histograma de intensidades
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist.ravel() / hist.sum()
        
        # Calcular entropia (medida de complexidade)
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
        
        # Método 4: Análise de textura usando LBP (Local Binary Patterns)
        # Implementação simplificada de LBP
        lbp_patterns = []
        for i in range(1, gray.shape[0] - 1):
            for j in range(1, gray.shape[1] - 1):
                center = gray[i, j]
                pattern = 0
                for k, (di, dj) in enumerate([(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]):
                    if gray[i + di, j + dj] >= center:
                        pattern |= (1 << k)
                lbp_patterns.append(pattern)
        
        # Calcular uniformidade dos padrões LBP
        lbp_uniformity = len(set(lbp_patterns)) / len(lbp_patterns) if lbp_patterns else 0
        
        # Critérios biométricos para pássaros vs mamíferos
        
        # Score para pássaros (alta complexidade estrutural, padrões variados)
        bird_score = 0.0
        if feature_density > 0.001:  # Alta densidade de características
            bird_score += 0.3
        if structural_complexity > 100:  # Alta complexidade estrutural
            bird_score += 0.3
        if entropy > 6.0:  # Alta entropia (padrões variados)
            bird_score += 0.2
        if lbp_uniformity < 0.8:  # Padrões não uniformes (penas)
            bird_score += 0.2
        
        # Score para mamíferos (baixa complexidade estrutural, padrões uniformes)
        mammal_score = 0.0
        if feature_density < 0.0005:  # Baixa densidade de características
            mammal_score += 0.3
        if structural_complexity < 50:  # Baixa complexidade estrutural
            mammal_score += 0.3
        if entropy < 5.0:  # Baixa entropia (padrões uniformes)
            mammal_score += 0.2
        if lbp_uniformity > 0.9:  # Padrões uniformes (pelo)
            mammal_score += 0.2
        
        characteristics['biometric_bird_score'] = bird_score
        characteristics['biometric_mammal_score'] = mammal_score
        
        return characteristics
    
    def _calculate_mammal_score(self, characteristics: Dict) -> float:
        """Calcula score de mamífero baseado em múltiplas características"""
        score = 0.0
        total_weight = 0.0
        
        # Peso 1: Ausência de características de pássaro
        bird_chars = [
            characteristics.get('has_wings', False),
            characteristics.get('has_beak', False),
            characteristics.get('has_feathers', False),
            characteristics.get('has_wings_ultra', False),
            characteristics.get('has_beak_ultra', False),
            characteristics.get('has_feathers_ultra', False)
        ]
        
        bird_count = sum(bird_chars)
        if bird_count == 0:
            score += 3.0  # Peso alto para ausência total
        elif bird_count <= 1:
            score += 1.0  # Peso médio para poucas características
        
        total_weight += 3.0
        
        # Peso 2: Detecção de textura de pelo
        if characteristics.get('has_fur_ultra', False):
            score += 2.0
        if characteristics.get('has_fur_texture', False):
            score += 1.0
        total_weight += 2.0
        
        # Peso 3: Score de forma de mamífero
        mammal_shape_score = characteristics.get('mammal_shape_score_ultra', 0)
        score += mammal_shape_score * 2.0
        total_weight += 2.0
        
        # Peso 4: Score biométrico de mamífero
        biometric_mammal_score = characteristics.get('biometric_mammal_score', 0)
        score += biometric_mammal_score * 1.5
        total_weight += 1.5
        
        # Peso 5: Ausência de textura de penas
        if not characteristics.get('has_feathers_ultra', False):
            score += 1.0
        if not characteristics.get('has_feathers_hybrid', False):
            score += 0.5
        total_weight += 1.5
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_bird_score(self, characteristics: Dict, yolo_confidence: float) -> float:
        """Calcula score de pássaro baseado em múltiplas características"""
        score = 0.0
        total_weight = 0.0
        
        # Peso 1: Detecção YOLO
        if characteristics.get('yolo_detection', False):
            score += yolo_confidence * 3.0  # Peso alto para YOLO
        total_weight += 3.0
        
        # Peso 2: Características visuais básicas
        visual_chars = [
            characteristics.get('has_wings', False),
            characteristics.get('has_beak', False),
            characteristics.get('has_feathers', False),
            characteristics.get('has_eyes', False)
        ]
        
        visual_count = sum(visual_chars)
        if visual_count >= 3:
            score += 2.0
        elif visual_count >= 2:
            score += 1.5
        elif visual_count >= 1:
            score += 1.0
        total_weight += 2.0
        
        # Peso 3: Características ultra-avançadas
        ultra_chars = [
            characteristics.get('has_wings_ultra', False),
            characteristics.get('has_beak_ultra', False),
            characteristics.get('has_feathers_ultra', False),
            characteristics.get('has_body_ultra', False)
        ]
        
        ultra_count = sum(ultra_chars)
        if ultra_count >= 3:
            score += 2.5
        elif ultra_count >= 2:
            score += 2.0
        elif ultra_count >= 1:
            score += 1.5
        total_weight += 2.5
        
        # Peso 4: Score de forma de pássaro
        bird_shape_score = characteristics.get('bird_shape_score_ultra', 0)
        score += bird_shape_score * 1.5
        total_weight += 1.5
        
        # Peso 5: Score biométrico de pássaro
        biometric_bird_score = characteristics.get('biometric_bird_score', 0)
        score += biometric_bird_score * 1.0
        total_weight += 1.0
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _multi_library_detection(self, image: np.ndarray) -> Dict[str, Any]:
        """Detecção usando múltiplas bibliotecas (YOLO, OpenCV, MediaPipe)"""
        detection_results = {
            'yolo_detection': False,
            'opencv_detection': False,
            'mediapipe_detection': False,
            'detection_votes': {},
            'total_detections': 0,
            'bird_confidence_avg': 0.0
        }
        
        bird_detections = []
        all_detections = []
        
        # 1. Detecção YOLO (todas as versões disponíveis)
        for model_name, model in self.detection_models.items():
            if 'yolo' in model_name.lower():
                try:
                    yolo_result = self._detect_with_yolo(model, image, model_name)
                    if yolo_result['detected']:
                        bird_detections.append(yolo_result['confidence'])
                        all_detections.append(f"YOLO_{model_name}")
                        detection_results['yolo_detection'] = True
                        detection_results['detection_votes'][f"yolo_{model_name}"] = yolo_result['confidence']
                except Exception as e:
                    logger.warning(f"⚠️ Erro na detecção YOLO {model_name}: {e}")
        
        # 2. Detecção OpenCV DNN
        for model_name, model in self.detection_models.items():
            if 'opencv' in model_name.lower():
                try:
                    opencv_result = self._detect_with_opencv(model, image, model_name)
                    if opencv_result['detected']:
                        bird_detections.append(opencv_result['confidence'])
                        all_detections.append(f"OpenCV_{model_name}")
                        detection_results['opencv_detection'] = True
                        detection_results['detection_votes'][f"opencv_{model_name}"] = opencv_result['confidence']
                except Exception as e:
                    logger.warning(f"⚠️ Erro na detecção OpenCV {model_name}: {e}")
        
        # 3. Detecção MediaPipe
        if 'mediapipe' in self.detection_models:
            try:
                mediapipe_result = self._detect_with_mediapipe(self.detection_models['mediapipe'], image)
                if mediapipe_result['detected']:
                    bird_detections.append(mediapipe_result['confidence'])
                    all_detections.append("MediaPipe")
                    detection_results['mediapipe_detection'] = True
                    detection_results['detection_votes']['mediapipe'] = mediapipe_result['confidence']
            except Exception as e:
                logger.warning(f"⚠️ Erro na detecção MediaPipe: {e}")
        
        # 4. Calcular estatísticas finais
        detection_results['total_detections'] = len(all_detections)
        if bird_detections:
            detection_results['bird_confidence_avg'] = sum(bird_detections) / len(bird_detections)
        
        return detection_results
    
    def _detect_with_yolo(self, model, image: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Detecção usando modelo YOLO com configurações avançadas"""
        try:
            # Configurações avançadas para melhor detecção
            detection_params = {
                'conf': 0.1,  # Confiança mínima muito baixa
                'iou': 0.3,   # IoU threshold
                'agnostic_nms': False,
                'max_det': 1000,
                'half': False,
                'device': 'cpu'
            }
            
            if 'v5' in model_name:
                # YOLOv5 tem interface diferente
                results = model(image, **detection_params)
                detections = results.pandas().xyxy[0]
                
                bird_detected = False
                max_confidence = 0.0
                bird_features = []
                
                for _, detection in detections.iterrows():
                    confidence = detection['confidence']
                    class_id = detection['class']
                    
                    if confidence > 0.1:  # Threshold muito baixo
                        # COCO classes expandidas para pássaros e características
                        bird_classes = [15]  # Bird class principal
                        bird_feature_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]  # Outras classes que podem indicar características
                        
                        if class_id in bird_classes:
                            bird_detected = True
                            max_confidence = max(max_confidence, confidence)
                        elif class_id in bird_feature_classes and confidence > 0.2:
                            bird_features.append({
                                'class': class_id,
                                'confidence': confidence,
                                'name': detection.get('name', f'class_{class_id}')
                            })
                
                return {
                    'detected': bird_detected,
                    'confidence': max_confidence,
                    'model': model_name,
                    'bird_features': bird_features,
                    'total_features': len(bird_features)
                }
            else:
                # YOLOv8+ tem interface diferente
                results = model(image, **detection_params)
                
                bird_detected = False
                max_confidence = 0.0
                bird_features = []
                
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            if confidence > 0.1:  # Threshold muito baixo
                                # COCO classes expandidas
                                bird_classes = [15]  # Bird class principal
                                bird_feature_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
                                
                                if class_id in bird_classes:
                                    bird_detected = True
                                    max_confidence = max(max_confidence, confidence)
                                elif class_id in bird_feature_classes and confidence > 0.2:
                                    bird_features.append({
                                        'class': class_id,
                                        'confidence': confidence,
                                        'name': result.names.get(class_id, f'class_{class_id}')
                                    })
                
                return {
                    'detected': bird_detected,
                    'confidence': max_confidence,
                    'model': model_name,
                    'bird_features': bird_features,
                    'total_features': len(bird_features)
                }
                
        except Exception as e:
            logger.warning(f"⚠️ Erro na detecção YOLO {model_name}: {e}")
            return {
                'detected': False, 
                'confidence': 0.0, 
                'model': model_name,
                'bird_features': [],
                'total_features': 0
            }
    
    def _detect_with_opencv(self, model, image: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Detecção usando OpenCV DNN"""
        try:
            # Preparar imagem para OpenCV DNN
            blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
            model.setInput(blob)
            
            # Detectar objetos
            outputs = model.forward()
            
            bird_detected = False
            max_confidence = 0.0
            
            # Processar detecções (formato YOLO)
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Assumindo que 0 é pássaro ou classe de pássaro
                    if confidence > 0.3 and class_id == 0:
                        bird_detected = True
                        max_confidence = max(max_confidence, confidence)
            
            return {
                'detected': bird_detected,
                'confidence': max_confidence,
                'model': model_name
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Erro na detecção OpenCV {model_name}: {e}")
            return {'detected': False, 'confidence': 0.0, 'model': model_name}
    
    def _detect_with_mediapipe(self, model, image: np.ndarray) -> Dict[str, Any]:
        """Detecção usando MediaPipe"""
        try:
            # Converter para RGB (MediaPipe requer RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detectar objetos
            results = model.process(rgb_image)
            
            bird_detected = False
            max_confidence = 0.0
            
            if results.detected_objects:
                for detected_object in results.detected_objects:
                    # MediaPipe Objectron detecta objetos 3D
                    # Assumir que qualquer detecção pode ser um pássaro
                    if detected_object.score > 0.3:
                        bird_detected = True
                        max_confidence = max(max_confidence, detected_object.score)
            
            return {
                'detected': bird_detected,
                'confidence': max_confidence,
                'model': 'mediapipe'
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Erro na detecção MediaPipe: {e}")
            return {'detected': False, 'confidence': 0.0, 'model': 'mediapipe'}
    
    def _calculate_model_consensus(self, characteristics: Dict, detection_results: Dict) -> Dict[str, Any]:
        """Calcula consenso entre todos os modelos de detecção"""
        consensus_result = {
            'model_consensus': 0.0,
            'detection_agreement': 0.0,
            'confidence_consensus': 0.0,
            'final_decision': 'unknown'
        }
        
        # Contar votos
        votes = detection_results.get('detection_votes', {})
        total_models = len(votes)
        
        if total_models == 0:
            return consensus_result
        
        # Calcular consenso de detecção
        positive_votes = sum(1 for conf in votes.values() if conf > 0.3)
        detection_agreement = positive_votes / total_models
        
        # Calcular consenso de confiança
        if votes:
            avg_confidence = sum(votes.values()) / len(votes)
            confidence_consensus = avg_confidence
        else:
            confidence_consensus = 0.0
        
        # Consenso geral
        model_consensus = (detection_agreement + confidence_consensus) / 2
        
        # Decisão final baseada no consenso
        if model_consensus > 0.6:
            final_decision = 'bird_high_confidence'
        elif model_consensus > 0.4:
            final_decision = 'bird_medium_confidence'
        elif model_consensus > 0.2:
            final_decision = 'uncertain'
        else:
            final_decision = 'not_bird'
        
        consensus_result.update({
            'model_consensus': model_consensus,
            'detection_agreement': detection_agreement,
            'confidence_consensus': confidence_consensus,
            'final_decision': final_decision
        })
        
        return consensus_result
    
    def _calculate_mammal_score_enhanced(self, characteristics: Dict) -> float:
        """Calcula score de mamífero melhorado com votação de modelos"""
        score = 0.0
        total_weight = 0.0
        
        # Peso 1: Consenso de modelos (novo)
        model_consensus = characteristics.get('model_consensus', 0)
        final_decision = characteristics.get('final_decision', 'unknown')
        
        if final_decision == 'not_bird':
            score += 2.0  # Peso alto se modelos concordam que não é pássaro
        elif final_decision == 'uncertain':
            score += 0.5  # Peso baixo se incerto
        total_weight += 2.0
        
        # Peso 2: Ausência de características de pássaro (mantido)
        bird_chars = [
            characteristics.get('has_wings', False),
            characteristics.get('has_beak', False),
            characteristics.get('has_feathers', False),
            characteristics.get('has_wings_ultra', False),
            characteristics.get('has_beak_ultra', False),
            characteristics.get('has_feathers_ultra', False)
        ]
        
        bird_count = sum(bird_chars)
        if bird_count == 0:
            score += 2.0  # Peso alto para ausência total
        elif bird_count <= 1:
            score += 1.0  # Peso médio para poucas características
        total_weight += 2.0
        
        # Peso 3: Detecção de textura de pelo (mantido)
        if characteristics.get('has_fur_ultra', False):
            score += 2.0
        if characteristics.get('has_fur_texture', False):
            score += 1.0
        total_weight += 2.0
        
        # Peso 4: Score de forma de mamífero (mantido)
        mammal_shape_score = characteristics.get('mammal_shape_score_ultra', 0)
        score += mammal_shape_score * 1.5
        total_weight += 1.5
        
        # Peso 5: Score biométrico de mamífero (mantido)
        biometric_mammal_score = characteristics.get('biometric_mammal_score', 0)
        score += biometric_mammal_score * 1.0
        total_weight += 1.0
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_bird_score_enhanced(self, characteristics: Dict) -> float:
        """Calcula score de pássaro melhorado com votação de modelos"""
        score = 0.0
        total_weight = 0.0
        
        # Peso 1: Consenso de modelos (novo)
        model_consensus = characteristics.get('model_consensus', 0)
        final_decision = characteristics.get('final_decision', 'unknown')
        
        if final_decision == 'bird_high_confidence':
            score += 3.0  # Peso alto se modelos concordam que é pássaro
        elif final_decision == 'bird_medium_confidence':
            score += 2.0  # Peso médio
        elif final_decision == 'uncertain':
            score += 0.5  # Peso baixo se incerto
        total_weight += 3.0
        
        # Peso 2: Detecção multi-biblioteca (novo)
        detection_votes = characteristics.get('detection_votes', {})
        if detection_votes:
            avg_detection_confidence = sum(detection_votes.values()) / len(detection_votes)
            score += avg_detection_confidence * 2.0
        total_weight += 2.0
        
        # Peso 3: Características visuais básicas (mantido)
        visual_chars = [
            characteristics.get('has_wings', False),
            characteristics.get('has_beak', False),
            characteristics.get('has_feathers', False),
            characteristics.get('has_eyes', False)
        ]
        
        visual_count = sum(visual_chars)
        if visual_count >= 3:
            score += 2.0
        elif visual_count >= 2:
            score += 1.5
        elif visual_count >= 1:
            score += 1.0
        total_weight += 2.0
        
        # Peso 4: Características ultra-avançadas (mantido)
        ultra_chars = [
            characteristics.get('has_wings_ultra', False),
            characteristics.get('has_beak_ultra', False),
            characteristics.get('has_feathers_ultra', False),
            characteristics.get('has_body_ultra', False)
        ]
        
        ultra_count = sum(ultra_chars)
        if ultra_count >= 3:
            score += 2.5
        elif ultra_count >= 2:
            score += 2.0
        elif ultra_count >= 1:
            score += 1.5
        total_weight += 2.5
        
        # Peso 5: Score de forma de pássaro (mantido)
        bird_shape_score = characteristics.get('bird_shape_score_ultra', 0)
        score += bird_shape_score * 1.0
        total_weight += 1.0
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_final_hybrid_confidence(self, characteristics: Dict, mammal_score: float, bird_score: float) -> float:
        """Calcula confiança híbrida final considerando votação de modelos"""
        confidence_factors = []
        
        # Fator 1: Consenso de modelos
        model_consensus = characteristics.get('model_consensus', 0)
        confidence_factors.append(model_consensus)
        
        # Fator 2: Diferença entre scores de pássaro e mamífero
        score_difference = bird_score - mammal_score
        if score_difference > 0.3:
            confidence_factors.append(0.8)  # Alta confiança se pássaro claramente maior
        elif score_difference > 0.1:
            confidence_factors.append(0.6)  # Confiança média
        elif score_difference < -0.3:
            confidence_factors.append(0.1)  # Baixa confiança (provavelmente mamífero)
        else:
            confidence_factors.append(0.4)  # Confiança baixa (caso duvidoso)
        
        # Fator 3: Detecção multi-biblioteca
        detection_votes = characteristics.get('detection_votes', {})
        if detection_votes:
            avg_detection_confidence = sum(detection_votes.values()) / len(detection_votes)
            confidence_factors.append(avg_detection_confidence)
        else:
            confidence_factors.append(0.3)  # Baixa confiança sem detecção
        
        # Fator 4: Características ultra-avançadas
        ultra_count = sum([
            characteristics.get('has_wings_ultra', False),
            characteristics.get('has_beak_ultra', False),
            characteristics.get('has_feathers_ultra', False),
            characteristics.get('has_body_ultra', False)
        ])
        
        if ultra_count >= 3:
            confidence_factors.append(0.9)
        elif ultra_count >= 2:
            confidence_factors.append(0.7)
        elif ultra_count >= 1:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.2)
        
        # Calcular confiança média ponderada
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.0
    
    def _calculate_ultra_hybrid_confidence(self, characteristics: Dict, yolo_detections: int, mammal_score: float, bird_score: float) -> float:
        """Calcula confiança híbrida ultra-rigorosa"""
        confidence_factors = []
        
        # Fator 1: Detecção YOLO
        if characteristics.get('yolo_detection', False):
            confidence_factors.append(0.9)  # Alta confiança se YOLO detectou
        
        # Fator 2: Diferença entre scores de pássaro e mamífero
        score_difference = bird_score - mammal_score
        if score_difference > 0.3:
            confidence_factors.append(0.8)  # Alta confiança se pássaro claramente maior
        elif score_difference > 0.1:
            confidence_factors.append(0.6)  # Confiança média
        elif score_difference < -0.3:
            confidence_factors.append(0.1)  # Baixa confiança (provavelmente mamífero)
        else:
            confidence_factors.append(0.4)  # Confiança baixa (caso duvidoso)
        
        # Fator 3: Características ultra-avançadas
        ultra_count = sum([
            characteristics.get('has_wings_ultra', False),
            characteristics.get('has_beak_ultra', False),
            characteristics.get('has_feathers_ultra', False),
            characteristics.get('has_body_ultra', False)
        ])
        
        if ultra_count >= 3:
            confidence_factors.append(0.9)
        elif ultra_count >= 2:
            confidence_factors.append(0.7)
        elif ultra_count >= 1:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.2)
        
        # Fator 4: Análise de textura ultra-híbrida
        if characteristics.get('has_feathers_ultra', False) and not characteristics.get('has_fur_ultra', False):
            confidence_factors.append(0.8)
        elif characteristics.get('has_fur_ultra', False) and not characteristics.get('has_feathers_ultra', False):
            confidence_factors.append(0.1)
        else:
            confidence_factors.append(0.4)
        
        # Calcular confiança média ponderada
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.0
    
    def _determine_species(self, visual_analysis: Dict, characteristics: Dict) -> str:
        """Determina espécie baseada em características visuais"""
        # Lógica simples baseada em cores e características
        dominant_color = visual_analysis.get('dominant_color', 'unknown')
        
        # Mapeamento básico de cores para espécies comuns
        color_species_map = {
            'brown': 'Pássaro marrom (possível rolinha, pardal)',
            'black': 'Pássaro preto (possível corvo, melro)',
            'white': 'Pássaro branco (possível pomba, gaivota)',
            'red': 'Pássaro vermelho (possível cardeal, beija-flor)',
            'blue': 'Pássaro azul (possível azulão, sabiá)',
            'yellow': 'Pássaro amarelo (possível canário, pintassilgo)',
            'green': 'Pássaro verde (possível papagaio, periquito)'
        }
        
        return color_species_map.get(dominant_color, 'Pássaro de espécie desconhecida')
    
    def _detect_learning_candidates(self, visual_analysis: Dict, characteristics: Dict, reasoning: Dict) -> List[LearningCandidate]:
        """Detecta candidatos para aprendizado contínuo"""
        candidates = []
        
        # Candidato 1: Análise visual interessante
        if visual_analysis.get('bird_like_features', 0) > 0.3:
            candidates.append(LearningCandidate(
                type=LearningCandidateType.VISUAL_ANALYSIS,
                confidence=visual_analysis['bird_like_features'],
                characteristics=visual_analysis,
                reasoning="Características visuais interessantes detectadas",
                image_path="",  # Será preenchido pelo chamador
                metadata={'analysis_type': 'visual'}
            ))
        
        # Candidato 2: Espécie desconhecida
        if reasoning.get('is_bird', False) and reasoning.get('species', '').startswith('Pássaro de espécie desconhecida'):
            candidates.append(LearningCandidate(
                type=LearningCandidateType.SPECIES_UNKNOWN,
                confidence=reasoning.get('confidence', 0),
                characteristics=characteristics,
                reasoning="Pássaro detectado mas espécie desconhecida",
                image_path="",
                metadata={'analysis_type': 'species_unknown'}
            ))
        
        # Candidato 3: Características para aprendizado
        if len(reasoning.get('characteristics_found', [])) > 0:
            candidates.append(LearningCandidate(
                type=LearningCandidateType.CHARACTERISTIC_LEARNING,
                confidence=reasoning.get('confidence', 0),
                characteristics={'found': reasoning['characteristics_found']},
                reasoning="Características específicas encontradas para aprendizado",
                image_path="",
                metadata={'analysis_type': 'characteristics'}
            ))
        
        return candidates
    
    def _recommend_action(self, candidates: List[LearningCandidate], reasoning: Dict) -> str:
        """Recomenda ação baseada na análise"""
        if not candidates:
            return "PROCESSAR_NORMALMENTE"
        
        # Priorizar candidatos com alta confiança
        high_confidence_candidates = [c for c in candidates if c.confidence > 0.7]
        
        if high_confidence_candidates:
            return "ANALISAR_MANUALMENTE"
        
        # Candidatos com confiança moderada
        medium_confidence_candidates = [c for c in candidates if c.confidence > 0.5]
        
        if medium_confidence_candidates:
            return "PROCESSAR_COM_CUIDADO"
        
        # Candidatos com baixa confiança
        return "REGISTRAR_PARA_ANALISE_FUTURA"
    
    def learn_from_feedback(self, image_path: str, human_feedback: Dict[str, Any]):
        """Aprende com feedback humano (como uma criança)"""
        try:
            # Extrair características da imagem
            visual_analysis = self._analyze_visual_characteristics(image_path)
            characteristics = self._detect_fundamental_characteristics(image_path)
            
            # Atualizar conhecimento baseado no feedback
            if human_feedback.get('is_bird', False):
                species = human_feedback.get('species', 'unknown')
                self.learned_patterns['known_species'].add(species)
                
                # Aprender padrões de características
                for char_name, char_value in characteristics.items():
                    if char_value and char_name != 'error':
                        if char_name not in self.learned_patterns['characteristic_patterns']:
                            self.learned_patterns['characteristic_patterns'][char_name] = []
                        
                        self.learned_patterns['characteristic_patterns'][char_name].append({
                            'visual_features': visual_analysis,
                            'species': species,
                            'confidence': human_feedback.get('confidence', 0.8)
                        })
                
                logger.info(f"🧠 Aprendizado: {species} adicionado ao conhecimento")
            
        except Exception as e:
            logger.error(f"❌ Erro no aprendizado: {e}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas de aprendizado"""
        return {
            'known_species_count': len(self.learned_patterns['known_species']),
            'known_species': list(self.learned_patterns['known_species']),
            'characteristic_patterns': len(self.learned_patterns['characteristic_patterns']),
            'total_learning_events': sum(
                len(patterns) for patterns in self.learned_patterns['characteristic_patterns'].values()
            )
        }
