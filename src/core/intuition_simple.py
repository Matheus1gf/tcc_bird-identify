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
        """Carrega modelos YOLO e Keras"""
        try:
            # Carregar YOLO
            from ultralytics import YOLO
            self.yolo_model = YOLO(self.yolo_model_path)
            logger.info("✅ Modelo YOLO carregado")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar YOLO: {e}")
            self.yolo_model = None
        
        try:
            # Carregar Keras
            import tensorflow as tf
            self.keras_model = tf.keras.models.load_model(self.keras_model_path)
            logger.info("✅ Modelo Keras carregado")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar Keras: {e}")
            self.keras_model = None
    
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
                'confidence': logical_reasoning.get('overall_confidence', 0.0),
                'species': logical_reasoning.get('species', 'Desconhecida'),
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
        """Analisa cores como uma criança reconheceria"""
        # Cores típicas de pássaros
        bird_colors = {
            'brown': [(10, 50, 20), (20, 150, 200)],
            'black': [(0, 0, 0), (50, 50, 50)],
            'white': [(200, 200, 200), (255, 255, 255)],
            'red': [(0, 50, 100), (10, 150, 255)],
            'blue': [(100, 50, 0), (150, 150, 255)],
            'yellow': [(0, 100, 100), (50, 255, 255)],
            'green': [(50, 100, 0), (150, 255, 150)]
        }
        
        color_scores = {}
        
        for color_name, (lower, upper) in bird_colors.items():
            mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
            score = np.sum(mask) / (mask.shape[0] * mask.shape[1] * 255)
            color_scores[color_name] = score
        
        # Encontrar cor dominante
        dominant_color = max(color_scores, key=color_scores.get)
        
        return {
            'dominant_color': dominant_color,
            'distribution': color_scores,
            'bird_color_score': sum(color_scores.values()) / len(color_scores)
        }
    
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
        aspect_ratio = w / h if h > 0 else 0
        
        # Análise de compactness
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        return {
            'aspect_ratio': aspect_ratio,
            'compactness': compactness,
            'area_ratio': area / (image.shape[0] * image.shape[1]),
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
        texture_uniformity = 1.0 / (1.0 + texture_variance)
        
        return {
            'texture_variance': texture_variance,
            'texture_uniformity': texture_uniformity,
            'feather_like_score': self._calculate_feather_score(texture_variance, texture_uniformity)
        }
    
    def _calculate_bird_like_score(self, color_analysis: Dict, shape_analysis: Dict, texture_analysis: Dict) -> float:
        """Calcula score geral de características de pássaro"""
        color_score = color_analysis.get('bird_color_score', 0)
        shape_score = shape_analysis.get('bird_shape_score', 0)
        texture_score = texture_analysis.get('feather_like_score', 0)
        
        # Pesos para diferentes características
        weights = {'color': 0.3, 'shape': 0.4, 'texture': 0.3}
        
        return (color_score * weights['color'] + 
                shape_score * weights['shape'] + 
                texture_score * weights['texture'])
    
    def _calculate_shape_score(self, aspect_ratio: float, compactness: float) -> float:
        """Calcula score de forma baseado em características de pássaro"""
        # Pássaros típicos têm aspect_ratio entre 0.5 e 2.0
        aspect_score = 1.0 - abs(aspect_ratio - 1.0) / 1.0
        aspect_score = max(0, min(1, aspect_score))
        
        # Compactness típica de pássaros (formas arredondadas)
        compactness_score = compactness if compactness <= 1.0 else 1.0
        
        return (aspect_score + compactness_score) / 2
    
    def _calculate_feather_score(self, variance: float, uniformity: float) -> float:
        """Calcula score de textura de penas"""
        # Penas têm textura variada mas não muito uniforme
        variance_score = min(1.0, variance / 1000.0)  # Normalizar
        uniformity_score = uniformity
        
        return (variance_score + uniformity_score) / 2
    
    def _detect_fundamental_characteristics(self, image_path: str) -> Dict[str, Any]:
        """Detecta características fundamentais de pássaros"""
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
                'bird_body_shape': False
            }
            
            # Usar YOLO para detectar partes do corpo se disponível
            if self.yolo_model:
                yolo_results = self.yolo_model(image)
                for result in yolo_results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            # Mapear classes YOLO para características
                            if confidence > 0.5:
                                if class_id == 0:  # Assumindo que 0 é pássaro
                                    characteristics['bird_body_shape'] = True
                                # Adicionar mais mapeamentos conforme necessário
            
            # Análise visual para detectar características
            characteristics.update(self._detect_visual_characteristics(image))
            
            return characteristics
            
        except Exception as e:
            logger.error(f"❌ Erro na detecção de características: {e}")
            return {'error': str(e)}
    
    def _detect_visual_characteristics(self, image: np.ndarray) -> Dict[str, bool]:
        """Detecta características visuais usando análise de imagem"""
        characteristics = {}
        
        # Detectar olhos (círculos escuros)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=5, maxRadius=50)
        characteristics['has_eyes'] = circles is not None and len(circles[0]) > 0
        
        # Detectar formas de asas (contornos alongados)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        wing_like_contours = 0
        for contour in contours:
            if len(contour) > 5:
                ellipse = cv2.fitEllipse(contour)
                (center, axes, orientation) = ellipse
                major_axis = max(axes)
                minor_axis = min(axes)
                if major_axis / minor_axis > 2.0:  # Forma alongada como asa
                    wing_like_contours += 1
        
        characteristics['has_wings'] = wing_like_contours > 0
        
        # Detectar textura de penas (padrões repetitivos)
        characteristics['has_feathers'] = self._detect_feather_texture(image)
        
        return characteristics
    
    def _detect_feather_texture(self, image: np.ndarray) -> bool:
        """Detecta textura de penas usando análise de padrões"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Usar transformada de Fourier para detectar padrões repetitivos
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Calcular energia em diferentes frequências
        center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
        
        # Energia em frequências médias (características de penas)
        medium_freq_energy = np.sum(magnitude_spectrum[center_y-20:center_y+20, center_x-20:center_x+20])
        total_energy = np.sum(magnitude_spectrum)
        
        feather_score = medium_freq_energy / total_energy if total_energy > 0 else 0
        
        return feather_score > 0.1  # Threshold para detectar textura de penas
    
    def _logical_reasoning(self, visual_analysis: Dict, characteristics: Dict) -> Dict[str, Any]:
        """Raciocínio lógico neuro-simbólico"""
        reasoning = {
            'is_bird': False,
            'confidence': 0.0,
            'species': 'Desconhecida',
            'reasoning_steps': [],
            'characteristics_found': [],
            'missing_characteristics': []
        }
        
        # Passo 1: Verificar características fundamentais
        fundamental_count = 0
        total_characteristics = 0
        
        for char_name, char_value in characteristics.items():
            if char_name != 'error':
                total_characteristics += 1
                if char_value:
                    fundamental_count += 1
                    reasoning['characteristics_found'].append(char_name)
                else:
                    reasoning['missing_characteristics'].append(char_name)
        
        # Passo 2: Raciocínio lógico
        if fundamental_count >= 3:  # Pelo menos 3 características fundamentais
            reasoning['is_bird'] = True
            reasoning['confidence'] = fundamental_count / total_characteristics
            reasoning['reasoning_steps'].append(
                f"Encontradas {fundamental_count} características fundamentais de pássaro"
            )
        elif fundamental_count >= 2:
            reasoning['is_bird'] = True
            reasoning['confidence'] = 0.6  # Confiança moderada
            reasoning['reasoning_steps'].append(
                f"Encontradas {fundamental_count} características, confiança moderada"
            )
        else:
            reasoning['is_bird'] = False
            reasoning['confidence'] = 0.2
            reasoning['reasoning_steps'].append(
                f"Apenas {fundamental_count} características encontradas, provavelmente não é um pássaro"
            )
        
        # Passo 3: Determinar espécie (se for pássaro)
        if reasoning['is_bird']:
            species = self._determine_species(visual_analysis, characteristics)
            reasoning['species'] = species
            reasoning['reasoning_steps'].append(f"Espécie identificada: {species}")
        
        # Passo 4: Calcular confiança geral
        visual_score = visual_analysis.get('bird_like_features', 0)
        reasoning['overall_confidence'] = (reasoning['confidence'] + visual_score) / 2
        
        return reasoning
    
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
