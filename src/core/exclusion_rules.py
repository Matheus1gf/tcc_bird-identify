#!/usr/bin/env python3
"""
Sistema de Regras de Exclusão
MELHORIA 5: Regras para reduzir falsos positivos identificando características de não-pássaros
"""

import numpy as np
import cv2
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ExclusionRules:
    """Sistema de regras de exclusão para reduzir falsos positivos"""
    
    def __init__(self):
        """Inicializa sistema de regras de exclusão"""
        # Penalidades por características de não-pássaro
        self.penalties = {
            'scales': 0.4,  # Escamas indicam réptil
            'fur': 0.5,  # Pelos indicam mamífero
            'smooth_skin': 0.3,  # Pele lisa pode indicar réptil ou anfíbio
            'insect_exoskeleton': 0.6,  # Exoesqueleto indica inseto
            'mammal_body_shape': 0.4,  # Forma de corpo de mamífero
            'reptile_head_shape': 0.3,  # Forma de cabeça de réptil
            'insect_antennae': 0.5,  # Antenas indicam inseto
            'mammal_limbs': 0.3,  # Membros de mamífero (4 patas)
            'reptile_tail': 0.2,  # Cauda de réptil
            'no_wings_but_bird_like': 0.2  # Sem asas mas parece pássaro (pode ser erro)
        }
    
    def detect_non_bird_features(self, image: np.ndarray, visual_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detecta características de não-pássaro na imagem
        
        Args:
            image: Imagem para análise
            visual_analysis: Análise visual existente
        
        Returns:
            Dicionário com características detectadas e penalidades
        """
        features = {
            'has_scales': False,
            'has_fur': False,
            'has_smooth_skin': False,
            'has_insect_exoskeleton': False,
            'has_mammal_body_shape': False,
            'has_reptile_head_shape': False,
            'has_insect_antennae': False,
            'has_mammal_limbs': False,
            'has_reptile_tail': False,
            'total_penalty': 0.0,
            'exclusion_reasons': []
        }
        
        try:
            # Detectar escamas
            features['has_scales'] = self._detect_scales(image)
            if features['has_scales']:
                features['total_penalty'] += self.penalties['scales']
                features['exclusion_reasons'].append('Escamas detectadas (réptil)')
            
            # Detectar pelos
            features['has_fur'] = self._detect_fur(image)
            if features['has_fur']:
                features['total_penalty'] += self.penalties['fur']
                features['exclusion_reasons'].append('Pelos detectados (mamífero)')
            
            # Detectar pele lisa
            features['has_smooth_skin'] = self._detect_smooth_skin(image)
            if features['has_smooth_skin']:
                features['total_penalty'] += self.penalties['smooth_skin']
                features['exclusion_reasons'].append('Pele lisa detectada (réptil/anfíbio)')
            
            # Detectar exoesqueleto de inseto
            features['has_insect_exoskeleton'] = self._detect_insect_exoskeleton(image)
            if features['has_insect_exoskeleton']:
                features['total_penalty'] += self.penalties['insect_exoskeleton']
                features['exclusion_reasons'].append('Exoesqueleto detectado (inseto)')
            
            # Detectar forma de corpo de mamífero
            features['has_mammal_body_shape'] = self._detect_mammal_body_shape(image, visual_analysis)
            if features['has_mammal_body_shape']:
                features['total_penalty'] += self.penalties['mammal_body_shape']
                features['exclusion_reasons'].append('Forma de corpo de mamífero detectada')
            
            # Detectar forma de cabeça de réptil
            features['has_reptile_head_shape'] = self._detect_reptile_head_shape(image, visual_analysis)
            if features['has_reptile_head_shape']:
                features['total_penalty'] += self.penalties['reptile_head_shape']
                features['exclusion_reasons'].append('Forma de cabeça de réptil detectada')
            
            # Detectar antenas de inseto
            features['has_insect_antennae'] = self._detect_insect_antennae(image)
            if features['has_insect_antennae']:
                features['total_penalty'] += self.penalties['insect_antennae']
                features['exclusion_reasons'].append('Antenas detectadas (inseto)')
            
            # Detectar membros de mamífero (4 patas)
            features['has_mammal_limbs'] = self._detect_mammal_limbs(image, visual_analysis)
            if features['has_mammal_limbs']:
                features['total_penalty'] += self.penalties['mammal_limbs']
                features['exclusion_reasons'].append('Membros de mamífero detectados (4 patas)')
            
            # Limitar penalidade total a 1.0
            features['total_penalty'] = min(features['total_penalty'], 1.0)
            
        except Exception as e:
            logger.error(f"[EXCLUSION_RULES] Erro ao detectar características de não-pássaro: {e}")
        
        return features
    
    def _detect_scales(self, image: np.ndarray) -> bool:
        """Detecta escamas na imagem"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Escamas têm padrões repetitivos regulares
            # Usar análise de Fourier para detectar padrões repetitivos
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Escamas têm energia em frequências médias específicas
            center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
            medium_freq_energy = np.sum(magnitude_spectrum[center_y-15:center_y+15, center_x-15:center_x+15])
            total_energy = np.sum(magnitude_spectrum)
            
            scale_score = medium_freq_energy / total_energy if total_energy > 0 else 0
            
            # Escamas têm padrões mais regulares que penas
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Escamas têm densidade de bordas média-alta e padrões regulares
            return scale_score > 0.15 and edge_density > 0.1 and edge_density < 0.3
        except:
            return False
    
    def _detect_fur(self, image: np.ndarray) -> bool:
        """Detecta pelos na imagem"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Pelos têm textura mais suave e menos regular que penas
            # Análise de gradientes
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Pelos têm variância de textura menor que penas
            texture_variance = np.var(magnitude)
            
            # Pelos têm padrões menos regulares
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Pelos têm densidade de bordas baixa e variância baixa
            return texture_variance < 500 and edge_density < 0.15
        except:
            return False
    
    def _detect_smooth_skin(self, image: np.ndarray) -> bool:
        """Detecta pele lisa (réptil/anfíbio)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Pele lisa tem textura muito uniforme
            texture_variance = np.var(gray)
            
            # Pele lisa tem poucas bordas
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Pele lisa: variância baixa e densidade de bordas muito baixa
            return texture_variance < 200 and edge_density < 0.05
        except:
            return False
    
    def _detect_insect_exoskeleton(self, image: np.ndarray) -> bool:
        """Detecta exoesqueleto de inseto"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Exoesqueleto tem textura muito rígida e brilhante
            # Análise de brilho
            brightness = np.mean(gray)
            
            # Exoesqueleto tem muitas bordas rígidas
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Exoesqueleto: brilho alto e densidade de bordas alta
            return brightness > 150 and edge_density > 0.25
        except:
            return False
    
    def _detect_mammal_body_shape(self, image: np.ndarray, visual_analysis: Dict[str, Any]) -> bool:
        """Detecta forma de corpo de mamífero"""
        try:
            # Mamíferos têm proporções diferentes de pássaros
            # Corpo mais horizontal, menos compacto
            shape_score = visual_analysis.get('bird_shape_score', 0)
            
            # Se forma não parece pássaro mas tem outras características, pode ser mamífero
            return shape_score < 0.3
        except:
            return False
    
    def _detect_reptile_head_shape(self, image: np.ndarray, visual_analysis: Dict[str, Any]) -> bool:
        """Detecta forma de cabeça de réptil"""
        try:
            # Répteis têm cabeças mais alongadas e menos arredondadas
            # Análise simplificada baseada em características visuais
            shape_score = visual_analysis.get('bird_shape_score', 0)
            
            # Se forma não parece pássaro, pode ser réptil
            return shape_score < 0.2
        except:
            return False
    
    def _detect_insect_antennae(self, image: np.ndarray) -> bool:
        """Detecta antenas de inseto"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Antenas são estruturas finas e alongadas
            edges = cv2.Canny(gray, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            antenna_count = 0
            for contour in contours:
                if len(contour) > 5:
                    # Calcular alongamento
                    area = cv2.contourArea(contour)
                    if area > 10:  # Filtrar ruído
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = max(w, h) / max(min(w, h), 1)
                        
                        # Antenas são muito alongadas (aspect ratio alto)
                        if aspect_ratio > 5 and area < 100:
                            antenna_count += 1
            
            return antenna_count >= 2  # Insetos geralmente têm 2 antenas
        except:
            return False
    
    def _detect_mammal_limbs(self, image: np.ndarray, visual_analysis: Dict[str, Any]) -> bool:
        """Detecta membros de mamífero (4 patas)"""
        try:
            # Mamíferos têm 4 membros, pássaros têm 2 asas + 2 pernas
            # Análise simplificada: se não detectou asas mas detectou membros, pode ser mamífero
            # Esta é uma heurística simples, pode ser melhorada
            return False  # Por enquanto, retornar False (análise complexa requer detecção de partes do corpo)
        except:
            return False
    
    def apply_penalties(self, bird_like_score: float, exclusion_features: Dict[str, Any]) -> float:
        """
        Aplica penalidades ao score de pássaro
        
        Args:
            bird_like_score: Score original de características de pássaro
            exclusion_features: Características de exclusão detectadas
        
        Returns:
            Score ajustado com penalidades
        """
        total_penalty = exclusion_features.get('total_penalty', 0.0)
        adjusted_score = max(0.0, bird_like_score - total_penalty)
        
        return adjusted_score

