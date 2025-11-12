#!/usr/bin/env python3
"""
Sistema de Aprendizado Incremental Adaptativo
MELHORIA 3: Sistema que começa permissivo e aprende a ser mais preciso com o tempo
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class LearningMode(Enum):
    """Modos de aprendizado"""
    INITIAL = "initial"  # Mais permissivo (thresholds baixos)
    INTERMEDIATE = "intermediate"  # Balanceado (thresholds médios)
    EXPERIENCED = "experienced"  # Mais rigoroso (thresholds altos)

class AdaptiveLearningSystem:
    """Sistema de aprendizado incremental adaptativo"""
    
    def __init__(self, config_file: str = "data/adaptive_learning_config.json"):
        """
        Inicializa sistema de aprendizado adaptativo
        
        Args:
            config_file: Caminho do arquivo de configuração
        """
        self.config_file = config_file
        self.config = self._load_config()
        
        # Thresholds iniciais (modo permissivo)
        self.thresholds = {
            'bird_like_features': self.config.get('bird_like_features_threshold', 0.3),
            'bird_shape_score': self.config.get('bird_shape_score_threshold', 0.2),
            'bird_color_score': self.config.get('bird_color_score_threshold', 0.2),
            'confidence_min': self.config.get('confidence_min_threshold', 0.3)
        }
        
        # Histórico de feedback
        self.feedback_history = []
        self.max_history_size = 100
        
        # Modo atual de aprendizado
        self.current_mode = LearningMode(self.config.get('current_mode', 'initial'))
        
        # Estatísticas
        self.total_feedback = 0
        self.false_positives_corrected = 0
        self.false_negatives_corrected = 0
        
        logger.info(f"[ADAPTIVE_LEARNING] Sistema inicializado em modo: {self.current_mode.value}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Carrega configuração do arquivo"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"[ADAPTIVE_LEARNING] Erro ao carregar configuração: {e}")
        
        # Configuração padrão
        return {
            'current_mode': 'initial',
            'bird_like_features_threshold': 0.3,
            'bird_shape_score_threshold': 0.2,
            'bird_color_score_threshold': 0.2,
            'confidence_min_threshold': 0.3,
            'learning_rate': 0.01,
            'min_feedback_for_adjustment': 10
        }
    
    def _save_config(self):
        """Salva configuração no arquivo"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            config_to_save = {
                'current_mode': self.current_mode.value,
                'bird_like_features_threshold': self.thresholds['bird_like_features'],
                'bird_shape_score_threshold': self.thresholds['bird_shape_score'],
                'bird_color_score_threshold': self.thresholds['bird_color_score'],
                'confidence_min_threshold': self.thresholds['confidence_min'],
                'learning_rate': self.config.get('learning_rate', 0.01),
                'min_feedback_for_adjustment': self.config.get('min_feedback_for_adjustment', 10),
                'last_updated': datetime.now().isoformat(),
                'total_feedback': self.total_feedback,
                'false_positives_corrected': self.false_positives_corrected,
                'false_negatives_corrected': self.false_negatives_corrected
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[ADAPTIVE_LEARNING] Erro ao salvar configuração: {e}")
    
    def record_feedback(self, predicted_bird: bool, actual_bird: bool, confidence: float):
        """
        Registra feedback para ajuste de thresholds
        
        Args:
            predicted_bird: Se o sistema previu pássaro
            actual_bird: Se realmente é pássaro
            confidence: Confiança da predição
        """
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'predicted_bird': predicted_bird,
            'actual_bird': actual_bird,
            'confidence': confidence,
            'is_correct': predicted_bird == actual_bird,
            'is_false_positive': predicted_bird and not actual_bird,
            'is_false_negative': not predicted_bird and actual_bird
        }
        
        self.feedback_history.append(feedback)
        self.total_feedback += 1
        
        # Manter apenas últimos registros
        if len(self.feedback_history) > self.max_history_size:
            self.feedback_history = self.feedback_history[-self.max_history_size:]
        
        # Contar correções
        if feedback['is_false_positive']:
            self.false_positives_corrected += 1
        elif feedback['is_false_negative']:
            self.false_negatives_corrected += 1
        
        # Ajustar thresholds se tiver feedback suficiente
        if len(self.feedback_history) >= self.config.get('min_feedback_for_adjustment', 10):
            self._adjust_thresholds_based_on_feedback()
    
    def _adjust_thresholds_based_on_feedback(self):
        """
        Ajusta thresholds baseado no histórico de feedback
        """
        if len(self.feedback_history) < self.config.get('min_feedback_for_adjustment', 10):
            return
        
        # Analisar últimos feedbacks
        recent_feedback = self.feedback_history[-100:]  # Últimos 100
        
        # Calcular taxas de erro
        false_positive_rate = sum(1 for f in recent_feedback if f['is_false_positive']) / len(recent_feedback)
        false_negative_rate = sum(1 for f in recent_feedback if f['is_false_negative']) / len(recent_feedback)
        
        learning_rate = self.config.get('learning_rate', 0.01)
        
        # Ajustar thresholds baseado em falsos positivos
        if false_positive_rate > 0.15:  # Mais de 15% de falsos positivos
            # Aumentar thresholds para ser mais rigoroso
            self.thresholds['bird_like_features'] = min(0.7, 
                self.thresholds['bird_like_features'] + learning_rate)
            self.thresholds['bird_shape_score'] = min(0.6,
                self.thresholds['bird_shape_score'] + learning_rate)
            self.thresholds['bird_color_score'] = min(0.6,
                self.thresholds['bird_color_score'] + learning_rate)
            self.thresholds['confidence_min'] = min(0.6,
                self.thresholds['confidence_min'] + learning_rate)
            
            logger.info(f"[ADAPTIVE_LEARNING] Thresholds aumentados (FP rate: {false_positive_rate:.2%})")
        
        # Ajustar thresholds baseado em falsos negativos
        elif false_negative_rate > 0.15:  # Mais de 15% de falsos negativos
            # Diminuir thresholds para ser mais permissivo
            self.thresholds['bird_like_features'] = max(0.2,
                self.thresholds['bird_like_features'] - learning_rate)
            self.thresholds['bird_shape_score'] = max(0.1,
                self.thresholds['bird_shape_score'] - learning_rate)
            self.thresholds['bird_color_score'] = max(0.1,
                self.thresholds['bird_color_score'] - learning_rate)
            self.thresholds['confidence_min'] = max(0.2,
                self.thresholds['confidence_min'] - learning_rate)
            
            logger.info(f"[ADAPTIVE_LEARNING] Thresholds diminuídos (FN rate: {false_negative_rate:.2%})")
        
        # Determinar modo baseado em performance
        self._update_learning_mode(false_positive_rate, false_negative_rate)
        
        # Salvar configuração
        self._save_config()
    
    def _update_learning_mode(self, false_positive_rate: float, false_negative_rate: float):
        """
        Atualiza modo de aprendizado baseado em performance
        
        Args:
            false_positive_rate: Taxa de falsos positivos
            false_negative_rate: Taxa de falsos negativos
        """
        # Critérios para mudança de modo
        if self.current_mode == LearningMode.INITIAL:
            # Transição para intermediário: pelo menos 50 feedbacks e taxa de erro < 20%
            if (self.total_feedback >= 50 and 
                false_positive_rate < 0.20 and false_negative_rate < 0.20):
                self.current_mode = LearningMode.INTERMEDIATE
                logger.info("[ADAPTIVE_LEARNING] Modo alterado para INTERMEDIATE")
        
        elif self.current_mode == LearningMode.INTERMEDIATE:
            # Transição para experiente: pelo menos 200 feedbacks e taxa de erro < 10%
            if (self.total_feedback >= 200 and 
                false_positive_rate < 0.10 and false_negative_rate < 0.10):
                self.current_mode = LearningMode.EXPERIENCED
                logger.info("[ADAPTIVE_LEARNING] Modo alterado para EXPERIENCED")
            
            # Voltar para inicial se performance piorar muito
            elif false_positive_rate > 0.30 or false_negative_rate > 0.30:
                self.current_mode = LearningMode.INITIAL
                logger.warning("[ADAPTIVE_LEARNING] Modo revertido para INITIAL (performance ruim)")
        
        elif self.current_mode == LearningMode.EXPERIENCED:
            # Voltar para intermediário se performance piorar
            if false_positive_rate > 0.20 or false_negative_rate > 0.20:
                self.current_mode = LearningMode.INTERMEDIATE
                logger.warning("[ADAPTIVE_LEARNING] Modo revertido para INTERMEDIATE (performance degradada)")
    
    def get_thresholds(self) -> Dict[str, float]:
        """
        Retorna thresholds atuais
        
        Returns:
            Dicionário com thresholds
        """
        return self.thresholds.copy()
    
    def get_learning_mode(self) -> LearningMode:
        """
        Retorna modo atual de aprendizado
        
        Returns:
            Modo de aprendizado atual
        """
        return self.current_mode
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do sistema
        
        Returns:
            Dicionário com estatísticas
        """
        recent_feedback = self.feedback_history[-100:] if self.feedback_history else []
        
        false_positive_rate = 0.0
        false_negative_rate = 0.0
        accuracy = 0.0
        
        if recent_feedback:
            false_positive_rate = sum(1 for f in recent_feedback if f['is_false_positive']) / len(recent_feedback)
            false_negative_rate = sum(1 for f in recent_feedback if f['is_false_negative']) / len(recent_feedback)
            accuracy = sum(1 for f in recent_feedback if f['is_correct']) / len(recent_feedback)
        
        return {
            'current_mode': self.current_mode.value,
            'total_feedback': self.total_feedback,
            'false_positives_corrected': self.false_positives_corrected,
            'false_negatives_corrected': self.false_negatives_corrected,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'accuracy': accuracy,
            'thresholds': self.thresholds.copy()
        }
    
    def reset_to_initial_mode(self):
        """Reseta sistema para modo inicial"""
        self.current_mode = LearningMode.INITIAL
        self.thresholds = {
            'bird_like_features': 0.3,
            'bird_shape_score': 0.2,
            'bird_color_score': 0.2,
            'confidence_min': 0.3
        }
        self._save_config()
        logger.info("[ADAPTIVE_LEARNING] Sistema resetado para modo inicial")

