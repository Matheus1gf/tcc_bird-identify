#!/usr/bin/env python3
"""
Sistema de Métricas de Aprendizado
MELHORIA 3: Sistema de Aprendizado Incremental Adaptativo
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)

class LearningMetrics:
    """Gerencia métricas de aprendizado ao longo do tempo"""
    
    def __init__(self, metrics_file: str = "data/learning_metrics.json"):
        """
        Inicializa sistema de métricas
        
        Args:
            metrics_file: Caminho do arquivo para salvar métricas
        """
        self.metrics_file = metrics_file
        self.metrics_history = []
        self.max_history_size = 1000  # Manter últimos 1000 registros
        
        # Carregar métricas existentes
        self._load_metrics()
    
    def _load_metrics(self):
        """Carrega métricas do arquivo"""
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.metrics_history = data.get('history', [])
                    # Manter apenas os últimos registros
                    if len(self.metrics_history) > self.max_history_size:
                        self.metrics_history = self.metrics_history[-self.max_history_size:]
                logger.info(f"[LEARNING_METRICS] {len(self.metrics_history)} métricas carregadas")
        except Exception as e:
            logger.error(f"[LEARNING_METRICS] Erro ao carregar métricas: {e}")
            self.metrics_history = []
    
    def _save_metrics(self):
        """Salva métricas no arquivo"""
        try:
            os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'last_updated': datetime.now().isoformat(),
                    'history': self.metrics_history[-self.max_history_size:]
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[LEARNING_METRICS] Erro ao salvar métricas: {e}")
    
    def record_feedback(self, is_bird: bool, predicted_bird: bool, confidence: float, 
                       feedback_type: str = "manual"):
        """
        Registra um feedback para análise
        
        Args:
            is_bird: Se realmente é um pássaro
            predicted_bird: Se o sistema previu que é pássaro
            confidence: Confiança da predição
            feedback_type: Tipo de feedback (manual, auto, etc)
        """
        metric = {
            'timestamp': datetime.now().isoformat(),
            'is_bird': is_bird,
            'predicted_bird': predicted_bird,
            'confidence': confidence,
            'feedback_type': feedback_type,
            'is_correct': is_bird == predicted_bird,
            'is_false_positive': predicted_bird and not is_bird,
            'is_false_negative': not predicted_bird and is_bird
        }
        
        self.metrics_history.append(metric)
        
        # Manter apenas últimos registros
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
        
        # Salvar periodicamente (a cada 10 registros)
        if len(self.metrics_history) % 10 == 0:
            self._save_metrics()
    
    def get_recent_metrics(self, days: int = 7, count: int = 100) -> List[Dict[str, Any]]:
        """
        Obtém métricas recentes
        
        Args:
            days: Número de dias para buscar
            count: Número máximo de registros
        
        Returns:
            Lista de métricas recentes
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()
        
        recent = [
            m for m in self.metrics_history 
            if m.get('timestamp', '') >= cutoff_str
        ]
        
        return recent[-count:]
    
    def calculate_false_positive_rate(self, days: int = 7) -> float:
        """
        Calcula taxa de falsos positivos nos últimos dias
        
        Args:
            days: Número de dias para analisar
        
        Returns:
            Taxa de falsos positivos (0.0 a 1.0)
        """
        recent = self.get_recent_metrics(days=days)
        if not recent:
            return 0.0
        
        false_positives = sum(1 for m in recent if m.get('is_false_positive', False))
        total_predictions = len(recent)
        
        if total_predictions == 0:
            return 0.0
        
        return false_positives / total_predictions
    
    def calculate_false_negative_rate(self, days: int = 7) -> float:
        """
        Calcula taxa de falsos negativos nos últimos dias
        
        Args:
            days: Número de dias para analisar
        
        Returns:
            Taxa de falsos negativos (0.0 a 1.0)
        """
        recent = self.get_recent_metrics(days=days)
        if not recent:
            return 0.0
        
        false_negatives = sum(1 for m in recent if m.get('is_false_negative', False))
        total_predictions = len(recent)
        
        if total_predictions == 0:
            return 0.0
        
        return false_negatives / total_predictions
    
    def calculate_accuracy(self, days: int = 7) -> float:
        """
        Calcula precisão nos últimos dias
        
        Args:
            days: Número de dias para analisar
        
        Returns:
            Precisão (0.0 a 1.0)
        """
        recent = self.get_recent_metrics(days=days)
        if not recent:
            return 0.0
        
        correct = sum(1 for m in recent if m.get('is_correct', False))
        total = len(recent)
        
        if total == 0:
            return 0.0
        
        return correct / total
    
    def get_metrics_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Obtém resumo de métricas
        
        Args:
            days: Número de dias para analisar
        
        Returns:
            Dicionário com resumo de métricas
        """
        recent = self.get_recent_metrics(days=days)
        
        if not recent:
            return {
                'total_feedback': 0,
                'accuracy': 0.0,
                'false_positive_rate': 0.0,
                'false_negative_rate': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        total = len(recent)
        correct = sum(1 for m in recent if m.get('is_correct', False))
        false_positives = sum(1 for m in recent if m.get('is_false_positive', False))
        false_negatives = sum(1 for m in recent if m.get('is_false_negative', False))
        true_positives = sum(1 for m in recent if m.get('is_bird', False) and m.get('predicted_bird', False))
        true_negatives = sum(1 for m in recent if not m.get('is_bird', False) and not m.get('predicted_bird', False))
        
        accuracy = correct / total if total > 0 else 0.0
        false_positive_rate = false_positives / total if total > 0 else 0.0
        false_negative_rate = false_negatives / total if total > 0 else 0.0
        
        # Precision = TP / (TP + FP)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        
        # Recall = TP / (TP + FN)
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'total_feedback': total,
            'accuracy': accuracy,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def save(self):
        """Salva métricas no arquivo"""
        self._save_metrics()

