#!/usr/bin/env python3
"""
Sistema de Auto-Otimização com Ajuste Automático de Thresholds
============================================================

Este sistema implementa:
- Monitoramento de performance em tempo real
- Ajuste automático de thresholds baseado em métricas
- Otimização adaptativa de parâmetros
- Sistema de feedback contínuo
- Aprendizado de padrões de performance
"""

import sys
import os
sys.path.append('.')

import logging
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import random

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Métricas de performance para otimização."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    processing_time: float
    confidence_variance: float
    detection_rate: float
    timestamp: datetime

@dataclass
class ThresholdConfig:
    """Configuração de thresholds para otimização."""
    bird_threshold: float
    confidence_threshold: float
    boost_factor: float
    color_weight: float
    shape_weight: float
    pattern_weight: float
    detection_sensitivity: float
    reasoning_threshold: float
    adaptation_rate: float

class PerformanceMonitor:
    """
    Monitor de performance que coleta métricas em tempo real.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history: deque = deque(maxlen=window_size)
        self.current_metrics: Optional[PerformanceMetrics] = None
        
        # Métricas acumuladas
        self.total_detections = 0
        self.correct_detections = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.total_processing_time = 0.0
        
        # Histórico de confianças
        self.confidence_history: deque = deque(maxlen=window_size)
        
    def update_metrics(self, 
                      is_bird: bool, 
                      predicted_bird: bool, 
                      confidence: float, 
                      processing_time: float):
        """
        Atualiza métricas com nova detecção.
        
        Args:
            is_bird: Se a imagem é realmente um pássaro
            predicted_bird: Se o sistema previu que é um pássaro
            confidence: Confiança da predição
            processing_time: Tempo de processamento
        """
        self.total_detections += 1
        self.total_processing_time += processing_time
        self.confidence_history.append(confidence)
        
        # Atualizar contadores
        if predicted_bird and is_bird:
            self.correct_detections += 1
        elif predicted_bird and not is_bird:
            self.false_positives += 1
        elif not predicted_bird and is_bird:
            self.false_negatives += 1
        
        # Calcular métricas atuais
        self._calculate_current_metrics()
        
        # Adicionar ao histórico
        if self.current_metrics:
            self.metrics_history.append(self.current_metrics)
    
    def _calculate_current_metrics(self):
        """Calcula métricas atuais baseadas nos dados acumulados."""
        if self.total_detections == 0:
            return
        
        # Métricas básicas
        accuracy = self.correct_detections / self.total_detections
        
        # Precision: TP / (TP + FP)
        precision = 0.0
        if self.correct_detections + self.false_positives > 0:
            precision = self.correct_detections / (self.correct_detections + self.false_positives)
        
        # Recall: TP / (TP + FN)
        recall = 0.0
        if self.correct_detections + self.false_negatives > 0:
            recall = self.correct_detections / (self.correct_detections + self.false_negatives)
        
        # F1 Score
        f1_score = 0.0
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        
        # Taxas de erro
        false_positive_rate = self.false_positives / self.total_detections
        false_negative_rate = self.false_negatives / self.total_detections
        
        # Tempo de processamento médio
        avg_processing_time = self.total_processing_time / self.total_detections
        
        # Variância de confiança
        confidence_variance = 0.0
        if len(self.confidence_history) > 1:
            confidence_variance = np.var(list(self.confidence_history))
        
        # Taxa de detecção
        detection_rate = (self.correct_detections + self.false_positives) / self.total_detections
        
        self.current_metrics = PerformanceMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            processing_time=avg_processing_time,
            confidence_variance=confidence_variance,
            detection_rate=detection_rate,
            timestamp=datetime.now()
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Retorna resumo das métricas de performance."""
        if not self.current_metrics:
            return {"error": "Nenhuma métrica disponível"}
        
        return {
            "total_detections": self.total_detections,
            "accuracy": self.current_metrics.accuracy,
            "precision": self.current_metrics.precision,
            "recall": self.current_metrics.recall,
            "f1_score": self.current_metrics.f1_score,
            "false_positive_rate": self.current_metrics.false_positive_rate,
            "false_negative_rate": self.current_metrics.false_negative_rate,
            "avg_processing_time": self.current_metrics.processing_time,
            "confidence_variance": self.current_metrics.confidence_variance,
            "detection_rate": self.current_metrics.detection_rate,
            "history_size": len(self.metrics_history)
        }
    
    def get_performance_trend(self) -> Dict[str, Any]:
        """Analisa tendência de performance."""
        if len(self.metrics_history) < 10:
            return {"error": "Histórico insuficiente para análise de tendência"}
        
        recent_metrics = list(self.metrics_history)[-10:]
        older_metrics = list(self.metrics_history)[-20:-10] if len(self.metrics_history) >= 20 else []
        
        # Calcular médias
        recent_accuracy = np.mean([m.accuracy for m in recent_metrics])
        recent_f1 = np.mean([m.f1_score for m in recent_metrics])
        
        trend_analysis = {
            "recent_accuracy": recent_accuracy,
            "recent_f1_score": recent_f1,
            "trend_direction": "stable"
        }
        
        if older_metrics:
            older_accuracy = np.mean([m.accuracy for m in older_metrics])
            older_f1 = np.mean([m.f1_score for m in older_metrics])
            
            accuracy_change = recent_accuracy - older_accuracy
            f1_change = recent_f1 - older_f1
            
            if accuracy_change > 0.05:
                trend_analysis["trend_direction"] = "improving"
            elif accuracy_change < -0.05:
                trend_analysis["trend_direction"] = "declining"
            
            trend_analysis.update({
                "accuracy_change": accuracy_change,
                "f1_change": f1_change,
                "older_accuracy": older_accuracy,
                "older_f1_score": older_f1
            })
        
        return trend_analysis

class ThresholdOptimizer:
    """
    Otimizador de thresholds que ajusta parâmetros baseado na performance.
    """
    
    def __init__(self):
        self.current_config = ThresholdConfig(
            bird_threshold=0.5,
            confidence_threshold=0.6,
            boost_factor=0.1,
            color_weight=0.3,
            shape_weight=0.4,
            pattern_weight=0.3,
            detection_sensitivity=0.8,
            reasoning_threshold=0.5,
            adaptation_rate=0.05
        )
        
        # Histórico de configurações
        self.config_history: List[Tuple[ThresholdConfig, float]] = []
        
        # Parâmetros de otimização
        self.optimization_strategies = {
            'gradient_descent': 0.3,
            'genetic_algorithm': 0.25,
            'bayesian_optimization': 0.2,
            'random_search': 0.15,
            'adaptive_mutation': 0.1
        }
        
        # Limites dos parâmetros
        self.parameter_bounds = {
            'bird_threshold': (0.1, 0.9),
            'confidence_threshold': (0.2, 0.95),
            'boost_factor': (0.01, 0.5),
            'color_weight': (0.1, 0.6),
            'shape_weight': (0.1, 0.6),
            'pattern_weight': (0.1, 0.6),
            'detection_sensitivity': (0.3, 1.0),
            'reasoning_threshold': (0.2, 0.8),
            'adaptation_rate': (0.01, 0.2)
        }
        
        # Estado da otimização
        self.optimization_active = False
        self.last_optimization_time = None
        self.optimization_frequency = 50  # Otimizar a cada 50 detecções
        
    def should_optimize(self, detection_count: int) -> bool:
        """Verifica se deve otimizar baseado no número de detecções."""
        if not self.optimization_active:
            return False
        
        if self.last_optimization_time is None:
            return detection_count >= self.optimization_frequency
        
        time_since_last = time.time() - self.last_optimization_time
        return (detection_count >= self.optimization_frequency and 
                time_since_last >= 300)  # Mínimo 5 minutos entre otimizações
    
    def optimize_thresholds(self, performance_metrics: PerformanceMetrics) -> ThresholdConfig:
        """
        Otimiza thresholds baseado nas métricas de performance.
        
        Args:
            performance_metrics: Métricas de performance atuais
            
        Returns:
            Nova configuração de thresholds otimizada
        """
        logger.info("Iniciando otimização de thresholds")
        
        # Selecionar estratégia de otimização
        strategy = self._select_optimization_strategy(performance_metrics)
        
        # Aplicar otimização
        new_config = self._apply_optimization_strategy(strategy, performance_metrics)
        
        # Validar nova configuração
        validated_config = self._validate_configuration(new_config)
        
        # Registrar no histórico
        fitness_score = self._calculate_fitness_score(performance_metrics)
        self.config_history.append((validated_config, fitness_score))
        
        # Manter apenas histórico recente
        if len(self.config_history) > 50:
            self.config_history = self.config_history[-50:]
        
        self.current_config = validated_config
        self.last_optimization_time = time.time()
        
        logger.info(f"Otimização concluída usando estratégia: {strategy}")
        return validated_config
    
    def _select_optimization_strategy(self, metrics: PerformanceMetrics) -> str:
        """Seleciona estratégia de otimização baseada nas métricas."""
        # Se accuracy está baixa, usar estratégias mais agressivas
        if metrics.accuracy < 0.7:
            if metrics.false_positive_rate > metrics.false_negative_rate:
                return 'gradient_descent'  # Reduzir falsos positivos
            else:
                return 'genetic_algorithm'  # Explorar mais espaço
        
        # Se accuracy está boa mas F1 score baixo, usar otimização balanceada
        elif metrics.f1_score < 0.8:
            return 'bayesian_optimization'
        
        # Se performance está boa, usar otimização sutil
        else:
            return 'adaptive_mutation'
    
    def _apply_optimization_strategy(self, strategy: str, metrics: PerformanceMetrics) -> ThresholdConfig:
        """Aplica estratégia de otimização específica."""
        if strategy == 'gradient_descent':
            return self._gradient_descent_optimization(metrics)
        elif strategy == 'genetic_algorithm':
            return self._genetic_algorithm_optimization(metrics)
        elif strategy == 'bayesian_optimization':
            return self._bayesian_optimization(metrics)
        elif strategy == 'random_search':
            return self._random_search_optimization(metrics)
        elif strategy == 'adaptive_mutation':
            return self._adaptive_mutation_optimization(metrics)
        else:
            return self._adaptive_mutation_optimization(metrics)
    
    def _gradient_descent_optimization(self, metrics: PerformanceMetrics) -> ThresholdConfig:
        """Otimização por gradiente descendente."""
        new_config = ThresholdConfig(
            bird_threshold=self.current_config.bird_threshold,
            confidence_threshold=self.current_config.confidence_threshold,
            boost_factor=self.current_config.boost_factor,
            color_weight=self.current_config.color_weight,
            shape_weight=self.current_config.shape_weight,
            pattern_weight=self.current_config.pattern_weight,
            detection_sensitivity=self.current_config.detection_sensitivity,
            reasoning_threshold=self.current_config.reasoning_threshold,
            adaptation_rate=self.current_config.adaptation_rate
        )
        
        # Ajustar thresholds baseado nos erros
        learning_rate = 0.01
        
        if metrics.false_positive_rate > 0.1:
            # Muitos falsos positivos - aumentar threshold
            new_config.bird_threshold = min(0.9, 
                self.current_config.bird_threshold + learning_rate)
            new_config.confidence_threshold = min(0.95,
                self.current_config.confidence_threshold + learning_rate)
        
        if metrics.false_negative_rate > 0.1:
            # Muitos falsos negativos - diminuir threshold
            new_config.bird_threshold = max(0.1,
                self.current_config.bird_threshold - learning_rate)
            new_config.confidence_threshold = max(0.2,
                self.current_config.confidence_threshold - learning_rate)
        
        return new_config
    
    def _genetic_algorithm_optimization(self, metrics: PerformanceMetrics) -> ThresholdConfig:
        """Otimização usando algoritmo genético."""
        # Criar população de configurações
        population_size = 10
        population = []
        
        # Adicionar configuração atual
        population.append(self.current_config)
        
        # Gerar configurações mutadas
        for _ in range(population_size - 1):
            mutated_config = self._mutate_configuration(self.current_config)
            population.append(mutated_config)
        
        # Avaliar população
        fitness_scores = []
        for config in population:
            fitness = self._evaluate_configuration(config, metrics)
            fitness_scores.append(fitness)
        
        # Selecionar melhor configuração
        best_idx = np.argmax(fitness_scores)
        return population[best_idx]
    
    def _bayesian_optimization(self, metrics: PerformanceMetrics) -> ThresholdConfig:
        """Otimização bayesiana."""
        # Implementação simplificada de otimização bayesiana
        new_config = ThresholdConfig(
            bird_threshold=self.current_config.bird_threshold,
            confidence_threshold=self.current_config.confidence_threshold,
            boost_factor=self.current_config.boost_factor,
            color_weight=self.current_config.color_weight,
            shape_weight=self.current_config.shape_weight,
            pattern_weight=self.current_config.pattern_weight,
            detection_sensitivity=self.current_config.detection_sensitivity,
            reasoning_threshold=self.current_config.reasoning_threshold,
            adaptation_rate=self.current_config.adaptation_rate
        )
        
        # Ajustar pesos baseado na performance
        if metrics.precision < metrics.recall:
            # Baixa precisão - ajustar pesos para ser mais conservador
            new_config.color_weight *= 0.95
            new_config.shape_weight *= 0.95
            new_config.pattern_weight *= 1.05
        else:
            # Baixo recall - ajustar pesos para ser mais sensível
            new_config.color_weight *= 1.05
            new_config.shape_weight *= 1.05
            new_config.pattern_weight *= 0.95
        
        return new_config
    
    def _random_search_optimization(self, metrics: PerformanceMetrics) -> ThresholdConfig:
        """Otimização por busca aleatória."""
        new_config = ThresholdConfig(
            bird_threshold=random.uniform(*self.parameter_bounds['bird_threshold']),
            confidence_threshold=random.uniform(*self.parameter_bounds['confidence_threshold']),
            boost_factor=random.uniform(*self.parameter_bounds['boost_factor']),
            color_weight=random.uniform(*self.parameter_bounds['color_weight']),
            shape_weight=random.uniform(*self.parameter_bounds['shape_weight']),
            pattern_weight=random.uniform(*self.parameter_bounds['pattern_weight']),
            detection_sensitivity=random.uniform(*self.parameter_bounds['detection_sensitivity']),
            reasoning_threshold=random.uniform(*self.parameter_bounds['reasoning_threshold']),
            adaptation_rate=random.uniform(*self.parameter_bounds['adaptation_rate'])
        )
        
        return new_config
    
    def _adaptive_mutation_optimization(self, metrics: PerformanceMetrics) -> ThresholdConfig:
        """Otimização por mutação adaptativa."""
        mutation_strength = 0.05
        
        # Ajustar força de mutação baseada na performance
        if metrics.accuracy < 0.8:
            mutation_strength = 0.1  # Mutação mais forte
        elif metrics.accuracy > 0.9:
            mutation_strength = 0.02  # Mutação mais sutil
        
        return self._mutate_configuration(self.current_config, mutation_strength)
    
    def _mutate_configuration(self, config: ThresholdConfig, mutation_strength: float = 0.05) -> ThresholdConfig:
        """Aplica mutação a uma configuração."""
        new_config = ThresholdConfig(
            bird_threshold=self._mutate_parameter(config.bird_threshold, 
                self.parameter_bounds['bird_threshold'], mutation_strength),
            confidence_threshold=self._mutate_parameter(config.confidence_threshold,
                self.parameter_bounds['confidence_threshold'], mutation_strength),
            boost_factor=self._mutate_parameter(config.boost_factor,
                self.parameter_bounds['boost_factor'], mutation_strength),
            color_weight=self._mutate_parameter(config.color_weight,
                self.parameter_bounds['color_weight'], mutation_strength),
            shape_weight=self._mutate_parameter(config.shape_weight,
                self.parameter_bounds['shape_weight'], mutation_strength),
            pattern_weight=self._mutate_parameter(config.pattern_weight,
                self.parameter_bounds['pattern_weight'], mutation_strength),
            detection_sensitivity=self._mutate_parameter(config.detection_sensitivity,
                self.parameter_bounds['detection_sensitivity'], mutation_strength),
            reasoning_threshold=self._mutate_parameter(config.reasoning_threshold,
                self.parameter_bounds['reasoning_threshold'], mutation_strength),
            adaptation_rate=self._mutate_parameter(config.adaptation_rate,
                self.parameter_bounds['adaptation_rate'], mutation_strength)
        )
        
        return new_config
    
    def _mutate_parameter(self, value: float, bounds: Tuple[float, float], 
                         mutation_strength: float) -> float:
        """Aplica mutação a um parâmetro individual."""
        min_val, max_val = bounds
        range_size = max_val - min_val
        
        # Mutação gaussiana
        mutation = np.random.normal(0, mutation_strength * range_size)
        new_value = value + mutation
        
        # Manter dentro dos limites
        return max(min_val, min(max_val, new_value))
    
    def _evaluate_configuration(self, config: ThresholdConfig, 
                              metrics: PerformanceMetrics) -> float:
        """Avalia uma configuração baseada nas métricas."""
        # Função de fitness simples
        fitness = metrics.f1_score * 0.4 + metrics.accuracy * 0.3 + (1 - metrics.false_positive_rate) * 0.3
        
        # Penalizar configurações extremas
        extreme_penalty = 0.0
        if config.bird_threshold < 0.2 or config.bird_threshold > 0.8:
            extreme_penalty += 0.1
        if config.confidence_threshold < 0.3 or config.confidence_threshold > 0.9:
            extreme_penalty += 0.1
        
        return fitness - extreme_penalty
    
    def _validate_configuration(self, config: ThresholdConfig) -> ThresholdConfig:
        """Valida e corrige uma configuração."""
        validated_config = ThresholdConfig(
            bird_threshold=max(self.parameter_bounds['bird_threshold'][0],
                             min(self.parameter_bounds['bird_threshold'][1], config.bird_threshold)),
            confidence_threshold=max(self.parameter_bounds['confidence_threshold'][0],
                                   min(self.parameter_bounds['confidence_threshold'][1], config.confidence_threshold)),
            boost_factor=max(self.parameter_bounds['boost_factor'][0],
                           min(self.parameter_bounds['boost_factor'][1], config.boost_factor)),
            color_weight=max(self.parameter_bounds['color_weight'][0],
                           min(self.parameter_bounds['color_weight'][1], config.color_weight)),
            shape_weight=max(self.parameter_bounds['shape_weight'][0],
                           min(self.parameter_bounds['shape_weight'][1], config.shape_weight)),
            pattern_weight=max(self.parameter_bounds['pattern_weight'][0],
                             min(self.parameter_bounds['pattern_weight'][1], config.pattern_weight)),
            detection_sensitivity=max(self.parameter_bounds['detection_sensitivity'][0],
                                    min(self.parameter_bounds['detection_sensitivity'][1], config.detection_sensitivity)),
            reasoning_threshold=max(self.parameter_bounds['reasoning_threshold'][0],
                                  min(self.parameter_bounds['reasoning_threshold'][1], config.reasoning_threshold)),
            adaptation_rate=max(self.parameter_bounds['adaptation_rate'][0],
                              min(self.parameter_bounds['adaptation_rate'][1], config.adaptation_rate))
        )
        
        return validated_config
    
    def _calculate_fitness_score(self, metrics: PerformanceMetrics) -> float:
        """Calcula score de fitness para uma configuração."""
        return self._evaluate_configuration(self.current_config, metrics)
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Retorna status da otimização."""
        return {
            "optimization_active": self.optimization_active,
            "current_config": {
                "bird_threshold": self.current_config.bird_threshold,
                "confidence_threshold": self.current_config.confidence_threshold,
                "boost_factor": self.current_config.boost_factor,
                "color_weight": self.current_config.color_weight,
                "shape_weight": self.current_config.shape_weight,
                "pattern_weight": self.current_config.pattern_weight,
                "detection_sensitivity": self.current_config.detection_sensitivity,
                "reasoning_threshold": self.current_config.reasoning_threshold,
                "adaptation_rate": self.current_config.adaptation_rate
            },
            "config_history_size": len(self.config_history),
            "last_optimization_time": self.last_optimization_time,
            "optimization_frequency": self.optimization_frequency
        }

class AutoOptimizationSystem:
    """
    Sistema principal de auto-otimização que coordena monitoramento e otimização.
    """
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.threshold_optimizer = ThresholdOptimizer()
        
        # Estado do sistema
        self.is_active = False
        self.optimization_enabled = True
        
        # Configurações
        self.save_config_path = "data/optimized_thresholds.json"
        self.load_config_path = "data/optimized_thresholds.json"
        
        # Carregar configuração salva se existir
        self._load_saved_configuration()
        
    def start_optimization(self):
        """Inicia o sistema de auto-otimização."""
        self.is_active = True
        self.threshold_optimizer.optimization_active = True
        logger.info("Sistema de auto-otimização iniciado")
    
    def stop_optimization(self):
        """Para o sistema de auto-otimização."""
        self.is_active = False
        self.threshold_optimizer.optimization_active = False
        logger.info("Sistema de auto-otimização parado")
    
    def process_detection(self, 
                         is_bird: bool, 
                         predicted_bird: bool, 
                         confidence: float, 
                         processing_time: float) -> Optional[ThresholdConfig]:
        """
        Processa uma detecção e retorna nova configuração se otimizada.
        
        Args:
            is_bird: Se a imagem é realmente um pássaro
            predicted_bird: Se o sistema previu que é um pássaro
            confidence: Confiança da predição
            processing_time: Tempo de processamento
            
        Returns:
            Nova configuração de thresholds se otimizada, None caso contrário
        """
        if not self.is_active:
            return None
        
        # Atualizar métricas
        self.performance_monitor.update_metrics(
            is_bird, predicted_bird, confidence, processing_time
        )
        
        # Verificar se deve otimizar
        if self.threshold_optimizer.should_optimize(self.performance_monitor.total_detections):
            # Obter métricas atuais
            current_metrics = self.performance_monitor.current_metrics
            if current_metrics:
                # Otimizar thresholds
                new_config = self.threshold_optimizer.optimize_thresholds(current_metrics)
                
                # Salvar configuração
                self._save_configuration(new_config)
                
                logger.info(f"Thresholds otimizados: accuracy={current_metrics.accuracy:.3f}, "
                           f"f1_score={current_metrics.f1_score:.3f}")
                
                return new_config
        
        return None
    
    def get_current_configuration(self) -> ThresholdConfig:
        """Retorna configuração atual de thresholds."""
        return self.threshold_optimizer.current_config
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Retorna relatório completo de performance."""
        performance_summary = self.performance_monitor.get_performance_summary()
        performance_trend = self.performance_monitor.get_performance_trend()
        optimization_status = self.threshold_optimizer.get_optimization_status()
        
        return {
            "performance_summary": performance_summary,
            "performance_trend": performance_trend,
            "optimization_status": optimization_status,
            "system_active": self.is_active,
            "optimization_enabled": self.optimization_enabled
        }
    
    def _save_configuration(self, config: ThresholdConfig):
        """Salva configuração em arquivo."""
        try:
            config_data = {
                "bird_threshold": config.bird_threshold,
                "confidence_threshold": config.confidence_threshold,
                "boost_factor": config.boost_factor,
                "color_weight": config.color_weight,
                "shape_weight": config.shape_weight,
                "pattern_weight": config.pattern_weight,
                "detection_sensitivity": config.detection_sensitivity,
                "reasoning_threshold": config.reasoning_threshold,
                "adaptation_rate": config.adaptation_rate,
                "timestamp": datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(self.save_config_path), exist_ok=True)
            with open(self.save_config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Erro ao salvar configuração: {e}")
    
    def _load_saved_configuration(self):
        """Carrega configuração salva do arquivo."""
        try:
            if os.path.exists(self.load_config_path):
                with open(self.load_config_path, 'r') as f:
                    config_data = json.load(f)
                
                self.threshold_optimizer.current_config = ThresholdConfig(
                    bird_threshold=config_data.get('bird_threshold', 0.5),
                    confidence_threshold=config_data.get('confidence_threshold', 0.6),
                    boost_factor=config_data.get('boost_factor', 0.1),
                    color_weight=config_data.get('color_weight', 0.3),
                    shape_weight=config_data.get('shape_weight', 0.4),
                    pattern_weight=config_data.get('pattern_weight', 0.3),
                    detection_sensitivity=config_data.get('detection_sensitivity', 0.8),
                    reasoning_threshold=config_data.get('reasoning_threshold', 0.5),
                    adaptation_rate=config_data.get('adaptation_rate', 0.05)
                )
                
                logger.info("Configuração de thresholds carregada com sucesso")
                
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {e}")
    
    def reset_optimization(self):
        """Reseta o sistema de otimização."""
        self.performance_monitor = PerformanceMonitor()
        self.threshold_optimizer = ThresholdOptimizer()
        logger.info("Sistema de otimização resetado")
