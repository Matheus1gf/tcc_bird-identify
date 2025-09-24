#!/usr/bin/env python3
"""
Sistema de Otimização Apurada dos Pesos
=======================================

Este sistema implementa:
- Otimização precisa dos pesos dos componentes
- Análise de importância dos pesos
- Otimização multi-objetivo
- Ajuste fino baseado em gradientes
- Sistema de pesos adaptativos
- Otimização por componentes específicos
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
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

@dataclass
class WeightConfiguration:
    """Configuração de pesos otimizada."""
    # Pesos principais
    color_weight: float
    shape_weight: float
    pattern_weight: float
    texture_weight: float
    size_weight: float
    
    # Pesos de confiança
    yolo_confidence_weight: float
    color_confidence_weight: float
    shape_confidence_weight: float
    pattern_confidence_weight: float
    
    # Pesos de características específicas
    beak_weight: float
    wing_weight: float
    tail_weight: float
    eye_weight: float
    
    # Pesos de contexto
    background_weight: float
    lighting_weight: float
    angle_weight: float
    
    # Pesos de aprendizado
    learned_pattern_weight: float
    species_boost_weight: float
    characteristic_boost_weight: float
    
    # Metadados
    optimization_score: float
    timestamp: datetime

@dataclass
class ComponentPerformance:
    """Performance de componentes individuais."""
    component_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    importance_score: float
    correlation_with_target: float
    stability_score: float

class WeightAnalyzer:
    """
    Analisador de importância e performance dos pesos.
    """
    
    def __init__(self):
        self.component_history: Dict[str, deque] = {}
        self.performance_history: deque = deque(maxlen=1000)
        self.correlation_matrix: Optional[np.ndarray] = None
        
        # Componentes para análise
        self.components = [
            'color', 'shape', 'pattern', 'texture', 'size',
            'yolo_confidence', 'color_confidence', 'shape_confidence', 'pattern_confidence',
            'beak', 'wing', 'tail', 'eye',
            'background', 'lighting', 'angle',
            'learned_pattern', 'species_boost', 'characteristic_boost'
        ]
        
        # Inicializar histórico para cada componente
        for component in self.components:
            self.component_history[component] = deque(maxlen=100)
    
    def analyze_component_importance(self, 
                                   detection_results: List[Dict[str, Any]], 
                                   ground_truth: List[bool]) -> Dict[str, ComponentPerformance]:
        """
        Analisa a importância de cada componente baseado nos resultados.
        
        Args:
            detection_results: Lista de resultados de detecção
            ground_truth: Lista de valores verdadeiros
            
        Returns:
            Dicionário com performance de cada componente
        """
        component_performance = {}
        
        for component in self.components:
            # Extrair scores do componente
            component_scores = []
            component_predictions = []
            
            for result in detection_results:
                if component in result.get('component_scores', {}):
                    score = result['component_scores'][component]
                    component_scores.append(score)
                    
                    # Converter score em predição binária
                    threshold = 0.5  # Threshold padrão
                    component_predictions.append(score > threshold)
            
            if len(component_scores) > 0 and len(component_predictions) == len(ground_truth):
                # Calcular métricas
                accuracy = accuracy_score(ground_truth, component_predictions)
                precision = precision_score(ground_truth, component_predictions, zero_division=0)
                recall = recall_score(ground_truth, component_predictions, zero_division=0)
                f1 = f1_score(ground_truth, component_predictions, zero_division=0)
                
                # Calcular importância baseada na correlação
                importance_score = self._calculate_importance_score(component_scores, ground_truth)
                
                # Calcular correlação com target
                correlation = self._calculate_correlation(component_scores, ground_truth)
                
                # Calcular estabilidade
                stability_score = self._calculate_stability_score(component_scores)
                
                component_performance[component] = ComponentPerformance(
                    component_name=component,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    importance_score=importance_score,
                    correlation_with_target=correlation,
                    stability_score=stability_score
                )
                
                # Atualizar histórico
                self.component_history[component].append({
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'importance': importance_score,
                    'timestamp': datetime.now()
                })
        
        return component_performance
    
    def _calculate_importance_score(self, scores: List[float], targets: List[bool]) -> float:
        """Calcula score de importância de um componente."""
        if len(scores) != len(targets):
            return 0.0
        
        # Converter targets para float
        targets_float = [float(t) for t in targets]
        
        # Calcular correlação
        correlation = np.corrcoef(scores, targets_float)[0, 1]
        
        # Calcular variância explicada
        explained_variance = correlation ** 2
        
        # Calcular estabilidade
        stability = 1.0 - np.std(scores) / (np.mean(scores) + 1e-8)
        
        # Score combinado
        importance_score = (explained_variance * 0.5 + 
                          abs(correlation) * 0.3 + 
                          stability * 0.2)
        
        return max(0.0, min(1.0, importance_score))
    
    def _calculate_correlation(self, scores: List[float], targets: List[bool]) -> float:
        """Calcula correlação entre scores e targets."""
        if len(scores) != len(targets):
            return 0.0
        
        targets_float = [float(t) for t in targets]
        correlation = np.corrcoef(scores, targets_float)[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def _calculate_stability_score(self, scores: List[float]) -> float:
        """Calcula score de estabilidade dos scores."""
        if len(scores) < 2:
            return 1.0
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Estabilidade inversamente proporcional ao coeficiente de variação
        cv = std_score / (mean_score + 1e-8)
        stability = 1.0 / (1.0 + cv)
        
        return max(0.0, min(1.0, stability))
    
    def get_component_trends(self) -> Dict[str, Dict[str, float]]:
        """Retorna tendências dos componentes ao longo do tempo."""
        trends = {}
        
        for component, history in self.component_history.items():
            if len(history) >= 10:
                recent_data = list(history)[-10:]
                older_data = list(history)[-20:-10] if len(history) >= 20 else []
                
                recent_importance = np.mean([h['importance'] for h in recent_data])
                recent_accuracy = np.mean([h['accuracy'] for h in recent_data])
                
                trend_data = {
                    'recent_importance': recent_importance,
                    'recent_accuracy': recent_accuracy,
                    'trend_direction': 'stable'
                }
                
                if older_data:
                    older_importance = np.mean([h['importance'] for h in older_data])
                    older_accuracy = np.mean([h['accuracy'] for h in older_data])
                    
                    importance_change = recent_importance - older_importance
                    accuracy_change = recent_accuracy - older_accuracy
                    
                    if importance_change > 0.05:
                        trend_data['trend_direction'] = 'increasing'
                    elif importance_change < -0.05:
                        trend_data['trend_direction'] = 'decreasing'
                    
                    trend_data.update({
                        'importance_change': importance_change,
                        'accuracy_change': accuracy_change
                    })
                
                trends[component] = trend_data
        
        return trends

class MultiObjectiveOptimizer:
    """
    Otimizador multi-objetivo para pesos.
    """
    
    def __init__(self):
        self.objectives = {
            'accuracy': 0.3,
            'precision': 0.2,
            'recall': 0.2,
            'f1_score': 0.2,
            'stability': 0.1
        }
        
        # Limites dos pesos
        self.weight_bounds = {
            'color_weight': (0.0, 1.0),
            'shape_weight': (0.0, 1.0),
            'pattern_weight': (0.0, 1.0),
            'texture_weight': (0.0, 1.0),
            'size_weight': (0.0, 1.0),
            'yolo_confidence_weight': (0.0, 1.0),
            'color_confidence_weight': (0.0, 1.0),
            'shape_confidence_weight': (0.0, 1.0),
            'pattern_confidence_weight': (0.0, 1.0),
            'beak_weight': (0.0, 1.0),
            'wing_weight': (0.0, 1.0),
            'tail_weight': (0.0, 1.0),
            'eye_weight': (0.0, 1.0),
            'background_weight': (0.0, 1.0),
            'lighting_weight': (0.0, 1.0),
            'angle_weight': (0.0, 1.0),
            'learned_pattern_weight': (0.0, 1.0),
            'species_boost_weight': (0.0, 1.0),
            'characteristic_boost_weight': (0.0, 1.0)
        }
        
        # Histórico de otimizações
        self.optimization_history: List[Tuple[WeightConfiguration, float]] = []
        
    def optimize_weights(self, 
                       component_performance: Dict[str, ComponentPerformance],
                       current_weights: WeightConfiguration) -> WeightConfiguration:
        """
        Otimiza pesos usando otimização multi-objetivo.
        
        Args:
            component_performance: Performance dos componentes
            current_weights: Configuração atual de pesos
            
        Returns:
            Nova configuração de pesos otimizada
        """
        logger.info("Iniciando otimização multi-objetivo dos pesos")
        
        # Preparar dados para otimização
        optimization_data = self._prepare_optimization_data(component_performance)
        
        # Definir função objetivo
        def objective_function(weights_array):
            return self._evaluate_weight_configuration(weights_array, optimization_data)
        
        # Definir limites
        bounds = list(self.weight_bounds.values())
        
        # Otimização usando algoritmo genético diferencial
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=100,
            popsize=15,
            seed=42,
            atol=1e-6,
            tol=1e-6
        )
        
        # Converter resultado para configuração de pesos
        optimized_weights = self._array_to_weight_configuration(result.x)
        optimized_weights.optimization_score = -result.fun  # Negativo porque minimizamos
        optimized_weights.timestamp = datetime.now()
        
        # Registrar no histórico
        self.optimization_history.append((optimized_weights, optimized_weights.optimization_score))
        
        # Manter apenas histórico recente
        if len(self.optimization_history) > 50:
            self.optimization_history = self.optimization_history[-50:]
        
        logger.info(f"Otimização concluída. Score: {optimized_weights.optimization_score:.4f}")
        
        return optimized_weights
    
    def _prepare_optimization_data(self, 
                                 component_performance: Dict[str, ComponentPerformance]) -> Dict[str, Any]:
        """Prepara dados para otimização."""
        optimization_data = {}
        
        for component, perf in component_performance.items():
            optimization_data[component] = {
                'accuracy': perf.accuracy,
                'precision': perf.precision,
                'recall': perf.recall,
                'f1_score': perf.f1_score,
                'importance_score': perf.importance_score,
                'correlation': perf.correlation_with_target,
                'stability': perf.stability_score
            }
        
        return optimization_data
    
    def _evaluate_weight_configuration(self, 
                                     weights_array: np.ndarray, 
                                     optimization_data: Dict[str, Any]) -> float:
        """
        Avalia uma configuração de pesos.
        
        Args:
            weights_array: Array de pesos
            optimization_data: Dados de otimização
            
        Returns:
            Score negativo (para minimização)
        """
        # Converter array para configuração
        weight_config = self._array_to_weight_configuration(weights_array)
        
        # Calcular score multi-objetivo
        total_score = 0.0
        
        for objective, weight in self.objectives.items():
            objective_score = self._calculate_objective_score(objective, weight_config, optimization_data)
            total_score += weight * objective_score
        
        # Adicionar penalidades por configurações inválidas
        penalty = self._calculate_penalty(weight_config)
        
        # Retornar negativo para minimização
        return -(total_score - penalty)
    
    def _calculate_objective_score(self, 
                                 objective: str, 
                                 weight_config: WeightConfiguration,
                                 optimization_data: Dict[str, Any]) -> float:
        """Calcula score para um objetivo específico."""
        if objective == 'accuracy':
            return self._calculate_accuracy_score(weight_config, optimization_data)
        elif objective == 'precision':
            return self._calculate_precision_score(weight_config, optimization_data)
        elif objective == 'recall':
            return self._calculate_recall_score(weight_config, optimization_data)
        elif objective == 'f1_score':
            return self._calculate_f1_score(weight_config, optimization_data)
        elif objective == 'stability':
            return self._calculate_stability_score(weight_config, optimization_data)
        else:
            return 0.0
    
    def _calculate_accuracy_score(self, 
                                weight_config: WeightConfiguration,
                                optimization_data: Dict[str, Any]) -> float:
        """Calcula score de accuracy."""
        weighted_accuracy = 0.0
        total_weight = 0.0
        
        for component, data in optimization_data.items():
            weight = getattr(weight_config, f"{component}_weight", 0.0)
            accuracy = data['accuracy']
            
            weighted_accuracy += weight * accuracy
            total_weight += weight
        
        return weighted_accuracy / (total_weight + 1e-8)
    
    def _calculate_precision_score(self, 
                                 weight_config: WeightConfiguration,
                                 optimization_data: Dict[str, Any]) -> float:
        """Calcula score de precision."""
        weighted_precision = 0.0
        total_weight = 0.0
        
        for component, data in optimization_data.items():
            weight = getattr(weight_config, f"{component}_weight", 0.0)
            precision = data['precision']
            
            weighted_precision += weight * precision
            total_weight += weight
        
        return weighted_precision / (total_weight + 1e-8)
    
    def _calculate_recall_score(self, 
                              weight_config: WeightConfiguration,
                              optimization_data: Dict[str, Any]) -> float:
        """Calcula score de recall."""
        weighted_recall = 0.0
        total_weight = 0.0
        
        for component, data in optimization_data.items():
            weight = getattr(weight_config, f"{component}_weight", 0.0)
            recall = data['recall']
            
            weighted_recall += weight * recall
            total_weight += weight
        
        return weighted_recall / (total_weight + 1e-8)
    
    def _calculate_f1_score(self, 
                           weight_config: WeightConfiguration,
                           optimization_data: Dict[str, Any]) -> float:
        """Calcula score de F1."""
        weighted_f1 = 0.0
        total_weight = 0.0
        
        for component, data in optimization_data.items():
            weight = getattr(weight_config, f"{component}_weight", 0.0)
            f1 = data['f1_score']
            
            weighted_f1 += weight * f1
            total_weight += weight
        
        return weighted_f1 / (total_weight + 1e-8)
    
    def _calculate_stability_score(self, 
                                  weight_config: WeightConfiguration,
                                  optimization_data: Dict[str, Any]) -> float:
        """Calcula score de estabilidade."""
        weighted_stability = 0.0
        total_weight = 0.0
        
        for component, data in optimization_data.items():
            weight = getattr(weight_config, f"{component}_weight", 0.0)
            stability = data['stability']
            
            weighted_stability += weight * stability
            total_weight += weight
        
        return weighted_stability / (total_weight + 1e-8)
    
    def _calculate_penalty(self, weight_config: WeightConfiguration) -> float:
        """Calcula penalidade por configurações inválidas."""
        penalty = 0.0
        
        # Penalizar pesos muito desbalanceados
        main_weights = [
            weight_config.color_weight,
            weight_config.shape_weight,
            weight_config.pattern_weight,
            weight_config.texture_weight,
            weight_config.size_weight
        ]
        
        weight_sum = sum(main_weights)
        if weight_sum < 0.1 or weight_sum > 2.0:
            penalty += 0.5
        
        # Penalizar pesos extremos
        for weight in main_weights:
            if weight > 0.8:
                penalty += 0.1
            if weight < 0.01:
                penalty += 0.1
        
        return penalty
    
    def _array_to_weight_configuration(self, weights_array: np.ndarray) -> WeightConfiguration:
        """Converte array de pesos para configuração."""
        weight_names = list(self.weight_bounds.keys())
        
        if len(weights_array) != len(weight_names):
            # Usar valores padrão se o array não tiver o tamanho correto
            weights_array = np.array([0.2] * len(weight_names))
        
        # Criar dicionário de pesos
        weights_dict = dict(zip(weight_names, weights_array))
        
        return WeightConfiguration(
            color_weight=weights_dict.get('color_weight', 0.2),
            shape_weight=weights_dict.get('shape_weight', 0.2),
            pattern_weight=weights_dict.get('pattern_weight', 0.2),
            texture_weight=weights_dict.get('texture_weight', 0.1),
            size_weight=weights_dict.get('size_weight', 0.1),
            yolo_confidence_weight=weights_dict.get('yolo_confidence_weight', 0.3),
            color_confidence_weight=weights_dict.get('color_confidence_weight', 0.2),
            shape_confidence_weight=weights_dict.get('shape_confidence_weight', 0.2),
            pattern_confidence_weight=weights_dict.get('pattern_confidence_weight', 0.2),
            beak_weight=weights_dict.get('beak_weight', 0.1),
            wing_weight=weights_dict.get('wing_weight', 0.1),
            tail_weight=weights_dict.get('tail_weight', 0.1),
            eye_weight=weights_dict.get('eye_weight', 0.1),
            background_weight=weights_dict.get('background_weight', 0.05),
            lighting_weight=weights_dict.get('lighting_weight', 0.05),
            angle_weight=weights_dict.get('angle_weight', 0.05),
            learned_pattern_weight=weights_dict.get('learned_pattern_weight', 0.2),
            species_boost_weight=weights_dict.get('species_boost_weight', 0.1),
            characteristic_boost_weight=weights_dict.get('characteristic_boost_weight', 0.1),
            optimization_score=0.0,
            timestamp=datetime.now()
        )

class GradientBasedOptimizer:
    """
    Otimizador baseado em gradientes para ajuste fino.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.gradient_history: deque = deque(maxlen=100)
        self.momentum = 0.9
        self.velocity = {}
        
    def fine_tune_weights(self, 
                         current_weights: WeightConfiguration,
                         gradient_data: Dict[str, float]) -> WeightConfiguration:
        """
        Ajusta pesos usando gradientes.
        
        Args:
            current_weights: Configuração atual de pesos
            gradient_data: Dados de gradientes
            
        Returns:
            Nova configuração de pesos ajustada
        """
        logger.info("Iniciando ajuste fino baseado em gradientes")
        
        # Converter configuração para dicionário
        weights_dict = self._weight_configuration_to_dict(current_weights)
        
        # Aplicar gradientes
        for weight_name, gradient in gradient_data.items():
            if weight_name in weights_dict:
                # Calcular velocidade com momentum
                if weight_name not in self.velocity:
                    self.velocity[weight_name] = 0.0
                
                self.velocity[weight_name] = (self.momentum * self.velocity[weight_name] + 
                                            self.learning_rate * gradient)
                
                # Aplicar atualização
                weights_dict[weight_name] += self.velocity[weight_name]
                
                # Manter dentro dos limites
                weights_dict[weight_name] = max(0.0, min(1.0, weights_dict[weight_name]))
        
        # Converter de volta para configuração
        new_weights = self._dict_to_weight_configuration(weights_dict)
        new_weights.optimization_score = current_weights.optimization_score
        new_weights.timestamp = datetime.now()
        
        # Registrar gradientes
        self.gradient_history.append({
            'gradients': gradient_data,
            'timestamp': datetime.now()
        })
        
        logger.info("Ajuste fino concluído")
        
        return new_weights
    
    def _weight_configuration_to_dict(self, config: WeightConfiguration) -> Dict[str, float]:
        """Converte configuração de pesos para dicionário."""
        return {
            'color_weight': config.color_weight,
            'shape_weight': config.shape_weight,
            'pattern_weight': config.pattern_weight,
            'texture_weight': config.texture_weight,
            'size_weight': config.size_weight,
            'yolo_confidence_weight': config.yolo_confidence_weight,
            'color_confidence_weight': config.color_confidence_weight,
            'shape_confidence_weight': config.shape_confidence_weight,
            'pattern_confidence_weight': config.pattern_confidence_weight,
            'beak_weight': config.beak_weight,
            'wing_weight': config.wing_weight,
            'tail_weight': config.tail_weight,
            'eye_weight': config.eye_weight,
            'background_weight': config.background_weight,
            'lighting_weight': config.lighting_weight,
            'angle_weight': config.angle_weight,
            'learned_pattern_weight': config.learned_pattern_weight,
            'species_boost_weight': config.species_boost_weight,
            'characteristic_boost_weight': config.characteristic_boost_weight
        }
    
    def _dict_to_weight_configuration(self, weights_dict: Dict[str, float]) -> WeightConfiguration:
        """Converte dicionário para configuração de pesos."""
        return WeightConfiguration(
            color_weight=weights_dict.get('color_weight', 0.2),
            shape_weight=weights_dict.get('shape_weight', 0.2),
            pattern_weight=weights_dict.get('pattern_weight', 0.2),
            texture_weight=weights_dict.get('texture_weight', 0.1),
            size_weight=weights_dict.get('size_weight', 0.1),
            yolo_confidence_weight=weights_dict.get('yolo_confidence_weight', 0.3),
            color_confidence_weight=weights_dict.get('color_confidence_weight', 0.2),
            shape_confidence_weight=weights_dict.get('shape_confidence_weight', 0.2),
            pattern_confidence_weight=weights_dict.get('pattern_confidence_weight', 0.2),
            beak_weight=weights_dict.get('beak_weight', 0.1),
            wing_weight=weights_dict.get('wing_weight', 0.1),
            tail_weight=weights_dict.get('tail_weight', 0.1),
            eye_weight=weights_dict.get('eye_weight', 0.1),
            background_weight=weights_dict.get('background_weight', 0.05),
            lighting_weight=weights_dict.get('lighting_weight', 0.05),
            angle_weight=weights_dict.get('angle_weight', 0.05),
            learned_pattern_weight=weights_dict.get('learned_pattern_weight', 0.2),
            species_boost_weight=weights_dict.get('species_boost_weight', 0.1),
            characteristic_boost_weight=weights_dict.get('characteristic_boost_weight', 0.1),
            optimization_score=0.0,
            timestamp=datetime.now()
        )

class AdvancedWeightOptimizationSystem:
    """
    Sistema principal de otimização apurada dos pesos.
    """
    
    def __init__(self):
        self.weight_analyzer = WeightAnalyzer()
        self.multi_objective_optimizer = MultiObjectiveOptimizer()
        self.gradient_optimizer = GradientBasedOptimizer()
        
        # Configuração atual de pesos
        self.current_weights = self._create_default_weights()
        
        # Estado do sistema
        self.is_active = False
        self.optimization_frequency = 100  # Otimizar a cada 100 detecções
        self.last_optimization_time = None
        
        # Configurações
        self.save_weights_path = "data/optimized_weights.json"
        self.load_weights_path = "data/optimized_weights.json"
        
        # Carregar pesos salvos se existirem
        self._load_saved_weights()
        
    def _create_default_weights(self) -> WeightConfiguration:
        """Cria configuração padrão de pesos."""
        return WeightConfiguration(
            color_weight=0.25,
            shape_weight=0.25,
            pattern_weight=0.25,
            texture_weight=0.1,
            size_weight=0.1,
            yolo_confidence_weight=0.3,
            color_confidence_weight=0.2,
            shape_confidence_weight=0.2,
            pattern_confidence_weight=0.2,
            beak_weight=0.1,
            wing_weight=0.1,
            tail_weight=0.1,
            eye_weight=0.1,
            background_weight=0.05,
            lighting_weight=0.05,
            angle_weight=0.05,
            learned_pattern_weight=0.2,
            species_boost_weight=0.1,
            characteristic_boost_weight=0.1,
            optimization_score=0.0,
            timestamp=datetime.now()
        )
    
    def start_optimization(self):
        """Inicia o sistema de otimização de pesos."""
        self.is_active = True
        logger.info("Sistema de otimização apurada de pesos iniciado")
    
    def stop_optimization(self):
        """Para o sistema de otimização de pesos."""
        self.is_active = False
        logger.info("Sistema de otimização apurada de pesos parado")
    
    def process_detection_batch(self, 
                              detection_results: List[Dict[str, Any]], 
                              ground_truth: List[bool]) -> Optional[WeightConfiguration]:
        """
        Processa um lote de detecções para otimização.
        
        Args:
            detection_results: Lista de resultados de detecção
            ground_truth: Lista de valores verdadeiros
            
        Returns:
            Nova configuração de pesos se otimizada, None caso contrário
        """
        if not self.is_active:
            return None
        
        # Analisar performance dos componentes
        component_performance = self.weight_analyzer.analyze_component_importance(
            detection_results, ground_truth
        )
        
        # Verificar se deve otimizar
        if self._should_optimize():
            # Otimização multi-objetivo
            optimized_weights = self.multi_objective_optimizer.optimize_weights(
                component_performance, self.current_weights
            )
            
            # Ajuste fino baseado em gradientes
            gradient_data = self._calculate_gradients(component_performance)
            final_weights = self.gradient_optimizer.fine_tune_weights(
                optimized_weights, gradient_data
            )
            
            # Atualizar pesos atuais
            self.current_weights = final_weights
            
            # Salvar configuração
            self._save_weights(final_weights)
            
            self.last_optimization_time = time.time()
            
            logger.info(f"Pesos otimizados: score={final_weights.optimization_score:.4f}")
            
            return final_weights
        
        return None
    
    def _should_optimize(self) -> bool:
        """Verifica se deve otimizar."""
        if self.last_optimization_time is None:
            return True
        
        time_since_last = time.time() - self.last_optimization_time
        return time_since_last >= 300  # Mínimo 5 minutos entre otimizações
    
    def _calculate_gradients(self, 
                           component_performance: Dict[str, ComponentPerformance]) -> Dict[str, float]:
        """Calcula gradientes para ajuste fino."""
        gradients = {}
        
        for component, perf in component_performance.items():
            # Gradiente baseado na importância e performance
            gradient = (perf.importance_score * 0.5 + 
                       perf.f1_score * 0.3 + 
                       perf.correlation_with_target * 0.2)
            
            # Normalizar gradiente
            gradient = (gradient - 0.5) * 0.1  # Escalar para [-0.05, 0.05]
            
            gradients[f"{component}_weight"] = gradient
        
        return gradients
    
    def get_current_weights(self) -> WeightConfiguration:
        """Retorna configuração atual de pesos."""
        return self.current_weights
    
    def get_component_analysis(self) -> Dict[str, Any]:
        """Retorna análise dos componentes."""
        trends = self.weight_analyzer.get_component_trends()
        
        return {
            "component_trends": trends,
            "optimization_history_size": len(self.multi_objective_optimizer.optimization_history),
            "gradient_history_size": len(self.gradient_optimizer.gradient_history),
            "current_weights": {
                "color_weight": self.current_weights.color_weight,
                "shape_weight": self.current_weights.shape_weight,
                "pattern_weight": self.current_weights.pattern_weight,
                "texture_weight": self.current_weights.texture_weight,
                "size_weight": self.current_weights.size_weight,
                "optimization_score": self.current_weights.optimization_score
            }
        }
    
    def _save_weights(self, weights: WeightConfiguration):
        """Salva pesos em arquivo."""
        try:
            weights_data = {
                "color_weight": weights.color_weight,
                "shape_weight": weights.shape_weight,
                "pattern_weight": weights.pattern_weight,
                "texture_weight": weights.texture_weight,
                "size_weight": weights.size_weight,
                "yolo_confidence_weight": weights.yolo_confidence_weight,
                "color_confidence_weight": weights.color_confidence_weight,
                "shape_confidence_weight": weights.shape_confidence_weight,
                "pattern_confidence_weight": weights.pattern_confidence_weight,
                "beak_weight": weights.beak_weight,
                "wing_weight": weights.wing_weight,
                "tail_weight": weights.tail_weight,
                "eye_weight": weights.eye_weight,
                "background_weight": weights.background_weight,
                "lighting_weight": weights.lighting_weight,
                "angle_weight": weights.angle_weight,
                "learned_pattern_weight": weights.learned_pattern_weight,
                "species_boost_weight": weights.species_boost_weight,
                "characteristic_boost_weight": weights.characteristic_boost_weight,
                "optimization_score": weights.optimization_score,
                "timestamp": weights.timestamp.isoformat()
            }
            
            os.makedirs(os.path.dirname(self.save_weights_path), exist_ok=True)
            with open(self.save_weights_path, 'w') as f:
                json.dump(weights_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Erro ao salvar pesos: {e}")
    
    def _load_saved_weights(self):
        """Carrega pesos salvos do arquivo."""
        try:
            if os.path.exists(self.load_weights_path):
                with open(self.load_weights_path, 'r') as f:
                    weights_data = json.load(f)
                
                self.current_weights = WeightConfiguration(
                    color_weight=weights_data.get('color_weight', 0.25),
                    shape_weight=weights_data.get('shape_weight', 0.25),
                    pattern_weight=weights_data.get('pattern_weight', 0.25),
                    texture_weight=weights_data.get('texture_weight', 0.1),
                    size_weight=weights_data.get('size_weight', 0.1),
                    yolo_confidence_weight=weights_data.get('yolo_confidence_weight', 0.3),
                    color_confidence_weight=weights_data.get('color_confidence_weight', 0.2),
                    shape_confidence_weight=weights_data.get('shape_confidence_weight', 0.2),
                    pattern_confidence_weight=weights_data.get('pattern_confidence_weight', 0.2),
                    beak_weight=weights_data.get('beak_weight', 0.1),
                    wing_weight=weights_data.get('wing_weight', 0.1),
                    tail_weight=weights_data.get('tail_weight', 0.1),
                    eye_weight=weights_data.get('eye_weight', 0.1),
                    background_weight=weights_data.get('background_weight', 0.05),
                    lighting_weight=weights_data.get('lighting_weight', 0.05),
                    angle_weight=weights_data.get('angle_weight', 0.05),
                    learned_pattern_weight=weights_data.get('learned_pattern_weight', 0.2),
                    species_boost_weight=weights_data.get('species_boost_weight', 0.1),
                    characteristic_boost_weight=weights_data.get('characteristic_boost_weight', 0.1),
                    optimization_score=weights_data.get('optimization_score', 0.0),
                    timestamp=datetime.now()
                )
                
                logger.info("Pesos otimizados carregados com sucesso")
                
        except Exception as e:
            logger.error(f"Erro ao carregar pesos: {e}")
    
    def reset_optimization(self):
        """Reseta o sistema de otimização."""
        self.weight_analyzer = WeightAnalyzer()
        self.multi_objective_optimizer = MultiObjectiveOptimizer()
        self.gradient_optimizer = GradientBasedOptimizer()
        self.current_weights = self._create_default_weights()
        logger.info("Sistema de otimização de pesos resetado")
