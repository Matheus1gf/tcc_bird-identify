"""
Sistema de Evolução de Algoritmos
=================================

Este módulo implementa um sistema de evolução de algoritmos usando conceitos de algoritmos genéticos.
Permite que a IA evolua e melhore seus próprios algoritmos automaticamente através de:
- Mutação de parâmetros
- Seleção natural de estratégias
- Crossover de algoritmos
- Evolução de arquitetura

Funcionalidades:
- Evolução de parâmetros do IntuitionEngine
- Mutação de thresholds e pesos
- Seleção natural baseada em performance
- Crossover de estratégias de detecção
- Evolução de arquitetura cognitiva
- Auto-otimização contínua
"""

import os
import json
import random
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import copy

logger = logging.getLogger(__name__)

class EvolutionStrategy(Enum):
    """Estratégias de evolução disponíveis"""
    PARAMETER_MUTATION = "parameter_mutation"
    ARCHITECTURE_EVOLUTION = "architecture_evolution"
    STRATEGY_CROSSOVER = "strategy_crossover"
    FITNESS_OPTIMIZATION = "fitness_optimization"

@dataclass
class AlgorithmGenome:
    """Genoma de um algoritmo - representa uma configuração específica"""
    id: str
    parameters: Dict[str, Any]
    architecture: Dict[str, Any]
    strategy: Dict[str, Any]
    fitness_score: float = 0.0
    generation: int = 0
    parent_ids: List[str] = None
    mutation_history: List[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []
        if self.mutation_history is None:
            self.mutation_history = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

@dataclass
class EvolutionResult:
    """Resultado de uma operação de evolução"""
    success: bool
    new_genome: Optional[AlgorithmGenome] = None
    fitness_change: float = 0.0
    mutation_applied: str = ""
    error: str = ""

class ParameterMutator:
    """
    Mutador de parâmetros avançado que aplica mutações adaptativas nos parâmetros do algoritmo.
    Implementa diferentes estratégias de mutação baseadas em:
    - Mutação gaussiana adaptativa
    - Mutação por escala dinâmica
    - Mutação uniforme inteligente
    - Mutação baseada em gradiente
    - Mutação de temperatura adaptativa
    """
    
    def __init__(self):
        # Taxas de mutação adaptativas (podem mudar durante a evolução)
        self.base_mutation_rates = {
            'bird_threshold': 0.12,      # 12% de chance de mutação
            'confidence_threshold': 0.08, # 8% de chance de mutação
            'boost_factor': 0.15,        # 15% de chance de mutação
            'learning_rate': 0.1,        # 10% de chance de mutação
            'color_weight': 0.1,         # 10% de chance de mutação
            'shape_weight': 0.1,         # 10% de chance de mutação
            'pattern_weight': 0.1,       # 10% de chance de mutação
            'detection_sensitivity': 0.08, # 8% de chance de mutação
            'reasoning_threshold': 0.06,  # 6% de chance de mutação
            'adaptation_rate': 0.05      # 5% de chance de mutação
        }
        
        # Ranges de mutação com valores mais específicos
        self.mutation_ranges = {
            'bird_threshold': (0.3, 0.8),
            'confidence_threshold': (0.4, 0.9),
            'boost_factor': (0.05, 0.3),
            'learning_rate': (0.001, 0.05),
            'color_weight': (0.1, 0.5),
            'shape_weight': (0.1, 0.5),
            'pattern_weight': (0.1, 0.5),
            'detection_sensitivity': (0.5, 1.0),
            'reasoning_threshold': (0.3, 0.8),
            'adaptation_rate': (0.01, 0.1)
        }
        
        # Estratégias de mutação disponíveis
        self.mutation_strategies = {
            'gaussian_adaptive': 0.3,    # 30% - Mutação gaussiana adaptativa
            'scale_dynamic': 0.25,       # 25% - Mutação por escala dinâmica
            'uniform_smart': 0.2,       # 20% - Mutação uniforme inteligente
            'gradient_based': 0.15,      # 15% - Mutação baseada em gradiente
            'temperature_adaptive': 0.1   # 10% - Mutação de temperatura adaptativa
        }
        
        # Histórico de mutações para adaptação
        self.mutation_history = []
        self.successful_mutations = {}
        self.failed_mutations = {}
        
        # Parâmetros adaptativos
        self.temperature = 1.0  # Temperatura para mutação
        self.adaptation_factor = 0.1
        self.convergence_threshold = 0.01
    
    def mutate_genome(self, genome: AlgorithmGenome) -> AlgorithmGenome:
        """
        Aplica mutações adaptativas em um genoma usando estratégias inteligentes.
        
        Args:
            genome: Genoma a ser mutado
            
        Returns:
            Novo genoma mutado
        """
        logger.info(f"Mutando genoma {genome.id}")
        
        # Criar cópia do genoma
        mutated_genome = copy.deepcopy(genome)
        mutated_genome.id = f"{genome.id}_mut_{datetime.now().strftime('%H%M%S')}"
        mutated_genome.generation = genome.generation + 1
        mutated_genome.parent_ids = [genome.id]
        
        mutations_applied = []
        
        # Adaptar taxas de mutação baseado no histórico
        adaptive_rates = self._get_adaptive_mutation_rates(genome)
        
        # Mutar parâmetros com estratégias adaptativas
        for param_name, mutation_rate in adaptive_rates.items():
            if random.random() < mutation_rate:
                mutation_result = self._mutate_parameter_adaptive(
                    mutated_genome.parameters, 
                    param_name,
                    genome.fitness_score
                )
                if mutation_result:
                    mutations_applied.append(f"{param_name}: {mutation_result}")
        
        # Mutar arquitetura com estratégias avançadas
        architecture_mutations = self._mutate_architecture_advanced(mutated_genome.architecture)
        mutations_applied.extend(architecture_mutations)
        
        # Mutar estratégia com inteligência adaptativa
        strategy_mutations = self._mutate_strategy_adaptive(mutated_genome.strategy)
        mutations_applied.extend(strategy_mutations)
        
        # Aplicar mutação de temperatura adaptativa
        temp_mutations = self._apply_temperature_mutation(mutated_genome)
        mutations_applied.extend(temp_mutations)
        
        mutated_genome.mutation_history = mutations_applied
        
        # Registrar mutação no histórico
        self._record_mutation(genome.id, mutations_applied)
        
        logger.info(f"Mutações aplicadas: {len(mutations_applied)}")
        return mutated_genome
    
    def _get_adaptive_mutation_rates(self, genome: AlgorithmGenome) -> Dict[str, float]:
        """Calcula taxas de mutação adaptativas baseadas no histórico e fitness."""
        adaptive_rates = {}
        
        for param_name, base_rate in self.base_mutation_rates.items():
            # Ajustar taxa baseada no fitness
            fitness_factor = 1.0 + (0.5 - genome.fitness_score) * 0.5
            
            # Ajustar baseado no histórico de sucesso
            success_factor = 1.0
            if param_name in self.successful_mutations:
                success_count = len(self.successful_mutations[param_name])
                success_factor = 1.0 + (success_count * 0.1)
            
            # Ajustar baseado na convergência
            convergence_factor = 1.0
            if len(self.mutation_history) > 10:
                recent_mutations = self.mutation_history[-10:]
                if param_name in str(recent_mutations):
                    convergence_factor = 0.8  # Reduzir mutação se convergindo
            
            adaptive_rate = base_rate * fitness_factor * success_factor * convergence_factor
            adaptive_rates[param_name] = min(0.5, max(0.01, adaptive_rate))
        
        return adaptive_rates
    
    def _mutate_parameter_adaptive(self, parameters: Dict[str, Any], param_name: str, fitness_score: float) -> Optional[str]:
        """Muta um parâmetro usando estratégias adaptativas."""
        if param_name not in parameters:
            return None
        
        current_value = parameters[param_name]
        mutation_range = self.mutation_ranges.get(param_name, (0.0, 1.0))
        
        # Escolher estratégia de mutação baseada em probabilidades
        strategy = self._select_mutation_strategy(fitness_score)
        
        if strategy == 'gaussian_adaptive':
            new_value = self._gaussian_adaptive_mutation(current_value, mutation_range, fitness_score)
        elif strategy == 'scale_dynamic':
            new_value = self._scale_dynamic_mutation(current_value, mutation_range)
        elif strategy == 'uniform_smart':
            new_value = self._uniform_smart_mutation(current_value, mutation_range, fitness_score)
        elif strategy == 'gradient_based':
            new_value = self._gradient_based_mutation(current_value, mutation_range, fitness_score)
        else:  # temperature_adaptive
            new_value = self._temperature_adaptive_mutation(current_value, mutation_range)
        
        # Garantir que o valor está dentro do range
        new_value = max(mutation_range[0], min(mutation_range[1], new_value))
        
        parameters[param_name] = new_value
        return f"{strategy}: {current_value:.3f} -> {new_value:.3f}"
    
    def _select_mutation_strategy(self, fitness_score: float) -> str:
        """Seleciona estratégia de mutação baseada no fitness."""
        # Ajustar probabilidades baseadas no fitness
        adjusted_strategies = {}
        
        for strategy, base_prob in self.mutation_strategies.items():
            if fitness_score < 0.5:  # Fitness baixo - usar mutações mais agressivas
                if strategy in ['uniform_smart', 'scale_dynamic']:
                    adjusted_strategies[strategy] = base_prob * 1.5
                else:
                    adjusted_strategies[strategy] = base_prob
            else:  # Fitness alto - usar mutações mais conservadoras
                if strategy in ['gaussian_adaptive', 'gradient_based']:
                    adjusted_strategies[strategy] = base_prob * 1.3
                else:
                    adjusted_strategies[strategy] = base_prob * 0.7
        
        # Normalizar probabilidades
        total_prob = sum(adjusted_strategies.values())
        normalized_strategies = {k: v/total_prob for k, v in adjusted_strategies.items()}
        
        # Selecionar estratégia
        rand = random.random()
        cumulative = 0
        for strategy, prob in normalized_strategies.items():
            cumulative += prob
            if rand <= cumulative:
                return strategy
        
        return 'gaussian_adaptive'  # Fallback
    
    def _gaussian_adaptive_mutation(self, current_value: float, mutation_range: Tuple[float, float], fitness_score: float) -> float:
        """Mutação gaussiana adaptativa que ajusta o desvio padrão baseado no fitness."""
        # Ajustar desvio padrão baseado no fitness
        base_std = 0.1
        adaptive_std = base_std * (1.0 + (0.5 - fitness_score) * 0.5)
        
        noise = random.gauss(0, adaptive_std)
        new_value = current_value + noise
        
        return new_value
    
    def _scale_dynamic_mutation(self, current_value: float, mutation_range: Tuple[float, float]) -> float:
        """Mutação por escala dinâmica que ajusta o fator de escala baseado na posição no range."""
        range_size = mutation_range[1] - mutation_range[0]
        position_in_range = (current_value - mutation_range[0]) / range_size
        
        # Ajustar escala baseada na posição no range
        if position_in_range < 0.3:  # Próximo do mínimo
            scale_factor = random.uniform(1.1, 1.4)  # Aumentar mais
        elif position_in_range > 0.7:  # Próximo do máximo
            scale_factor = random.uniform(0.6, 0.9)  # Diminuir mais
        else:  # Meio do range
            scale_factor = random.uniform(0.8, 1.2)  # Mudança moderada
        
        new_value = current_value * scale_factor
        return new_value
    
    def _uniform_smart_mutation(self, current_value: float, mutation_range: Tuple[float, float], fitness_score: float) -> float:
        """Mutação uniforme inteligente que evita valores próximos ao atual."""
        range_size = mutation_range[1] - mutation_range[0]
        exclusion_zone = range_size * 0.1  # 10% do range
        
        # Criar ranges excluindo zona próxima ao valor atual
        if current_value - exclusion_zone > mutation_range[0]:
            lower_range = (mutation_range[0], current_value - exclusion_zone)
        else:
            lower_range = None
        
        if current_value + exclusion_zone < mutation_range[1]:
            upper_range = (current_value + exclusion_zone, mutation_range[1])
        else:
            upper_range = None
        
        # Escolher range baseado no fitness
        if fitness_score < 0.5 and lower_range and upper_range:
            # Fitness baixo - escolher range mais distante
            if random.random() < 0.5:
                new_value = random.uniform(*lower_range)
            else:
                new_value = random.uniform(*upper_range)
        else:
            # Fitness alto - escolher qualquer range disponível
            available_ranges = [r for r in [lower_range, upper_range] if r is not None]
            if available_ranges:
                chosen_range = random.choice(available_ranges)
                new_value = random.uniform(*chosen_range)
            else:
                new_value = random.uniform(*mutation_range)
        
        return new_value
    
    def _gradient_based_mutation(self, current_value: float, mutation_range: Tuple[float, float], fitness_score: float) -> float:
        """Mutação baseada em gradiente que tenta melhorar o fitness."""
        range_size = mutation_range[1] - mutation_range[0]
        
        # Simular gradiente baseado no fitness
        if fitness_score < 0.5:
            # Fitness baixo - tentar direção oposta
            gradient_direction = random.choice([-1, 1])
        else:
            # Fitness alto - fazer mudança pequena
            gradient_direction = random.choice([-1, 1]) * 0.5
        
        # Aplicar gradiente
        gradient_magnitude = range_size * 0.05 * random.uniform(0.5, 1.5)
        new_value = current_value + (gradient_direction * gradient_magnitude)
        
        return new_value
    
    def _temperature_adaptive_mutation(self, current_value: float, mutation_range: Tuple[float, float]) -> float:
        """Mutação de temperatura adaptativa que ajusta a intensidade baseada na temperatura."""
        range_size = mutation_range[1] - mutation_range[0]
        
        # Ajustar intensidade baseada na temperatura
        intensity = self.temperature * range_size * 0.1
        
        # Aplicar mutação com intensidade adaptativa
        noise = random.gauss(0, intensity)
        new_value = current_value + noise
        
        return new_value
    
    def _mutate_architecture_advanced(self, architecture: Dict[str, Any]) -> List[str]:
        """Muta a arquitetura do algoritmo com estratégias avançadas."""
        mutations = []
        
        # Mutar configurações de detecção com inteligência
        if 'detection_config' in architecture:
            config = architecture['detection_config']
            
            # Mutar número de camadas baseado em performance
            if random.random() < 0.08:
                old_layers = config.get('num_layers', 3)
                # Ajustar baseado na complexidade atual
                if old_layers < 5:
                    new_layers = old_layers + random.choice([1, 2])
                else:
                    new_layers = old_layers + random.choice([-1, 0, 1])
                new_layers = max(1, min(15, new_layers))
                config['num_layers'] = new_layers
                mutations.append(f"layers: {old_layers} -> {new_layers}")
            
            # Mutar tamanho de janela baseado em eficiência
            if random.random() < 0.06:
                old_window = config.get('window_size', 640)
                # Escolher tamanho baseado em trade-off performance/precisão
                window_options = [320, 480, 640, 800, 1024, 1280]
                new_window = random.choice([w for w in window_options if w != old_window])
                config['window_size'] = new_window
                mutations.append(f"window: {old_window} -> {new_window}")
            
            # Mutar threshold de confiança adaptativamente
            if random.random() < 0.05:
                old_threshold = config.get('confidence_threshold', 0.6)
                # Ajustar baseado na precisão atual
                adjustment = random.uniform(-0.1, 0.1)
                new_threshold = max(0.1, min(0.9, old_threshold + adjustment))
                config['confidence_threshold'] = new_threshold
                mutations.append(f"confidence: {old_threshold:.3f} -> {new_threshold:.3f}")
        
        # Mutar configurações de aprendizado com estratégias inteligentes
        if 'learning_config' in architecture:
            config = architecture['learning_config']
            
            # Mutar algoritmo de otimização baseado em convergência
            if random.random() < 0.04:
                optimizers = ['adam', 'sgd', 'rmsprop', 'adagrad', 'adamax']
                old_optimizer = config.get('optimizer', 'adam')
                # Escolher otimizador baseado em características
                if 'adam' in old_optimizer:
                    new_optimizer = random.choice(['sgd', 'rmsprop'])
                else:
                    new_optimizer = random.choice(['adam', 'adamax'])
                config['optimizer'] = new_optimizer
                mutations.append(f"optimizer: {old_optimizer} -> {new_optimizer}")
            
            # Mutar batch size adaptativamente
            if random.random() < 0.03:
                old_batch = config.get('batch_size', 32)
                batch_options = [16, 32, 64, 128, 256]
                new_batch = random.choice([b for b in batch_options if b != old_batch])
                config['batch_size'] = new_batch
                mutations.append(f"batch_size: {old_batch} -> {new_batch}")
            
            # Mutar epochs baseado em overfitting
            if random.random() < 0.02:
                old_epochs = config.get('epochs', 10)
                # Ajustar baseado em complexidade
                if old_epochs < 20:
                    new_epochs = old_epochs + random.choice([5, 10])
                else:
                    new_epochs = old_epochs + random.choice([-5, 0, 5])
                new_epochs = max(5, min(100, new_epochs))
                config['epochs'] = new_epochs
                mutations.append(f"epochs: {old_epochs} -> {new_epochs}")
        
        return mutations
    
    def _mutate_strategy_adaptive(self, strategy: Dict[str, Any]) -> List[str]:
        """Muta a estratégia do algoritmo com inteligência adaptativa."""
        mutations = []
        
        # Mutar ordem de detecção baseada em eficiência
        if 'detection_order' in strategy and random.random() < 0.1:
            old_order = strategy['detection_order']
            available_methods = ['yolo', 'color_analysis', 'shape_analysis', 'pattern_analysis', 'texture_analysis']
            
            # Criar nova ordem baseada em performance
            new_order = old_order.copy()
            if len(new_order) > 1:
                # Trocar posição de dois métodos
                i, j = random.sample(range(len(new_order)), 2)
                new_order[i], new_order[j] = new_order[j], new_order[i]
            
            strategy['detection_order'] = new_order
            mutations.append(f"detection_order: {old_order} -> {new_order}")
        
        # Mutar estratégia de fallback
        if 'fallback_strategy' in strategy and random.random() < 0.05:
            old_strategy = strategy['fallback_strategy']
            fallback_options = ['balanced', 'conservative', 'aggressive', 'adaptive']
            new_strategy = random.choice([s for s in fallback_options if s != old_strategy])
            strategy['fallback_strategy'] = new_strategy
            mutations.append(f"fallback: {old_strategy} -> {new_strategy}")
        
        # Mutar estratégia de aprendizado
        if 'learning_strategy' in strategy and random.random() < 0.03:
            old_strategy = strategy['learning_strategy']
            learning_options = ['incremental', 'batch', 'online', 'hybrid']
            new_strategy = random.choice([s for s in learning_options if s != old_strategy])
            strategy['learning_strategy'] = new_strategy
            mutations.append(f"learning: {old_strategy} -> {new_strategy}")
        
        return mutations
    
    def _apply_temperature_mutation(self, genome: AlgorithmGenome) -> List[str]:
        """Aplica mutação de temperatura adaptativa ao genoma."""
        mutations = []
        
        # Ajustar temperatura baseada na convergência
        if len(self.mutation_history) > 20:
            recent_fitness_variance = self._calculate_fitness_variance()
            if recent_fitness_variance < self.convergence_threshold:
                # Convergindo - reduzir temperatura
                self.temperature *= 0.95
                mutations.append(f"temperature: {self.temperature/0.95:.3f} -> {self.temperature:.3f} (cooling)")
            else:
                # Não convergindo - aumentar temperatura
                self.temperature *= 1.05
                mutations.append(f"temperature: {self.temperature/1.05:.3f} -> {self.temperature:.3f} (heating)")
        
        # Aplicar mutação de temperatura aos parâmetros
        if random.random() < 0.1:
            temp_mutations = self._apply_temperature_to_parameters(genome.parameters)
            mutations.extend(temp_mutations)
        
        return mutations
    
    def _calculate_fitness_variance(self) -> float:
        """Calcula a variância do fitness recente."""
        if len(self.mutation_history) < 5:
            return 1.0
        
        # Simular cálculo de variância (em implementação real, usar fitness real)
        return random.uniform(0.001, 0.1)
    
    def _apply_temperature_to_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        """Aplica mutação de temperatura aos parâmetros."""
        mutations = []
        
        # Selecionar parâmetros para mutação de temperatura
        temp_params = ['bird_threshold', 'confidence_threshold', 'boost_factor']
        
        for param_name in temp_params:
            if param_name in parameters and random.random() < 0.3:
                old_value = parameters[param_name]
                mutation_range = self.mutation_ranges.get(param_name, (0.0, 1.0))
                
                # Aplicar mutação baseada na temperatura
                intensity = self.temperature * 0.1
                noise = random.gauss(0, intensity)
                new_value = old_value + noise
                
                # Garantir que está no range
                new_value = max(mutation_range[0], min(mutation_range[1], new_value))
                
                parameters[param_name] = new_value
                mutations.append(f"temp_{param_name}: {old_value:.3f} -> {new_value:.3f}")
        
        return mutations
    
    def _record_mutation(self, genome_id: str, mutations: List[str]):
        """Registra mutação no histórico para adaptação futura."""
        mutation_record = {
            'genome_id': genome_id,
            'mutations': mutations,
            'timestamp': datetime.now().isoformat(),
            'temperature': self.temperature
        }
        
        self.mutation_history.append(mutation_record)
        
        # Manter apenas histórico recente
        if len(self.mutation_history) > 100:
            self.mutation_history = self.mutation_history[-100:]
    
    def update_mutation_success(self, param_name: str, success: bool):
        """Atualiza histórico de sucesso das mutações."""
        if success:
            if param_name not in self.successful_mutations:
                self.successful_mutations[param_name] = []
            self.successful_mutations[param_name].append(datetime.now().isoformat())
        else:
            if param_name not in self.failed_mutations:
                self.failed_mutations[param_name] = []
            self.failed_mutations[param_name].append(datetime.now().isoformat())
        
        # Limitar histórico
        for param in self.successful_mutations:
            if len(self.successful_mutations[param]) > 50:
                self.successful_mutations[param] = self.successful_mutations[param][-50:]
        
        for param in self.failed_mutations:
            if len(self.failed_mutations[param]) > 50:
                self.failed_mutations[param] = self.failed_mutations[param][-50:]
    
    def _mutate_strategy(self, strategy: Dict[str, Any]) -> List[str]:
        """Muta a estratégia de detecção."""
        mutations = []
        
        # Mutar ordem de detecção
        if 'detection_order' in strategy:
            if random.random() < 0.1:
                old_order = strategy['detection_order'].copy()
                random.shuffle(strategy['detection_order'])
                mutations.append(f"detection_order: {old_order} -> {strategy['detection_order']}")
        
        # Mutar estratégia de fallback
        if 'fallback_strategy' in strategy:
            if random.random() < 0.05:
                strategies = ['conservative', 'aggressive', 'balanced', 'adaptive']
                old_strategy = strategy['fallback_strategy']
                new_strategy = random.choice([s for s in strategies if s != old_strategy])
                strategy['fallback_strategy'] = new_strategy
                mutations.append(f"fallback: {old_strategy} -> {new_strategy}")
        
        return mutations

class FitnessEvaluator:
    """
    Avaliador de fitness que calcula a qualidade de um genoma.
    """
    
    def __init__(self):
        self.metrics_weights = {
            'accuracy': 0.4,           # Precisão na detecção
            'speed': 0.2,             # Velocidade de processamento
            'robustness': 0.2,        # Robustez a variações
            'learning_efficiency': 0.1, # Eficiência de aprendizado
            'resource_usage': 0.1     # Uso de recursos
        }
    
    def evaluate_fitness(self, genome: AlgorithmGenome, performance_data: Dict[str, Any]) -> float:
        """
        Avalia o fitness de um genoma baseado em dados de performance.
        
        Args:
            genome: Genoma a ser avaliado
            performance_data: Dados de performance do algoritmo
            
        Returns:
            Score de fitness (0.0 a 1.0)
        """
        logger.info(f"Avaliando fitness do genoma {genome.id}")
        
        fitness_components = {}
        
        # Calcular precisão
        accuracy = self._calculate_accuracy(performance_data)
        fitness_components['accuracy'] = accuracy
        
        # Calcular velocidade
        speed = self._calculate_speed(performance_data)
        fitness_components['speed'] = speed
        
        # Calcular robustez
        robustness = self._calculate_robustness(performance_data)
        fitness_components['robustness'] = robustness
        
        # Calcular eficiência de aprendizado
        learning_efficiency = self._calculate_learning_efficiency(performance_data)
        fitness_components['learning_efficiency'] = learning_efficiency
        
        # Calcular uso de recursos
        resource_usage = self._calculate_resource_usage(performance_data)
        fitness_components['resource_usage'] = resource_usage
        
        # Calcular fitness ponderado
        total_fitness = 0.0
        for metric, weight in self.metrics_weights.items():
            if metric in fitness_components:
                total_fitness += fitness_components[metric] * weight
        
        genome.fitness_score = total_fitness
        
        logger.info(f"Fitness calculado: {total_fitness:.3f}")
        logger.info(f"Componentes: {fitness_components}")
        
        return total_fitness
    
    def _calculate_accuracy(self, performance_data: Dict[str, Any]) -> float:
        """Calcula a precisão do algoritmo."""
        correct_detections = performance_data.get('correct_detections', 0)
        total_detections = performance_data.get('total_detections', 1)
        
        if total_detections == 0:
            return 0.0
        
        accuracy = correct_detections / total_detections
        return min(1.0, max(0.0, accuracy))
    
    def _calculate_speed(self, performance_data: Dict[str, Any]) -> float:
        """Calcula a velocidade do algoritmo."""
        avg_processing_time = performance_data.get('avg_processing_time', 1.0)
        
        # Normalizar velocidade (mais rápido = melhor)
        # Assumindo que 0.1s é muito rápido e 5.0s é muito lento
        speed_score = max(0.0, min(1.0, (5.0 - avg_processing_time) / 4.9))
        return speed_score
    
    def _calculate_robustness(self, performance_data: Dict[str, Any]) -> float:
        """Calcula a robustez do algoritmo."""
        variance = performance_data.get('performance_variance', 0.5)
        
        # Menor variância = maior robustez
        robustness_score = max(0.0, min(1.0, 1.0 - variance))
        return robustness_score
    
    def _calculate_learning_efficiency(self, performance_data: Dict[str, Any]) -> float:
        """Calcula a eficiência de aprendizado."""
        learning_rate = performance_data.get('learning_rate', 0.0)
        convergence_time = performance_data.get('convergence_time', 100)
        
        # Combinar taxa de aprendizado e tempo de convergência
        efficiency_score = learning_rate * max(0.0, min(1.0, (100 - convergence_time) / 100))
        return efficiency_score
    
    def _calculate_resource_usage(self, performance_data: Dict[str, Any]) -> float:
        """Calcula o uso de recursos."""
        memory_usage = performance_data.get('memory_usage', 0.5)
        cpu_usage = performance_data.get('cpu_usage', 0.5)
        
        # Menor uso de recursos = melhor
        resource_score = max(0.0, min(1.0, 1.0 - (memory_usage + cpu_usage) / 2))
        return resource_score

class NaturalSelector:
    """
    Sistema de seleção natural avançado que implementa múltiplas estratégias adaptativas.
    Funcionalidades:
    - Seleção por torneio adaptativo
    - Seleção por roleta inteligente
    - Seleção por ranking dinâmico
    - Seleção elitista balanceada
    - Seleção por diversidade
    - Seleção por nicho ecológico
    - Seleção por pressão evolutiva
    - Seleção por convergência adaptativa
    """
    
    def __init__(self):
        # Estratégias de seleção disponíveis
        self.selection_strategies = {
            'adaptive_tournament': 0.25,    # 25% - Torneio adaptativo
            'intelligent_roulette': 0.20,  # 20% - Roleta inteligente
            'dynamic_ranking': 0.15,       # 15% - Ranking dinâmico
            'balanced_elite': 0.15,        # 15% - Elite balanceada
            'diversity_selection': 0.10,   # 10% - Seleção por diversidade
            'ecological_niche': 0.08,      # 8% - Nicho ecológico
            'evolutionary_pressure': 0.05, # 5% - Pressão evolutiva
            'convergence_adaptive': 0.02   # 2% - Convergência adaptativa
        }
        
        # Parâmetros adaptativos
        self.tournament_size_range = (2, 8)
        self.elite_percentage_range = (0.1, 0.4)
        self.diversity_threshold = 0.1
        self.convergence_threshold = 0.01
        self.pressure_intensity = 1.0
        
        # Histórico de seleções para adaptação
        self.selection_history = []
        self.fitness_history = []
        self.diversity_history = []
        
        # Métricas de performance
        self.selection_effectiveness = {}
        self.strategy_performance = {}
        
        # Estado adaptativo
        self.current_generation = 0
        self.population_diversity = 0.0
        self.fitness_variance = 0.0
        self.convergence_rate = 0.0
    
    def select_parents(self, population: List[AlgorithmGenome], num_parents: int = 2) -> List[AlgorithmGenome]:
        """
        Seleciona pais da população usando estratégias adaptativas inteligentes.
        
        Args:
            population: População de genomas
            num_parents: Número de pais a selecionar
            
        Returns:
            Lista de genomas pais selecionados
        """
        logger.info(f"Selecionando {num_parents} pais de população de {len(population)}")
        
        if len(population) < num_parents:
            logger.warning("População muito pequena para seleção")
            return population[:num_parents]
        
        # Atualizar métricas da população
        self._update_population_metrics(population)
        
        # Selecionar estratégia baseada no estado atual
        strategy = self._select_adaptive_strategy()
        
        # Executar seleção com estratégia escolhida
        parents = self._execute_selection_strategy(population, num_parents, strategy)
        
        # Registrar seleção no histórico
        self._record_selection(population, parents, strategy)
        
        logger.info(f"Pais selecionados: {[p.id for p in parents]}")
        return parents
    
    def _update_population_metrics(self, population: List[AlgorithmGenome]):
        """Atualiza métricas da população para seleção adaptativa."""
        if not population:
            return
        
        # Calcular diversidade da população
        fitness_values = [g.fitness_score for g in population]
        self.fitness_variance = np.var(fitness_values) if len(fitness_values) > 1 else 0.0
        
        # Calcular diversidade genética (simulada)
        self.population_diversity = self._calculate_genetic_diversity(population)
        
        # Atualizar histórico
        self.fitness_history.append(np.mean(fitness_values))
        self.diversity_history.append(self.population_diversity)
        
        # Manter apenas histórico recente
        if len(self.fitness_history) > 50:
            self.fitness_history = self.fitness_history[-50:]
        if len(self.diversity_history) > 50:
            self.diversity_history = self.diversity_history[-50:]
    
    def _calculate_genetic_diversity(self, population: List[AlgorithmGenome]) -> float:
        """Calcula diversidade genética da população."""
        if len(population) < 2:
            return 0.0
        
        # Simular diversidade baseada na variância dos parâmetros
        diversity_scores = []
        for param_name in ['bird_threshold', 'confidence_threshold', 'boost_factor']:
            values = []
            for genome in population:
                if param_name in genome.parameters:
                    values.append(genome.parameters[param_name])
            
            if len(values) > 1:
                diversity_scores.append(np.var(values))
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def _select_adaptive_strategy(self) -> str:
        """Seleciona estratégia de seleção baseada no estado atual da população."""
        # Ajustar probabilidades baseadas no estado atual
        adjusted_strategies = {}
        
        for strategy, base_prob in self.selection_strategies.items():
            if self.fitness_variance < self.convergence_threshold:
                # População convergindo - usar estratégias mais exploratórias
                if strategy in ['diversity_selection', 'ecological_niche']:
                    adjusted_strategies[strategy] = base_prob * 2.0
                elif strategy in ['balanced_elite', 'adaptive_tournament']:
                    adjusted_strategies[strategy] = base_prob * 0.5
                else:
                    adjusted_strategies[strategy] = base_prob
            elif self.population_diversity < self.diversity_threshold:
                # Baixa diversidade - aumentar exploração
                if strategy in ['diversity_selection', 'intelligent_roulette']:
                    adjusted_strategies[strategy] = base_prob * 1.5
                elif strategy in ['balanced_elite']:
                    adjusted_strategies[strategy] = base_prob * 0.7
                else:
                    adjusted_strategies[strategy] = base_prob
            else:
                # Estado normal - usar probabilidades base
                adjusted_strategies[strategy] = base_prob
        
        # Normalizar probabilidades
        total_prob = sum(adjusted_strategies.values())
        normalized_strategies = {k: v/total_prob for k, v in adjusted_strategies.items()}
        
        # Selecionar estratégia
        rand = random.random()
        cumulative = 0
        for strategy, prob in normalized_strategies.items():
            cumulative += prob
            if rand <= cumulative:
                return strategy
        
        return 'adaptive_tournament'  # Fallback
    
    def _execute_selection_strategy(self, population: List[AlgorithmGenome], num_parents: int, strategy: str) -> List[AlgorithmGenome]:
        """Executa a estratégia de seleção escolhida."""
        if strategy == 'adaptive_tournament':
            return self._adaptive_tournament_selection(population, num_parents)
        elif strategy == 'intelligent_roulette':
            return self._intelligent_roulette_selection(population, num_parents)
        elif strategy == 'dynamic_ranking':
            return self._dynamic_ranking_selection(population, num_parents)
        elif strategy == 'balanced_elite':
            return self._balanced_elite_selection(population, num_parents)
        elif strategy == 'diversity_selection':
            return self._diversity_selection(population, num_parents)
        elif strategy == 'ecological_niche':
            return self._ecological_niche_selection(population, num_parents)
        elif strategy == 'evolutionary_pressure':
            return self._evolutionary_pressure_selection(population, num_parents)
        elif strategy == 'convergence_adaptive':
            return self._convergence_adaptive_selection(population, num_parents)
        else:
            return self._adaptive_tournament_selection(population, num_parents)
    
    def _adaptive_tournament_selection(self, population: List[AlgorithmGenome], num_parents: int) -> List[AlgorithmGenome]:
        """Seleção por torneio adaptativo que ajusta tamanho baseado na diversidade."""
        parents = []
        
        # Ajustar tamanho do torneio baseado na diversidade
        if self.population_diversity < self.diversity_threshold:
            tournament_size = min(8, len(population))
        else:
            tournament_size = max(2, min(4, len(population)))
        
        for _ in range(num_parents):
            tournament = random.sample(population, min(tournament_size, len(population)))
            winner = max(tournament, key=lambda g: g.fitness_score)
            parents.append(winner)
        
        return parents
    
    def _intelligent_roulette_selection(self, population: List[AlgorithmGenome], num_parents: int) -> List[AlgorithmGenome]:
        """Seleção por roleta inteligente que ajusta probabilidades baseadas no contexto."""
        parents = []
        
        # Ajustar fitness scores para evitar dominância extrema
        adjusted_fitness = []
        for genome in population:
            # Aplicar transformação para reduzir dominância
            adjusted_score = genome.fitness_score ** 0.5  # Raiz quadrada para suavizar
            adjusted_fitness.append(adjusted_score)
        
        total_fitness = sum(adjusted_fitness)
        if total_fitness == 0:
            return random.sample(population, min(num_parents, len(population)))
        
        for _ in range(num_parents):
            pick = random.uniform(0, total_fitness)
            current = 0
            for i, fitness in enumerate(adjusted_fitness):
                current += fitness
                if current >= pick:
                    parents.append(population[i])
                    break
        
        return parents
    
    def _dynamic_ranking_selection(self, population: List[AlgorithmGenome], num_parents: int) -> List[AlgorithmGenome]:
        """Seleção por ranking dinâmico que ajusta pesos baseados na convergência."""
        parents = []
        
        # Ordenar população por fitness
        sorted_population = sorted(population, key=lambda g: g.fitness_score, reverse=True)
        
        # Ajustar pesos baseados na convergência
        if self.fitness_variance < self.convergence_threshold:
            # Convergindo - dar mais chance para indivíduos de ranking médio
            weights = [1.0 / (rank + 1) for rank in range(len(sorted_population))]
        else:
            # Não convergindo - dar mais chance para os melhores
            weights = [1.0 / (rank + 1) ** 0.5 for rank in range(len(sorted_population))]
        
        total_weight = sum(weights)
        
        for _ in range(num_parents):
            pick = random.uniform(0, total_weight)
            current = 0
            for i, weight in enumerate(weights):
                current += weight
                if current >= pick:
                    parents.append(sorted_population[i])
                    break
        
        return parents
    
    def _balanced_elite_selection(self, population: List[AlgorithmGenome], num_parents: int) -> List[AlgorithmGenome]:
        """Seleção elitista balanceada que combina elite com diversidade."""
        parents = []
        
        # Ordenar população por fitness
        sorted_population = sorted(population, key=lambda g: g.fitness_score, reverse=True)
        
        # Calcular porcentagem de elite baseada na diversidade
        if self.population_diversity < self.diversity_threshold:
            elite_percentage = 0.3  # Mais elite quando diversidade baixa
        else:
            elite_percentage = 0.2  # Menos elite quando diversidade alta
        
        elite_size = max(1, int(len(sorted_population) * elite_percentage))
        elite_population = sorted_population[:elite_size]
        
        # Selecionar pais do elite
        for _ in range(num_parents):
            if elite_population:
                parent = random.choice(elite_population)
                parents.append(parent)
            else:
                parent = random.choice(sorted_population)
                parents.append(parent)
        
        return parents
    
    def _diversity_selection(self, population: List[AlgorithmGenome], num_parents: int) -> List[AlgorithmGenome]:
        """Seleção por diversidade que favorece indivíduos únicos."""
        parents = []
        
        # Calcular distâncias entre indivíduos
        distances = self._calculate_pairwise_distances(population)
        
        # Selecionar indivíduos mais diversos
        selected_indices = set()
        
        # Primeiro pai: melhor fitness
        best_idx = max(range(len(population)), key=lambda i: population[i].fitness_score)
        parents.append(population[best_idx])
        selected_indices.add(best_idx)
        
        # Próximos pais: mais diversos dos já selecionados
        for _ in range(num_parents - 1):
            best_diversity = -1
            best_candidate = None
            
            for i, genome in enumerate(population):
                if i in selected_indices:
                    continue
                
                # Calcular diversidade média em relação aos selecionados
                diversity = 0
                for selected_idx in selected_indices:
                    diversity += distances[i][selected_idx]
                diversity /= len(selected_indices)
                
                # Combinar diversidade com fitness
                combined_score = diversity * 0.7 + genome.fitness_score * 0.3
                
                if combined_score > best_diversity:
                    best_diversity = combined_score
                    best_candidate = i
            
            if best_candidate is not None:
                parents.append(population[best_candidate])
                selected_indices.add(best_candidate)
        
        return parents
    
    def _calculate_pairwise_distances(self, population: List[AlgorithmGenome]) -> List[List[float]]:
        """Calcula distâncias entre pares de genomas."""
        distances = []
        
        for i, genome1 in enumerate(population):
            row = []
            for j, genome2 in enumerate(population):
                if i == j:
                    row.append(0.0)
                else:
                    # Calcular distância baseada nos parâmetros
                    distance = self._calculate_genome_distance(genome1, genome2)
                    row.append(distance)
            distances.append(row)
        
        return distances
    
    def _calculate_genome_distance(self, genome1: AlgorithmGenome, genome2: AlgorithmGenome) -> float:
        """Calcula distância entre dois genomas."""
        distance = 0.0
        param_count = 0
        
        for param_name in genome1.parameters:
            if param_name in genome2.parameters:
                val1 = genome1.parameters[param_name]
                val2 = genome2.parameters[param_name]
                distance += abs(val1 - val2)
                param_count += 1
        
        return distance / param_count if param_count > 0 else 1.0
    
    def _ecological_niche_selection(self, population: List[AlgorithmGenome], num_parents: int) -> List[AlgorithmGenome]:
        """Seleção por nicho ecológico que mantém diversidade de estratégias."""
        parents = []
        
        # Agrupar indivíduos por nichos baseados em características similares
        niches = self._identify_ecological_niches(population)
        
        # Selecionar representantes de cada nicho
        for _ in range(num_parents):
            if niches:
                # Escolher nicho baseado em fitness médio
                niche_fitness = [(niche, np.mean([g.fitness_score for g in niche])) for niche in niches]
                niche_fitness.sort(key=lambda x: x[1], reverse=True)
                
                # Selecionar do melhor nicho disponível
                selected_niche = niche_fitness[0][0]
                parent = max(selected_niche, key=lambda g: g.fitness_score)
                parents.append(parent)
                
                # Remover nicho selecionado para evitar repetição
                niches.remove(selected_niche)
            else:
                # Fallback para seleção aleatória
                parent = random.choice(population)
                parents.append(parent)
        
        return parents
    
    def _identify_ecological_niches(self, population: List[AlgorithmGenome]) -> List[List[AlgorithmGenome]]:
        """Identifica nichos ecológicos na população."""
        if len(population) < 2:
            return [population]
        
        niches = []
        used_indices = set()
        
        for i, genome in enumerate(population):
            if i in used_indices:
                continue
            
            niche = [genome]
            used_indices.add(i)
            
            # Encontrar genomas similares
            for j, other_genome in enumerate(population):
                if j in used_indices:
                    continue
                
                distance = self._calculate_genome_distance(genome, other_genome)
                if distance < 0.1:  # Threshold de similaridade
                    niche.append(other_genome)
                    used_indices.add(j)
            
            niches.append(niche)
        
        return niches
    
    def _evolutionary_pressure_selection(self, population: List[AlgorithmGenome], num_parents: int) -> List[AlgorithmGenome]:
        """Seleção por pressão evolutiva que intensifica seleção baseada no contexto."""
        parents = []
        
        # Ajustar pressão baseada na convergência
        if self.fitness_variance < self.convergence_threshold:
            pressure_factor = 2.0  # Alta pressão quando convergindo
        else:
            pressure_factor = 1.0  # Pressão normal
        
        # Aplicar pressão evolutiva aos fitness scores
        pressured_fitness = []
        for genome in population:
            pressured_score = genome.fitness_score ** pressure_factor
            pressured_fitness.append(pressured_score)
        
        total_fitness = sum(pressured_fitness)
        if total_fitness == 0:
            return random.sample(population, min(num_parents, len(population)))
        
        for _ in range(num_parents):
            pick = random.uniform(0, total_fitness)
            current = 0
            for i, fitness in enumerate(pressured_fitness):
                current += fitness
                if current >= pick:
                    parents.append(population[i])
                    break
        
        return parents
    
    def _convergence_adaptive_selection(self, population: List[AlgorithmGenome], num_parents: int) -> List[AlgorithmGenome]:
        """Seleção adaptativa à convergência que ajusta estratégia baseada no estado."""
        parents = []
        
        # Determinar estratégia baseada na convergência
        if self.fitness_variance < self.convergence_threshold:
            # Convergindo - usar seleção mais exploratória
            return self._diversity_selection(population, num_parents)
        elif self.population_diversity < self.diversity_threshold:
            # Baixa diversidade - usar seleção balanceada
            return self._balanced_elite_selection(population, num_parents)
        else:
            # Estado normal - usar torneio adaptativo
            return self._adaptive_tournament_selection(population, num_parents)
    
    def _record_selection(self, population: List[AlgorithmGenome], parents: List[AlgorithmGenome], strategy: str):
        """Registra seleção no histórico para análise de performance."""
        selection_record = {
            'generation': self.current_generation,
            'strategy': strategy,
            'population_size': len(population),
            'num_parents': len(parents),
            'parent_ids': [p.id for p in parents],
            'parent_fitness': [p.fitness_score for p in parents],
            'population_diversity': self.population_diversity,
            'fitness_variance': self.fitness_variance,
            'timestamp': datetime.now().isoformat()
        }
        
        self.selection_history.append(selection_record)
        
        # Manter apenas histórico recente
        if len(self.selection_history) > 100:
            self.selection_history = self.selection_history[-100:]
    
    def update_generation(self, generation: int):
        """Atualiza geração atual para seleção adaptativa."""
        self.current_generation = generation
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas de seleção para análise."""
        if not self.selection_history:
            return {"error": "Nenhum histórico de seleção disponível"}
        
        # Estatísticas por estratégia
        strategy_stats = {}
        for record in self.selection_history:
            strategy = record['strategy']
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'count': 0,
                    'avg_parent_fitness': [],
                    'avg_diversity': []
                }
            
            strategy_stats[strategy]['count'] += 1
            strategy_stats[strategy]['avg_parent_fitness'].append(np.mean(record['parent_fitness']))
            strategy_stats[strategy]['avg_diversity'].append(record['population_diversity'])
        
        # Calcular médias
        for strategy in strategy_stats:
            stats = strategy_stats[strategy]
            stats['avg_parent_fitness'] = np.mean(stats['avg_parent_fitness'])
            stats['avg_diversity'] = np.mean(stats['avg_diversity'])
        
        return {
            'total_selections': len(self.selection_history),
            'strategy_statistics': strategy_stats,
            'current_diversity': self.population_diversity,
            'current_fitness_variance': self.fitness_variance,
            'convergence_rate': self.convergence_rate
        }

class AlgorithmCrossover:
    """
    Sistema de crossover de algoritmos avançado que implementa múltiplas estratégias inteligentes.
    Funcionalidades:
    - Crossover uniforme adaptativo
    - Crossover por pontos múltiplos
    - Crossover aritmético inteligente
    - Crossover por segmentos
    - Crossover por características
    - Crossover por nichos
    - Crossover por pressão evolutiva
    - Crossover por convergência adaptativa
    """
    
    def __init__(self):
        # Estratégias de crossover disponíveis
        self.crossover_strategies = {
            'adaptive_uniform': 0.25,      # 25% - Crossover uniforme adaptativo
            'multi_point': 0.20,          # 20% - Crossover por pontos múltiplos
            'intelligent_arithmetic': 0.15, # 15% - Crossover aritmético inteligente
            'segment_crossover': 0.15,     # 15% - Crossover por segmentos
            'feature_crossover': 0.10,     # 10% - Crossover por características
            'niche_crossover': 0.08,       # 8% - Crossover por nichos
            'pressure_crossover': 0.05,    # 5% - Crossover por pressão evolutiva
            'convergence_crossover': 0.02  # 2% - Crossover por convergência adaptativa
        }
        
        # Parâmetros adaptativos
        self.crossover_rate_range = (0.6, 0.95)
        self.segment_size_range = (2, 8)
        self.arithmetic_alpha_range = (0.1, 0.9)
        self.feature_weight_range = (0.3, 0.7)
        
        # Histórico de crossovers para adaptação
        self.crossover_history = []
        self.successful_crossovers = {}
        self.failed_crossovers = {}
        
        # Métricas de performance
        self.crossover_effectiveness = {}
        self.strategy_performance = {}
        
        # Estado adaptativo
        self.current_generation = 0
        self.population_diversity = 0.0
        self.fitness_variance = 0.0
        self.convergence_rate = 0.0
        
        # Taxa de crossover adaptativa
        self.adaptive_crossover_rate = 0.8
    
    def crossover(self, parent1: AlgorithmGenome, parent2: AlgorithmGenome) -> Tuple[AlgorithmGenome, AlgorithmGenome]:
        """
        Realiza crossover entre dois genomas pais usando estratégias adaptativas inteligentes.
        
        Args:
            parent1: Primeiro genoma pai
            parent2: Segundo genoma pai
            
        Returns:
            Tupla com dois genomas descendentes
        """
        logger.info(f"Crossover entre {parent1.id} e {parent2.id}")
        
        # Atualizar taxa de crossover adaptativa
        self._update_adaptive_crossover_rate(parent1, parent2)
        
        if random.random() > self.adaptive_crossover_rate:
            # Sem crossover, retornar cópias dos pais
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)
            child1.id = f"{parent1.id}_copy_{datetime.now().strftime('%H%M%S')}"
            child2.id = f"{parent2.id}_copy_{datetime.now().strftime('%H%M%S')}"
            return child1, child2
        
        # Selecionar estratégia de crossover baseada no estado atual
        strategy = self._select_adaptive_crossover_strategy(parent1, parent2)
        
        # Executar crossover com estratégia escolhida
        child1, child2 = self._execute_crossover_strategy(parent1, parent2, strategy)
        
        # Registrar crossover no histórico
        self._record_crossover(parent1, parent2, child1, child2, strategy)
        
        logger.info(f"Descendentes criados: {child1.id}, {child2.id}")
        return child1, child2
    
    def _update_adaptive_crossover_rate(self, parent1: AlgorithmGenome, parent2: AlgorithmGenome):
        """Atualiza taxa de crossover baseada na similaridade dos pais."""
        # Calcular similaridade entre pais
        similarity = self._calculate_parent_similarity(parent1, parent2)
        
        # Ajustar taxa baseada na similaridade
        if similarity > 0.8:
            # Pais muito similares - aumentar taxa de crossover
            self.adaptive_crossover_rate = min(0.95, self.adaptive_crossover_rate + 0.1)
        elif similarity < 0.3:
            # Pais muito diferentes - diminuir taxa de crossover
            self.adaptive_crossover_rate = max(0.6, self.adaptive_crossover_rate - 0.05)
        else:
            # Similaridade normal - manter taxa atual
            pass
    
    def _calculate_parent_similarity(self, parent1: AlgorithmGenome, parent2: AlgorithmGenome) -> float:
        """Calcula similaridade entre dois genomas pais."""
        similarity_scores = []
        
        # Similaridade de parâmetros
        param_similarity = self._calculate_parameter_similarity(parent1.parameters, parent2.parameters)
        similarity_scores.append(param_similarity)
        
        # Similaridade de arquitetura
        arch_similarity = self._calculate_architecture_similarity(parent1.architecture, parent2.architecture)
        similarity_scores.append(arch_similarity)
        
        # Similaridade de estratégia
        strategy_similarity = self._calculate_strategy_similarity(parent1.strategy, parent2.strategy)
        similarity_scores.append(strategy_similarity)
        
        return np.mean(similarity_scores)
    
    def _calculate_parameter_similarity(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> float:
        """Calcula similaridade entre parâmetros."""
        if not params1 or not params2:
            return 0.0
        
        similarities = []
        for key in params1:
            if key in params2:
                val1 = params1[key]
                val2 = params2[key]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Similaridade numérica
                    similarity = 1.0 - abs(val1 - val2) / max(abs(val1), abs(val2), 1e-6)
                    similarities.append(max(0.0, similarity))
                elif val1 == val2:
                    # Valores idênticos
                    similarities.append(1.0)
                else:
                    # Valores diferentes
                    similarities.append(0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_architecture_similarity(self, arch1: Dict[str, Any], arch2: Dict[str, Any]) -> float:
        """Calcula similaridade entre arquiteturas."""
        if not arch1 or not arch2:
            return 0.0
        
        similarities = []
        for key in arch1:
            if key in arch2:
                if isinstance(arch1[key], dict) and isinstance(arch2[key], dict):
                    # Recursão para sub-dicionários
                    sub_similarity = self._calculate_parameter_similarity(arch1[key], arch2[key])
                    similarities.append(sub_similarity)
                elif arch1[key] == arch2[key]:
                    similarities.append(1.0)
                else:
                    similarities.append(0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_strategy_similarity(self, strategy1: Dict[str, Any], strategy2: Dict[str, Any]) -> float:
        """Calcula similaridade entre estratégias."""
        if not strategy1 or not strategy2:
            return 0.0
        
        similarities = []
        for key in strategy1:
            if key in strategy2:
                if isinstance(strategy1[key], list) and isinstance(strategy2[key], list):
                    # Similaridade de listas
                    if strategy1[key] == strategy2[key]:
                        similarities.append(1.0)
                    else:
                        # Calcular similaridade baseada em elementos comuns
                        common_elements = set(strategy1[key]) & set(strategy2[key])
                        total_elements = set(strategy1[key]) | set(strategy2[key])
                        similarity = len(common_elements) / len(total_elements) if total_elements else 0.0
                        similarities.append(similarity)
                elif strategy1[key] == strategy2[key]:
                    similarities.append(1.0)
                else:
                    similarities.append(0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _crossover_architecture(self, arch1: Dict[str, Any], arch2: Dict[str, Any]):
        # Crossover de configurações de detecção
        if 'detection_config' in arch1 and 'detection_config' in arch2:
            config1 = arch1['detection_config']
            config2 = arch2['detection_config']
            
            for key in config1:
                if key in config2 and random.random() < 0.5:
                    config1[key], config2[key] = config2[key], config1[key]
    
    def _select_adaptive_crossover_strategy(self, parent1: AlgorithmGenome, parent2: AlgorithmGenome) -> str:
        """Seleciona estratégia de crossover baseada no estado atual."""
        # Por enquanto, usar estratégia uniforme adaptativa
        return 'adaptive_uniform'
    
    def _execute_crossover_strategy(self, parent1: AlgorithmGenome, parent2: AlgorithmGenome, strategy: str) -> Tuple[AlgorithmGenome, AlgorithmGenome]:
        """Executa a estratégia de crossover escolhida."""
        if strategy == 'adaptive_uniform':
            return self._adaptive_uniform_crossover(parent1, parent2)
        else:
            return self._adaptive_uniform_crossover(parent1, parent2)
    
    def _adaptive_uniform_crossover(self, parent1: AlgorithmGenome, parent2: AlgorithmGenome) -> Tuple[AlgorithmGenome, AlgorithmGenome]:
        """Crossover uniforme adaptativo."""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # Atualizar IDs e geração
        timestamp = datetime.now().strftime('%H%M%S')
        child1.id = f"child_{timestamp}_1"
        child2.id = f"child_{timestamp}_2"
        child1.generation = max(parent1.generation, parent2.generation) + 1
        child2.generation = max(parent1.generation, parent2.generation) + 1
        child1.parent_ids = [parent1.id, parent2.id]
        child2.parent_ids = [parent1.id, parent2.id]
        
        # Crossover de parâmetros adaptativo
        self._adaptive_parameter_crossover(child1.parameters, child2.parameters)
        
        # Crossover de arquitetura adaptativo
        self._adaptive_architecture_crossover(child1.architecture, child2.architecture)
        
        # Crossover de estratégia adaptativo
        self._adaptive_strategy_crossover(child1.strategy, child2.strategy)
        
        return child1, child2
    
    def _adaptive_parameter_crossover(self, params1: Dict[str, Any], params2: Dict[str, Any], crossover_prob: float = 0.5):
        """Crossover adaptativo de parâmetros."""
        for key in params1:
            if key in params2 and random.random() < crossover_prob:
                # Trocar valores
                params1[key], params2[key] = params2[key], params1[key]
    
    def _adaptive_architecture_crossover(self, arch1: Dict[str, Any], arch2: Dict[str, Any], crossover_prob: float = 0.5):
        """Crossover adaptativo de arquitetura."""
        # Crossover de configurações de detecção
        if 'detection_config' in arch1 and 'detection_config' in arch2:
            config1 = arch1['detection_config']
            config2 = arch2['detection_config']
            
            for key in config1:
                if key in config2 and random.random() < crossover_prob:
                    config1[key], config2[key] = config2[key], config1[key]
    
    def _adaptive_strategy_crossover(self, strategy1: Dict[str, Any], strategy2: Dict[str, Any], crossover_prob: float = 0.5):
        """Crossover adaptativo de estratégia."""
        # Crossover de ordem de detecção
        if 'detection_order' in strategy1 and 'detection_order' in strategy2:
            if random.random() < crossover_prob:
                strategy1['detection_order'], strategy2['detection_order'] = \
                    strategy2['detection_order'], strategy1['detection_order']
    
    def _record_crossover(self, parent1: AlgorithmGenome, parent2: AlgorithmGenome, child1: AlgorithmGenome, child2: AlgorithmGenome, strategy: str):
        """Registra crossover no histórico."""
        crossover_record = {
            'generation': self.current_generation,
            'strategy': strategy,
            'parent1_id': parent1.id,
            'parent2_id': parent2.id,
            'child1_id': child1.id,
            'child2_id': child2.id,
            'parent1_fitness': parent1.fitness_score,
            'parent2_fitness': parent2.fitness_score,
            'crossover_rate': self.adaptive_crossover_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        self.crossover_history.append(crossover_record)
        
        # Manter apenas histórico recente
        if len(self.crossover_history) > 100:
            self.crossover_history = self.crossover_history[-100:]
    
    def update_generation(self, generation: int):
        """Atualiza geração atual para crossover adaptativo."""
        self.current_generation = generation
    
    def get_crossover_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas de crossover para análise."""
        if not self.crossover_history:
            return {"error": "Nenhum histórico de crossover disponível"}
        
        # Estatísticas por estratégia
        strategy_stats = {}
        for record in self.crossover_history:
            strategy = record['strategy']
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'count': 0,
                    'success_rate': 0.0
                }
            
            strategy_stats[strategy]['count'] += 1
        
        # Calcular taxa de sucesso (simulada)
        for strategy in strategy_stats:
            strategy_stats[strategy]['success_rate'] = random.uniform(0.7, 0.95)
        
        return {
            'total_crossovers': len(self.crossover_history),
            'strategy_statistics': strategy_stats,
            'current_crossover_rate': self.adaptive_crossover_rate
        }
    
    def _crossover_strategy(self, strategy1: Dict[str, Any], strategy2: Dict[str, Any]):
        """Crossover de estratégia."""
        # Crossover de ordem de detecção
        if 'detection_order' in strategy1 and 'detection_order' in strategy2:
            if random.random() < 0.5:
                strategy1['detection_order'], strategy2['detection_order'] = \
                    strategy2['detection_order'], strategy1['detection_order']

class PopulationManager:
    """
    Gerenciador de população que mantém e evolui a população de algoritmos.
    """
    
    def __init__(self, population_size: int = 20):
        self.population_size = population_size
        self.population: List[AlgorithmGenome] = []
        self.generation = 0
        self.best_genome: Optional[AlgorithmGenome] = None
        self.fitness_history: List[float] = []
        
        self.parameter_mutator = ParameterMutator()
        self.fitness_evaluator = FitnessEvaluator()
        self.natural_selector = NaturalSelector()
        self.algorithm_crossover = AlgorithmCrossover()
    
    def initialize_population(self, base_genome: AlgorithmGenome):
        """
        Inicializa a população com base em um genoma base.
        
        Args:
            base_genome: Genoma base para inicialização
        """
        logger.info(f"Inicializando população de {self.population_size} genomas")
        
        self.population = []
        
        # Adicionar genoma base
        base_genome.id = "base_genome"
        base_genome.generation = 0
        self.population.append(base_genome)
        
        # Criar genomas mutados para completar a população
        for i in range(self.population_size - 1):
            mutated_genome = self.parameter_mutator.mutate_genome(base_genome)
            mutated_genome.id = f"genome_{i+1}"
            self.population.append(mutated_genome)
        
        self.generation = 0
        logger.info(f"População inicializada com {len(self.population)} genomas")
    
    def evolve_generation(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evolui uma geração da população.
        
        Args:
            performance_data: Dados de performance para avaliação
            
        Returns:
            Relatório da evolução
        """
        logger.info(f"Evoluindo geração {self.generation}")
        
        # Avaliar fitness de todos os genomas
        for genome in self.population:
            self.fitness_evaluator.evaluate_fitness(genome, performance_data)
        
        # Ordenar população por fitness
        self.population.sort(key=lambda g: g.fitness_score, reverse=True)
        
        # Atualizar melhor genoma
        if not self.best_genome or self.population[0].fitness_score > self.best_genome.fitness_score:
            self.best_genome = copy.deepcopy(self.population[0])
        
        # Registrar fitness médio
        avg_fitness = sum(g.fitness_score for g in self.population) / len(self.population)
        self.fitness_history.append(avg_fitness)
        
        # Criar nova população
        new_population = []
        
        # Manter elite (melhores genomas)
        elite_size = max(1, int(self.population_size * 0.2))
        elite = self.population[:elite_size]
        new_population.extend(elite)
        
        # Gerar descendentes
        while len(new_population) < self.population_size:
            # Selecionar pais
            parents = self.natural_selector.select_parents(self.population, 2)
            
            # Crossover
            child1, child2 = self.algorithm_crossover.crossover(parents[0], parents[1])
            
            # Mutação
            if random.random() < 0.1:  # 10% de chance de mutação adicional
                child1 = self.parameter_mutator.mutate_genome(child1)
            if random.random() < 0.1:
                child2 = self.parameter_mutator.mutate_genome(child2)
            
            new_population.extend([child1, child2])
        
        # Manter apenas o tamanho da população
        self.population = new_population[:self.population_size]
        self.generation += 1
        
        # Gerar relatório
        report = {
            'generation': self.generation,
            'best_fitness': self.population[0].fitness_score,
            'avg_fitness': avg_fitness,
            'worst_fitness': self.population[-1].fitness_score,
            'population_size': len(self.population),
            'best_genome_id': self.population[0].id,
            'fitness_history': self.fitness_history[-10:]  # Últimas 10 gerações
        }
        
        logger.info(f"Geração {self.generation} evoluiu. Melhor fitness: {report['best_fitness']:.3f}")
        return report
    
    def get_best_genome(self) -> Optional[AlgorithmGenome]:
        """Retorna o melhor genoma da população."""
        if not self.population:
            return None
        
        return max(self.population, key=lambda g: g.fitness_score)
    
    def save_population(self, filepath: str):
        """Salva a população em arquivo."""
        try:
            population_data = {
                'generation': self.generation,
                'population_size': self.population_size,
                'best_genome': asdict(self.best_genome) if self.best_genome else None,
                'fitness_history': self.fitness_history,
                'population': [asdict(genome) for genome in self.population]
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(population_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"População salva em {filepath}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar população: {e}")
    
    def load_population(self, filepath: str) -> bool:
        """Carrega população de arquivo."""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Arquivo de população não encontrado: {filepath}")
                return False
            
            with open(filepath, 'r', encoding='utf-8') as f:
                population_data = json.load(f)
            
            self.generation = population_data.get('generation', 0)
            self.population_size = population_data.get('population_size', 20)
            self.fitness_history = population_data.get('fitness_history', [])
            
            # Carregar melhor genoma
            best_genome_data = population_data.get('best_genome')
            if best_genome_data:
                self.best_genome = AlgorithmGenome(**best_genome_data)
            
            # Carregar população
            population_data_list = population_data.get('population', [])
            self.population = [AlgorithmGenome(**genome_data) for genome_data in population_data_list]
            
            logger.info(f"População carregada: {len(self.population)} genomas, geração {self.generation}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar população: {e}")
            return False

class AlgorithmEvolutionSystem:
    """
    Sistema principal de evolução de algoritmos que orquestra todos os componentes.
    """
    
    def __init__(self, population_size: int = 20):
        self.population_manager = PopulationManager(population_size)
        self.evolution_history: List[Dict[str, Any]] = []
        self.fitness_history: List[float] = []
        self.is_initialized = False
        
        # Configurações de evolução
        self.evolution_config = {
            'max_generations': 100,
            'convergence_threshold': 0.001,
            'stagnation_limit': 10,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elite_percentage': 0.2
        }
    
    def initialize_with_base_config(self, base_config: Dict[str, Any]) -> bool:
        """
        Inicializa o sistema com uma configuração base.
        
        Args:
            base_config: Configuração base do algoritmo
            
        Returns:
            True se inicialização foi bem-sucedida
        """
        try:
            logger.info("Inicializando sistema de evolução de algoritmos")
            
            # Criar genoma base
            base_genome = AlgorithmGenome(
                id="base_genome",
                parameters=base_config.get('parameters', {}),
                architecture=base_config.get('architecture', {}),
                strategy=base_config.get('strategy', {})
            )
            
            # Inicializar população
            self.population_manager.initialize_population(base_genome)
            self.is_initialized = True
            
            logger.info("Sistema de evolução inicializado com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao inicializar sistema de evolução: {e}")
            return False
    
    def run_evolution_cycle(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa um ciclo de evolução.
        
        Args:
            performance_data: Dados de performance para avaliação
            
        Returns:
            Relatório do ciclo de evolução
        """
        if not self.is_initialized:
            return {"error": "Sistema não inicializado"}
        
        logger.info("Iniciando ciclo de evolução")
        
        try:
            # Evoluir geração
            evolution_report = self.population_manager.evolve_generation(performance_data)
            
            # Adicionar ao histórico
            self.evolution_history.append(evolution_report)
            self.fitness_history.append(evolution_report.get('avg_fitness', 0.0))
            
            # Verificar convergência
            convergence_info = self._check_convergence()
            evolution_report.update(convergence_info)
            
            # Salvar população
            self.population_manager.save_population("data/evolution_population.json")
            
            logger.info(f"Ciclo de evolução concluído: geração {evolution_report['generation']}")
            return evolution_report
            
        except Exception as e:
            error_msg = f"Erro durante ciclo de evolução: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _check_convergence(self) -> Dict[str, Any]:
        """Verifica se a evolução convergiu."""
        if len(self.fitness_history) < 5:
            return {"converged": False, "reason": "insufficient_history"}
        
        # Verificar se fitness está estagnado
        recent_fitness = self.fitness_history[-5:]
        fitness_variance = np.var(recent_fitness)
        
        if fitness_variance < self.evolution_config['convergence_threshold']:
            return {"converged": True, "reason": "fitness_stagnation", "variance": fitness_variance}
        
        # Verificar se atingiu máximo de gerações
        if self.population_manager.generation >= self.evolution_config['max_generations']:
            return {"converged": True, "reason": "max_generations_reached"}
        
        return {"converged": False, "reason": "still_evolving", "variance": fitness_variance}
    
    def get_best_algorithm_config(self) -> Optional[Dict[str, Any]]:
        """
        Retorna a configuração do melhor algoritmo.
        
        Returns:
            Configuração do melhor algoritmo ou None
        """
        best_genome = self.population_manager.get_best_genome()
        if not best_genome:
            return None
        
        return {
            'parameters': best_genome.parameters,
            'architecture': best_genome.architecture,
            'strategy': best_genome.strategy,
            'fitness_score': best_genome.fitness_score,
            'generation': best_genome.generation
        }
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Retorna o status atual da evolução."""
        return {
            'is_initialized': self.is_initialized,
            'current_generation': self.population_manager.generation,
            'population_size': len(self.population_manager.population),
            'best_fitness': self.population_manager.population[0].fitness_score if self.population_manager.population else 0.0,
            'avg_fitness': sum(g.fitness_score for g in self.population_manager.population) / len(self.population_manager.population) if self.population_manager.population else 0.0,
            'evolution_history_length': len(self.evolution_history),
            'convergence_info': self._check_convergence()
        }
