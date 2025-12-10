import logging
import json
import os
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np
from collections import deque
from enum import Enum
import copy

# Configurar logging
logger = logging.getLogger(__name__)

class ArchitectureComponent(Enum):
    """Componentes da arquitetura cognitiva."""
    INTUITION_ENGINE = "intuition_engine"
    YOLO_DETECTOR = "yolo_detector"
    COLOR_ANALYZER = "color_analyzer"
    SHAPE_ANALYZER = "shape_analyzer"
    PATTERN_ANALYZER = "pattern_analyzer"
    TEXTURE_ANALYZER = "texture_analyzer"
    SIZE_ANALYZER = "size_analyzer"
    CONFIDENCE_CALCULATOR = "confidence_calculator"
    LEARNING_SYSTEM = "learning_system"
    REASONING_ENGINE = "reasoning_engine"

class ArchitectureStrategy(Enum):
    """Estratégias de evolução de arquitetura."""
    ADD_COMPONENT = "add_component"
    REMOVE_COMPONENT = "remove_component"
    MODIFY_COMPONENT = "modify_component"
    REORGANIZE_CONNECTIONS = "reorganize_connections"
    OPTIMIZE_PARAMETERS = "optimize_parameters"
    HYBRID_APPROACH = "hybrid_approach"

@dataclass
class ArchitectureConfig:
    """Configuração da arquitetura cognitiva."""
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    connections: Dict[str, List[str]] = field(default_factory=dict)
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    fitness_score: float = 0.0
    complexity_score: float = 0.0
    efficiency_score: float = 0.0

@dataclass
class ComponentSpec:
    """Especificação de um componente da arquitetura."""
    name: str
    component_type: ArchitectureComponent
    parameters: Dict[str, Any]
    connections: List[str]
    weight: float
    enabled: bool
    performance_history: List[float] = field(default_factory=list)

class ArchitectureAnalyzer:
    """Analisa a arquitetura atual e identifica oportunidades de melhoria."""
    
    def __init__(self):
        self.component_performance: Dict[str, deque] = {}
        self.connection_analysis: Dict[str, Dict[str, float]] = {}
        self.bottleneck_history: List[Dict[str, Any]] = []
        
    def analyze_architecture(self, config: ArchitectureConfig) -> Dict[str, Any]:
        """
        Analisa a arquitetura atual e retorna insights.
        
        Args:
            config: Configuração da arquitetura atual
            
        Returns:
            Análise detalhada da arquitetura
        """
        analysis = {
            "component_count": len(config.components),
            "connection_count": sum(len(conns) for conns in config.connections.values()),
            "performance_distribution": {},
            "bottlenecks": [],
            "optimization_opportunities": [],
            "complexity_analysis": {},
            "efficiency_metrics": {}
        }
        
        # Analisar performance dos componentes
        for component_name, component_data in config.components.items():
            if "performance_history" in component_data:
                history = component_data["performance_history"]
                if history:
                    analysis["performance_distribution"][component_name] = {
                        "mean": np.mean(history),
                        "std": np.std(history),
                        "trend": self._calculate_trend(history),
                        "stability": 1.0 - np.std(history) if len(history) > 1 else 1.0
                    }
        
        # Identificar gargalos
        analysis["bottlenecks"] = self._identify_bottlenecks(config)
        
        # Identificar oportunidades de otimização
        analysis["optimization_opportunities"] = self._identify_optimization_opportunities(config)
        
        # Análise de complexidade
        analysis["complexity_analysis"] = self._analyze_complexity(config)
        
        # Métricas de eficiência
        analysis["efficiency_metrics"] = self._calculate_efficiency_metrics(config)
        
        return analysis
    
    def _calculate_trend(self, history: List[float]) -> str:
        """Calcula a tendência de performance."""
        if len(history) < 2:
            return "insufficient_data"
        
        recent = np.mean(history[-3:]) if len(history) >= 3 else history[-1]
        older = np.mean(history[:-3]) if len(history) >= 6 else history[0]
        
        change = recent - older
        if change > 0.05:
            return "improving"
        elif change < -0.05:
            return "declining"
        else:
            return "stable"
    
    def _identify_bottlenecks(self, config: ArchitectureConfig) -> List[Dict[str, Any]]:
        """Identifica gargalos na arquitetura."""
        bottlenecks = []
        
        for component_name, component_data in config.components.items():
            if "performance_history" in component_data:
                history = component_data["performance_history"]
                if history and len(history) > 5:
                    avg_performance = np.mean(history)
                    if avg_performance < 0.3:  # Threshold para gargalo
                        bottlenecks.append({
                            "component": component_name,
                            "performance": avg_performance,
                            "severity": "high" if avg_performance < 0.2 else "medium",
                            "suggestion": self._suggest_bottleneck_fix(component_name, avg_performance)
                        })
        
        return bottlenecks
    
    def _suggest_bottleneck_fix(self, component_name: str, performance: float) -> str:
        """Sugere correções para gargalos."""
        if performance < 0.1:
            return f"Considerar remoção ou substituição completa do componente {component_name}"
        elif performance < 0.2:
            return f"Otimizar parâmetros do componente {component_name}"
        else:
            return f"Melhorar conexões do componente {component_name}"
    
    def _identify_optimization_opportunities(self, config: ArchitectureConfig) -> List[Dict[str, Any]]:
        """Identifica oportunidades de otimização."""
        opportunities = []
        
        # Verificar componentes subutilizados
        for component_name, component_data in config.components.items():
            if "performance_history" in component_data:
                history = component_data["performance_history"]
                if history and len(history) > 3:
                    avg_performance = np.mean(history)
                    if avg_performance > 0.8:  # Componente muito eficiente
                        opportunities.append({
                            "type": "component_optimization",
                            "component": component_name,
                            "action": "increase_weight",
                            "reason": f"Componente {component_name} tem alta performance ({avg_performance:.3f})"
                        })
        
        # Verificar conexões desnecessárias
        for source, targets in config.connections.items():
            if len(targets) > 5:  # Muitas conexões
                opportunities.append({
                    "type": "connection_optimization",
                    "component": source,
                    "action": "reduce_connections",
                    "reason": f"Componente {source} tem muitas conexões ({len(targets)})"
                })
        
        return opportunities
    
    def _analyze_complexity(self, config: ArchitectureConfig) -> Dict[str, Any]:
        """Analisa a complexidade da arquitetura."""
        complexity = {
            "component_complexity": len(config.components),
            "connection_complexity": sum(len(conns) for conns in config.connections.values()),
            "parameter_complexity": sum(len(params) for params in config.parameters.values()),
            "overall_complexity": 0.0
        }
        
        # Calcular complexidade geral (normalizada)
        max_components = 20  # Máximo esperado
        max_connections = 50  # Máximo esperado
        max_parameters = 100  # Máximo esperado
        
        complexity["overall_complexity"] = (
            (complexity["component_complexity"] / max_components) * 0.4 +
            (complexity["connection_complexity"] / max_connections) * 0.4 +
            (complexity["parameter_complexity"] / max_parameters) * 0.2
        )
        
        return complexity
    
    def _calculate_efficiency_metrics(self, config: ArchitectureConfig) -> Dict[str, float]:
        """Calcula métricas de eficiência."""
        metrics = {
            "component_efficiency": 0.0,
            "connection_efficiency": 0.0,
            "parameter_efficiency": 0.0,
            "overall_efficiency": 0.0
        }
        
        # Eficiência dos componentes
        if config.components:
            component_performances = []
            for component_data in config.components.values():
                if "performance_history" in component_data and component_data["performance_history"]:
                    component_performances.append(np.mean(component_data["performance_history"]))
            
            if component_performances:
                metrics["component_efficiency"] = np.mean(component_performances)
        
        # Eficiência das conexões (simplificada)
        if config.connections:
            connection_count = sum(len(conns) for conns in config.connections.values())
            optimal_connections = len(config.components) * 2  # Conexões ótimas estimadas
            metrics["connection_efficiency"] = min(1.0, optimal_connections / max(connection_count, 1))
        
        # Eficiência geral
        metrics["overall_efficiency"] = (
            metrics["component_efficiency"] * 0.6 +
            metrics["connection_efficiency"] * 0.4
        )
        
        return metrics

class ArchitectureMutator:
    """Aplica mutações na arquitetura para evolução."""
    
    def __init__(self):
        self.mutation_history: List[Dict[str, Any]] = []
        self.mutation_strategies = {
            ArchitectureStrategy.ADD_COMPONENT: self._add_component,
            ArchitectureStrategy.REMOVE_COMPONENT: self._remove_component,
            ArchitectureStrategy.MODIFY_COMPONENT: self._modify_component,
            ArchitectureStrategy.REORGANIZE_CONNECTIONS: self._reorganize_connections,
            ArchitectureStrategy.OPTIMIZE_PARAMETERS: self._optimize_parameters,
            ArchitectureStrategy.HYBRID_APPROACH: self._hybrid_approach
        }
    
    def mutate_architecture(self, 
                           config: ArchitectureConfig, 
                           strategy: ArchitectureStrategy,
                           analysis: Dict[str, Any]) -> ArchitectureConfig:
        """
        Aplica mutação na arquitetura.
        
        Args:
            config: Configuração atual da arquitetura
            strategy: Estratégia de mutação
            analysis: Análise da arquitetura atual
            
        Returns:
            Nova configuração da arquitetura
        """
        logger.info(f"Aplicando mutação de arquitetura: {strategy.value}")
        
        # Criar cópia da configuração
        new_config = copy.deepcopy(config)
        
        # Aplicar mutação baseada na estratégia
        if strategy in self.mutation_strategies:
            mutation_result = self.mutation_strategies[strategy](new_config, analysis)
            
            # Registrar mutação
            self.mutation_history.append({
                "strategy": strategy.value,
                "timestamp": os.path.getmtime("data/architecture_evolution_config.json") if os.path.exists("data/architecture_evolution_config.json") else None,
                "result": mutation_result,
                "config_before": {
                    "components": len(config.components),
                    "connections": sum(len(conns) for conns in config.connections.values()),
                    "fitness": config.fitness_score
                },
                "config_after": {
                    "components": len(new_config.components),
                    "connections": sum(len(conns) for conns in new_config.connections.values()),
                    "fitness": new_config.fitness_score
                }
            })
        
        return new_config
    
    def _add_component(self, config: ArchitectureConfig, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Adiciona um novo componente à arquitetura."""
        # Identificar onde adicionar componente baseado na análise
        bottlenecks = analysis.get("bottlenecks", [])
        
        if bottlenecks:
            # Adicionar componente para resolver gargalo
            bottleneck = random.choice(bottlenecks)
            component_name = f"optimizer_{bottleneck['component']}_{len(config.components)}"
            
            new_component = {
                "name": component_name,
                "type": "optimizer",
                "parameters": {
                    "target_component": bottleneck["component"],
                    "optimization_factor": 1.5,
                    "enabled": True
                },
                "connections": [bottleneck["component"]],
                "weight": 0.1,
                "performance_history": []
            }
            
            config.components[component_name] = new_component
            config.connections[component_name] = [bottleneck["component"]]
            
            return {
                "action": "add_component",
                "component": component_name,
                "reason": f"Resolver gargalo em {bottleneck['component']}",
                "success": True
            }
        else:
            # Adicionar componente genérico
            component_name = f"generic_component_{len(config.components)}"
            
            new_component = {
                "name": component_name,
                "type": "generic",
                "parameters": {
                    "efficiency": 0.5,
                    "enabled": True
                },
                "connections": [],
                "weight": 0.05,
                "performance_history": []
            }
            
            config.components[component_name] = new_component
            config.connections[component_name] = []
            
            return {
                "action": "add_component",
                "component": component_name,
                "reason": "Expandir capacidade da arquitetura",
                "success": True
            }
    
    def _remove_component(self, config: ArchitectureConfig, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Remove um componente da arquitetura."""
        # Identificar componentes para remoção
        low_performance_components = []
        
        for component_name, component_data in config.components.items():
            if "performance_history" in component_data:
                history = component_data["performance_history"]
                if history and np.mean(history) < 0.2:  # Baixa performance
                    low_performance_components.append(component_name)
        
        if low_performance_components and len(config.components) > 3:  # Manter mínimo de componentes
            component_to_remove = random.choice(low_performance_components)
            
            # Remover componente e suas conexões
            del config.components[component_to_remove]
            if component_to_remove in config.connections:
                del config.connections[component_to_remove]
            
            # Remover conexões para este componente
            for source, targets in config.connections.items():
                if component_to_remove in targets:
                    targets.remove(component_to_remove)
            
            return {
                "action": "remove_component",
                "component": component_to_remove,
                "reason": "Componente com baixa performance",
                "success": True
            }
        else:
            return {
                "action": "remove_component",
                "reason": "Nenhum componente adequado para remoção",
                "success": False
            }
    
    def _modify_component(self, config: ArchitectureConfig, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Modifica um componente existente."""
        if not config.components:
            return {"action": "modify_component", "reason": "Nenhum componente para modificar", "success": False}
        
        component_name = random.choice(list(config.components.keys()))
        component_data = config.components[component_name]
        
        # Modificar parâmetros
        if "parameters" in component_data:
            for param_name, param_value in component_data["parameters"].items():
                if isinstance(param_value, (int, float)):
                    # Aplicar mutação gaussiana
                    mutation_factor = random.gauss(1.0, 0.1)
                    new_value = param_value * mutation_factor
                    
                    # Limitar valores
                    if isinstance(param_value, int):
                        new_value = max(1, int(new_value))
                    else:
                        new_value = max(0.01, min(1.0, new_value))
                    
                    component_data["parameters"][param_name] = new_value
        
        # Modificar peso
        if "weight" in component_data:
            weight_mutation = random.gauss(1.0, 0.15)
            component_data["weight"] = max(0.01, min(1.0, component_data["weight"] * weight_mutation))
        
        return {
            "action": "modify_component",
            "component": component_name,
            "reason": "Otimização de parâmetros",
            "success": True
        }
    
    def _reorganize_connections(self, config: ArchitectureConfig, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Reorganiza as conexões entre componentes."""
        if len(config.components) < 2:
            return {"action": "reorganize_connections", "reason": "Poucos componentes para reorganizar", "success": False}
        
        # Identificar componentes com muitas ou poucas conexões
        connection_counts = {name: len(conns) for name, conns in config.connections.items()}
        
        if connection_counts:
            # Reorganizar conexões
            source_component = random.choice(list(config.components.keys()))
            target_component = random.choice([name for name in config.components.keys() if name != source_component])
            
            if source_component not in config.connections:
                config.connections[source_component] = []
            
            if target_component not in config.connections[source_component]:
                config.connections[source_component].append(target_component)
            else:
                config.connections[source_component].remove(target_component)
        
        return {
            "action": "reorganize_connections",
            "reason": "Otimização de conectividade",
            "success": True
        }
    
    def _optimize_parameters(self, config: ArchitectureConfig, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Otimiza parâmetros da arquitetura."""
        optimized_count = 0
        
        for component_name, component_data in config.components.items():
            if "parameters" in component_data:
                for param_name, param_value in component_data["parameters"].items():
                    if isinstance(param_value, (int, float)):
                        # Otimização baseada em análise
                        if param_name in ["efficiency", "performance"]:
                            # Aumentar eficiência
                            component_data["parameters"][param_name] = min(1.0, param_value * 1.1)
                        elif param_name in ["complexity", "overhead"]:
                            # Reduzir complexidade
                            component_data["parameters"][param_name] = max(0.01, param_value * 0.9)
                        
                        optimized_count += 1
        
        return {
            "action": "optimize_parameters",
            "optimized_count": optimized_count,
            "reason": "Otimização geral de parâmetros",
            "success": optimized_count > 0
        }
    
    def _hybrid_approach(self, config: ArchitectureConfig, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica abordagem híbrida combinando múltiplas estratégias."""
        strategies_applied = []
        
        # Aplicar múltiplas estratégias
        if random.random() < 0.3:
            result = self._modify_component(config, analysis)
            if result["success"]:
                strategies_applied.append("modify_component")
        
        if random.random() < 0.2:
            result = self._optimize_parameters(config, analysis)
            if result["success"]:
                strategies_applied.append("optimize_parameters")
        
        if random.random() < 0.1:
            result = self._reorganize_connections(config, analysis)
            if result["success"]:
                strategies_applied.append("reorganize_connections")
        
        return {
            "action": "hybrid_approach",
            "strategies_applied": strategies_applied,
            "reason": f"Aplicação híbrida: {', '.join(strategies_applied)}",
            "success": len(strategies_applied) > 0
        }

class ArchitectureEvaluator:
    """Avalia a fitness da arquitetura."""
    
    def __init__(self):
        self.evaluation_history: List[Dict[str, Any]] = []
        self.performance_weights = {
            "accuracy": 0.4,
            "efficiency": 0.3,
            "complexity": 0.2,
            "stability": 0.1
        }
    
    def evaluate_architecture(self, 
                             config: ArchitectureConfig, 
                             performance_data: Dict[str, Any]) -> float:
        """
        Avalia a fitness da arquitetura.
        
        Args:
            config: Configuração da arquitetura
            performance_data: Dados de performance do sistema
            
        Returns:
            Score de fitness da arquitetura
        """
        # Calcular métricas de performance
        accuracy_score = self._calculate_accuracy_score(config, performance_data)
        efficiency_score = self._calculate_efficiency_score(config, performance_data)
        complexity_score = self._calculate_complexity_score(config)
        stability_score = self._calculate_stability_score(config, performance_data)
        
        # Calcular fitness geral
        fitness = (
            accuracy_score * self.performance_weights["accuracy"] +
            efficiency_score * self.performance_weights["efficiency"] +
            complexity_score * self.performance_weights["complexity"] +
            stability_score * self.performance_weights["stability"]
        )
        
        # Registrar avaliação
        evaluation = {
            "timestamp": os.path.getmtime("data/architecture_evolution_config.json") if os.path.exists("data/architecture_evolution_config.json") else None,
            "fitness": fitness,
            "accuracy": accuracy_score,
            "efficiency": efficiency_score,
            "complexity": complexity_score,
            "stability": stability_score,
            "component_count": len(config.components),
            "connection_count": sum(len(conns) for conns in config.connections.values())
        }
        
        self.evaluation_history.append(evaluation)
        
        # Atualizar configuração
        config.fitness_score = fitness
        config.performance_metrics = {
            "accuracy": accuracy_score,
            "efficiency": efficiency_score,
            "complexity": complexity_score,
            "stability": stability_score
        }
        
        return fitness
    
    def _calculate_accuracy_score(self, config: ArchitectureConfig, performance_data: Dict[str, Any]) -> float:
        """Calcula score de acurácia."""
        # Simular score baseado na configuração e dados de performance
        base_accuracy = performance_data.get("accuracy", 0.5)
        
        # Ajustar baseado na arquitetura
        component_bonus = min(0.2, len(config.components) * 0.02)
        connection_bonus = min(0.1, sum(len(conns) for conns in config.connections.values()) * 0.005)
        
        return min(1.0, base_accuracy + component_bonus + connection_bonus)
    
    def _calculate_efficiency_score(self, config: ArchitectureConfig, performance_data: Dict[str, Any]) -> float:
        """Calcula score de eficiência."""
        # Simular score baseado na eficiência da arquitetura
        base_efficiency = performance_data.get("efficiency", 0.5)
        
        # Penalizar arquiteturas muito complexas
        complexity_penalty = max(0, (len(config.components) - 5) * 0.05)
        
        return max(0.0, base_efficiency - complexity_penalty)
    
    def _calculate_complexity_score(self, config: ArchitectureConfig) -> float:
        """Calcula score de complexidade (menor é melhor)."""
        # Penalizar arquiteturas muito complexas
        component_penalty = max(0, (len(config.components) - 8) * 0.1)
        connection_penalty = max(0, (sum(len(conns) for conns in config.connections.values()) - 15) * 0.05)
        
        complexity_score = 1.0 - (component_penalty + connection_penalty)
        return max(0.0, complexity_score)
    
    def _calculate_stability_score(self, config: ArchitectureConfig, performance_data: Dict[str, Any]) -> float:
        """Calcula score de estabilidade."""
        # Simular score baseado na estabilidade dos componentes
        stability_scores = []
        
        for component_data in config.components.values():
            if "performance_history" in component_data:
                history = component_data["performance_history"]
                if len(history) > 1:
                    stability = 1.0 - np.std(history)
                    stability_scores.append(max(0.0, stability))
        
        if stability_scores:
            return np.mean(stability_scores)
        else:
            return 0.5  # Score padrão

class ArchitectureEvolutionSystem:
    """Sistema principal de evolução de arquitetura."""
    
    def __init__(self, 
                 config_file: str = "data/architecture_evolution_config.json",
                 evolution_frequency: int = 20):
        self.config_file = config_file
        self.evolution_frequency = evolution_frequency
        self.is_active = False
        
        self.analyzer = ArchitectureAnalyzer()
        self.mutator = ArchitectureMutator()
        self.evaluator = ArchitectureEvaluator()
        
        self.current_architecture = self._load_configuration()
        self.evolution_history: List[Dict[str, Any]] = []
        
        logger.info("Sistema de evolução de arquitetura inicializado")
    
    def _load_configuration(self) -> ArchitectureConfig:
        """Carrega configuração da arquitetura."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    
                config = ArchitectureConfig(
                    components=data.get("components", {}),
                    connections=data.get("connections", {}),
                    parameters=data.get("parameters", {}),
                    performance_metrics=data.get("performance_metrics", {}),
                    evolution_history=data.get("evolution_history", []),
                    fitness_score=data.get("fitness_score", 0.0),
                    complexity_score=data.get("complexity_score", 0.0),
                    efficiency_score=data.get("efficiency_score", 0.0)
                )
                
                logger.info(f"Configuração de arquitetura carregada de {self.config_file}")
                return config
                
            except Exception as e:
                logger.error(f"Erro ao carregar configuração de arquitetura: {e}")
                return self._create_default_architecture()
        else:
            logger.info("Nenhuma configuração de arquitetura encontrada. Criando arquitetura padrão.")
            return self._create_default_architecture()
    
    def _create_default_architecture(self) -> ArchitectureConfig:
        """Cria arquitetura padrão."""
        default_components = {
            "intuition_engine": {
                "name": "intuition_engine",
                "type": "core",
                "parameters": {
                    "efficiency": 0.8,
                    "complexity": 0.6,
                    "enabled": True
                },
                "connections": ["yolo_detector", "color_analyzer", "shape_analyzer"],
                "weight": 0.3,
                "performance_history": []
            },
            "yolo_detector": {
                "name": "yolo_detector",
                "type": "detector",
                "parameters": {
                    "confidence_threshold": 0.5,
                    "enabled": True
                },
                "connections": ["intuition_engine"],
                "weight": 0.25,
                "performance_history": []
            },
            "color_analyzer": {
                "name": "color_analyzer",
                "type": "analyzer",
                "parameters": {
                    "sensitivity": 0.7,
                    "enabled": True
                },
                "connections": ["intuition_engine"],
                "weight": 0.2,
                "performance_history": []
            },
            "shape_analyzer": {
                "name": "shape_analyzer",
                "type": "analyzer",
                "parameters": {
                    "sensitivity": 0.7,
                    "enabled": True
                },
                "connections": ["intuition_engine"],
                "weight": 0.2,
                "performance_history": []
            }
        }
        
        default_connections = {
            "intuition_engine": ["yolo_detector", "color_analyzer", "shape_analyzer"],
            "yolo_detector": ["intuition_engine"],
            "color_analyzer": ["intuition_engine"],
            "shape_analyzer": ["intuition_engine"]
        }
        
        return ArchitectureConfig(
            components=default_components,
            connections=default_connections,
            parameters={},
            performance_metrics={},
            evolution_history=[],
            fitness_score=0.0,
            complexity_score=0.0,
            efficiency_score=0.0
        )
    
    def _save_configuration(self):
        """Salva configuração da arquitetura."""
        try:
            config_data = {
                "components": self.current_architecture.components,
                "connections": self.current_architecture.connections,
                "parameters": self.current_architecture.parameters,
                "performance_metrics": self.current_architecture.performance_metrics,
                "evolution_history": self.current_architecture.evolution_history,
                "fitness_score": self.current_architecture.fitness_score,
                "complexity_score": self.current_architecture.complexity_score,
                "efficiency_score": self.current_architecture.efficiency_score
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
            
            logger.info(f"Configuração de arquitetura salva em {self.config_file}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar configuração de arquitetura: {e}")
    
    def start_evolution(self):
        """Inicia o sistema de evolução de arquitetura."""
        self.is_active = True
        logger.info("Sistema de evolução de arquitetura iniciado")
    
    def stop_evolution(self):
        """Para o sistema de evolução de arquitetura."""
        self.is_active = False
        logger.info("Sistema de evolução de arquitetura parado")
    
    def run_evolution_cycle(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa um ciclo de evolução de arquitetura.
        
        Args:
            performance_data: Dados de performance do sistema
            
        Returns:
            Resultado do ciclo de evolução
        """
        if not self.is_active:
            return {"error": "Sistema de evolução de arquitetura não está ativo"}
        
        try:
            logger.info("Iniciando ciclo de evolução de arquitetura")
            
            # 1. Analisar arquitetura atual
            analysis = self.analyzer.analyze_architecture(self.current_architecture)
            
            # 2. Avaliar fitness atual
            current_fitness = self.evaluator.evaluate_architecture(self.current_architecture, performance_data)
            
            # 3. Decidir se evoluir
            should_evolve = self._should_evolve(analysis, current_fitness)
            
            if should_evolve:
                # 4. Selecionar estratégia de evolução
                strategy = self._select_evolution_strategy(analysis)
                
                # 5. Aplicar mutação
                new_architecture = self.mutator.mutate_architecture(
                    self.current_architecture, strategy, analysis
                )
                
                # 6. Avaliar nova arquitetura
                new_fitness = self.evaluator.evaluate_architecture(new_architecture, performance_data)
                
                # 7. Decidir se aceitar mudança
                if new_fitness > current_fitness:
                    self.current_architecture = new_architecture
                    self.current_architecture.evolution_history.append({
                        "timestamp": os.path.getmtime(self.config_file) if os.path.exists(self.config_file) else None,
                        "strategy": strategy.value,
                        "fitness_before": current_fitness,
                        "fitness_after": new_fitness,
                        "improvement": new_fitness - current_fitness
                    })
                    
                    self._save_configuration()
                    
                    logger.info(f"Arquitetura evoluída! Fitness: {current_fitness:.3f} → {new_fitness:.3f}")
                    
                    return {
                        "success": True,
                        "message": "Arquitetura evoluída com sucesso",
                        "strategy": strategy.value,
                        "fitness_before": current_fitness,
                        "fitness_after": new_fitness,
                        "improvement": new_fitness - current_fitness,
                        "analysis": analysis
                    }
                else:
                    logger.info(f"Evolução rejeitada. Fitness: {current_fitness:.3f} → {new_fitness:.3f}")
                    
                    return {
                        "success": False,
                        "message": "Evolução rejeitada - sem melhoria",
                        "strategy": strategy.value,
                        "fitness_before": current_fitness,
                        "fitness_after": new_fitness,
                        "analysis": analysis
                    }
            else:
                logger.info("Evolução não necessária neste ciclo")
                
                return {
                    "success": True,
                    "message": "Evolução não necessária",
                    "fitness": current_fitness,
                    "analysis": analysis
                }
                
        except Exception as e:
            error_msg = f"Erro durante ciclo de evolução: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _should_evolve(self, analysis: Dict[str, Any], current_fitness: float) -> bool:
        """Decide se a arquitetura deve evoluir."""
        # Evoluir se há gargalos
        if analysis.get("bottlenecks"):
            return True
        
        # Evoluir se fitness é baixo
        if current_fitness < 0.6:
            return True
        
        # Evoluir se há oportunidades de otimização
        if analysis.get("optimization_opportunities"):
            return True
        
        # Evoluir aleatoriamente com baixa probabilidade
        return random.random() < 0.1
    
    def _select_evolution_strategy(self, analysis: Dict[str, Any]) -> ArchitectureStrategy:
        """Seleciona estratégia de evolução baseada na análise."""
        strategies = []
        
        # Adicionar estratégias baseadas na análise
        if analysis.get("bottlenecks"):
            strategies.extend([ArchitectureStrategy.ADD_COMPONENT, ArchitectureStrategy.MODIFY_COMPONENT])
        
        if analysis.get("optimization_opportunities"):
            strategies.extend([ArchitectureStrategy.OPTIMIZE_PARAMETERS, ArchitectureStrategy.REORGANIZE_CONNECTIONS])
        
        if len(self.current_architecture.components) > 8:
            strategies.append(ArchitectureStrategy.REMOVE_COMPONENT)
        
        # Estratégia padrão
        if not strategies:
            strategies = [ArchitectureStrategy.MODIFY_COMPONENT, ArchitectureStrategy.OPTIMIZE_PARAMETERS]
        
        return random.choice(strategies)
    
    def get_current_architecture(self) -> Dict[str, Any]:
        """Retorna a arquitetura atual."""
        return {
            "components": self.current_architecture.components,
            "connections": self.current_architecture.connections,
            "fitness_score": self.current_architecture.fitness_score,
            "performance_metrics": self.current_architecture.performance_metrics,
            "evolution_history": self.current_architecture.evolution_history
        }
    
    def get_architecture_analysis(self) -> Dict[str, Any]:
        """Retorna análise da arquitetura atual."""
        analysis = self.analyzer.analyze_architecture(self.current_architecture)
        
        return {
            "current_architecture": self.get_current_architecture(),
            "analysis": analysis,
            "evaluation_history": self.evaluator.evaluation_history,
            "mutation_history": self.mutator.mutation_history,
            "system_active": self.is_active
        }
