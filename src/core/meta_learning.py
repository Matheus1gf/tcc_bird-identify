#!/usr/bin/env python3
"""
Sistema de Meta-Aprendizado
==========================

Este módulo implementa um sistema avançado de meta-aprendizado que permite:
- Aprendizado sobre como aprender melhor
- Adaptação de estratégias de aprendizado baseado no histórico
- Auto-reflexão sobre o próprio processo de aprendizado
- Otimização de hiperparâmetros de aprendizado
- Sistema de feedback loop para melhoria contínua
- Análise de padrões de sucesso e falha
- Sistema de recomendações de estratégias
"""

import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict, Counter
import random
import time
import pickle

# Configurar logging primeiro
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tentar importar sklearn, mas não falhar se não estiver disponível
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn não disponível - funcionalidades de otimização limitadas")

class MetaLearningStrategy(Enum):
    """Estratégias de meta-aprendizado."""
    STRATEGY_OPTIMIZATION = "strategy_optimization"    # Otimização de estratégias
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"    # Ajuste de hiperparâmetros
    FEATURE_SELECTION = "feature_selection"             # Seleção de características
    LEARNING_RATE_ADAPTATION = "learning_rate_adaptation"  # Adaptação de taxa de aprendizado
    ARCHITECTURE_SEARCH = "architecture_search"         # Busca de arquitetura
    TRANSFER_OPTIMIZATION = "transfer_optimization"     # Otimização de transferência

class LearningContext(Enum):
    """Contextos de aprendizado."""
    NEW_CONCEPT = "new_concept"           # Conceito novo
    SIMILAR_CONCEPT = "similar_concept"   # Conceito similar
    COMPLEX_CONCEPT = "complex_concept"   # Conceito complexo
    SIMPLE_CONCEPT = "simple_concept"     # Conceito simples
    FEW_EXAMPLES = "few_examples"         # Poucos exemplos
    MANY_EXAMPLES = "many_examples"       # Muitos exemplos

class PerformanceMetric(Enum):
    """Métricas de performance."""
    ACCURACY = "accuracy"                 # Precisão
    CONFIDENCE = "confidence"            # Confiança
    LEARNING_SPEED = "learning_speed"    # Velocidade de aprendizado
    GENERALIZATION = "generalization"     # Generalização
    STABILITY = "stability"              # Estabilidade
    EFFICIENCY = "efficiency"            # Eficiência

@dataclass
class LearningSession:
    """Sessão de aprendizado."""
    session_id: str
    concept_name: str
    concept_type: str
    strategy_used: str
    context: LearningContext
    examples_count: int
    performance_metrics: Dict[str, float]
    success: bool
    learning_time: float
    timestamp: float
    metadata: Dict[str, Any]

@dataclass
class MetaLearningRule:
    """Regra de meta-aprendizado."""
    rule_id: str
    condition: Dict[str, Any]
    recommendation: Dict[str, Any]
    confidence: float
    usage_count: int
    success_rate: float
    created_at: float
    last_updated: float

@dataclass
class MetaLearningResult:
    """Resultado de meta-aprendizado."""
    session_id: str
    strategy_recommendation: str
    hyperparameters: Dict[str, Any]
    confidence: float
    reasoning: List[str]
    expected_improvement: float
    timestamp: float

class PerformanceAnalyzer:
    """Analisador de performance para meta-aprendizado."""
    
    def __init__(self):
        self.metric_weights = {
            PerformanceMetric.ACCURACY: 0.3,
            PerformanceMetric.CONFIDENCE: 0.2,
            PerformanceMetric.LEARNING_SPEED: 0.15,
            PerformanceMetric.GENERALIZATION: 0.15,
            PerformanceMetric.STABILITY: 0.1,
            PerformanceMetric.EFFICIENCY: 0.1
        }
    
    def analyze_session_performance(self, session: LearningSession) -> Dict[str, Any]:
        """Analisa performance de uma sessão de aprendizado."""
        try:
            # Calcular score geral de performance
            overall_score = self._calculate_overall_score(session.performance_metrics)
            
            # Identificar pontos fortes e fracos
            strengths, weaknesses = self._identify_strengths_weaknesses(session.performance_metrics)
            
            # Calcular eficiência de aprendizado
            learning_efficiency = self._calculate_learning_efficiency(session)
            
            # Identificar padrões de sucesso/falha
            success_patterns = self._identify_success_patterns(session)
            
            return {
                "overall_score": overall_score,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "learning_efficiency": learning_efficiency,
                "success_patterns": success_patterns,
                "recommendations": self._generate_recommendations(session, overall_score)
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de performance: {e}")
            return {"error": str(e)}
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calcula score geral de performance."""
        total_score = 0.0
        total_weight = 0.0
        
        for metric_name, value in metrics.items():
            if metric_name in [m.value for m in PerformanceMetric]:
                metric_enum = PerformanceMetric(metric_name)
                weight = self.metric_weights.get(metric_enum, 0.1)
                total_score += value * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _identify_strengths_weaknesses(self, metrics: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """Identifica pontos fortes e fracos."""
        strengths = []
        weaknesses = []
        
        for metric_name, value in metrics.items():
            if value >= 0.8:
                strengths.append(f"{metric_name}: {value:.3f}")
            elif value <= 0.4:
                weaknesses.append(f"{metric_name}: {value:.3f}")
        
        return strengths, weaknesses
    
    def _calculate_learning_efficiency(self, session: LearningSession) -> float:
        """Calcula eficiência de aprendizado."""
        if session.learning_time <= 0:
            return 0.0
        
        # Eficiência baseada na performance vs tempo
        performance_score = self._calculate_overall_score(session.performance_metrics)
        efficiency = performance_score / session.learning_time
        
        return min(efficiency, 1.0)
    
    def _identify_success_patterns(self, session: LearningSession) -> List[str]:
        """Identifica padrões de sucesso."""
        patterns = []
        
        if session.success:
            if session.examples_count <= 3:
                patterns.append("few_examples_success")
            if session.performance_metrics.get("confidence", 0) >= 0.8:
                patterns.append("high_confidence_success")
            if session.learning_time <= 1.0:
                patterns.append("fast_learning_success")
        
        return patterns
    
    def _generate_recommendations(self, session: LearningSession, overall_score: float) -> List[str]:
        """Gera recomendações baseadas na análise."""
        recommendations = []
        
        if overall_score < 0.5:
            recommendations.append("Considerar estratégia diferente")
            recommendations.append("Aumentar número de exemplos")
            recommendations.append("Revisar características utilizadas")
        elif overall_score >= 0.8:
            recommendations.append("Estratégia funcionando bem")
            recommendations.append("Manter configuração atual")
        else:
            recommendations.append("Estratégia moderadamente eficaz")
            recommendations.append("Considerar pequenos ajustes")
        
        return recommendations

class StrategyOptimizer:
    """Otimizador de estratégias para meta-aprendizado."""
    
    def __init__(self):
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        self.context_strategy_map: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.strategy_recommendations: Dict[str, Dict[str, Any]] = {}
    
    def update_strategy_performance(self, strategy: str, context: LearningContext, 
                                  performance: float, session: LearningSession):
        """Atualiza performance de uma estratégia."""
        try:
            # Atualizar histórico de performance
            self.strategy_performance[strategy].append(performance)
            
            # Atualizar mapeamento contexto-estratégia
            context_key = f"{context.value}_{session.concept_type}_{session.examples_count}"
            self.context_strategy_map[context_key][strategy] = performance
            
            # Gerar recomendações atualizadas
            self._update_recommendations(strategy, context, performance)
            
        except Exception as e:
            logger.error(f"Erro ao atualizar performance da estratégia: {e}")
    
    def recommend_strategy(self, context: LearningContext, concept_type: str, 
                         examples_count: int) -> Dict[str, Any]:
        """Recomenda estratégia baseada no contexto."""
        try:
            context_key = f"{context.value}_{concept_type}_{examples_count}"
            
            # Buscar estratégias para o contexto
            if context_key in self.context_strategy_map:
                strategies = self.context_strategy_map[context_key]
                best_strategy = max(strategies.items(), key=lambda x: x[1])
                
                return {
                    "recommended_strategy": best_strategy[0],
                    "confidence": best_strategy[1],
                    "alternative_strategies": sorted(strategies.items(), key=lambda x: x[1], reverse=True)[1:3],
                    "reasoning": self._generate_reasoning(best_strategy[0], context, concept_type)
                }
            else:
                # Estratégia padrão baseada no contexto
                default_strategy = self._get_default_strategy(context, concept_type, examples_count)
                
                return {
                    "recommended_strategy": default_strategy,
                    "confidence": 0.5,
                    "alternative_strategies": [],
                    "reasoning": ["Estratégia padrão baseada no contexto"]
                }
                
        except Exception as e:
            logger.error(f"Erro ao recomendar estratégia: {e}")
            return {"error": str(e)}
    
    def _update_recommendations(self, strategy: str, context: LearningContext, performance: float):
        """Atualiza recomendações de estratégias."""
        context_key = f"{context.value}"
        
        if context_key not in self.strategy_recommendations:
            self.strategy_recommendations[context_key] = {}
        
        if strategy not in self.strategy_recommendations[context_key]:
            self.strategy_recommendations[context_key][strategy] = {
                "total_performance": 0.0,
                "usage_count": 0,
                "average_performance": 0.0
            }
        
        # Atualizar estatísticas
        stats = self.strategy_recommendations[context_key][strategy]
        stats["total_performance"] += performance
        stats["usage_count"] += 1
        stats["average_performance"] = stats["total_performance"] / stats["usage_count"]
    
    def _generate_reasoning(self, strategy: str, context: LearningContext, concept_type: str) -> List[str]:
        """Gera raciocínio para a recomendação."""
        reasoning = []
        
        if context == LearningContext.NEW_CONCEPT:
            reasoning.append("Conceito novo requer estratégia robusta")
        elif context == LearningContext.SIMILAR_CONCEPT:
            reasoning.append("Conceito similar pode usar transferência")
        elif context == LearningContext.COMPLEX_CONCEPT:
            reasoning.append("Conceito complexo requer estratégia adaptativa")
        
        if concept_type == "species":
            reasoning.append("Espécies requerem análise morfológica detalhada")
        elif concept_type == "behavior":
            reasoning.append("Comportamentos requerem análise temporal")
        
        reasoning.append(f"Estratégia '{strategy}' tem histórico positivo neste contexto")
        
        return reasoning
    
    def _get_default_strategy(self, context: LearningContext, concept_type: str, examples_count: int) -> str:
        """Obtém estratégia padrão baseada no contexto."""
        if context == LearningContext.NEW_CONCEPT:
            return "prototype_based"
        elif context == LearningContext.SIMILAR_CONCEPT:
            return "transfer_learning"
        elif context == LearningContext.COMPLEX_CONCEPT:
            return "adaptive_fusion"
        elif examples_count <= 3:
            return "similarity_based"
        else:
            return "meta_learning"

class HyperparameterOptimizer:
    """Otimizador de hiperparâmetros para meta-aprendizado."""
    
    def __init__(self):
        self.hyperparameter_history: List[Dict[str, Any]] = []
        self.optimization_model = None
        self.parameter_ranges = {
            "learning_rate": (0.001, 0.1),
            "confidence_threshold": (0.3, 0.9),
            "similarity_threshold": (0.5, 0.95),
            "adaptation_rate": (0.01, 0.5),
            "fusion_weight": (0.1, 0.9)
        }
    
    def optimize_hyperparameters(self, context: LearningContext, concept_type: str, 
                                performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Otimiza hiperparâmetros baseado no histórico."""
        try:
            if len(performance_history) < 5:
                # Retornar configuração padrão se histórico insuficiente
                return self._get_default_hyperparameters(context, concept_type)
            
            # Preparar dados para otimização
            X, y = self._prepare_optimization_data(performance_history)
            
            if len(X) < 3:
                return self._get_default_hyperparameters(context, concept_type)
            
            # Treinar modelo de otimização
            if SKLEARN_AVAILABLE:
                self.optimization_model = RandomForestRegressor(n_estimators=50, random_state=42)
                self.optimization_model.fit(X, y)
            else:
                # Usar modelo simples sem sklearn
                self.optimization_model = None
            
            # Otimizar hiperparâmetros
            optimized_params = self._optimize_parameters(context, concept_type)
            
            return {
                "optimized_parameters": optimized_params,
                "confidence": self._calculate_optimization_confidence(),
                "improvement_expected": self._estimate_improvement(optimized_params),
                "optimization_method": "random_forest_regression"
            }
            
        except Exception as e:
            logger.error(f"Erro na otimização de hiperparâmetros: {e}")
            return self._get_default_hyperparameters(context, concept_type)
    
    def _prepare_optimization_data(self, performance_history: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[float]]:
        """Prepara dados para otimização."""
        X = []
        y = []
        
        for record in performance_history:
            # Extrair características do contexto
            features = [
                record.get("examples_count", 0),
                record.get("learning_time", 0),
                record.get("concept_complexity", 0.5),
                record.get("context_similarity", 0.5)
            ]
            
            # Extrair hiperparâmetros
            hyperparams = record.get("hyperparameters", {})
            for param_name in self.parameter_ranges.keys():
                features.append(hyperparams.get(param_name, 0.5))
            
            X.append(features)
            y.append(record.get("performance_score", 0.5))
        
        return X, y
    
    def _optimize_parameters(self, context: LearningContext, concept_type: str) -> Dict[str, float]:
        """Otimiza parâmetros usando o modelo treinado."""
        # Gerar configurações candidatas
        candidates = self._generate_parameter_candidates()
        
        best_params = None
        best_score = -1
        
        for params in candidates:
            # Preparar features para predição
            features = self._prepare_prediction_features(params, context, concept_type)
            
            if self.optimization_model and SKLEARN_AVAILABLE:
                predicted_score = self.optimization_model.predict([features])[0]
                
                if predicted_score > best_score:
                    best_score = predicted_score
                    best_params = params
            else:
                # Usar heurística simples sem sklearn
                heuristic_score = self._calculate_heuristic_score(params, context, concept_type)
                
                if heuristic_score > best_score:
                    best_score = heuristic_score
                    best_params = params
        
        return best_params if best_params else self._get_default_hyperparameters(context, concept_type)["optimized_parameters"]
    
    def _generate_parameter_candidates(self) -> List[Dict[str, float]]:
        """Gera candidatos para otimização de parâmetros."""
        candidates = []
        
        # Gerar configurações aleatórias
        for _ in range(20):
            params = {}
            for param_name, (min_val, max_val) in self.parameter_ranges.items():
                params[param_name] = random.uniform(min_val, max_val)
            candidates.append(params)
        
        return candidates
    
    def _prepare_prediction_features(self, params: Dict[str, float], 
                                   context: LearningContext, concept_type: str) -> List[float]:
        """Prepara features para predição."""
        features = [
            3.0,  # examples_count médio
            1.0,  # learning_time médio
            0.5,  # concept_complexity médio
            0.5   # context_similarity médio
        ]
        
        # Adicionar hiperparâmetros
        for param_name in self.parameter_ranges.keys():
            features.append(params.get(param_name, 0.5))
        
        return features
    
    def _calculate_heuristic_score(self, params: Dict[str, float], 
                                 context: LearningContext, concept_type: str) -> float:
        """Calcula score heurístico para parâmetros sem sklearn."""
        score = 0.5  # Score base
        
        # Ajustar baseado no contexto
        if context == LearningContext.NEW_CONCEPT:
            if params.get("learning_rate", 0.5) > 0.05:
                score += 0.1
            if params.get("confidence_threshold", 0.5) < 0.7:
                score += 0.1
        elif context == LearningContext.SIMILAR_CONCEPT:
            if params.get("similarity_threshold", 0.5) < 0.8:
                score += 0.1
            if params.get("adaptation_rate", 0.5) > 0.1:
                score += 0.1
        
        # Ajustar baseado no tipo de conceito
        if concept_type == "species":
            if params.get("fusion_weight", 0.5) > 0.3:
                score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_optimization_confidence(self) -> float:
        """Calcula confiança na otimização."""
        if not self.optimization_model:
            return 0.0
        
        # Usar validação cruzada para estimar confiança
        if SKLEARN_AVAILABLE and self.optimization_model:
            try:
                X, y = self._prepare_optimization_data(self.hyperparameter_history)
                if len(X) >= 3:
                    scores = cross_val_score(self.optimization_model, X, y, cv=min(3, len(X)))
                    return float(np.mean(scores))
            except:
                pass
        
        return 0.5
    
    def _estimate_improvement(self, params: Dict[str, float]) -> float:
        """Estima melhoria esperada."""
        # Estimativa simples baseada na diferença dos parâmetros
        default_params = self._get_default_hyperparameters(LearningContext.NEW_CONCEPT, "species")
        default_values = default_params["optimized_parameters"]
        
        total_diff = 0.0
        for param_name in self.parameter_ranges.keys():
            diff = abs(params.get(param_name, 0.5) - default_values.get(param_name, 0.5))
            total_diff += diff
        
        # Converter diferença em estimativa de melhoria
        improvement = min(total_diff * 0.1, 0.3)  # Máximo 30% de melhoria
        return improvement
    
    def _get_default_hyperparameters(self, context: LearningContext, concept_type: str) -> Dict[str, Any]:
        """Obtém hiperparâmetros padrão."""
        defaults = {
            "learning_rate": 0.01,
            "confidence_threshold": 0.7,
            "similarity_threshold": 0.8,
            "adaptation_rate": 0.1,
            "fusion_weight": 0.5
        }
        
        # Ajustar baseado no contexto
        if context == LearningContext.NEW_CONCEPT:
            defaults["learning_rate"] = 0.05
            defaults["confidence_threshold"] = 0.6
        elif context == LearningContext.SIMILAR_CONCEPT:
            defaults["similarity_threshold"] = 0.7
            defaults["adaptation_rate"] = 0.2
        
        return {
            "optimized_parameters": defaults,
            "confidence": 0.5,
            "improvement_expected": 0.0,
            "optimization_method": "default"
        }

class MetaLearningSystem:
    """Sistema principal de meta-aprendizado."""
    
    def __init__(self):
        self.performance_analyzer = PerformanceAnalyzer()
        self.strategy_optimizer = StrategyOptimizer()
        self.hyperparameter_optimizer = HyperparameterOptimizer()
        self.learning_sessions: List[LearningSession] = []
        self.meta_rules: List[MetaLearningRule] = []
        self.meta_learning_history: List[MetaLearningResult] = []
        
        # Carregar dados existentes
        self._load_data()
    
    def analyze_learning_session(self, session: LearningSession) -> Dict[str, Any]:
        """Analisa uma sessão de aprendizado para meta-aprendizado."""
        try:
            # Analisar performance da sessão
            performance_analysis = self.performance_analyzer.analyze_session_performance(session)
            
            # Atualizar performance da estratégia
            self.strategy_optimizer.update_strategy_performance(
                session.strategy_used, 
                session.context, 
                performance_analysis.get("overall_score", 0.0),
                session
            )
            
            # Adicionar à lista de sessões
            self.learning_sessions.append(session)
            
            # Gerar regras de meta-aprendizado
            new_rules = self._generate_meta_rules(session, performance_analysis)
            self.meta_rules.extend(new_rules)
            
            # Atualizar histórico de hiperparâmetros
            self.hyperparameter_optimizer.hyperparameter_history.append({
                "examples_count": session.examples_count,
                "learning_time": session.learning_time,
                "concept_complexity": self._estimate_concept_complexity(session),
                "context_similarity": self._estimate_context_similarity(session),
                "hyperparameters": session.metadata.get("hyperparameters", {}),
                "performance_score": performance_analysis.get("overall_score", 0.0)
            })
            
            # Salvar dados
            self._save_data()
            
            return {
                "performance_analysis": performance_analysis,
                "new_rules_generated": len(new_rules),
                "strategy_performance_updated": True,
                "meta_learning_insights": self._generate_insights(session, performance_analysis)
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de sessão de aprendizado: {e}")
            return {"error": str(e)}
    
    def recommend_learning_strategy(self, concept_name: str, concept_type: str, 
                                  examples_count: int, context: LearningContext = LearningContext.NEW_CONCEPT) -> Dict[str, Any]:
        """Recomenda estratégia de aprendizado baseada em meta-aprendizado."""
        try:
            # Recomendar estratégia
            strategy_recommendation = self.strategy_optimizer.recommend_strategy(
                context, concept_type, examples_count
            )
            
            # Otimizar hiperparâmetros
            hyperparameter_optimization = self.hyperparameter_optimizer.optimize_hyperparameters(
                context, concept_type, self.hyperparameter_optimizer.hyperparameter_history
            )
            
            # Gerar raciocínio
            reasoning = self._generate_meta_reasoning(
                strategy_recommendation, hyperparameter_optimization, context, concept_type
            )
            
            # Calcular confiança geral
            overall_confidence = (
                strategy_recommendation.get("confidence", 0.5) + 
                hyperparameter_optimization.get("confidence", 0.5)
            ) / 2.0
            
            # Calcular melhoria esperada
            expected_improvement = hyperparameter_optimization.get("improvement_expected", 0.0)
            
            # Criar resultado
            result = MetaLearningResult(
                session_id=f"meta_{int(time.time())}",
                strategy_recommendation=strategy_recommendation.get("recommended_strategy", "prototype_based"),
                hyperparameters=hyperparameter_optimization.get("optimized_parameters", {}),
                confidence=overall_confidence,
                reasoning=reasoning,
                expected_improvement=expected_improvement,
                timestamp=time.time()
            )
            
            # Adicionar ao histórico
            self.meta_learning_history.append(result)
            
            return {
                "strategy_recommendation": strategy_recommendation,
                "hyperparameter_optimization": hyperparameter_optimization,
                "overall_confidence": overall_confidence,
                "expected_improvement": expected_improvement,
                "reasoning": reasoning,
                "meta_learning_result": asdict(result)
            }
            
        except Exception as e:
            logger.error(f"Erro na recomendação de estratégia: {e}")
            return {"error": str(e)}
    
    def get_meta_learning_analysis(self) -> Dict[str, Any]:
        """Obtém análise geral do sistema de meta-aprendizado."""
        try:
            # Estatísticas gerais
            total_sessions = len(self.learning_sessions)
            total_rules = len(self.meta_rules)
            total_recommendations = len(self.meta_learning_history)
            
            # Análise de performance por estratégia
            strategy_performance = {}
            for strategy, performances in self.strategy_optimizer.strategy_performance.items():
                if performances:
                    strategy_performance[strategy] = {
                        "average_performance": np.mean(performances),
                        "usage_count": len(performances),
                        "best_performance": max(performances),
                        "stability": 1.0 - np.std(performances) if len(performances) > 1 else 1.0
                    }
            
            # Análise de regras de meta-aprendizado
            rule_analysis = {}
            for rule in self.meta_rules:
                rule_type = rule.condition.get("type", "unknown")
                if rule_type not in rule_analysis:
                    rule_analysis[rule_type] = {
                        "count": 0,
                        "average_confidence": 0.0,
                        "average_success_rate": 0.0
                    }
                
                rule_analysis[rule_type]["count"] += 1
                rule_analysis[rule_type]["average_confidence"] += rule.confidence
                rule_analysis[rule_type]["average_success_rate"] += rule.success_rate
            
            # Normalizar médias
            for rule_type in rule_analysis:
                count = rule_analysis[rule_type]["count"]
                rule_analysis[rule_type]["average_confidence"] /= count
                rule_analysis[rule_type]["average_success_rate"] /= count
            
            # Análise de tendências
            recent_sessions = self.learning_sessions[-10:] if len(self.learning_sessions) >= 10 else self.learning_sessions
            recent_performance = [self.performance_analyzer.analyze_session_performance(session).get("overall_score", 0.0) 
                                for session in recent_sessions]
            
            performance_trend = "improving" if len(recent_performance) >= 2 and recent_performance[-1] > recent_performance[0] else "stable"
            
            return {
                "total_sessions": total_sessions,
                "total_rules": total_rules,
                "total_recommendations": total_recommendations,
                "strategy_performance": strategy_performance,
                "rule_analysis": rule_analysis,
                "performance_trend": performance_trend,
                "recent_performance": recent_performance,
                "meta_learning_effectiveness": self._calculate_meta_learning_effectiveness(),
                "recommendations": self._generate_system_recommendations(),
                "analysis_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de meta-aprendizado: {e}")
            return {"error": str(e)}
    
    def _generate_meta_rules(self, session: LearningSession, performance_analysis: Dict[str, Any]) -> List[MetaLearningRule]:
        """Gera regras de meta-aprendizado baseadas na sessão."""
        rules = []
        
        # Regra baseada no contexto
        if performance_analysis.get("overall_score", 0) >= 0.8:
            rule = MetaLearningRule(
                rule_id=f"success_{session.session_id}",
                condition={
                    "type": "context_success",
                    "context": session.context.value,
                    "concept_type": session.concept_type,
                    "examples_count_range": f"{session.examples_count-1}-{session.examples_count+1}"
                },
                recommendation={
                    "strategy": session.strategy_used,
                    "confidence_threshold": 0.7,
                    "adaptation_rate": 0.1
                },
                confidence=performance_analysis.get("overall_score", 0.0),
                usage_count=1,
                success_rate=1.0,
                created_at=time.time(),
                last_updated=time.time()
            )
            rules.append(rule)
        
        # Regra baseada em padrões de sucesso
        success_patterns = performance_analysis.get("success_patterns", [])
        for pattern in success_patterns:
            rule = MetaLearningRule(
                rule_id=f"pattern_{pattern}_{session.session_id}",
                condition={
                    "type": "pattern_success",
                    "pattern": pattern,
                    "context": session.context.value
                },
                recommendation={
                    "strategy": session.strategy_used,
                    "pattern_boost": 0.1
                },
                confidence=0.8,
                usage_count=1,
                success_rate=1.0,
                created_at=time.time(),
                last_updated=time.time()
            )
            rules.append(rule)
        
        return rules
    
    def _estimate_concept_complexity(self, session: LearningSession) -> float:
        """Estima complexidade do conceito."""
        # Complexidade baseada no número de exemplos e tempo de aprendizado
        complexity = 0.5
        
        if session.examples_count > 5:
            complexity += 0.2
        if session.learning_time > 2.0:
            complexity += 0.2
        if session.concept_type == "behavior":
            complexity += 0.1
        
        return min(complexity, 1.0)
    
    def _estimate_context_similarity(self, session: LearningSession) -> float:
        """Estima similaridade do contexto."""
        # Similaridade baseada no contexto
        if session.context == LearningContext.SIMILAR_CONCEPT:
            return 0.8
        elif session.context == LearningContext.NEW_CONCEPT:
            return 0.2
        else:
            return 0.5
    
    def _generate_insights(self, session: LearningSession, performance_analysis: Dict[str, Any]) -> List[str]:
        """Gera insights de meta-aprendizado."""
        insights = []
        
        overall_score = performance_analysis.get("overall_score", 0.0)
        
        if overall_score >= 0.8:
            insights.append(f"Estratégia '{session.strategy_used}' muito eficaz para {session.concept_type}")
        elif overall_score <= 0.4:
            insights.append(f"Estratégia '{session.strategy_used}' precisa de ajustes para {session.concept_type}")
        
        strengths = performance_analysis.get("strengths", [])
        if strengths:
            insights.append(f"Pontos fortes: {', '.join(strengths[:2])}")
        
        weaknesses = performance_analysis.get("weaknesses", [])
        if weaknesses:
            insights.append(f"Pontos fracos: {', '.join(weaknesses[:2])}")
        
        return insights
    
    def _generate_meta_reasoning(self, strategy_recommendation: Dict[str, Any], 
                               hyperparameter_optimization: Dict[str, Any],
                               context: LearningContext, concept_type: str) -> List[str]:
        """Gera raciocínio de meta-aprendizado."""
        reasoning = []
        
        # Raciocínio da estratégia
        strategy_reasoning = strategy_recommendation.get("reasoning", [])
        reasoning.extend(strategy_reasoning)
        
        # Raciocínio dos hiperparâmetros
        optimization_method = hyperparameter_optimization.get("optimization_method", "default")
        if optimization_method != "default":
            reasoning.append(f"Hiperparâmetros otimizados usando {optimization_method}")
        
        # Raciocínio contextual
        reasoning.append(f"Contexto '{context.value}' requer abordagem específica")
        reasoning.append(f"Tipo de conceito '{concept_type}' influencia estratégia")
        
        return reasoning
    
    def _calculate_meta_learning_effectiveness(self) -> float:
        """Calcula efetividade do meta-aprendizado."""
        if len(self.learning_sessions) < 3:
            return 0.5
        
        # Calcular melhoria ao longo do tempo
        early_sessions = self.learning_sessions[:len(self.learning_sessions)//2]
        recent_sessions = self.learning_sessions[len(self.learning_sessions)//2:]
        
        early_performance = np.mean([
            self.performance_analyzer.analyze_session_performance(session).get("overall_score", 0.0)
            for session in early_sessions
        ])
        
        recent_performance = np.mean([
            self.performance_analyzer.analyze_session_performance(session).get("overall_score", 0.0)
            for session in recent_sessions
        ])
        
        improvement = recent_performance - early_performance
        effectiveness = 0.5 + improvement  # Base 0.5 + melhoria
        
        return max(0.0, min(effectiveness, 1.0))
    
    def _generate_system_recommendations(self) -> List[str]:
        """Gera recomendações para o sistema."""
        recommendations = []
        
        # Análise de tendências
        if len(self.learning_sessions) >= 5:
            recent_performance = [
                self.performance_analyzer.analyze_session_performance(session).get("overall_score", 0.0)
                for session in self.learning_sessions[-5:]
            ]
            
            if np.mean(recent_performance) < 0.6:
                recommendations.append("Considerar revisão das estratégias de aprendizado")
                recommendations.append("Aumentar diversidade de exemplos de treinamento")
            
            if np.std(recent_performance) > 0.2:
                recommendations.append("Melhorar consistência das estratégias")
        
        # Análise de regras
        if len(self.meta_rules) > 10:
            recommendations.append("Sistema de regras bem desenvolvido")
        else:
            recommendations.append("Continuar coleta de dados para regras")
        
        return recommendations
    
    def _load_data(self):
        """Carrega dados salvos."""
        try:
            data_file = "data/meta_learning.json"
            if os.path.exists(data_file):
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Carregar sessões de aprendizado
                if "learning_sessions" in data:
                    self.learning_sessions = [
                        LearningSession(**session) for session in data["learning_sessions"]
                    ]
                
                # Carregar regras de meta-aprendizado
                if "meta_rules" in data:
                    self.meta_rules = [
                        MetaLearningRule(**rule) for rule in data["meta_rules"]
                    ]
                
                # Carregar histórico de meta-aprendizado
                if "meta_learning_history" in data:
                    self.meta_learning_history = [
                        MetaLearningResult(**result) for result in data["meta_learning_history"]
                    ]
                
                logger.info(f"Dados de meta-aprendizado carregados: {len(self.learning_sessions)} sessões, {len(self.meta_rules)} regras")
                
        except Exception as e:
            logger.warning(f"Erro ao carregar dados de meta-aprendizado: {e}")
    
    def _save_data(self):
        """Salva dados."""
        try:
            data_file = "data/meta_learning.json"
            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            
            data = {
                "learning_sessions": [{
                    "session_id": session.session_id,
                    "concept_name": session.concept_name,
                    "concept_type": session.concept_type,
                    "strategy_used": session.strategy_used,
                    "context": session.context.value,
                    "examples_count": session.examples_count,
                    "performance_metrics": session.performance_metrics,
                    "success": session.success,
                    "learning_time": session.learning_time,
                    "timestamp": session.timestamp,
                    "metadata": session.metadata
                } for session in self.learning_sessions],
                "meta_rules": [asdict(rule) for rule in self.meta_rules],
                "meta_learning_history": [asdict(result) for result in self.meta_learning_history],
                "last_saved": time.time()
            }
            
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dados de meta-aprendizado salvos: {len(self.learning_sessions)} sessões, {len(self.meta_rules)} regras")
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados de meta-aprendizado: {e}")
