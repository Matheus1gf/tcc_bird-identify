#!/usr/bin/env python3
"""
Sistema de Meta-Cognição Avançada
Permite ao sistema refletir sobre seu próprio funcionamento e adaptar estratégias
"""

import logging
import json
import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
import random
import time
import hashlib

# Configurar logging
logger = logging.getLogger(__name__)

class MetaCognitiveProcess(Enum):
    """Processos meta-cognitivos."""
    LEARNING_TO_LEARN = "learning_to_learn"  # Aprender a aprender
    SELF_REFLECTION = "self_reflection"  # Auto-reflexão
    STRATEGY_ADAPTATION = "strategy_adaptation"  # Adaptação de estratégias
    PERFORMANCE_MONITORING = "performance_monitoring"  # Monitoramento de performance
    METACOGNITIVE_CONTROL = "metacognitive_control"  # Controle meta-cognitivo
    KNOWLEDGE_MONITORING = "knowledge_monitoring"  # Monitoramento de conhecimento

class ReflectionType(Enum):
    """Tipos de reflexão."""
    PERFORMANCE_REFLECTION = "performance_reflection"  # Reflexão sobre performance
    STRATEGY_REFLECTION = "strategy_reflection"  # Reflexão sobre estratégias
    LEARNING_REFLECTION = "learning_reflection"  # Reflexão sobre aprendizado
    ERROR_REFLECTION = "error_reflection"  # Reflexão sobre erros
    SUCCESS_REFLECTION = "success_reflection"  # Reflexão sobre sucessos
    METACOGNITIVE_REFLECTION = "metacognitive_reflection"  # Reflexão meta-cognitiva

class AdaptationStrategy(Enum):
    """Estratégias de adaptação."""
    PARAMETER_ADJUSTMENT = "parameter_adjustment"  # Ajuste de parâmetros
    STRATEGY_SWITCHING = "strategy_switching"  # Mudança de estratégia
    LEARNING_RATE_ADAPTATION = "learning_rate_adaptation"  # Adaptação de taxa de aprendizado
    ARCHITECTURE_MODIFICATION = "architecture_modification"  # Modificação de arquitetura
    KNOWLEDGE_RESTRUCTURING = "knowledge_restructuring"  # Reestruturação de conhecimento
    METACOGNITIVE_ADAPTATION = "metacognitive_adaptation"  # Adaptação meta-cognitiva

class MetaCognitiveLevel(Enum):
    """Níveis meta-cognitivos."""
    BASIC = "basic"  # Básico
    INTERMEDIATE = "intermediate"  # Intermediário
    ADVANCED = "advanced"  # Avançado
    EXPERT = "expert"  # Especialista
    MASTER = "master"  # Mestre

@dataclass
class SelfReflection:
    """Reflexão sobre o próprio funcionamento."""
    reflection_id: str
    reflection_type: ReflectionType
    target_process: str
    reflection_content: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    confidence: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyAdaptation:
    """Adaptação de estratégias."""
    adaptation_id: str
    adaptation_type: AdaptationStrategy
    original_strategy: str
    adapted_strategy: str
    adaptation_reason: str
    expected_improvement: float
    confidence: float
    success: bool = False
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LearningToLearnInsight:
    """Insight sobre como aprender melhor."""
    insight_id: str
    learning_context: str
    insight_type: str
    insight_content: str
    applicability: List[str]
    confidence: float
    validation_count: int = 0
    success_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class MetaCognitiveSystem:
    """Sistema de meta-cognição avançada."""
    
    def __init__(self):
        self.self_reflections: Dict[str, SelfReflection] = {}
        self.strategy_adaptations: Dict[str, StrategyAdaptation] = {}
        self.learning_insights: Dict[str, LearningToLearnInsight] = {}
        self.metacognitive_level: MetaCognitiveLevel = MetaCognitiveLevel.BASIC
        self.performance_history: List[Dict[str, Any]] = []
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        self.reflection_patterns: Dict[str, List[str]] = defaultdict(list)
        
        # Carregar dados existentes
        self._load_data()
        
        logger.info("Sistema de meta-cognição avançada inicializado")
    
    def perform_self_reflection(self, process_name: str, process_data: Dict[str, Any],
                               reflection_type: ReflectionType = ReflectionType.PERFORMANCE_REFLECTION) -> Dict[str, Any]:
        """
        Realiza auto-reflexão sobre um processo.
        
        Args:
            process_name: Nome do processo
            process_data: Dados do processo
            reflection_type: Tipo de reflexão
            
        Returns:
            Resultado da reflexão
        """
        try:
            # Gerar ID único para a reflexão
            reflection_id = self._generate_reflection_id(process_name, reflection_type)
            
            # Realizar reflexão baseada no tipo
            if reflection_type == ReflectionType.PERFORMANCE_REFLECTION:
                insights, recommendations = self._reflect_on_performance(process_data)
            elif reflection_type == ReflectionType.STRATEGY_REFLECTION:
                insights, recommendations = self._reflect_on_strategy(process_data)
            elif reflection_type == ReflectionType.LEARNING_REFLECTION:
                insights, recommendations = self._reflect_on_learning(process_data)
            elif reflection_type == ReflectionType.ERROR_REFLECTION:
                insights, recommendations = self._reflect_on_errors(process_data)
            elif reflection_type == ReflectionType.SUCCESS_REFLECTION:
                insights, recommendations = self._reflect_on_success(process_data)
            elif reflection_type == ReflectionType.METACOGNITIVE_REFLECTION:
                insights, recommendations = self._reflect_on_metacognition(process_data)
            else:
                insights, recommendations = [], []
            
            # Calcular confiança da reflexão
            confidence = self._calculate_reflection_confidence(insights, recommendations, process_data)
            
            # Criar reflexão
            reflection = SelfReflection(
                reflection_id=reflection_id,
                reflection_type=reflection_type,
                target_process=process_name,
                reflection_content=process_data,
                insights=insights,
                recommendations=recommendations,
                confidence=confidence
            )
            
            # Armazenar reflexão
            self.self_reflections[reflection_id] = reflection
            
            # Atualizar padrões de reflexão
            self._update_reflection_patterns(reflection)
            
            # Salvar dados
            self._save_data()
            
            return {
                "success": True,
                "reflection_id": reflection_id,
                "insights": insights,
                "recommendations": recommendations,
                "confidence": confidence,
                "reflection_type": reflection_type.value
            }
            
        except Exception as e:
            logger.error(f"Erro na auto-reflexão: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def adapt_strategy(self, original_strategy: str, performance_data: Dict[str, Any],
                      adaptation_type: AdaptationStrategy = AdaptationStrategy.PARAMETER_ADJUSTMENT) -> Dict[str, Any]:
        """
        Adapta uma estratégia baseada na performance.
        
        Args:
            original_strategy: Estratégia original
            performance_data: Dados de performance
            adaptation_type: Tipo de adaptação
            
        Returns:
            Resultado da adaptação
        """
        try:
            # Gerar ID único para a adaptação
            adaptation_id = self._generate_adaptation_id(original_strategy, adaptation_type)
            
            # Determinar estratégia adaptada
            adapted_strategy, adaptation_reason = self._determine_adapted_strategy(
                original_strategy, performance_data, adaptation_type
            )
            
            # Calcular melhoria esperada
            expected_improvement = self._calculate_expected_improvement(
                original_strategy, adapted_strategy, performance_data
            )
            
            # Calcular confiança da adaptação
            confidence = self._calculate_adaptation_confidence(
                original_strategy, adapted_strategy, performance_data
            )
            
            # Criar adaptação
            adaptation = StrategyAdaptation(
                adaptation_id=adaptation_id,
                adaptation_type=adaptation_type,
                original_strategy=original_strategy,
                adapted_strategy=adapted_strategy,
                adaptation_reason=adaptation_reason,
                expected_improvement=expected_improvement,
                confidence=confidence
            )
            
            # Armazenar adaptação
            self.strategy_adaptations[adaptation_id] = adaptation
            
            # Atualizar histórico de performance da estratégia
            self._update_strategy_performance(original_strategy, performance_data)
            
            # Salvar dados
            self._save_data()
            
            return {
                "success": True,
                "adaptation_id": adaptation_id,
                "original_strategy": original_strategy,
                "adapted_strategy": adapted_strategy,
                "adaptation_reason": adaptation_reason,
                "expected_improvement": expected_improvement,
                "confidence": confidence,
                "adaptation_type": adaptation_type.value
            }
            
        except Exception as e:
            logger.error(f"Erro na adaptação de estratégia: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def learn_to_learn(self, learning_context: str, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aprende como aprender melhor.
        
        Args:
            learning_context: Contexto de aprendizado
            learning_data: Dados de aprendizado
            
        Returns:
            Resultado do aprendizado
        """
        try:
            # Analisar dados de aprendizado
            learning_analysis = self._analyze_learning_data(learning_data)
            
            # Gerar insights sobre como aprender melhor
            insights = self._generate_learning_insights(learning_context, learning_analysis)
            
            # Criar insights de aprendizado
            learning_insights = []
            for insight_content, insight_type, applicability in insights:
                insight_id = self._generate_insight_id(learning_context, insight_type)
                
                insight = LearningToLearnInsight(
                    insight_id=insight_id,
                    learning_context=learning_context,
                    insight_type=insight_type,
                    insight_content=insight_content,
                    applicability=applicability,
                    confidence=self._calculate_insight_confidence(insight_content, learning_analysis)
                )
                
                learning_insights.append(insight)
                self.learning_insights[insight_id] = insight
            
            # Atualizar nível meta-cognitivo
            self._update_metacognitive_level()
            
            # Salvar dados
            self._save_data()
            
            return {
                "success": True,
                "insights_count": len(learning_insights),
                "insights": [
                    {
                        "insight_id": insight.insight_id,
                        "insight_type": insight.insight_type,
                        "insight_content": insight.insight_content,
                        "applicability": insight.applicability,
                        "confidence": insight.confidence
                    } for insight in learning_insights
                ],
                "metacognitive_level": self.metacognitive_level.value
            }
            
        except Exception as e:
            logger.error(f"Erro no aprendizado de como aprender: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_metacognitive_analysis(self) -> Dict[str, Any]:
        """
        Obtém análise meta-cognitiva do sistema.
        
        Returns:
            Análise meta-cognitiva
        """
        try:
            # Estatísticas gerais
            total_reflections = len(self.self_reflections)
            total_adaptations = len(self.strategy_adaptations)
            total_insights = len(self.learning_insights)
            
            # Análise de reflexões
            reflection_types = defaultdict(int)
            reflection_confidence = []
            for reflection in self.self_reflections.values():
                reflection_types[reflection.reflection_type.value] += 1
                reflection_confidence.append(reflection.confidence)
            
            avg_reflection_confidence = np.mean(reflection_confidence) if reflection_confidence else 0.0
            
            # Análise de adaptações
            adaptation_types = defaultdict(int)
            adaptation_success = []
            for adaptation in self.strategy_adaptations.values():
                adaptation_types[adaptation.adaptation_type.value] += 1
                adaptation_success.append(adaptation.success)
            
            success_rate = np.mean(adaptation_success) if adaptation_success else 0.0
            
            # Análise de insights
            insight_types = defaultdict(int)
            insight_confidence = []
            for insight in self.learning_insights.values():
                insight_types[insight.insight_type] += 1
                insight_confidence.append(insight.confidence)
            
            avg_insight_confidence = np.mean(insight_confidence) if insight_confidence else 0.0
            
            # Padrões de reflexão
            reflection_patterns = dict(self.reflection_patterns)
            
            # Estratégias mais adaptadas
            strategy_adaptations = defaultdict(int)
            for adaptation in self.strategy_adaptations.values():
                strategy_adaptations[adaptation.original_strategy] += 1
            
            most_adapted_strategies = sorted(strategy_adaptations.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "success": True,
                "analysis": {
                    "metacognitive_level": self.metacognitive_level.value,
                    "total_reflections": total_reflections,
                    "total_adaptations": total_adaptations,
                    "total_insights": total_insights,
                    "reflection_types": dict(reflection_types),
                    "avg_reflection_confidence": avg_reflection_confidence,
                    "adaptation_types": dict(adaptation_types),
                    "adaptation_success_rate": success_rate,
                    "insight_types": dict(insight_types),
                    "avg_insight_confidence": avg_insight_confidence,
                    "reflection_patterns": reflection_patterns,
                    "most_adapted_strategies": most_adapted_strategies
                }
            }
            
        except Exception as e:
            logger.error(f"Erro na análise meta-cognitiva: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_reflection_id(self, process_name: str, reflection_type: ReflectionType) -> str:
        """Gera ID único para uma reflexão."""
        content = f"{process_name}_{reflection_type.value}_{int(time.time())}"
        return f"reflection_{hashlib.md5(content.encode()).hexdigest()[:12]}"
    
    def _generate_adaptation_id(self, strategy: str, adaptation_type: AdaptationStrategy) -> str:
        """Gera ID único para uma adaptação."""
        content = f"{strategy}_{adaptation_type.value}_{int(time.time())}"
        return f"adaptation_{hashlib.md5(content.encode()).hexdigest()[:12]}"
    
    def _generate_insight_id(self, context: str, insight_type: str) -> str:
        """Gera ID único para um insight."""
        content = f"{context}_{insight_type}_{int(time.time())}"
        return f"insight_{hashlib.md5(content.encode()).hexdigest()[:12]}"
    
    def _reflect_on_performance(self, performance_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Reflete sobre performance."""
        insights = []
        recommendations = []
        
        # Analisar métricas de performance
        accuracy = performance_data.get("accuracy", 0.0)
        precision = performance_data.get("precision", 0.0)
        recall = performance_data.get("recall", 0.0)
        f1_score = performance_data.get("f1_score", 0.0)
        
        # Gerar insights baseados na performance
        if accuracy < 0.7:
            insights.append("Performance de precisão abaixo do ideal")
            recommendations.append("Considerar ajuste de thresholds de confiança")
        
        if precision < 0.6:
            insights.append("Muitos falsos positivos detectados")
            recommendations.append("Refinar critérios de detecção")
        
        if recall < 0.6:
            insights.append("Muitos falsos negativos detectados")
            recommendations.append("Ajustar sensibilidade do sistema")
        
        if f1_score < 0.65:
            insights.append("Equilíbrio entre precisão e recall precisa ser melhorado")
            recommendations.append("Otimizar parâmetros do modelo")
        
        # Insights positivos
        if accuracy > 0.8:
            insights.append("Performance de precisão excelente")
            recommendations.append("Manter estratégia atual")
        
        if f1_score > 0.8:
            insights.append("Equilíbrio entre precisão e recall muito bom")
            recommendations.append("Considerar aplicar estratégia a outros contextos")
        
        return insights, recommendations
    
    def _reflect_on_strategy(self, strategy_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Reflete sobre estratégias."""
        insights = []
        recommendations = []
        
        strategy_name = strategy_data.get("strategy", "unknown")
        success_rate = strategy_data.get("success_rate", 0.0)
        efficiency = strategy_data.get("efficiency", 0.0)
        
        # Gerar insights sobre estratégia
        if success_rate < 0.6:
            insights.append(f"Estratégia {strategy_name} tem taxa de sucesso baixa")
            recommendations.append("Considerar mudança de estratégia")
        
        if efficiency < 0.5:
            insights.append(f"Estratégia {strategy_name} é ineficiente")
            recommendations.append("Otimizar parâmetros da estratégia")
        
        if success_rate > 0.8:
            insights.append(f"Estratégia {strategy_name} é muito eficaz")
            recommendations.append("Aplicar estratégia em mais contextos")
        
        return insights, recommendations
    
    def _reflect_on_learning(self, learning_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Reflete sobre aprendizado."""
        insights = []
        recommendations = []
        
        learning_speed = learning_data.get("learning_speed", 0.0)
        retention_rate = learning_data.get("retention_rate", 0.0)
        generalization = learning_data.get("generalization", 0.0)
        
        # Gerar insights sobre aprendizado
        if learning_speed < 0.5:
            insights.append("Velocidade de aprendizado lenta")
            recommendations.append("Ajustar taxa de aprendizado")
        
        if retention_rate < 0.6:
            insights.append("Taxa de retenção baixa")
            recommendations.append("Implementar repetição espaçada")
        
        if generalization < 0.5:
            insights.append("Generalização limitada")
            recommendations.append("Aumentar diversidade dos exemplos")
        
        return insights, recommendations
    
    def _reflect_on_errors(self, error_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Reflete sobre erros."""
        insights = []
        recommendations = []
        
        error_types = error_data.get("error_types", {})
        error_frequency = error_data.get("error_frequency", 0.0)
        
        # Gerar insights sobre erros
        if error_frequency > 0.3:
            insights.append("Frequência de erros alta")
            recommendations.append("Investigar causas raiz dos erros")
        
        for error_type, count in error_types.items():
            if count > 5:
                insights.append(f"Erro {error_type} ocorre frequentemente")
                recommendations.append(f"Implementar correção específica para {error_type}")
        
        return insights, recommendations
    
    def _reflect_on_success(self, success_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Reflete sobre sucessos."""
        insights = []
        recommendations = []
        
        success_factors = success_data.get("success_factors", [])
        success_rate = success_data.get("success_rate", 0.0)
        
        # Gerar insights sobre sucessos
        if success_rate > 0.8:
            insights.append("Taxa de sucesso excelente")
            recommendations.append("Documentar fatores de sucesso")
        
        for factor in success_factors:
            insights.append(f"Fator de sucesso: {factor}")
            recommendations.append(f"Aplicar {factor} em outros contextos")
        
        return insights, recommendations
    
    def _reflect_on_metacognition(self, metacognitive_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Reflete sobre meta-cognição."""
        insights = []
        recommendations = []
        
        metacognitive_level = metacognitive_data.get("metacognitive_level", "basic")
        reflection_frequency = metacognitive_data.get("reflection_frequency", 0.0)
        
        # Gerar insights sobre meta-cognição
        if reflection_frequency < 0.5:
            insights.append("Frequência de reflexão baixa")
            recommendations.append("Aumentar frequência de auto-reflexão")
        
        if metacognitive_level == "basic":
            insights.append("Nível meta-cognitivo básico")
            recommendations.append("Desenvolver habilidades meta-cognitivas")
        
        return insights, recommendations
    
    def _calculate_reflection_confidence(self, insights: List[str], recommendations: List[str], 
                                        process_data: Dict[str, Any]) -> float:
        """Calcula confiança da reflexão."""
        # Fator baseado no número de insights
        insight_factor = min(len(insights) / 5.0, 1.0)
        
        # Fator baseado no número de recomendações
        recommendation_factor = min(len(recommendations) / 5.0, 1.0)
        
        # Fator baseado na qualidade dos dados
        data_quality = len(process_data) / 10.0
        data_factor = min(data_quality, 1.0)
        
        # Confiança geral
        confidence = (insight_factor * 0.4 + recommendation_factor * 0.3 + data_factor * 0.3)
        
        return min(confidence, 1.0)
    
    def _determine_adapted_strategy(self, original_strategy: str, performance_data: Dict[str, Any],
                                   adaptation_type: AdaptationStrategy) -> Tuple[str, str]:
        """Determina estratégia adaptada."""
        if adaptation_type == AdaptationStrategy.PARAMETER_ADJUSTMENT:
            adapted_strategy = f"{original_strategy}_adjusted"
            reason = "Ajuste de parâmetros baseado na performance"
        elif adaptation_type == AdaptationStrategy.STRATEGY_SWITCHING:
            adapted_strategy = self._get_alternative_strategy(original_strategy)
            reason = "Mudança de estratégia devido à performance baixa"
        elif adaptation_type == AdaptationStrategy.LEARNING_RATE_ADAPTATION:
            adapted_strategy = f"{original_strategy}_adaptive_lr"
            reason = "Adaptação da taxa de aprendizado"
        elif adaptation_type == AdaptationStrategy.ARCHITECTURE_MODIFICATION:
            adapted_strategy = f"{original_strategy}_modified_arch"
            reason = "Modificação da arquitetura"
        elif adaptation_type == AdaptationStrategy.KNOWLEDGE_RESTRUCTURING:
            adapted_strategy = f"{original_strategy}_restructured"
            reason = "Reestruturação do conhecimento"
        elif adaptation_type == AdaptationStrategy.METACOGNITIVE_ADAPTATION:
            adapted_strategy = f"{original_strategy}_metacognitive"
            reason = "Adaptação meta-cognitiva"
        else:
            adapted_strategy = original_strategy
            reason = "Nenhuma adaptação necessária"
        
        return adapted_strategy, reason
    
    def _get_alternative_strategy(self, original_strategy: str) -> str:
        """Obtém estratégia alternativa."""
        alternatives = {
            "prototype_based": "similarity_based",
            "similarity_based": "meta_learning",
            "meta_learning": "transfer_learning",
            "transfer_learning": "adaptive_fusion",
            "adaptive_fusion": "prototype_based"
        }
        return alternatives.get(original_strategy, "adaptive_fusion")
    
    def _calculate_expected_improvement(self, original_strategy: str, adapted_strategy: str,
                                     performance_data: Dict[str, Any]) -> float:
        """Calcula melhoria esperada."""
        # Melhoria baseada no tipo de adaptação
        improvement_factors = {
            AdaptationStrategy.PARAMETER_ADJUSTMENT: 0.1,
            AdaptationStrategy.STRATEGY_SWITCHING: 0.2,
            AdaptationStrategy.LEARNING_RATE_ADAPTATION: 0.15,
            AdaptationStrategy.ARCHITECTURE_MODIFICATION: 0.25,
            AdaptationStrategy.KNOWLEDGE_RESTRUCTURING: 0.3,
            AdaptationStrategy.METACOGNITIVE_ADAPTATION: 0.2
        }
        
        base_improvement = improvement_factors.get(AdaptationStrategy.PARAMETER_ADJUSTMENT, 0.1)
        
        # Ajustar baseado na performance atual
        current_performance = performance_data.get("accuracy", 0.5)
        if current_performance < 0.5:
            base_improvement *= 1.5  # Maior potencial de melhoria
        
        return min(base_improvement, 0.5)
    
    def _calculate_adaptation_confidence(self, original_strategy: str, adapted_strategy: str,
                                       performance_data: Dict[str, Any]) -> float:
        """Calcula confiança da adaptação."""
        # Confiança baseada na performance atual
        current_performance = performance_data.get("accuracy", 0.5)
        
        # Confiança baseada na mudança de estratégia
        if original_strategy != adapted_strategy:
            confidence = 0.7
        else:
            confidence = 0.5
        
        # Ajustar baseado na performance
        if current_performance < 0.5:
            confidence += 0.2  # Maior confiança em melhorar
        
        return min(confidence, 1.0)
    
    def _analyze_learning_data(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa dados de aprendizado."""
        analysis = {
            "learning_speed": learning_data.get("learning_speed", 0.0),
            "accuracy": learning_data.get("accuracy", 0.0),
            "efficiency": learning_data.get("efficiency", 0.0),
            "generalization": learning_data.get("generalization", 0.0),
            "retention": learning_data.get("retention", 0.0)
        }
        
        # Calcular métricas derivadas
        analysis["overall_performance"] = np.mean(list(analysis.values()))
        analysis["learning_quality"] = analysis["accuracy"] * analysis["generalization"]
        
        return analysis
    
    def _generate_learning_insights(self, learning_context: str, learning_analysis: Dict[str, Any]) -> List[Tuple[str, str, List[str]]]:
        """Gera insights sobre como aprender melhor."""
        insights = []
        
        # Insight sobre velocidade de aprendizado
        if learning_analysis["learning_speed"] < 0.5:
            insights.append((
                "Velocidade de aprendizado pode ser melhorada aumentando a taxa de aprendizado",
                "learning_speed",
                ["few_shot_learning", "meta_learning"]
            ))
        
        # Insight sobre precisão
        if learning_analysis["accuracy"] < 0.7:
            insights.append((
                "Precisão pode ser melhorada com mais exemplos de treinamento",
                "accuracy",
                ["prototype_based", "similarity_based"]
            ))
        
        # Insight sobre generalização
        if learning_analysis["generalization"] < 0.6:
            insights.append((
                "Generalização pode ser melhorada com exemplos mais diversos",
                "generalization",
                ["transfer_learning", "meta_learning"]
            ))
        
        # Insight sobre eficiência
        if learning_analysis["efficiency"] < 0.6:
            insights.append((
                "Eficiência pode ser melhorada otimizando hiperparâmetros",
                "efficiency",
                ["hyperparameter_tuning", "strategy_optimization"]
            ))
        
        return insights
    
    def _calculate_insight_confidence(self, insight_content: str, learning_analysis: Dict[str, Any]) -> float:
        """Calcula confiança de um insight."""
        # Confiança baseada na qualidade da análise
        analysis_quality = learning_analysis.get("overall_performance", 0.5)
        
        # Confiança baseada no conteúdo do insight
        content_confidence = 0.7  # Base
        
        # Ajustar baseado na análise
        if analysis_quality > 0.7:
            content_confidence += 0.2
        elif analysis_quality < 0.4:
            content_confidence -= 0.1
        
        return min(content_confidence, 1.0)
    
    def _update_metacognitive_level(self):
        """Atualiza nível meta-cognitivo."""
        # Calcular pontuação meta-cognitiva
        reflection_count = len(self.self_reflections)
        adaptation_count = len(self.strategy_adaptations)
        insight_count = len(self.learning_insights)
        
        metacognitive_score = (reflection_count * 0.4 + adaptation_count * 0.3 + insight_count * 0.3) / 100.0
        
        # Determinar nível
        if metacognitive_score >= 0.8:
            self.metacognitive_level = MetaCognitiveLevel.MASTER
        elif metacognitive_score >= 0.6:
            self.metacognitive_level = MetaCognitiveLevel.EXPERT
        elif metacognitive_score >= 0.4:
            self.metacognitive_level = MetaCognitiveLevel.ADVANCED
        elif metacognitive_score >= 0.2:
            self.metacognitive_level = MetaCognitiveLevel.INTERMEDIATE
        else:
            self.metacognitive_level = MetaCognitiveLevel.BASIC
    
    def _update_reflection_patterns(self, reflection: SelfReflection):
        """Atualiza padrões de reflexão."""
        pattern_key = f"{reflection.reflection_type.value}_{reflection.target_process}"
        self.reflection_patterns[pattern_key].append(reflection.reflection_id)
    
    def _update_strategy_performance(self, strategy: str, performance_data: Dict[str, Any]):
        """Atualiza performance da estratégia."""
        accuracy = performance_data.get("accuracy", 0.0)
        self.strategy_performance[strategy].append(accuracy)
    
    def _save_data(self):
        """Salva dados do sistema."""
        try:
            data = {
                "self_reflections": {},
                "strategy_adaptations": {},
                "learning_insights": {},
                "metacognitive_level": self.metacognitive_level.value,
                "performance_history": self.performance_history[-100:],  # Últimos 100 registros
                "strategy_performance": dict(self.strategy_performance),
                "reflection_patterns": dict(self.reflection_patterns)
            }
            
            # Converter reflexões para formato serializável
            for reflection_id, reflection in self.self_reflections.items():
                data["self_reflections"][reflection_id] = {
                    "reflection_id": reflection.reflection_id,
                    "reflection_type": reflection.reflection_type.value,
                    "target_process": reflection.target_process,
                    "reflection_content": reflection.reflection_content,
                    "insights": reflection.insights,
                    "recommendations": reflection.recommendations,
                    "confidence": reflection.confidence,
                    "timestamp": reflection.timestamp,
                    "metadata": reflection.metadata
                }
            
            # Converter adaptações para formato serializável
            for adaptation_id, adaptation in self.strategy_adaptations.items():
                data["strategy_adaptations"][adaptation_id] = {
                    "adaptation_id": adaptation.adaptation_id,
                    "adaptation_type": adaptation.adaptation_type.value,
                    "original_strategy": adaptation.original_strategy,
                    "adapted_strategy": adaptation.adapted_strategy,
                    "adaptation_reason": adaptation.adaptation_reason,
                    "expected_improvement": adaptation.expected_improvement,
                    "confidence": adaptation.confidence,
                    "success": adaptation.success,
                    "timestamp": adaptation.timestamp,
                    "metadata": adaptation.metadata
                }
            
            # Converter insights para formato serializável
            for insight_id, insight in self.learning_insights.items():
                data["learning_insights"][insight_id] = {
                    "insight_id": insight.insight_id,
                    "learning_context": insight.learning_context,
                    "insight_type": insight.insight_type,
                    "insight_content": insight.insight_content,
                    "applicability": insight.applicability,
                    "confidence": insight.confidence,
                    "validation_count": insight.validation_count,
                    "success_rate": insight.success_rate,
                    "timestamp": insight.timestamp,
                    "metadata": insight.metadata
                }
            
            # Salvar arquivo
            os.makedirs("data", exist_ok=True)
            with open("data/metacognitive_data.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dados meta-cognitivos salvos: {len(self.self_reflections)} reflexões, {len(self.strategy_adaptations)} adaptações, {len(self.learning_insights)} insights")
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados meta-cognitivos: {e}")
    
    def _load_data(self):
        """Carrega dados do sistema."""
        try:
            if not os.path.exists("data/metacognitive_data.json"):
                return
            
            with open("data/metacognitive_data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Carregar reflexões
            for reflection_id, reflection_data in data.get("self_reflections", {}).items():
                reflection = SelfReflection(
                    reflection_id=reflection_data["reflection_id"],
                    reflection_type=ReflectionType(reflection_data["reflection_type"]),
                    target_process=reflection_data["target_process"],
                    reflection_content=reflection_data["reflection_content"],
                    insights=reflection_data["insights"],
                    recommendations=reflection_data["recommendations"],
                    confidence=reflection_data["confidence"],
                    timestamp=reflection_data["timestamp"],
                    metadata=reflection_data.get("metadata", {})
                )
                self.self_reflections[reflection_id] = reflection
            
            # Carregar adaptações
            for adaptation_id, adaptation_data in data.get("strategy_adaptations", {}).items():
                adaptation = StrategyAdaptation(
                    adaptation_id=adaptation_data["adaptation_id"],
                    adaptation_type=AdaptationStrategy(adaptation_data["adaptation_type"]),
                    original_strategy=adaptation_data["original_strategy"],
                    adapted_strategy=adaptation_data["adapted_strategy"],
                    adaptation_reason=adaptation_data["adaptation_reason"],
                    expected_improvement=adaptation_data["expected_improvement"],
                    confidence=adaptation_data["confidence"],
                    success=adaptation_data["success"],
                    timestamp=adaptation_data["timestamp"],
                    metadata=adaptation_data.get("metadata", {})
                )
                self.strategy_adaptations[adaptation_id] = adaptation
            
            # Carregar insights
            for insight_id, insight_data in data.get("learning_insights", {}).items():
                insight = LearningToLearnInsight(
                    insight_id=insight_data["insight_id"],
                    learning_context=insight_data["learning_context"],
                    insight_type=insight_data["insight_type"],
                    insight_content=insight_data["insight_content"],
                    applicability=insight_data["applicability"],
                    confidence=insight_data["confidence"],
                    validation_count=insight_data["validation_count"],
                    success_rate=insight_data["success_rate"],
                    timestamp=insight_data["timestamp"],
                    metadata=insight_data.get("metadata", {})
                )
                self.learning_insights[insight_id] = insight
            
            # Carregar outros dados
            self.metacognitive_level = MetaCognitiveLevel(data.get("metacognitive_level", "basic"))
            self.performance_history = data.get("performance_history", [])
            self.strategy_performance = defaultdict(list, data.get("strategy_performance", {}))
            self.reflection_patterns = defaultdict(list, data.get("reflection_patterns", {}))
            
            logger.info(f"Dados meta-cognitivos carregados: {len(self.self_reflections)} reflexões, {len(self.strategy_adaptations)} adaptações, {len(self.learning_insights)} insights")
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados meta-cognitivos: {e}")
