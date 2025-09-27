#!/usr/bin/env python3
"""
Sistema de Raciocínio Causal
============================

Este módulo implementa um sistema avançado de raciocínio causal que permite:
- Identificação de relações causais entre eventos
- Inferência causal baseada em evidências
- Raciocínio sobre causa e efeito
- Sistema de confiança causal
- Detecção de correlação vs causalidade
- Raciocínio contrafactual
- Sistema de evidências causais
- Integração com outros sistemas cognitivos
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
import re
import hashlib

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CausalRelationType(Enum):
    """Tipos de relações causais."""
    NECESSARY = "necessary"                 # Causa necessária
    SUFFICIENT = "sufficient"              # Causa suficiente
    CONTRIBUTORY = "contributory"          # Causa contributiva
    PREVENTIVE = "preventive"              # Causa preventiva
    TEMPORAL = "temporal"                  # Relação temporal
    CORRELATIONAL = "correlational"        # Correlação (não causal)

class CausalStrength(Enum):
    """Força da relação causal."""
    WEAK = "weak"                         # Fraca (0.0 - 0.3)
    MODERATE = "moderate"                 # Moderada (0.3 - 0.6)
    STRONG = "strong"                     # Forte (0.6 - 0.8)
    VERY_STRONG = "very_strong"           # Muito forte (0.8 - 1.0)

class EvidenceType(Enum):
    """Tipos de evidências causais."""
    TEMPORAL_ORDER = "temporal_order"      # Ordem temporal
    CORRELATION = "correlation"           # Correlação
    INTERVENTION = "intervention"         # Intervenção
    COUNTERFACTUAL = "counterfactual"     # Contrafactual
    MECHANISM = "mechanism"               # Mecanismo
    STATISTICAL = "statistical"           # Evidência estatística

@dataclass
class CausalRelation:
    """Representa uma relação causal."""
    relation_id: str
    cause: str
    effect: str
    relation_type: CausalRelationType
    strength: CausalStrength
    confidence: float
    evidence: List[Dict[str, Any]]
    temporal_order: bool
    mechanism: Optional[str]
    context: Dict[str, Any]
    created_at: float
    last_updated: float

@dataclass
class CausalInference:
    """Resultado de uma inferência causal."""
    inference_id: str
    query: Dict[str, Any]
    inferred_relations: List[str]
    confidence: float
    evidence_used: List[str]
    reasoning_steps: List[str]
    timestamp: float

class CausalReasoningSystem:
    """Sistema de raciocínio causal."""
    
    def __init__(self):
        self.causal_relations: Dict[str, CausalRelation] = {}
        self.inference_history: List[CausalInference] = []
        self.evidence_base: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.causal_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Carregar dados existentes
        self._load_data()
        
        # Inicializar relações causais básicas
        self._initialize_basic_causal_relations()
    
    def identify_causal_relation(self, cause: str, effect: str, 
                              evidence: List[Dict[str, Any]] = None,
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Identifica uma relação causal entre causa e efeito."""
        try:
            # Gerar ID único para a relação
            relation_id = self._generate_relation_id(cause, effect, evidence)
            
            # Analisar evidências para determinar tipo e força
            relation_analysis = self._analyze_causal_evidence(evidence or [])
            
            # Determinar tipo de relação causal
            relation_type = self._determine_causal_type(relation_analysis)
            
            # Calcular força da relação
            strength_score = self._calculate_causal_strength(relation_analysis)
            strength_level = self._determine_strength_level(strength_score)
            
            # Calcular confiança
            confidence = self._calculate_causal_confidence(relation_analysis, evidence)
            
            # Verificar ordem temporal
            temporal_order = self._check_temporal_order(evidence)
            
            # Identificar mecanismo se possível
            mechanism = self._identify_causal_mechanism(cause, effect, evidence)
            
            # Criar relação causal
            relation = CausalRelation(
                relation_id=relation_id,
                cause=cause,
                effect=effect,
                relation_type=relation_type,
                strength=strength_level,
                confidence=confidence,
                evidence=evidence or [],
                temporal_order=temporal_order,
                mechanism=mechanism,
                context=context or {},
                created_at=time.time(),
                last_updated=time.time()
            )
            
            # Armazenar relação
            self.causal_relations[relation_id] = relation
            
            # Atualizar grafo causal
            self._update_causal_graph(relation)
            
            # Armazenar evidências
            self._store_evidence(relation_id, evidence or [])
            
            # Salvar dados
            self._save_data()
            
            return {
                "success": True,
                "relation_id": relation_id,
                "cause": cause,
                "effect": effect,
                "relation_type": relation_type.value,
                "strength": strength_score,
                "strength_level": strength_level.value,
                "confidence": confidence,
                "temporal_order": temporal_order,
                "mechanism": mechanism,
                "evidence_count": len(evidence or []),
                "message": f"Relação causal '{cause} → {effect}' identificada com sucesso"
            }
            
        except Exception as e:
            logger.error(f"Erro na identificação de relação causal: {e}")
            return {"error": str(e)}
    
    def _generate_relation_id(self, cause: str, effect: str, 
                            evidence: List[Dict[str, Any]]) -> str:
        """Gera ID único para a relação causal."""
        try:
            # Combinar causa, efeito e evidências para gerar hash
            combined_data = {
                "cause": cause,
                "effect": effect,
                "evidence": evidence,
                "timestamp": time.time()
            }
            
            # Gerar hash MD5
            data_string = json.dumps(combined_data, sort_keys=True)
            hash_object = hashlib.md5(data_string.encode())
            relation_id = f"causal_{hash_object.hexdigest()[:12]}"
            
            return relation_id
            
        except Exception as e:
            logger.error(f"Erro na geração de ID de relação: {e}")
            return f"causal_{int(time.time())}_{random.randint(1000, 9999)}"
    
    def _analyze_causal_evidence(self, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analisa evidências para determinar propriedades causais."""
        try:
            analysis = {
                "temporal_order_score": 0.0,
                "correlation_score": 0.0,
                "intervention_score": 0.0,
                "mechanism_score": 0.0,
                "statistical_score": 0.0,
                "evidence_types": [],
                "total_evidence": len(evidence)
            }
            
            for ev in evidence:
                ev_type = ev.get("type", "unknown")
                ev_strength = ev.get("strength", 0.5)
                
                analysis["evidence_types"].append(ev_type)
                
                if ev_type == "temporal_order":
                    analysis["temporal_order_score"] += ev_strength
                elif ev_type == "correlation":
                    analysis["correlation_score"] += ev_strength
                elif ev_type == "intervention":
                    analysis["intervention_score"] += ev_strength
                elif ev_type == "mechanism":
                    analysis["mechanism_score"] += ev_strength
                elif ev_type == "statistical":
                    analysis["statistical_score"] += ev_strength
            
            # Normalizar scores
            for key in analysis:
                if key.endswith("_score") and analysis["total_evidence"] > 0:
                    analysis[key] = min(analysis[key] / analysis["total_evidence"], 1.0)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Erro na análise de evidências causais: {e}")
            return {"temporal_order_score": 0.0, "correlation_score": 0.0, 
                   "intervention_score": 0.0, "mechanism_score": 0.0, 
                   "statistical_score": 0.0, "evidence_types": [], "total_evidence": 0}
    
    def _determine_causal_type(self, analysis: Dict[str, Any]) -> CausalRelationType:
        """Determina o tipo de relação causal baseado na análise."""
        try:
            # Priorizar evidências de intervenção e mecanismo
            if analysis["intervention_score"] >= 0.7:
                return CausalRelationType.SUFFICIENT
            elif analysis["mechanism_score"] >= 0.6:
                return CausalRelationType.NECESSARY
            elif analysis["temporal_order_score"] >= 0.5:
                return CausalRelationType.CONTRIBUTORY
            elif analysis["correlation_score"] >= 0.8:
                return CausalRelationType.CORRELATIONAL
            else:
                return CausalRelationType.CONTRIBUTORY
                
        except Exception as e:
            logger.error(f"Erro na determinação do tipo causal: {e}")
            return CausalRelationType.CONTRIBUTORY
    
    def _calculate_causal_strength(self, analysis: Dict[str, Any]) -> float:
        """Calcula a força da relação causal."""
        try:
            # Pesos para diferentes tipos de evidência
            weights = {
                "intervention": 0.4,
                "mechanism": 0.3,
                "temporal_order": 0.2,
                "statistical": 0.1
            }
            
            # Calcular força ponderada
            strength = 0.0
            for ev_type, weight in weights.items():
                score_key = f"{ev_type}_score"
                if score_key in analysis:
                    strength += analysis[score_key] * weight
            
            # Penalizar se há apenas correlação
            if analysis["correlation_score"] > 0.7 and analysis["intervention_score"] < 0.3:
                strength *= 0.5
            
            return min(strength, 1.0)
            
        except Exception as e:
            logger.error(f"Erro no cálculo de força causal: {e}")
            return 0.5
    
    def _determine_strength_level(self, strength: float) -> CausalStrength:
        """Determina nível de força da relação causal."""
        if strength >= 0.8:
            return CausalStrength.VERY_STRONG
        elif strength >= 0.6:
            return CausalStrength.STRONG
        elif strength >= 0.3:
            return CausalStrength.MODERATE
        else:
            return CausalStrength.WEAK
    
    def _calculate_causal_confidence(self, analysis: Dict[str, Any], 
                                  evidence: List[Dict[str, Any]]) -> float:
        """Calcula confiança na relação causal."""
        try:
            # Baseado na qualidade e quantidade de evidências
            evidence_quality = 0.0
            evidence_count = len(evidence)
            
            # Calcular qualidade média das evidências
            if evidence_count > 0:
                for ev in evidence:
                    ev_strength = ev.get("strength", 0.5)
                    evidence_quality += ev_strength
                evidence_quality /= evidence_count
            
            # Fator de diversidade de evidências
            diversity_factor = len(set(analysis.get("evidence_types", []))) / 5.0
            
            # Fator de quantidade
            quantity_factor = min(evidence_count / 10.0, 1.0)
            
            # Calcular confiança
            confidence = (evidence_quality * 0.5 + 
                        diversity_factor * 0.3 + 
                        quantity_factor * 0.2)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Erro no cálculo de confiança causal: {e}")
            return 0.5
    
    def _check_temporal_order(self, evidence: List[Dict[str, Any]]) -> bool:
        """Verifica se há evidência de ordem temporal."""
        try:
            for ev in evidence:
                if ev.get("type") == "temporal_order":
                    return ev.get("temporal_order", False)
            return False
            
        except Exception as e:
            logger.error(f"Erro na verificação de ordem temporal: {e}")
            return False
    
    def _identify_causal_mechanism(self, cause: str, effect: str, 
                                 evidence: List[Dict[str, Any]]) -> Optional[str]:
        """Identifica o mecanismo causal se possível."""
        try:
            for ev in evidence:
                if ev.get("type") == "mechanism":
                    return ev.get("mechanism_description")
            
            # Tentar inferir mecanismo baseado em padrões conhecidos
            known_mechanisms = {
                ("predator", "flight"): "Detecção de predador causa resposta de fuga",
                ("hunger", "foraging"): "Fome causa comportamento de busca por comida",
                ("threat", "defense"): "Ameaça causa comportamento defensivo",
                ("mating_season", "courtship"): "Época de acasalamento causa comportamento de corte"
            }
            
            for (known_cause, known_effect), mechanism in known_mechanisms.items():
                if known_cause in cause.lower() and known_effect in effect.lower():
                    return mechanism
            
            return None
            
        except Exception as e:
            logger.error(f"Erro na identificação de mecanismo causal: {e}")
            return None
    
    def _update_causal_graph(self, relation: CausalRelation):
        """Atualiza o grafo causal."""
        try:
            self.causal_graph[relation.cause].add(relation.effect)
        except Exception as e:
            logger.error(f"Erro na atualização do grafo causal: {e}")
    
    def _store_evidence(self, relation_id: str, evidence: List[Dict[str, Any]]):
        """Armazena evidências para uma relação."""
        try:
            self.evidence_base[relation_id] = evidence
        except Exception as e:
            logger.error(f"Erro no armazenamento de evidências: {e}")
    
    def infer_causal_relations(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Faz inferências causais baseadas em uma consulta."""
        try:
            # Extrair informações da consulta
            target_event = query.get("event", "")
            context = query.get("context", {})
            evidence = query.get("evidence", [])
            
            # Encontrar relações causais relevantes
            relevant_relations = self._find_relevant_causal_relations(target_event, context)
            
            # Fazer inferências baseadas nas relações encontradas
            inferences = self._make_causal_inferences(target_event, relevant_relations, evidence)
            
            # Gerar passos de raciocínio
            reasoning_steps = self._generate_reasoning_steps(target_event, inferences)
            
            # Calcular confiança geral
            overall_confidence = self._calculate_inference_confidence(inferences)
            
            # Criar resultado de inferência
            inference_result = CausalInference(
                inference_id=f"inference_{int(time.time())}_{random.randint(1000, 9999)}",
                query=query,
                inferred_relations=[rel["relation_id"] for rel in inferences],
                confidence=overall_confidence,
                evidence_used=[ev["type"] for ev in evidence],
                reasoning_steps=reasoning_steps,
                timestamp=time.time()
            )
            
            self.inference_history.append(inference_result)
            
            # Salvar dados
            self._save_data()
            
            return {
                "success": True,
                "inference_id": inference_result.inference_id,
                "target_event": target_event,
                "inferred_relations": len(inferences),
                "relations": inferences,
                "reasoning_steps": reasoning_steps,
                "confidence": overall_confidence,
                "evidence_used": len(evidence)
            }
            
        except Exception as e:
            logger.error(f"Erro na inferência causal: {e}")
            return {"error": str(e)}
    
    def _find_relevant_causal_relations(self, target_event: str, 
                                      context: Dict[str, Any]) -> List[CausalRelation]:
        """Encontra relações causais relevantes para o evento."""
        relevant_relations = []
        
        try:
            for relation in self.causal_relations.values():
                # Verificar se a relação é relevante
                if (target_event.lower() in relation.cause.lower() or 
                    target_event.lower() in relation.effect.lower()):
                    relevant_relations.append(relation)
                
                # Verificar relevância contextual
                if context:
                    for key, value in context.items():
                        if (str(value).lower() in relation.cause.lower() or 
                            str(value).lower() in relation.effect.lower()):
                            relevant_relations.append(relation)
            
            return relevant_relations
            
        except Exception as e:
            logger.error(f"Erro na busca de relações relevantes: {e}")
            return []
    
    def _make_causal_inferences(self, target_event: str, 
                              relations: List[CausalRelation],
                              evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Faz inferências causais baseadas nas relações."""
        inferences = []
        
        try:
            for relation in relations:
                # Determinar tipo de inferência
                if target_event.lower() in relation.cause.lower():
                    # Inferir efeitos possíveis
                    inference_type = "effect_prediction"
                    confidence = relation.confidence * 0.8
                elif target_event.lower() in relation.effect.lower():
                    # Inferir causas possíveis
                    inference_type = "cause_inference"
                    confidence = relation.confidence * 0.7
                else:
                    # Inferência indireta
                    inference_type = "indirect_inference"
                    confidence = relation.confidence * 0.5
                
                # Ajustar confiança baseada em evidências
                evidence_match = self._check_evidence_match(relation, evidence)
                confidence *= evidence_match
                
                inference = {
                    "relation_id": relation.relation_id,
                    "cause": relation.cause,
                    "effect": relation.effect,
                    "relation_type": relation.relation_type.value,
                    "strength": relation.strength.value,
                    "inference_type": inference_type,
                    "confidence": confidence,
                    "mechanism": relation.mechanism
                }
                
                inferences.append(inference)
            
            # Ordenar por confiança
            inferences.sort(key=lambda x: x["confidence"], reverse=True)
            
            return inferences
            
        except Exception as e:
            logger.error(f"Erro na criação de inferências: {e}")
            return []
    
    def _check_evidence_match(self, relation: CausalRelation, 
                            evidence: List[Dict[str, Any]]) -> float:
        """Verifica correspondência entre evidências e relação."""
        try:
            if not evidence:
                return 1.0
            
            match_score = 0.0
            for ev in evidence:
                for rel_ev in relation.evidence:
                    if ev.get("type") == rel_ev.get("type"):
                        match_score += 0.3
                    if ev.get("strength", 0) > 0.5 and rel_ev.get("strength", 0) > 0.5:
                        match_score += 0.2
            
            return min(match_score, 1.0)
            
        except Exception as e:
            logger.error(f"Erro na verificação de correspondência de evidências: {e}")
            return 0.5
    
    def _generate_reasoning_steps(self, target_event: str, 
                                inferences: List[Dict[str, Any]]) -> List[str]:
        """Gera passos de raciocínio causal."""
        steps = []
        
        try:
            steps.append(f"Analisando evento: '{target_event}'")
            
            if inferences:
                steps.append(f"Encontradas {len(inferences)} relações causais relevantes")
                
                for i, inference in enumerate(inferences[:3]):  # Top 3
                    if inference["inference_type"] == "effect_prediction":
                        steps.append(f"Relação {i+1}: '{inference['cause']}' pode causar '{inference['effect']}' (confiança: {inference['confidence']:.2f})")
                    elif inference["inference_type"] == "cause_inference":
                        steps.append(f"Relação {i+1}: '{inference['cause']}' pode ser causa de '{inference['effect']}' (confiança: {inference['confidence']:.2f})")
                    
                    if inference["mechanism"]:
                        steps.append(f"  Mecanismo: {inference['mechanism']}")
            else:
                steps.append("Nenhuma relação causal relevante encontrada")
            
            return steps
            
        except Exception as e:
            logger.error(f"Erro na geração de passos de raciocínio: {e}")
            return [f"Erro no raciocínio causal: {e}"]
    
    def _calculate_inference_confidence(self, inferences: List[Dict[str, Any]]) -> float:
        """Calcula confiança geral da inferência."""
        try:
            if not inferences:
                return 0.0
            
            # Média ponderada das confianças
            total_confidence = sum(inf["confidence"] for inf in inferences)
            return total_confidence / len(inferences)
            
        except Exception as e:
            logger.error(f"Erro no cálculo de confiança de inferência: {e}")
            return 0.0
    
    def detect_correlation_vs_causation(self, event_a: str, event_b: str, 
                                       evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detecta se a relação é correlação ou causalidade."""
        try:
            # Analisar evidências
            analysis = self._analyze_causal_evidence(evidence)
            
            # Determinar se é causalidade
            is_causal = (analysis["intervention_score"] >= 0.5 or 
                        analysis["mechanism_score"] >= 0.6)
            
            # Determinar força da relação
            if is_causal:
                strength = self._calculate_causal_strength(analysis)
                relation_type = "causal"
            else:
                strength = analysis["correlation_score"]
                relation_type = "correlational"
            
            # Gerar explicação
            explanation = self._generate_correlation_explanation(
                event_a, event_b, is_causal, analysis
            )
            
            return {
                "success": True,
                "event_a": event_a,
                "event_b": event_b,
                "relation_type": relation_type,
                "is_causal": is_causal,
                "strength": strength,
                "confidence": analysis.get("total_evidence", 0) / 10.0,
                "explanation": explanation,
                "evidence_analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Erro na detecção de correlação vs causalidade: {e}")
            return {"error": str(e)}
    
    def _generate_correlation_explanation(self, event_a: str, event_b: str, 
                                        is_causal: bool, analysis: Dict[str, Any]) -> str:
        """Gera explicação para a relação."""
        try:
            if is_causal:
                if analysis["intervention_score"] >= 0.5:
                    return f"'{event_a}' causa '{event_b}' - evidência de intervenção encontrada"
                elif analysis["mechanism_score"] >= 0.6:
                    return f"'{event_a}' causa '{event_b}' - mecanismo causal identificado"
                else:
                    return f"'{event_a}' causa '{event_b}' - relação causal estabelecida"
            else:
                if analysis["correlation_score"] >= 0.7:
                    return f"'{event_a}' e '{event_b}' estão correlacionados mas não há evidência de causalidade"
                else:
                    return f"'{event_a}' e '{event_b}' podem estar relacionados mas evidências são insuficientes"
                    
        except Exception as e:
            logger.error(f"Erro na geração de explicação: {e}")
            return "Erro na análise da relação"
    
    def get_causal_analysis(self, relation_id: str = None) -> Dict[str, Any]:
        """Obtém análise de relações causais."""
        try:
            if relation_id:
                # Análise específica de uma relação
                if relation_id not in self.causal_relations:
                    return {"error": f"Relação causal '{relation_id}' não encontrada"}
                
                relation = self.causal_relations[relation_id]
                
                return {
                    "relation": asdict(relation),
                    "evidence": self.evidence_base.get(relation_id, []),
                    "age_days": (time.time() - relation.created_at) / (24 * 3600),
                    "graph_connections": len(self.causal_graph.get(relation.cause, set()))
                }
            else:
                # Análise geral
                return {
                    "total_relations": len(self.causal_relations),
                    "total_inferences": len(self.inference_history),
                    "relation_distribution": self._get_relation_distribution(),
                    "strength_distribution": self._get_strength_distribution(),
                    "causal_graph_stats": self._get_graph_statistics(),
                    "causal_statistics": self._get_causal_statistics()
                }
                
        except Exception as e:
            logger.error(f"Erro na análise causal: {e}")
            return {"error": str(e)}
    
    def _get_relation_distribution(self) -> Dict[str, int]:
        """Obtém distribuição de relações por tipo."""
        distribution = defaultdict(int)
        
        for relation in self.causal_relations.values():
            distribution[relation.relation_type.value] += 1
        
        return dict(distribution)
    
    def _get_strength_distribution(self) -> Dict[str, int]:
        """Obtém distribuição de relações por força."""
        distribution = defaultdict(int)
        
        for relation in self.causal_relations.values():
            distribution[relation.strength.value] += 1
        
        return dict(distribution)
    
    def _get_graph_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas do grafo causal."""
        try:
            total_nodes = len(self.causal_graph)
            total_edges = sum(len(connections) for connections in self.causal_graph.values())
            
            # Calcular conectividade média
            avg_connectivity = total_edges / total_nodes if total_nodes > 0 else 0
            
            return {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "average_connectivity": avg_connectivity,
                "most_connected_nodes": self._get_most_connected_nodes()
            }
            
        except Exception as e:
            logger.error(f"Erro nas estatísticas do grafo: {e}")
            return {"error": str(e)}
    
    def _get_most_connected_nodes(self) -> List[Dict[str, Any]]:
        """Obtém nós mais conectados do grafo."""
        try:
            node_connections = [
                {"node": node, "connections": len(connections)}
                for node, connections in self.causal_graph.items()
            ]
            
            node_connections.sort(key=lambda x: x["connections"], reverse=True)
            return node_connections[:5]  # Top 5
            
        except Exception as e:
            logger.error(f"Erro na obtenção de nós mais conectados: {e}")
            return []
    
    def _get_causal_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas das relações causais."""
        try:
            if not self.causal_relations:
                return {"total_relations": 0, "average_confidence": 0, "average_strength": 0}
            
            confidences = [rel.confidence for rel in self.causal_relations.values()]
            strengths = [self._get_strength_factor(rel.strength) for rel in self.causal_relations.values()]
            
            return {
                "total_relations": len(self.causal_relations),
                "average_confidence": sum(confidences) / len(confidences),
                "average_strength": sum(strengths) / len(strengths),
                "max_confidence": max(confidences),
                "min_confidence": min(confidences),
                "high_confidence_count": len([c for c in confidences if c >= 0.7])
            }
            
        except Exception as e:
            logger.error(f"Erro nas estatísticas causais: {e}")
            return {"error": str(e)}
    
    def _get_strength_factor(self, strength: CausalStrength) -> float:
        """Obtém fator numérico da força."""
        strength_factors = {
            CausalStrength.WEAK: 0.2,
            CausalStrength.MODERATE: 0.5,
            CausalStrength.STRONG: 0.8,
            CausalStrength.VERY_STRONG: 1.0
        }
        return strength_factors.get(strength, 0.5)
    
    def _initialize_basic_causal_relations(self):
        """Inicializa relações causais básicas."""
        # Relações causais básicas para pássaros
        basic_relations = [
            {
                "cause": "predator_detection",
                "effect": "flight_response",
                "evidence": [
                    {"type": "temporal_order", "strength": 0.9, "temporal_order": True},
                    {"type": "mechanism", "strength": 0.8, "mechanism_description": "Detecção de ameaça ativa resposta de fuga"}
                ],
                "context": {"domain": "animal_behavior"}
            },
            {
                "cause": "hunger",
                "effect": "foraging_behavior",
                "evidence": [
                    {"type": "mechanism", "strength": 0.7, "mechanism_description": "Fome ativa comportamento de busca por comida"},
                    {"type": "temporal_order", "strength": 0.6, "temporal_order": True}
                ],
                "context": {"domain": "animal_behavior"}
            },
            {
                "cause": "mating_season",
                "effect": "courtship_display",
                "evidence": [
                    {"type": "temporal_order", "strength": 0.8, "temporal_order": True},
                    {"type": "correlation", "strength": 0.9}
                ],
                "context": {"domain": "animal_behavior"}
            }
        ]
        
        for relation_data in basic_relations:
            try:
                self.identify_causal_relation(**relation_data)
            except Exception as e:
                logger.error(f"Erro na inicialização de relação causal básica: {e}")
    
    def _load_data(self):
        """Carrega dados salvos."""
        try:
            data_file = "data/causal_reasoning.json"
            if os.path.exists(data_file):
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Carregar relações causais
                if "causal_relations" in data:
                    for relation_id, relation_data in data["causal_relations"].items():
                        relation_data["relation_type"] = CausalRelationType(relation_data["relation_type"])
                        relation_data["strength"] = CausalStrength(relation_data["strength"])
                        self.causal_relations[relation_id] = CausalRelation(**relation_data)
                
                # Carregar histórico de inferências
                if "inference_history" in data:
                    for inference_data in data["inference_history"]:
                        self.inference_history.append(CausalInference(**inference_data))
                
                # Carregar outros dados
                if "evidence_base" in data:
                    self.evidence_base = defaultdict(list, data["evidence_base"])
                
                if "causal_graph" in data:
                    self.causal_graph = defaultdict(set, data["causal_graph"])
                
                logger.info(f"Dados de raciocínio causal carregados: {len(self.causal_relations)} relações")
                
        except Exception as e:
            logger.warning(f"Erro ao carregar dados de raciocínio causal: {e}")
    
    def _save_data(self):
        """Salva dados."""
        try:
            data_file = "data/causal_reasoning.json"
            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            
            data = {
                "causal_relations": {
                    relation_id: {
                        "relation_id": relation.relation_id,
                        "cause": relation.cause,
                        "effect": relation.effect,
                        "relation_type": relation.relation_type.value,
                        "strength": relation.strength.value,
                        "confidence": relation.confidence,
                        "evidence": relation.evidence,
                        "temporal_order": relation.temporal_order,
                        "mechanism": relation.mechanism,
                        "context": relation.context,
                        "created_at": relation.created_at,
                        "last_updated": relation.last_updated
                    }
                    for relation_id, relation in self.causal_relations.items()
                },
                "inference_history": [asdict(inference) for inference in self.inference_history],
                "evidence_base": dict(self.evidence_base),
                "causal_graph": {k: list(v) for k, v in self.causal_graph.items()},
                "last_saved": time.time()
            }
            
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dados de raciocínio causal salvos: {len(self.causal_relations)} relações")
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados de raciocínio causal: {e}")
