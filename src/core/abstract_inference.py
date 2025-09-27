#!/usr/bin/env python3
"""
Sistema de Inferência Abstrata
==============================

Este módulo implementa um sistema avançado de inferência abstrata que permite:
- Inferência dedutiva abstrata
- Inferência indutiva abstrata
- Inferência abdutiva abstrata
- Raciocínio sobre conceitos abstratos
- Dedução lógica avançada
- Sistema de regras abstratas
- Validação de inferências
- Sistema de confiança abstrata
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

class InferenceType(Enum):
    """Tipos de inferência abstrata."""
    DEDUCTIVE = "deductive"                 # Inferência dedutiva
    INDUCTIVE = "inductive"                 # Inferência indutiva
    ABDUCTIVE = "abductive"                # Inferência abdutiva
    ANALOGICAL = "analogical"              # Inferência analógica
    CAUSAL = "causal"                      # Inferência causal
    PROBABILISTIC = "probabilistic"        # Inferência probabilística

class AbstractionLevel(Enum):
    """Níveis de abstração."""
    CONCRETE = "concrete"                  # Concreto
    SPECIFIC = "specific"                 # Específico
    GENERAL = "general"                    # Geral
    ABSTRACT = "abstract"                  # Abstrato
    UNIVERSAL = "universal"                # Universal

class ConfidenceLevel(Enum):
    """Níveis de confiança."""
    VERY_LOW = "very_low"                  # Muito baixa (0.0 - 0.2)
    LOW = "low"                           # Baixa (0.2 - 0.4)
    MODERATE = "moderate"                 # Moderada (0.4 - 0.6)
    HIGH = "high"                         # Alta (0.6 - 0.8)
    VERY_HIGH = "very_high"               # Muito alta (0.8 - 1.0)

@dataclass
class AbstractConcept:
    """Representa um conceito abstrato."""
    concept_id: str
    name: str
    description: str
    abstraction_level: AbstractionLevel
    properties: Dict[str, Any]
    relationships: List[str]
    examples: List[str]
    created_at: float
    last_updated: float

@dataclass
class InferenceRule:
    """Representa uma regra de inferência."""
    rule_id: str
    name: str
    premise_patterns: List[str]
    conclusion_pattern: str
    inference_type: InferenceType
    confidence: float
    abstraction_level: AbstractionLevel
    conditions: List[str]
    exceptions: List[str]
    created_at: float
    last_updated: float

@dataclass
class AbstractInference:
    """Resultado de uma inferência abstrata."""
    inference_id: str
    query: Dict[str, Any]
    inference_type: InferenceType
    premises: List[str]
    conclusion: str
    confidence: float
    confidence_level: ConfidenceLevel
    abstraction_level: AbstractionLevel
    reasoning_steps: List[str]
    supporting_evidence: List[str]
    timestamp: float

class AbstractInferenceEngine:
    """Sistema de inferência abstrata."""
    
    def __init__(self):
        self.abstract_concepts: Dict[str, AbstractConcept] = {}
        self.inference_rules: Dict[str, InferenceRule] = {}
        self.inference_history: List[AbstractInference] = []
        self.concept_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Carregar dados existentes
        self._load_data()
        
        # Inicializar conceitos e regras básicas
        self._initialize_basic_concepts()
        self._initialize_basic_rules()
    
    def add_abstract_concept(self, name: str, description: str, 
                           abstraction_level: AbstractionLevel,
                           properties: Dict[str, Any] = None,
                           examples: List[str] = None) -> Dict[str, Any]:
        """Adiciona um conceito abstrato."""
        try:
            # Gerar ID único
            concept_id = self._generate_concept_id(name, description)
            
            # Criar conceito abstrato
            concept = AbstractConcept(
                concept_id=concept_id,
                name=name,
                description=description,
                abstraction_level=abstraction_level,
                properties=properties or {},
                relationships=[],
                examples=examples or [],
                created_at=time.time(),
                last_updated=time.time()
            )
            
            # Armazenar conceito
            self.abstract_concepts[concept_id] = concept
            
            # Atualizar grafo de conceitos
            self._update_concept_graph(concept)
            
            # Salvar dados
            self._save_data()
            
            return {
                "success": True,
                "concept_id": concept_id,
                "name": name,
                "description": description,
                "abstraction_level": abstraction_level.value,
                "properties_count": len(properties or {}),
                "examples_count": len(examples or []),
                "message": f"Conceito abstrato '{name}' adicionado com sucesso"
            }
            
        except Exception as e:
            logger.error(f"Erro ao adicionar conceito abstrato: {e}")
            return {"error": str(e)}
    
    def _generate_concept_id(self, name: str, description: str) -> str:
        """Gera ID único para conceito."""
        try:
            combined_data = {
                "name": name,
                "description": description,
                "timestamp": time.time()
            }
            
            data_string = json.dumps(combined_data, sort_keys=True)
            hash_object = hashlib.md5(data_string.encode())
            concept_id = f"concept_{hash_object.hexdigest()[:12]}"
            
            return concept_id
            
        except Exception as e:
            logger.error(f"Erro na geração de ID de conceito: {e}")
            return f"concept_{int(time.time())}_{random.randint(1000, 9999)}"
    
    def _update_concept_graph(self, concept: AbstractConcept):
        """Atualiza grafo de conceitos."""
        try:
            # Adicionar relacionamentos baseados em propriedades
            for prop_name, prop_value in concept.properties.items():
                if isinstance(prop_value, str) and prop_value in self.abstract_concepts:
                    self.concept_graph[concept.concept_id].add(prop_value)
                    
        except Exception as e:
            logger.error(f"Erro na atualização do grafo de conceitos: {e}")
    
    def add_inference_rule(self, name: str, premise_patterns: List[str],
                          conclusion_pattern: str, inference_type: InferenceType,
                          abstraction_level: AbstractionLevel,
                          conditions: List[str] = None,
                          exceptions: List[str] = None) -> Dict[str, Any]:
        """Adiciona uma regra de inferência."""
        try:
            # Gerar ID único
            rule_id = self._generate_rule_id(name, premise_patterns, conclusion_pattern)
            
            # Calcular confiança inicial
            confidence = self._calculate_rule_confidence(premise_patterns, conclusion_pattern)
            
            # Criar regra de inferência
            rule = InferenceRule(
                rule_id=rule_id,
                name=name,
                premise_patterns=premise_patterns,
                conclusion_pattern=conclusion_pattern,
                inference_type=inference_type,
                confidence=confidence,
                abstraction_level=abstraction_level,
                conditions=conditions or [],
                exceptions=exceptions or [],
                created_at=time.time(),
                last_updated=time.time()
            )
            
            # Armazenar regra
            self.inference_rules[rule_id] = rule
            
            # Salvar dados
            self._save_data()
            
            return {
                "success": True,
                "rule_id": rule_id,
                "name": name,
                "premise_patterns": premise_patterns,
                "conclusion_pattern": conclusion_pattern,
                "inference_type": inference_type.value,
                "confidence": confidence,
                "abstraction_level": abstraction_level.value,
                "conditions_count": len(conditions or []),
                "exceptions_count": len(exceptions or []),
                "message": f"Regra de inferência '{name}' adicionada com sucesso"
            }
            
        except Exception as e:
            logger.error(f"Erro ao adicionar regra de inferência: {e}")
            return {"error": str(e)}
    
    def _generate_rule_id(self, name: str, premises: List[str], conclusion: str) -> str:
        """Gera ID único para regra."""
        try:
            combined_data = {
                "name": name,
                "premises": premises,
                "conclusion": conclusion,
                "timestamp": time.time()
            }
            
            data_string = json.dumps(combined_data, sort_keys=True)
            hash_object = hashlib.md5(data_string.encode())
            rule_id = f"rule_{hash_object.hexdigest()[:12]}"
            
            return rule_id
            
        except Exception as e:
            logger.error(f"Erro na geração de ID de regra: {e}")
            return f"rule_{int(time.time())}_{random.randint(1000, 9999)}"
    
    def _calculate_rule_confidence(self, premises: List[str], conclusion: str) -> float:
        """Calcula confiança inicial da regra."""
        try:
            # Baseado na complexidade e clareza da regra
            premise_complexity = len(premises) / 10.0
            conclusion_clarity = len(conclusion.split()) / 20.0
            
            # Confiança baseada em padrões conhecidos
            known_patterns = {
                "if": 0.8,
                "then": 0.7,
                "all": 0.9,
                "some": 0.6,
                "none": 0.8,
                "always": 0.9,
                "never": 0.9,
                "sometimes": 0.5
            }
            
            pattern_confidence = 0.0
            for pattern, conf in known_patterns.items():
                if pattern in conclusion.lower():
                    pattern_confidence = max(pattern_confidence, conf)
            
            # Calcular confiança final
            confidence = min(premise_complexity + conclusion_clarity + pattern_confidence, 1.0)
            
            return max(confidence, 0.3)  # Mínimo de 0.3
            
        except Exception as e:
            logger.error(f"Erro no cálculo de confiança da regra: {e}")
            return 0.5
    
    def perform_abstract_inference(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza inferência abstrata."""
        try:
            # Extrair informações da consulta
            premises = query.get("premises", [])
            target_concept = query.get("target_concept", "")
            inference_type = query.get("inference_type", InferenceType.DEDUCTIVE)
            abstraction_level = query.get("abstraction_level", AbstractionLevel.GENERAL)
            
            # Encontrar regras aplicáveis
            applicable_rules = self._find_applicable_rules(premises, inference_type, abstraction_level)
            
            # Realizar inferência
            inference_result = self._execute_inference(premises, applicable_rules, target_concept)
            
            # Gerar passos de raciocínio
            reasoning_steps = self._generate_reasoning_steps(premises, inference_result, applicable_rules)
            
            # Calcular confiança
            confidence = self._calculate_inference_confidence(inference_result, applicable_rules)
            confidence_level = self._determine_confidence_level(confidence)
            
            # Criar resultado de inferência
            abstract_inference = AbstractInference(
                inference_id=f"inference_{int(time.time())}_{random.randint(1000, 9999)}",
                query=query,
                inference_type=inference_type,
                premises=premises,
                conclusion=inference_result.get("conclusion", ""),
                confidence=confidence,
                confidence_level=confidence_level,
                abstraction_level=abstraction_level,
                reasoning_steps=reasoning_steps,
                supporting_evidence=inference_result.get("evidence", []),
                timestamp=time.time()
            )
            
            self.inference_history.append(abstract_inference)
            
            # Salvar dados
            self._save_data()
            
            return {
                "success": True,
                "inference_id": abstract_inference.inference_id,
                "premises": premises,
                "conclusion": inference_result.get("conclusion", ""),
                "inference_type": inference_type.value,
                "confidence": confidence,
                "confidence_level": confidence_level.value,
                "abstraction_level": abstraction_level.value,
                "reasoning_steps": reasoning_steps,
                "supporting_evidence": inference_result.get("evidence", []),
                "applicable_rules": len(applicable_rules)
            }
            
        except Exception as e:
            logger.error(f"Erro na inferência abstrata: {e}")
            return {"error": str(e)}
    
    def _find_applicable_rules(self, premises: List[str], inference_type: InferenceType,
                              abstraction_level: AbstractionLevel) -> List[InferenceRule]:
        """Encontra regras aplicáveis."""
        applicable_rules = []
        
        try:
            for rule in self.inference_rules.values():
                # Verificar tipo de inferência
                if rule.inference_type != inference_type:
                    continue
                
                # Verificar nível de abstração
                if rule.abstraction_level != abstraction_level:
                    continue
                
                # Verificar se as premissas correspondem aos padrões
                if self._premises_match_patterns(premises, rule.premise_patterns):
                    applicable_rules.append(rule)
            
            # Ordenar por confiança
            applicable_rules.sort(key=lambda r: r.confidence, reverse=True)
            
            return applicable_rules
            
        except Exception as e:
            logger.error(f"Erro na busca de regras aplicáveis: {e}")
            return []
    
    def _premises_match_patterns(self, premises: List[str], patterns: List[str]) -> bool:
        """Verifica se as premissas correspondem aos padrões."""
        try:
            if len(premises) != len(patterns):
                return False
            
            for premise, pattern in zip(premises, patterns):
                if not self._premise_matches_pattern(premise, pattern):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erro na verificação de correspondência de premissas: {e}")
            return False
    
    def _premise_matches_pattern(self, premise: str, pattern: str) -> bool:
        """Verifica se uma premissa corresponde a um padrão."""
        try:
            # Padrões simples baseados em palavras-chave
            pattern_keywords = pattern.lower().split()
            premise_lower = premise.lower()
            
            # Verificar se todas as palavras-chave estão na premissa
            for keyword in pattern_keywords:
                if keyword not in premise_lower:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erro na verificação de correspondência de premissa: {e}")
            return False
    
    def _execute_inference(self, premises: List[str], rules: List[InferenceRule],
                          target_concept: str) -> Dict[str, Any]:
        """Executa a inferência usando as regras."""
        try:
            if not rules:
                return {
                    "conclusion": "Nenhuma regra aplicável encontrada",
                    "evidence": [],
                    "confidence": 0.0
                }
            
            # Usar a regra com maior confiança
            best_rule = rules[0]
            
            # Gerar conclusão baseada na regra
            conclusion = self._generate_conclusion(premises, best_rule, target_concept)
            
            # Coletar evidências
            evidence = self._collect_evidence(premises, best_rule)
            
            return {
                "conclusion": conclusion,
                "evidence": evidence,
                "confidence": best_rule.confidence,
                "rule_used": best_rule.rule_id
            }
            
        except Exception as e:
            logger.error(f"Erro na execução da inferência: {e}")
            return {
                "conclusion": "Erro na execução da inferência",
                "evidence": [],
                "confidence": 0.0
            }
    
    def _generate_conclusion(self, premises: List[str], rule: InferenceRule,
                           target_concept: str) -> str:
        """Gera conclusão baseada na regra."""
        try:
            # Substituir variáveis na conclusão
            conclusion = rule.conclusion_pattern
            
            # Mapear premissas para variáveis
            premise_mapping = {}
            for i, premise in enumerate(premises):
                premise_mapping[f"premise_{i+1}"] = premise
            
            # Substituir variáveis
            for var, value in premise_mapping.items():
                conclusion = conclusion.replace(f"{{{var}}}", value)
            
            # Se há conceito alvo, incorporar na conclusão
            if target_concept:
                conclusion = f"Portanto, {target_concept}: {conclusion}"
            
            return conclusion
            
        except Exception as e:
            logger.error(f"Erro na geração de conclusão: {e}")
            return rule.conclusion_pattern
    
    def _collect_evidence(self, premises: List[str], rule: InferenceRule) -> List[str]:
        """Coleta evidências para a inferência."""
        try:
            evidence = []
            
            # Adicionar premissas como evidências
            for i, premise in enumerate(premises):
                evidence.append(f"Premissa {i+1}: {premise}")
            
            # Adicionar regra como evidência
            evidence.append(f"Regra aplicada: {rule.name}")
            
            # Adicionar conceitos relacionados
            for concept_id in self.concept_graph.get(rule.rule_id, []):
                if concept_id in self.abstract_concepts:
                    concept = self.abstract_concepts[concept_id]
                    evidence.append(f"Conceito relacionado: {concept.name}")
            
            return evidence
            
        except Exception as e:
            logger.error(f"Erro na coleta de evidências: {e}")
            return []
    
    def _generate_reasoning_steps(self, premises: List[str], inference_result: Dict[str, Any],
                                rules: List[InferenceRule]) -> List[str]:
        """Gera passos de raciocínio."""
        steps = []
        
        try:
            steps.append("Iniciando inferência abstrata...")
            
            # Adicionar premissas
            for i, premise in enumerate(premises):
                steps.append(f"Premissa {i+1}: {premise}")
            
            # Adicionar regras aplicáveis
            if rules:
                steps.append(f"Regras aplicáveis encontradas: {len(rules)}")
                for i, rule in enumerate(rules[:3]):  # Top 3
                    steps.append(f"Regra {i+1}: {rule.name} (confiança: {rule.confidence:.2f})")
            
            # Adicionar conclusão
            conclusion = inference_result.get("conclusion", "")
            if conclusion:
                steps.append(f"Conclusão: {conclusion}")
            
            # Adicionar evidências
            evidence = inference_result.get("evidence", [])
            if evidence:
                steps.append(f"Evidências: {len(evidence)} evidências encontradas")
            
            return steps
            
        except Exception as e:
            logger.error(f"Erro na geração de passos de raciocínio: {e}")
            return [f"Erro no raciocínio abstrato: {e}"]
    
    def _calculate_inference_confidence(self, inference_result: Dict[str, Any],
                                      rules: List[InferenceRule]) -> float:
        """Calcula confiança da inferência."""
        try:
            if not rules:
                return 0.0
            
            # Confiança baseada na regra usada
            rule_confidence = inference_result.get("confidence", 0.0)
            
            # Fator de número de regras aplicáveis
            rules_factor = min(len(rules) / 5.0, 1.0)
            
            # Fator de evidências
            evidence_count = len(inference_result.get("evidence", []))
            evidence_factor = min(evidence_count / 10.0, 1.0)
            
            # Calcular confiança final
            confidence = (rule_confidence * 0.6 + 
                         rules_factor * 0.2 + 
                         evidence_factor * 0.2)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Erro no cálculo de confiança da inferência: {e}")
            return 0.0
    
    def _determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Determina nível de confiança."""
        if confidence >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.6:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.4:
            return ConfidenceLevel.MODERATE
        elif confidence >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def validate_abstract_inference(self, inference_id: str) -> Dict[str, Any]:
        """Valida uma inferência abstrata."""
        try:
            # Encontrar inferência
            inference = None
            for inf in self.inference_history:
                if inf.inference_id == inference_id:
                    inference = inf
                    break
            
            if not inference:
                return {"error": f"Inferência '{inference_id}' não encontrada"}
            
            # Validar premissas
            premise_validation = self._validate_premises(inference.premises)
            
            # Validar conclusão
            conclusion_validation = self._validate_conclusion(inference.conclusion)
            
            # Validar raciocínio
            reasoning_validation = self._validate_reasoning(inference.reasoning_steps)
            
            # Calcular score de validação
            validation_score = (
                premise_validation["score"] * 0.4 +
                conclusion_validation["score"] * 0.4 +
                reasoning_validation["score"] * 0.2
            )
            
            return {
                "success": True,
                "inference_id": inference_id,
                "validation_score": validation_score,
                "premise_validation": premise_validation,
                "conclusion_validation": conclusion_validation,
                "reasoning_validation": reasoning_validation,
                "is_valid": validation_score >= 0.7
            }
            
        except Exception as e:
            logger.error(f"Erro na validação da inferência abstrata: {e}")
            return {"error": str(e)}
    
    def _validate_premises(self, premises: List[str]) -> Dict[str, Any]:
        """Valida premissas."""
        try:
            if not premises:
                return {"score": 0.0, "issues": ["Nenhuma premissa fornecida"]}
            
            issues = []
            score = 1.0
            
            for i, premise in enumerate(premises):
                if not premise.strip():
                    issues.append(f"Premissa {i+1} está vazia")
                    score -= 0.2
                elif len(premise.split()) < 3:
                    issues.append(f"Premissa {i+1} muito curta")
                    score -= 0.1
            
            return {
                "score": max(score, 0.0),
                "issues": issues,
                "premise_count": len(premises)
            }
            
        except Exception as e:
            logger.error(f"Erro na validação de premissas: {e}")
            return {"score": 0.0, "issues": [f"Erro na validação: {e}"]}
    
    def _validate_conclusion(self, conclusion: str) -> Dict[str, Any]:
        """Valida conclusão."""
        try:
            if not conclusion.strip():
                return {"score": 0.0, "issues": ["Conclusão vazia"]}
            
            issues = []
            score = 1.0
            
            if len(conclusion.split()) < 5:
                issues.append("Conclusão muito curta")
                score -= 0.3
            
            # Verificar palavras-chave de conclusão
            conclusion_keywords = ["portanto", "logo", "assim", "consequentemente", "então"]
            if not any(keyword in conclusion.lower() for keyword in conclusion_keywords):
                issues.append("Conclusão não contém palavras-chave apropriadas")
                score -= 0.2
            
            return {
                "score": max(score, 0.0),
                "issues": issues,
                "conclusion_length": len(conclusion.split())
            }
            
        except Exception as e:
            logger.error(f"Erro na validação de conclusão: {e}")
            return {"score": 0.0, "issues": [f"Erro na validação: {e}"]}
    
    def _validate_reasoning(self, reasoning_steps: List[str]) -> Dict[str, Any]:
        """Valida passos de raciocínio."""
        try:
            if not reasoning_steps:
                return {"score": 0.0, "issues": ["Nenhum passo de raciocínio"]}
            
            issues = []
            score = 1.0
            
            if len(reasoning_steps) < 3:
                issues.append("Poucos passos de raciocínio")
                score -= 0.3
            
            # Verificar se há progressão lógica
            logical_keywords = ["premissa", "regra", "conclusão", "evidência"]
            logical_steps = sum(1 for step in reasoning_steps 
                             if any(keyword in step.lower() for keyword in logical_keywords))
            
            if logical_steps < len(reasoning_steps) * 0.5:
                issues.append("Falta progressão lógica nos passos")
                score -= 0.2
            
            return {
                "score": max(score, 0.0),
                "issues": issues,
                "step_count": len(reasoning_steps),
                "logical_steps": logical_steps
            }
            
        except Exception as e:
            logger.error(f"Erro na validação de raciocínio: {e}")
            return {"score": 0.0, "issues": [f"Erro na validação: {e}"]}
    
    def get_abstract_inference_analysis(self, inference_id: str = None) -> Dict[str, Any]:
        """Obtém análise de inferências abstratas."""
        try:
            if inference_id:
                # Análise específica de uma inferência
                inference = None
                for inf in self.inference_history:
                    if inf.inference_id == inference_id:
                        inference = inf
                        break
                
                if not inference:
                    return {"error": f"Inferência '{inference_id}' não encontrada"}
                
                return {
                    "inference": asdict(inference),
                    "age_hours": (time.time() - inference.timestamp) / 3600,
                    "concept_connections": len(self.concept_graph.get(inference_id, set())),
                    "evidence_count": len(inference.supporting_evidence)
                }
            else:
                # Análise geral
                return {
                    "total_concepts": len(self.abstract_concepts),
                    "total_rules": len(self.inference_rules),
                    "total_inferences": len(self.inference_history),
                    "concept_distribution": self._get_concept_distribution(),
                    "rule_distribution": self._get_rule_distribution(),
                    "inference_distribution": self._get_inference_distribution(),
                    "confidence_statistics": self._get_confidence_statistics(),
                    "abstraction_statistics": self._get_abstraction_statistics()
                }
                
        except Exception as e:
            logger.error(f"Erro na análise de inferência abstrata: {e}")
            return {"error": str(e)}
    
    def _get_concept_distribution(self) -> Dict[str, int]:
        """Obtém distribuição de conceitos por nível de abstração."""
        distribution = defaultdict(int)
        
        for concept in self.abstract_concepts.values():
            distribution[concept.abstraction_level.value] += 1
        
        return dict(distribution)
    
    def _get_rule_distribution(self) -> Dict[str, int]:
        """Obtém distribuição de regras por tipo de inferência."""
        distribution = defaultdict(int)
        
        for rule in self.inference_rules.values():
            distribution[rule.inference_type.value] += 1
        
        return dict(distribution)
    
    def _get_inference_distribution(self) -> Dict[str, int]:
        """Obtém distribuição de inferências por tipo."""
        distribution = defaultdict(int)
        
        for inference in self.inference_history:
            distribution[inference.inference_type.value] += 1
        
        return dict(distribution)
    
    def _get_confidence_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas de confiança."""
        try:
            if not self.inference_history:
                return {"average_confidence": 0, "confidence_distribution": {}}
            
            confidences = [inf.confidence for inf in self.inference_history]
            confidence_levels = [inf.confidence_level.value for inf in self.inference_history]
            
            confidence_distribution = defaultdict(int)
            for level in confidence_levels:
                confidence_distribution[level] += 1
            
            return {
                "average_confidence": sum(confidences) / len(confidences),
                "max_confidence": max(confidences),
                "min_confidence": min(confidences),
                "confidence_distribution": dict(confidence_distribution)
            }
            
        except Exception as e:
            logger.error(f"Erro nas estatísticas de confiança: {e}")
            return {"error": str(e)}
    
    def _get_abstraction_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas de abstração."""
        try:
            abstraction_levels = [inf.abstraction_level.value for inf in self.inference_history]
            
            abstraction_distribution = defaultdict(int)
            for level in abstraction_levels:
                abstraction_distribution[level] += 1
            
            return {
                "abstraction_distribution": dict(abstraction_distribution),
                "most_common_level": max(abstraction_distribution.items(), key=lambda x: x[1])[0] if abstraction_distribution else None
            }
            
        except Exception as e:
            logger.error(f"Erro nas estatísticas de abstração: {e}")
            return {"error": str(e)}
    
    def _initialize_basic_concepts(self):
        """Inicializa conceitos abstratos básicos."""
        basic_concepts = [
            {
                "name": "animal",
                "description": "Ser vivo que se move e respira",
                "abstraction_level": AbstractionLevel.GENERAL,
                "properties": {"movement": True, "breathing": True, "consumption": True},
                "examples": ["cachorro", "gato", "pássaro", "peixe"]
            },
            {
                "name": "bird",
                "description": "Animal com penas e capacidade de voar",
                "abstraction_level": AbstractionLevel.SPECIFIC,
                "properties": {"feathers": True, "flying": True, "beak": True},
                "examples": ["canário", "papagaio", "águia", "pinguim"]
            },
            {
                "name": "behavior",
                "description": "Padrão de ações observáveis em animais",
                "abstraction_level": AbstractionLevel.ABSTRACT,
                "properties": {"observable": True, "pattern": True, "adaptive": True},
                "examples": ["voo", "canto", "nidificação", "migração"]
            }
        ]
        
        for concept_data in basic_concepts:
            try:
                self.add_abstract_concept(**concept_data)
            except Exception as e:
                logger.error(f"Erro na inicialização de conceito básico: {e}")
    
    def _initialize_basic_rules(self):
        """Inicializa regras de inferência básicas."""
        basic_rules = [
            {
                "name": "Classificação de Animais",
                "premise_patterns": ["{premise_1} tem penas", "{premise_2} pode voar"],
                "conclusion_pattern": "Portanto, {premise_1} é um pássaro",
                "inference_type": InferenceType.DEDUCTIVE,
                "abstraction_level": AbstractionLevel.GENERAL
            },
            {
                "name": "Padrão de Comportamento",
                "premise_patterns": ["{premise_1} canta", "{premise_2} constrói ninho"],
                "conclusion_pattern": "Portanto, {premise_1} está em época de acasalamento",
                "inference_type": InferenceType.INDUCTIVE,
                "abstraction_level": AbstractionLevel.ABSTRACT
            },
            {
                "name": "Inferência Causal",
                "premise_patterns": ["{premise_1} detecta predador", "{premise_2} foge"],
                "conclusion_pattern": "Portanto, {premise_1} causa {premise_2}",
                "inference_type": InferenceType.CAUSAL,
                "abstraction_level": AbstractionLevel.ABSTRACT
            }
        ]
        
        for rule_data in basic_rules:
            try:
                self.add_inference_rule(**rule_data)
            except Exception as e:
                logger.error(f"Erro na inicialização de regra básica: {e}")
    
    def _load_data(self):
        """Carrega dados salvos."""
        try:
            data_file = "data/abstract_inference.json"
            if os.path.exists(data_file):
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Carregar conceitos abstratos
                if "abstract_concepts" in data:
                    for concept_id, concept_data in data["abstract_concepts"].items():
                        concept_data["abstraction_level"] = AbstractionLevel(concept_data["abstraction_level"])
                        self.abstract_concepts[concept_id] = AbstractConcept(**concept_data)
                
                # Carregar regras de inferência
                if "inference_rules" in data:
                    for rule_id, rule_data in data["inference_rules"].items():
                        rule_data["inference_type"] = InferenceType(rule_data["inference_type"])
                        rule_data["abstraction_level"] = AbstractionLevel(rule_data["abstraction_level"])
                        self.inference_rules[rule_id] = InferenceRule(**rule_data)
                
                # Carregar histórico de inferências
                if "inference_history" in data:
                    for inference_data in data["inference_history"]:
                        inference_data["inference_type"] = InferenceType(inference_data["inference_type"])
                        inference_data["confidence_level"] = ConfidenceLevel(inference_data["confidence_level"])
                        inference_data["abstraction_level"] = AbstractionLevel(inference_data["abstraction_level"])
                        self.inference_history.append(AbstractInference(**inference_data))
                
                # Carregar grafo de conceitos
                if "concept_graph" in data:
                    self.concept_graph = defaultdict(set, data["concept_graph"])
                
                logger.info(f"Dados de inferência abstrata carregados: {len(self.abstract_concepts)} conceitos, {len(self.inference_rules)} regras")
                
        except Exception as e:
            logger.warning(f"Erro ao carregar dados de inferência abstrata: {e}")
    
    def _save_data(self):
        """Salva dados."""
        try:
            data_file = "data/abstract_inference.json"
            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            
            data = {
                "abstract_concepts": {
                    concept_id: {
                        "concept_id": concept.concept_id,
                        "name": concept.name,
                        "description": concept.description,
                        "abstraction_level": concept.abstraction_level.value,
                        "properties": concept.properties,
                        "relationships": concept.relationships,
                        "examples": concept.examples,
                        "created_at": concept.created_at,
                        "last_updated": concept.last_updated
                    }
                    for concept_id, concept in self.abstract_concepts.items()
                },
                "inference_rules": {
                    rule_id: {
                        "rule_id": rule.rule_id,
                        "name": rule.name,
                        "premise_patterns": rule.premise_patterns,
                        "conclusion_pattern": rule.conclusion_pattern,
                        "inference_type": rule.inference_type.value,
                        "confidence": rule.confidence,
                        "abstraction_level": rule.abstraction_level.value,
                        "conditions": rule.conditions,
                        "exceptions": rule.exceptions,
                        "created_at": rule.created_at,
                        "last_updated": rule.last_updated
                    }
                    for rule_id, rule in self.inference_rules.items()
                },
                "inference_history": [
                    {
                        "inference_id": inference.inference_id,
                        "query": inference.query,
                        "inference_type": inference.inference_type.value,
                        "premises": inference.premises,
                        "conclusion": inference.conclusion,
                        "confidence": inference.confidence,
                        "confidence_level": inference.confidence_level.value,
                        "abstraction_level": inference.abstraction_level.value,
                        "reasoning_steps": inference.reasoning_steps,
                        "supporting_evidence": inference.supporting_evidence,
                        "timestamp": inference.timestamp
                    }
                    for inference in self.inference_history
                ],
                "concept_graph": {k: list(v) for k, v in self.concept_graph.items()},
                "last_saved": time.time()
            }
            
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dados de inferência abstrata salvos: {len(self.abstract_concepts)} conceitos, {len(self.inference_rules)} regras")
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados de inferência abstrata: {e}")
