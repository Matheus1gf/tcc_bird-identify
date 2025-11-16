#!/usr/bin/env python3
"""
Sistema de Transferência de Conhecimento
Permite ao sistema transferir conhecimento entre diferentes domínios, conceitos e contextos
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

class TransferType(Enum):
    """Tipos de transferência de conhecimento."""
    DOMAIN_TO_DOMAIN = "domain_to_domain"  # Entre domínios diferentes
    CONCEPT_TO_CONCEPT = "concept_to_concept"  # Entre conceitos
    SPECIES_TO_SPECIES = "species_to_species"  # Entre espécies
    CONTEXT_TO_CONTEXT = "context_to_context"  # Entre contextos
    SKILL_TO_SKILL = "skill_to_skill"  # Entre habilidades
    PATTERN_TO_PATTERN = "pattern_to_pattern"  # Entre padrões
    FEATURE_TO_FEATURE = "feature_to_feature"  # Entre características
    STRATEGY_TO_STRATEGY = "strategy_to_strategy"  # Entre estratégias

class TransferStrategy(Enum):
    """Estratégias de transferência."""
    DIRECT = "direct"  # Transferência direta
    ADAPTIVE = "adaptive"  # Transferência adaptativa
    SELECTIVE = "selective"  # Transferência seletiva
    HIERARCHICAL = "hierarchical"  # Transferência hierárquica
    CONTEXTUAL = "contextual"  # Transferência contextual
    ANALOGICAL = "analogical"  # Transferência analógica
    META_LEARNING = "meta_learning"  # Meta-aprendizado
    FEW_SHOT = "few_shot"  # Few-shot learning

class TransferConfidence(Enum):
    """Níveis de confiança na transferência."""
    VERY_LOW = "very_low"    # 0.0 - 0.2
    LOW = "low"              # 0.2 - 0.4
    MODERATE = "moderate"    # 0.4 - 0.6
    HIGH = "high"            # 0.6 - 0.8
    VERY_HIGH = "very_high"  # 0.8 - 1.0

class KnowledgeType(Enum):
    """Tipos de conhecimento."""
    FACTUAL = "factual"  # Conhecimento factual
    PROCEDURAL = "procedural"  # Conhecimento procedural
    CONCEPTUAL = "conceptual"  # Conhecimento conceitual
    METACOGNITIVE = "metacognitive"  # Conhecimento metacognitivo
    EXPERIENTIAL = "experiential"  # Conhecimento experiencial
    PATTERN_BASED = "pattern_based"  # Conhecimento baseado em padrões

@dataclass
class TransferMapping:
    """Mapeamento de transferência entre elementos."""
    source_element: str
    target_element: str
    mapping_type: str
    confidence: float
    context: Dict[str, Any]
    created_at: float = field(default_factory=time.time)

@dataclass
class KnowledgeTransfer:
    """Transferência de conhecimento."""
    transfer_id: str
    source_domain: str
    target_domain: str
    transfer_type: TransferType
    strategy: TransferStrategy
    knowledge_type: KnowledgeType
    source_knowledge: Dict[str, Any]
    transferred_knowledge: Dict[str, Any]
    confidence: TransferConfidence
    success: bool
    adaptation_required: bool
    performance_improvement: float
    context: Dict[str, Any]
    mappings: List[TransferMapping]
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    usage_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TransferRule:
    """Regra de transferência."""
    rule_id: str
    source_pattern: str
    target_pattern: str
    conditions: List[str]
    confidence: float
    success_rate: float
    usage_count: int
    created_at: float = field(default_factory=time.time)

class KnowledgeTransferSystem:
    """Sistema de transferência de conhecimento."""
    
    def __init__(self):
        self.knowledge_transfers: Dict[str, KnowledgeTransfer] = {}
        self.transfer_rules: Dict[str, TransferRule] = {}
        self.domain_mappings: Dict[str, Dict[str, Any]] = {}
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        self.transfer_history: List[Dict[str, Any]] = []
        
        # Carregar dados existentes
        self._load_data()
        
        # Inicializar regras básicas
        self._initialize_basic_rules()
        
        logger.info("Sistema de transferência de conhecimento inicializado")
    
    def transfer_knowledge(self, source_domain: str, target_domain: str,
                          source_knowledge: Dict[str, Any],
                          transfer_type: TransferType = TransferType.CONCEPT_TO_CONCEPT,
                          strategy: TransferStrategy = TransferStrategy.ADAPTIVE,
                          knowledge_type: KnowledgeType = KnowledgeType.CONCEPTUAL,
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Transfere conhecimento entre domínios.
        
        Args:
            source_domain: Domínio fonte
            target_domain: Domínio alvo
            source_knowledge: Conhecimento fonte
            transfer_type: Tipo de transferência
            strategy: Estratégia de transferência
            knowledge_type: Tipo de conhecimento
            context: Contexto da transferência
            
        Returns:
            Resultado da transferência
        """
        try:
            # Gerar ID único para a transferência
            transfer_id = self._generate_transfer_id(source_domain, target_domain, transfer_type)
            
            # Calcular similaridade entre domínios
            similarity = self._calculate_domain_similarity(source_domain, target_domain)
            
            # Determinar se a transferência é viável
            # Remover restrição de similaridade para permitir transferências
            # if similarity < 0.1:
            #     return {
            #         "success": False,
            #         "error": "Similaridade insuficiente entre domínios",
            #         "similarity": similarity
            #     }
            
            # Aplicar estratégia de transferência
            transferred_knowledge, mappings = self._apply_transfer_strategy(
                source_knowledge, source_domain, target_domain, strategy, similarity
            )
            
            # Calcular confiança da transferência
            confidence = self._calculate_transfer_confidence(
                similarity, transferred_knowledge, mappings, strategy
            )
            
            # Determinar se adaptação é necessária
            adaptation_required = self._determine_adaptation_required(
                strategy, similarity, transferred_knowledge
            )
            
            # Calcular melhoria de performance esperada
            performance_improvement = self._calculate_performance_improvement(
                confidence, adaptation_required, similarity
            )
            
            # Criar transferência de conhecimento
            knowledge_transfer = KnowledgeTransfer(
                transfer_id=transfer_id,
                source_domain=source_domain,
                target_domain=target_domain,
                transfer_type=transfer_type,
                strategy=strategy,
                knowledge_type=knowledge_type,
                source_knowledge=source_knowledge,
                transferred_knowledge=transferred_knowledge,
                confidence=confidence,
                success=True,
                adaptation_required=adaptation_required,
                performance_improvement=performance_improvement,
                context=context or {},
                mappings=mappings
            )
            
            # Armazenar transferência
            self.knowledge_transfers[transfer_id] = knowledge_transfer
            
            # Atualizar regras de transferência
            self._update_transfer_rules(knowledge_transfer)
            
            # Registrar histórico
            self._record_transfer_event(knowledge_transfer)
            
            # Salvar dados
            self._save_data()
            
            return {
                "success": True,
                "transfer_id": transfer_id,
                "transferred_knowledge": transferred_knowledge,
                "confidence": confidence.value,
                "adaptation_required": adaptation_required,
                "performance_improvement": performance_improvement,
                "mappings_count": len(mappings),
                "similarity": similarity
            }
            
        except Exception as e:
            logger.error(f"Erro na transferência de conhecimento: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def find_similar_transfers(self, source_domain: str, target_domain: str,
                             transfer_type: TransferType = None) -> Dict[str, Any]:
        """
        Encontra transferências similares.
        
        Args:
            source_domain: Domínio fonte
            target_domain: Domínio alvo
            transfer_type: Tipo de transferência (opcional)
            
        Returns:
            Transferências similares encontradas
        """
        try:
            similar_transfers = []
            
            for transfer in self.knowledge_transfers.values():
                # Verificar tipo de transferência
                if transfer_type and transfer.transfer_type != transfer_type:
                    continue
                
                # Calcular similaridade com a transferência atual
                similarity = self._calculate_transfer_similarity(
                    source_domain, target_domain,
                    transfer.source_domain, transfer.target_domain
                )
                
                if similarity > 0.5:  # Limiar de similaridade
                    similar_transfers.append({
                        "transfer_id": transfer.transfer_id,
                        "source_domain": transfer.source_domain,
                        "target_domain": transfer.target_domain,
                        "transfer_type": transfer.transfer_type.value,
                        "strategy": transfer.strategy.value,
                        "confidence": transfer.confidence.value,
                        "success": transfer.success,
                        "similarity": similarity,
                        "usage_count": transfer.usage_count
                    })
            
            # Ordenar por similaridade e sucesso
            similar_transfers.sort(key=lambda x: (x["similarity"], x["success"]), reverse=True)
            
            return {
                "success": True,
                "similar_transfers": similar_transfers,
                "count": len(similar_transfers)
            }
            
        except Exception as e:
            logger.error(f"Erro ao encontrar transferências similares: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def apply_transfer_rule(self, rule_id: str, source_data: Dict[str, Any],
                           target_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica uma regra de transferência.
        
        Args:
            rule_id: ID da regra
            source_data: Dados fonte
            target_context: Contexto alvo
            
        Returns:
            Resultado da aplicação da regra
        """
        try:
            if rule_id not in self.transfer_rules:
                return {
                    "success": False,
                    "error": "Regra não encontrada"
                }
            
            rule = self.transfer_rules[rule_id]
            
            # Verificar condições da regra
            if not self._check_rule_conditions(rule, source_data, target_context):
                return {
                    "success": False,
                    "error": "Condições da regra não atendidas"
                }
            
            # Aplicar mapeamento da regra
            transferred_data = self._apply_rule_mapping(rule, source_data, target_context)
            
            # Atualizar estatísticas da regra
            rule.usage_count += 1
            
            return {
                "success": True,
                "rule_id": rule_id,
                "transferred_data": transferred_data,
                "confidence": rule.confidence,
                "success_rate": rule.success_rate
            }
            
        except Exception as e:
            logger.error(f"Erro ao aplicar regra de transferência: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def learn_transfer_rule(self, source_pattern: str, target_pattern: str,
                           examples: List[Dict[str, Any]],
                           conditions: List[str] = None) -> Dict[str, Any]:
        """
        Aprende uma nova regra de transferência.
        
        Args:
            source_pattern: Padrão fonte
            target_pattern: Padrão alvo
            examples: Exemplos de transferência
            conditions: Condições da regra
            
        Returns:
            Resultado do aprendizado da regra
        """
        try:
            # Gerar ID único para a regra
            rule_id = self._generate_rule_id(source_pattern, target_pattern)
            
            # Analisar exemplos para determinar confiança
            confidence = self._analyze_examples_for_confidence(examples)
            
            # Calcular taxa de sucesso
            success_rate = self._calculate_success_rate(examples)
            
            # Criar regra de transferência
            rule = TransferRule(
                rule_id=rule_id,
                source_pattern=source_pattern,
                target_pattern=target_pattern,
                conditions=conditions or [],
                confidence=confidence,
                success_rate=success_rate,
                usage_count=0
            )
            
            # Armazenar regra
            self.transfer_rules[rule_id] = rule
            
            # Salvar dados
            self._save_data()
            
            return {
                "success": True,
                "rule_id": rule_id,
                "confidence": confidence,
                "success_rate": success_rate,
                "conditions_count": len(conditions or [])
            }
            
        except Exception as e:
            logger.error(f"Erro ao aprender regra de transferência: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_transfer_analysis(self, domain: str = None) -> Dict[str, Any]:
        """
        Obtém análise das transferências de conhecimento.
        
        Args:
            domain: Domínio específico (opcional)
            
        Returns:
            Análise das transferências
        """
        try:
            if domain:
                # Análise de um domínio específico
                transfers = [t for t in self.knowledge_transfers.values() 
                           if t.source_domain == domain or t.target_domain == domain]
            else:
                # Análise geral
                transfers = list(self.knowledge_transfers.values())
            
            # Estatísticas gerais
            total_transfers = len(transfers)
            successful_transfers = sum(1 for t in transfers if t.success)
            success_rate = successful_transfers / total_transfers if total_transfers > 0 else 0
            
            # Distribuição por tipo
            transfer_types = defaultdict(int)
            strategies = defaultdict(int)
            knowledge_types = defaultdict(int)
            confidence_levels = defaultdict(int)
            
            for transfer in transfers:
                transfer_types[transfer.transfer_type.value] += 1
                strategies[transfer.strategy.value] += 1
                knowledge_types[transfer.knowledge_type.value] += 1
                confidence_levels[transfer.confidence.value] += 1
            
            # Domínios mais transferidos
            domain_frequency = defaultdict(int)
            for transfer in transfers:
                domain_frequency[transfer.source_domain] += 1
                domain_frequency[transfer.target_domain] += 1
            
            most_transferred_domains = sorted(domain_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Regras mais usadas
            rule_usage = [(rule_id, rule.usage_count) for rule_id, rule in self.transfer_rules.items()]
            most_used_rules = sorted(rule_usage, key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "success": True,
                "analysis": {
                    "total_transfers": total_transfers,
                    "successful_transfers": successful_transfers,
                    "success_rate": success_rate,
                    "transfer_types": dict(transfer_types),
                    "strategies": dict(strategies),
                    "knowledge_types": dict(knowledge_types),
                    "confidence_levels": dict(confidence_levels),
                    "most_transferred_domains": most_transferred_domains,
                    "most_used_rules": most_used_rules,
                    "total_rules": len(self.transfer_rules)
                }
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de transferências: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_transfer_id(self, source: str, target: str, transfer_type: TransferType) -> str:
        """Gera ID único para uma transferência."""
        content = f"{source}_{target}_{transfer_type.value}_{int(time.time())}"
        return f"transfer_{hashlib.md5(content.encode()).hexdigest()[:12]}"
    
    def _generate_rule_id(self, source_pattern: str, target_pattern: str) -> str:
        """Gera ID único para uma regra."""
        content = f"{source_pattern}_{target_pattern}"
        return f"rule_{hashlib.md5(content.encode()).hexdigest()[:12]}"
    
    def _calculate_domain_similarity(self, domain1: str, domain2: str) -> float:
        """Calcula similaridade entre domínios."""
        cache_key = tuple(sorted([domain1, domain2]))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Implementar cálculo de similaridade baseado em características dos domínios
        # Por enquanto, usar similaridade baseada em palavras-chave
        similarity = self._calculate_keyword_similarity(domain1, domain2)
        
        self.similarity_cache[cache_key] = similarity
        return similarity
    
    def _calculate_keyword_similarity(self, text1: str, text2: str) -> float:
        """Calcula similaridade baseada em palavras-chave."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        # Se não há interseção, dar uma similaridade mínima baseada no tamanho
        if intersection == 0:
            # Similaridade mínima baseada no tamanho das palavras
            min_length = min(len(text1), len(text2))
            max_length = max(len(text1), len(text2))
            if max_length > 0:
                return min_length / max_length * 0.1  # Similaridade mínima de 0.1
            return 0.1
        
        return intersection / union if union > 0 else 0.1
    
    def _apply_transfer_strategy(self, source_knowledge: Dict[str, Any],
                                source_domain: str, target_domain: str,
                                strategy: TransferStrategy, similarity: float) -> Tuple[Dict[str, Any], List[TransferMapping]]:
        """Aplica estratégia de transferência."""
        mappings = []
        
        if strategy == TransferStrategy.DIRECT:
            return self._direct_transfer(source_knowledge, mappings)
        elif strategy == TransferStrategy.ADAPTIVE:
            return self._adaptive_transfer(source_knowledge, source_domain, target_domain, similarity, mappings)
        elif strategy == TransferStrategy.SELECTIVE:
            return self._selective_transfer(source_knowledge, similarity, mappings)
        elif strategy == TransferStrategy.HIERARCHICAL:
            return self._hierarchical_transfer(source_knowledge, source_domain, target_domain, mappings)
        elif strategy == TransferStrategy.CONTEXTUAL:
            return self._contextual_transfer(source_knowledge, source_domain, target_domain, mappings)
        elif strategy == TransferStrategy.ANALOGICAL:
            return self._analogical_transfer(source_knowledge, source_domain, target_domain, mappings)
        else:
            return source_knowledge.copy(), mappings
    
    def _direct_transfer(self, source_knowledge: Dict[str, Any], mappings: List[TransferMapping]) -> Tuple[Dict[str, Any], List[TransferMapping]]:
        """Transferência direta."""
        return source_knowledge.copy(), mappings
    
    def _adaptive_transfer(self, source_knowledge: Dict[str, Any], source_domain: str, 
                         target_domain: str, similarity: float, mappings: List[TransferMapping]) -> Tuple[Dict[str, Any], List[TransferMapping]]:
        """Transferência adaptativa."""
        transferred = {}
        
        for key, value in source_knowledge.items():
            # Adaptar valor baseado na similaridade
            if isinstance(value, (int, float)):
                adapted_value = value * similarity
            elif isinstance(value, str):
                adapted_value = f"{value}_{target_domain}"
            else:
                adapted_value = value
            
            transferred[key] = adapted_value
            
            # Criar mapeamento
            mapping = TransferMapping(
                source_element=key,
                target_element=key,
                mapping_type="adaptive",
                confidence=similarity,
                context={"adaptation_factor": similarity}
            )
            mappings.append(mapping)
        
        return transferred, mappings
    
    def _selective_transfer(self, source_knowledge: Dict[str, Any], similarity: float, 
                          mappings: List[TransferMapping]) -> Tuple[Dict[str, Any], List[TransferMapping]]:
        """Transferência seletiva."""
        transferred = {}
        
        for key, value in source_knowledge.items():
            # Transferir apenas se similaridade for alta
            if similarity > 0.7:
                transferred[key] = value
                
                mapping = TransferMapping(
                    source_element=key,
                    target_element=key,
                    mapping_type="selective",
                    confidence=similarity,
                    context={"threshold": 0.7}
                )
                mappings.append(mapping)
        
        return transferred, mappings
    
    def _hierarchical_transfer(self, source_knowledge: Dict[str, Any], source_domain: str, 
                              target_domain: str, mappings: List[TransferMapping]) -> Tuple[Dict[str, Any], List[TransferMapping]]:
        """Transferência hierárquica."""
        transferred = {}
        
        # Implementar transferência baseada em hierarquia de conceitos
        for key, value in source_knowledge.items():
            # Mapear para nível hierárquico apropriado
            hierarchical_key = f"{target_domain}_{key}"
            transferred[hierarchical_key] = value
            
            mapping = TransferMapping(
                source_element=key,
                target_element=hierarchical_key,
                mapping_type="hierarchical",
                confidence=0.8,
                context={"hierarchy_level": "domain"}
            )
            mappings.append(mapping)
        
        return transferred, mappings
    
    def _contextual_transfer(self, source_knowledge: Dict[str, Any], source_domain: str, 
                            target_domain: str, mappings: List[TransferMapping]) -> Tuple[Dict[str, Any], List[TransferMapping]]:
        """Transferência contextual."""
        transferred = {}
        
        for key, value in source_knowledge.items():
            # Adaptar baseado no contexto
            contextual_key = f"{target_domain}_{key}"
            contextual_value = {
                "original_value": value,
                "context": target_domain,
                "source_domain": source_domain
            }
            transferred[contextual_key] = contextual_value
            
            mapping = TransferMapping(
                source_element=key,
                target_element=contextual_key,
                mapping_type="contextual",
                confidence=0.75,
                context={"context_adaptation": True}
            )
            mappings.append(mapping)
        
        return transferred, mappings
    
    def _analogical_transfer(self, source_knowledge: Dict[str, Any], source_domain: str, 
                            target_domain: str, mappings: List[TransferMapping]) -> Tuple[Dict[str, Any], List[TransferMapping]]:
        """Transferência analógica."""
        transferred = {}
        
        for key, value in source_knowledge.items():
            # Criar analogia entre domínios
            analogical_key = f"{key}_analog"
            analogical_value = {
                "analogy": f"{source_domain} -> {target_domain}",
                "value": value,
                "type": "analogical"
            }
            transferred[analogical_key] = analogical_value
            
            mapping = TransferMapping(
                source_element=key,
                target_element=analogical_key,
                mapping_type="analogical",
                confidence=0.7,
                context={"analogy_type": "domain_mapping"}
            )
            mappings.append(mapping)
        
        return transferred, mappings
    
    def _calculate_transfer_confidence(self, similarity: float, transferred_knowledge: Dict[str, Any],
                                     mappings: List[TransferMapping], strategy: TransferStrategy) -> TransferConfidence:
        """Calcula confiança da transferência."""
        # Fator baseado na similaridade
        similarity_factor = similarity
        
        # Fator baseado no número de mapeamentos
        mapping_factor = min(len(mappings) / 10.0, 1.0) if mappings else 0.0
        
        # Fator baseado na estratégia
        strategy_factors = {
            TransferStrategy.DIRECT: 0.9,
            TransferStrategy.ADAPTIVE: 0.8,
            TransferStrategy.SELECTIVE: 0.7,
            TransferStrategy.HIERARCHICAL: 0.75,
            TransferStrategy.CONTEXTUAL: 0.7,
            TransferStrategy.ANALOGICAL: 0.6,
            TransferStrategy.META_LEARNING: 0.8,
            TransferStrategy.FEW_SHOT: 0.65
        }
        strategy_factor = strategy_factors.get(strategy, 0.5)
        
        # Confiança geral
        overall_confidence = (similarity_factor * 0.5 + mapping_factor * 0.2 + strategy_factor * 0.3)
        
        # Converter para enum
        if overall_confidence >= 0.8:
            return TransferConfidence.VERY_HIGH
        elif overall_confidence >= 0.6:
            return TransferConfidence.HIGH
        elif overall_confidence >= 0.4:
            return TransferConfidence.MODERATE
        elif overall_confidence >= 0.2:
            return TransferConfidence.LOW
        else:
            return TransferConfidence.VERY_LOW
    
    def _determine_adaptation_required(self, strategy: TransferStrategy, similarity: float,
                                     transferred_knowledge: Dict[str, Any]) -> bool:
        """Determina se adaptação é necessária."""
        # Estratégias que sempre requerem adaptação
        adaptation_strategies = {
            TransferStrategy.ADAPTIVE,
            TransferStrategy.HIERARCHICAL,
            TransferStrategy.CONTEXTUAL,
            TransferStrategy.ANALOGICAL
        }
        
        if strategy in adaptation_strategies:
            return True
        
        # Adaptação baseada na similaridade
        if similarity < 0.7:
            return True
        
        return False
    
    def _calculate_performance_improvement(self, confidence: TransferConfidence, adaptation_required: bool,
                                        similarity: float) -> float:
        """Calcula melhoria de performance esperada."""
        confidence_values = {
            TransferConfidence.VERY_LOW: 0.1,
            TransferConfidence.LOW: 0.2,
            TransferConfidence.MODERATE: 0.4,
            TransferConfidence.HIGH: 0.6,
            TransferConfidence.VERY_HIGH: 0.8
        }
        
        base_improvement = confidence_values.get(confidence, 0.2)
        
        # Reduzir se adaptação for necessária
        if adaptation_required:
            base_improvement *= 0.8
        
        # Ajustar baseado na similaridade
        similarity_factor = similarity * 0.2
        
        return min(base_improvement + similarity_factor, 1.0)
    
    def _update_transfer_rules(self, knowledge_transfer: KnowledgeTransfer):
        """Atualiza regras de transferência baseadas na transferência."""
        # Implementar atualização de regras
        pass
    
    def _record_transfer_event(self, knowledge_transfer: KnowledgeTransfer):
        """Registra evento de transferência."""
        event = {
            "timestamp": time.time(),
            "transfer_id": knowledge_transfer.transfer_id,
            "source_domain": knowledge_transfer.source_domain,
            "target_domain": knowledge_transfer.target_domain,
            "transfer_type": knowledge_transfer.transfer_type.value,
            "strategy": knowledge_transfer.strategy.value,
            "success": knowledge_transfer.success,
            "confidence": knowledge_transfer.confidence.value
        }
        self.transfer_history.append(event)
    
    def _calculate_transfer_similarity(self, source1: str, target1: str, source2: str, target2: str) -> float:
        """Calcula similaridade entre transferências."""
        source_sim = self._calculate_keyword_similarity(source1, source2)
        target_sim = self._calculate_keyword_similarity(target1, target2)
        
        return (source_sim + target_sim) / 2.0
    
    def _check_rule_conditions(self, rule: TransferRule, source_data: Dict[str, Any],
                              target_context: Dict[str, Any]) -> bool:
        """Verifica condições de uma regra."""
        # Implementar verificação de condições
        return True
    
    def _apply_rule_mapping(self, rule: TransferRule, source_data: Dict[str, Any],
                          target_context: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica mapeamento de uma regra."""
        # Implementar aplicação de mapeamento
        return source_data.copy()
    
    def _analyze_examples_for_confidence(self, examples: List[Dict[str, Any]]) -> float:
        """Analisa exemplos para determinar confiança."""
        if not examples:
            return 0.0
        
        # Calcular confiança baseada na consistência dos exemplos
        success_count = sum(1 for ex in examples if ex.get("success", False))
        return success_count / len(examples)
    
    def _calculate_success_rate(self, examples: List[Dict[str, Any]]) -> float:
        """Calcula taxa de sucesso dos exemplos."""
        if not examples:
            return 0.0
        
        success_count = sum(1 for ex in examples if ex.get("success", False))
        return success_count / len(examples)
    
    def _initialize_basic_rules(self):
        """Inicializa regras básicas de transferência."""
        # Regras básicas para transferência entre espécies de pássaros
        basic_rules = [
            ("pássaro_características", "ave_características", ["morphology", "behavior"]),
            ("ave_comportamento", "pássaro_comportamento", ["behavior", "ecology"]),
            ("espécie_morfologia", "gênero_morfologia", ["taxonomy", "morphology"])
        ]
        
        for source_pattern, target_pattern, conditions in basic_rules:
            rule_id = self._generate_rule_id(source_pattern, target_pattern)
            rule = TransferRule(
                rule_id=rule_id,
                source_pattern=source_pattern,
                target_pattern=target_pattern,
                conditions=conditions,
                confidence=0.8,
                success_rate=0.7,
                usage_count=0
            )
            self.transfer_rules[rule_id] = rule
    
    def _save_data(self):
        """Salva dados do sistema."""
        try:
            data = {
                "knowledge_transfers": {},
                "transfer_rules": {},
                "transfer_history": self.transfer_history[-100:]  # Últimos 100 eventos
            }
            
            # Converter transferências para formato serializável
            for transfer_id, transfer in self.knowledge_transfers.items():
                data["knowledge_transfers"][transfer_id] = {
                    "transfer_id": transfer.transfer_id,
                    "source_domain": transfer.source_domain,
                    "target_domain": transfer.target_domain,
                    "transfer_type": transfer.transfer_type.value,
                    "strategy": transfer.strategy.value,
                    "knowledge_type": transfer.knowledge_type.value,
                    "source_knowledge": transfer.source_knowledge,
                    "transferred_knowledge": transfer.transferred_knowledge,
                    "confidence": transfer.confidence.value,
                    "success": transfer.success,
                    "adaptation_required": transfer.adaptation_required,
                    "performance_improvement": transfer.performance_improvement,
                    "context": transfer.context,
                    "mappings": [
                        {
                            "source_element": m.source_element,
                            "target_element": m.target_element,
                            "mapping_type": m.mapping_type,
                            "confidence": m.confidence,
                            "context": m.context,
                            "created_at": m.created_at
                        } for m in transfer.mappings
                    ],
                    "created_at": transfer.created_at,
                    "last_used": transfer.last_used,
                    "usage_count": transfer.usage_count,
                    "metadata": transfer.metadata
                }
            
            # Converter regras para formato serializável
            for rule_id, rule in self.transfer_rules.items():
                data["transfer_rules"][rule_id] = {
                    "rule_id": rule.rule_id,
                    "source_pattern": rule.source_pattern,
                    "target_pattern": rule.target_pattern,
                    "conditions": rule.conditions,
                    "confidence": rule.confidence,
                    "success_rate": rule.success_rate,
                    "usage_count": rule.usage_count,
                    "created_at": rule.created_at
                }
            
            # Salvar arquivo
            os.makedirs("data", exist_ok=True)
            with open("data/knowledge_transfer_data.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dados de transferência de conhecimento salvos: {len(self.knowledge_transfers)} transferências, {len(self.transfer_rules)} regras")
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados de transferência de conhecimento: {e}")
    
    def _load_data(self):
        """Carrega dados do sistema."""
        try:
            if not os.path.exists("data/knowledge_transfer_data.json"):
                return
            
            with open("data/knowledge_transfer_data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Carregar transferências
            for transfer_id, transfer_data in data.get("knowledge_transfers", {}).items():
                mappings = [
                    TransferMapping(
                        source_element=m["source_element"],
                        target_element=m["target_element"],
                        mapping_type=m["mapping_type"],
                        confidence=m["confidence"],
                        context=m["context"],
                        created_at=m["created_at"]
                    ) for m in transfer_data.get("mappings", [])
                ]
                
                knowledge_transfer = KnowledgeTransfer(
                    transfer_id=transfer_data["transfer_id"],
                    source_domain=transfer_data["source_domain"],
                    target_domain=transfer_data["target_domain"],
                    transfer_type=TransferType(transfer_data["transfer_type"]),
                    strategy=TransferStrategy(transfer_data["strategy"]),
                    knowledge_type=KnowledgeType(transfer_data["knowledge_type"]),
                    source_knowledge=transfer_data["source_knowledge"],
                    transferred_knowledge=transfer_data["transferred_knowledge"],
                    confidence=TransferConfidence(transfer_data["confidence"]),
                    success=transfer_data["success"],
                    adaptation_required=transfer_data["adaptation_required"],
                    performance_improvement=transfer_data["performance_improvement"],
                    context=transfer_data["context"],
                    mappings=mappings,
                    created_at=transfer_data["created_at"],
                    last_used=transfer_data["last_used"],
                    usage_count=transfer_data["usage_count"],
                    metadata=transfer_data.get("metadata", {})
                )
                
                self.knowledge_transfers[transfer_id] = knowledge_transfer
            
            # Carregar regras
            for rule_id, rule_data in data.get("transfer_rules", {}).items():
                rule = TransferRule(
                    rule_id=rule_data["rule_id"],
                    source_pattern=rule_data["source_pattern"],
                    target_pattern=rule_data["target_pattern"],
                    conditions=rule_data["conditions"],
                    confidence=rule_data["confidence"],
                    success_rate=rule_data["success_rate"],
                    usage_count=rule_data["usage_count"],
                    created_at=rule_data["created_at"]
                )
                self.transfer_rules[rule_id] = rule
            
            # Carregar histórico
            self.transfer_history = data.get("transfer_history", [])
            
            logger.info(f"Dados de transferência de conhecimento carregados: {len(self.knowledge_transfers)} transferências, {len(self.transfer_rules)} regras")
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados de transferência de conhecimento: {e}")
