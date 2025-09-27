import logging
import json
import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
import random
import time

# Configurar logging
logger = logging.getLogger(__name__)

class UniversalFeatureType(Enum):
    """Tipos de características universais."""
    MORPHOLOGICAL = "morphological"  # Características morfológicas universais
    BEHAVIORAL = "behavioral"  # Comportamentos universais
    ECOLOGICAL = "ecological"  # Características ecológicas universais
    PHYSIOLOGICAL = "physiological"  # Características fisiológicas universais
    EVOLUTIONARY = "evolutionary"  # Características evolutivas universais

class UniversalFeatureLevel(Enum):
    """Níveis de universalidade das características."""
    SPECIES_LEVEL = "species_level"  # Universal entre espécies
    GENUS_LEVEL = "genus_level"  # Universal entre gêneros
    FAMILY_LEVEL = "family_level"  # Universal entre famílias
    ORDER_LEVEL = "order_level"  # Universal entre ordens
    CLASS_LEVEL = "class_level"  # Universal entre classes
    KINGDOM_LEVEL = "kingdom_level"  # Universal entre reinos

@dataclass
class UniversalFeature:
    """Representa uma característica universal."""
    name: str
    feature_type: UniversalFeatureType
    universal_level: UniversalFeatureLevel
    description: str
    detection_methods: List[str]
    weight: float
    confidence_threshold: float
    examples: List[Dict[str, Any]] = field(default_factory=list)
    universality_score: float = 0.0
    transfer_power: float = 0.0
    generalization_strength: float = 0.0

@dataclass
class UniversalFeaturePattern:
    """Padrão de características universais."""
    pattern_name: str
    features: List[str]
    pattern_type: str
    confidence: float
    examples_count: int
    universality_level: UniversalFeatureLevel

class UniversalFeatureAnalyzer:
    """Analisador de características universais."""
    
    def __init__(self):
        self.universal_features = self._initialize_universal_features()
        self.feature_patterns: Dict[str, UniversalFeaturePattern] = {}
        self.transfer_history: List[Dict[str, Any]] = []
        self.universality_scores: Dict[str, float] = {}
        
    def _initialize_universal_features(self) -> Dict[str, UniversalFeature]:
        """Inicializa características universais."""
        features = {}
        
        # Características morfológicas universais
        features["bilateral_symmetry"] = UniversalFeature(
            name="bilateral_symmetry",
            feature_type=UniversalFeatureType.MORPHOLOGICAL,
            universal_level=UniversalFeatureLevel.KINGDOM_LEVEL,
            description="Simetria bilateral - característica universal em animais",
            detection_methods=["shape_analysis", "symmetry_analysis"],
            weight=0.9,
            confidence_threshold=0.7
        )
        
        features["head_structure"] = UniversalFeature(
            name="head_structure",
            feature_type=UniversalFeatureType.MORPHOLOGICAL,
            universal_level=UniversalFeatureLevel.CLASS_LEVEL,
            description="Estrutura cefálica - cabeça com órgãos sensoriais",
            detection_methods=["shape_analysis", "position_analysis"],
            weight=0.8,
            confidence_threshold=0.6
        )
        
        features["limb_structure"] = UniversalFeature(
            name="limb_structure",
            feature_type=UniversalFeatureType.MORPHOLOGICAL,
            universal_level=UniversalFeatureLevel.CLASS_LEVEL,
            description="Estruturas locomotivas - membros para movimento",
            detection_methods=["shape_analysis", "position_analysis"],
            weight=0.7,
            confidence_threshold=0.5
        )
        
        features["body_axis"] = UniversalFeature(
            name="body_axis",
            feature_type=UniversalFeatureType.MORPHOLOGICAL,
            universal_level=UniversalFeatureLevel.CLASS_LEVEL,
            description="Eixo corporal principal - estrutura longitudinal",
            detection_methods=["shape_analysis", "size_analysis"],
            weight=0.6,
            confidence_threshold=0.5
        )
        
        # Características comportamentais universais
        features["movement_patterns"] = UniversalFeature(
            name="movement_patterns",
            feature_type=UniversalFeatureType.BEHAVIORAL,
            universal_level=UniversalFeatureLevel.CLASS_LEVEL,
            description="Padrões de movimento característicos",
            detection_methods=["motion_analysis", "pattern_analysis"],
            weight=0.7,
            confidence_threshold=0.6
        )
        
        features["feeding_behavior"] = UniversalFeature(
            name="feeding_behavior",
            feature_type=UniversalFeatureType.BEHAVIORAL,
            universal_level=UniversalFeatureLevel.ORDER_LEVEL,
            description="Comportamento alimentar característico",
            detection_methods=["behavior_analysis", "pattern_analysis"],
            weight=0.6,
            confidence_threshold=0.5
        )
        
        # Características ecológicas universais
        features["habitat_adaptation"] = UniversalFeature(
            name="habitat_adaptation",
            feature_type=UniversalFeatureType.ECOLOGICAL,
            universal_level=UniversalFeatureLevel.FAMILY_LEVEL,
            description="Adaptações ao habitat específico",
            detection_methods=["environment_analysis", "pattern_analysis"],
            weight=0.5,
            confidence_threshold=0.4
        )
        
        features["ecological_niche"] = UniversalFeature(
            name="ecological_niche",
            feature_type=UniversalFeatureType.ECOLOGICAL,
            universal_level=UniversalFeatureLevel.SPECIES_LEVEL,
            description="Nicho ecológico específico",
            detection_methods=["environment_analysis", "behavior_analysis"],
            weight=0.4,
            confidence_threshold=0.3
        )
        
        # Características fisiológicas universais
        features["metabolic_signatures"] = UniversalFeature(
            name="metabolic_signatures",
            feature_type=UniversalFeatureType.PHYSIOLOGICAL,
            universal_level=UniversalFeatureLevel.CLASS_LEVEL,
            description="Assinaturas metabólicas características",
            detection_methods=["physiological_analysis", "pattern_analysis"],
            weight=0.6,
            confidence_threshold=0.5
        )
        
        features["respiratory_patterns"] = UniversalFeature(
            name="respiratory_patterns",
            feature_type=UniversalFeatureType.PHYSIOLOGICAL,
            universal_level=UniversalFeatureLevel.CLASS_LEVEL,
            description="Padrões respiratórios característicos",
            detection_methods=["physiological_analysis", "motion_analysis"],
            weight=0.5,
            confidence_threshold=0.4
        )
        
        # Características evolutivas universais
        features["evolutionary_markers"] = UniversalFeature(
            name="evolutionary_markers",
            feature_type=UniversalFeatureType.EVOLUTIONARY,
            universal_level=UniversalFeatureLevel.FAMILY_LEVEL,
            description="Marcadores evolutivos característicos",
            detection_methods=["evolutionary_analysis", "pattern_analysis"],
            weight=0.7,
            confidence_threshold=0.6
        )
        
        features["adaptive_traits"] = UniversalFeature(
            name="adaptive_traits",
            feature_type=UniversalFeatureType.EVOLUTIONARY,
            universal_level=UniversalFeatureLevel.ORDER_LEVEL,
            description="Traços adaptativos característicos",
            detection_methods=["evolutionary_analysis", "morphological_analysis"],
            weight=0.6,
            confidence_threshold=0.5
        )
        
        return features
    
    def analyze_universal_features(self, detection_data: Dict[str, Any], species_context: str = "") -> Dict[str, Any]:
        """
        Analisa características universais em um objeto.
        
        Args:
            detection_data: Dados de detecção do objeto
            species_context: Contexto da espécie para análise
            
        Returns:
            Análise de características universais
        """
        analysis = {
            "universal_features_detected": {},
            "universality_score": 0.0,
            "transfer_potential": 0.0,
            "generalization_strength": 0.0,
            "feature_patterns": [],
            "universal_reasoning": [],
            "transfer_suggestions": []
        }
        
        total_weight = 0.0
        weighted_score = 0.0
        
        # Analisar cada característica universal
        for feature_name, feature in self.universal_features.items():
            feature_analysis = self._analyze_feature(feature, detection_data, species_context)
            analysis["universal_features_detected"][feature_name] = feature_analysis
            
            if feature_analysis["detected"]:
                weighted_score += feature.weight * feature_analysis["confidence"]
                total_weight += feature.weight
                
                # Adicionar ao raciocínio universal
                analysis["universal_reasoning"].append(
                    f"✅ {feature_name}: {feature_analysis['confidence']:.3f} "
                    f"({feature.universal_level.value})"
                )
            else:
                analysis["universal_reasoning"].append(
                    f"❌ {feature_name}: não detectado"
                )
        
        # Calcular score de universalidade
        if total_weight > 0:
            analysis["universality_score"] = weighted_score / total_weight
        else:
            analysis["universality_score"] = 0.0
        
        # Calcular potencial de transferência
        analysis["transfer_potential"] = self._calculate_transfer_potential(analysis)
        
        # Calcular força de generalização
        analysis["generalization_strength"] = self._calculate_generalization_strength(analysis)
        
        # Identificar padrões universais
        analysis["feature_patterns"] = self._identify_universal_patterns(analysis)
        
        # Gerar sugestões de transferência
        analysis["transfer_suggestions"] = self._generate_transfer_suggestions(analysis, species_context)
        
        return analysis
    
    def _analyze_feature(self, feature: UniversalFeature, detection_data: Dict[str, Any], species_context: str) -> Dict[str, Any]:
        """Analisa uma característica universal específica."""
        confidence = 0.0
        detected = False
        
        # Aplicar métodos de detecção específicos
        for method in feature.detection_methods:
            if method in detection_data:
                method_confidence = detection_data[method].get("confidence", 0.0)
                confidence = max(confidence, method_confidence)
        
        # Aplicar lógica específica para cada característica
        if feature.name == "bilateral_symmetry":
            confidence = self._detect_bilateral_symmetry(detection_data)
        elif feature.name == "head_structure":
            confidence = self._detect_head_structure(detection_data)
        elif feature.name == "limb_structure":
            confidence = self._detect_limb_structure(detection_data)
        elif feature.name == "body_axis":
            confidence = self._detect_body_axis(detection_data)
        elif feature.name == "movement_patterns":
            confidence = self._detect_movement_patterns(detection_data)
        elif feature.name == "feeding_behavior":
            confidence = self._detect_feeding_behavior(detection_data)
        elif feature.name == "habitat_adaptation":
            confidence = self._detect_habitat_adaptation(detection_data)
        elif feature.name == "ecological_niche":
            confidence = self._detect_ecological_niche(detection_data)
        elif feature.name == "metabolic_signatures":
            confidence = self._detect_metabolic_signatures(detection_data)
        elif feature.name == "respiratory_patterns":
            confidence = self._detect_respiratory_patterns(detection_data)
        elif feature.name == "evolutionary_markers":
            confidence = self._detect_evolutionary_markers(detection_data)
        elif feature.name == "adaptive_traits":
            confidence = self._detect_adaptive_traits(detection_data)
        
        detected = confidence >= feature.confidence_threshold
        
        return {
            "detected": detected,
            "confidence": confidence,
            "feature_type": feature.feature_type.value,
            "universal_level": feature.universal_level.value,
            "description": feature.description,
            "weight": feature.weight
        }
    
    def _detect_bilateral_symmetry(self, detection_data: Dict[str, Any]) -> float:
        """Detecta simetria bilateral."""
        confidence = 0.0
        
        # Análise de forma
        if "shape_analysis" in detection_data:
            shapes = detection_data["shape_analysis"].get("detected_shapes", [])
            if "symmetrical" in shapes or "balanced" in shapes:
                confidence += 0.6
        
        # Análise de simetria
        if "symmetry_analysis" in detection_data:
            symmetry_data = detection_data["symmetry_analysis"]
            if symmetry_data.get("bilateral_symmetry", False):
                confidence += 0.8
        
        return min(1.0, confidence)
    
    def _detect_head_structure(self, detection_data: Dict[str, Any]) -> float:
        """Detecta estrutura cefálica."""
        confidence = 0.0
        
        # Análise de forma
        if "shape_analysis" in detection_data:
            shapes = detection_data["shape_analysis"].get("detected_shapes", [])
            if "head" in shapes or "cephalic" in shapes:
                confidence += 0.7
        
        # Análise de posição
        if "position_analysis" in detection_data:
            positions = detection_data["position_analysis"].get("detected_positions", [])
            if "front" in positions or "head" in positions:
                confidence += 0.5
        
        return min(1.0, confidence)
    
    def _detect_limb_structure(self, detection_data: Dict[str, Any]) -> float:
        """Detecta estruturas locomotivas."""
        confidence = 0.0
        
        # Análise de forma
        if "shape_analysis" in detection_data:
            shapes = detection_data["shape_analysis"].get("detected_shapes", [])
            if "limb" in shapes or "appendage" in shapes or "leg" in shapes:
                confidence += 0.8
        
        # Análise de posição
        if "position_analysis" in detection_data:
            positions = detection_data["position_analysis"].get("detected_positions", [])
            if "side" in positions or "peripheral" in positions:
                confidence += 0.4
        
        return min(1.0, confidence)
    
    def _detect_body_axis(self, detection_data: Dict[str, Any]) -> float:
        """Detecta eixo corporal principal."""
        confidence = 0.0
        
        # Análise de forma
        if "shape_analysis" in detection_data:
            shapes = detection_data["shape_analysis"].get("detected_shapes", [])
            if "elongated" in shapes or "linear" in shapes:
                confidence += 0.6
        
        # Análise de tamanho
        if "size_analysis" in detection_data:
            size_data = detection_data["size_analysis"]
            aspect_ratio = size_data.get("aspect_ratio", 1.0)
            if aspect_ratio > 1.2:  # Corpo mais longo que largo
                confidence += 0.7
        
        return min(1.0, confidence)
    
    def _detect_movement_patterns(self, detection_data: Dict[str, Any]) -> float:
        """Detecta padrões de movimento."""
        confidence = 0.0
        
        # Análise de movimento
        if "motion_analysis" in detection_data:
            motion_data = detection_data["motion_analysis"]
            if motion_data.get("movement_detected", False):
                confidence += 0.8
        
        # Análise de padrão
        if "pattern_analysis" in detection_data:
            patterns = detection_data["pattern_analysis"].get("detected_patterns", [])
            if "rhythmic" in patterns or "periodic" in patterns:
                confidence += 0.6
        
        return min(1.0, confidence)
    
    def _detect_feeding_behavior(self, detection_data: Dict[str, Any]) -> float:
        """Detecta comportamento alimentar."""
        confidence = 0.0
        
        # Análise de comportamento
        if "behavior_analysis" in detection_data:
            behavior_data = detection_data["behavior_analysis"]
            if behavior_data.get("feeding_behavior", False):
                confidence += 0.9
        
        # Análise de padrão
        if "pattern_analysis" in detection_data:
            patterns = detection_data["pattern_analysis"].get("detected_patterns", [])
            if "foraging" in patterns or "hunting" in patterns:
                confidence += 0.7
        
        return min(1.0, confidence)
    
    def _detect_habitat_adaptation(self, detection_data: Dict[str, Any]) -> float:
        """Detecta adaptações ao habitat."""
        confidence = 0.0
        
        # Análise ambiental
        if "environment_analysis" in detection_data:
            env_data = detection_data["environment_analysis"]
            if env_data.get("habitat_adaptation", False):
                confidence += 0.8
        
        # Análise de padrão
        if "pattern_analysis" in detection_data:
            patterns = detection_data["pattern_analysis"].get("detected_patterns", [])
            if "camouflage" in patterns or "adaptation" in patterns:
                confidence += 0.6
        
        return min(1.0, confidence)
    
    def _detect_ecological_niche(self, detection_data: Dict[str, Any]) -> float:
        """Detecta nicho ecológico."""
        confidence = 0.0
        
        # Análise ambiental
        if "environment_analysis" in detection_data:
            env_data = detection_data["environment_analysis"]
            if env_data.get("ecological_niche", False):
                confidence += 0.9
        
        # Análise de comportamento
        if "behavior_analysis" in detection_data:
            behavior_data = detection_data["behavior_analysis"]
            if behavior_data.get("niche_behavior", False):
                confidence += 0.7
        
        return min(1.0, confidence)
    
    def _detect_metabolic_signatures(self, detection_data: Dict[str, Any]) -> float:
        """Detecta assinaturas metabólicas."""
        confidence = 0.0
        
        # Análise fisiológica
        if "physiological_analysis" in detection_data:
            phys_data = detection_data["physiological_analysis"]
            if phys_data.get("metabolic_signature", False):
                confidence += 0.8
        
        # Análise de padrão
        if "pattern_analysis" in detection_data:
            patterns = detection_data["pattern_analysis"].get("detected_patterns", [])
            if "metabolic" in patterns or "physiological" in patterns:
                confidence += 0.6
        
        return min(1.0, confidence)
    
    def _detect_respiratory_patterns(self, detection_data: Dict[str, Any]) -> float:
        """Detecta padrões respiratórios."""
        confidence = 0.0
        
        # Análise fisiológica
        if "physiological_analysis" in detection_data:
            phys_data = detection_data["physiological_analysis"]
            if phys_data.get("respiratory_pattern", False):
                confidence += 0.8
        
        # Análise de movimento
        if "motion_analysis" in detection_data:
            motion_data = detection_data["motion_analysis"]
            if motion_data.get("breathing_pattern", False):
                confidence += 0.7
        
        return min(1.0, confidence)
    
    def _detect_evolutionary_markers(self, detection_data: Dict[str, Any]) -> float:
        """Detecta marcadores evolutivos."""
        confidence = 0.0
        
        # Análise evolutiva
        if "evolutionary_analysis" in detection_data:
            evo_data = detection_data["evolutionary_analysis"]
            if evo_data.get("evolutionary_marker", False):
                confidence += 0.9
        
        # Análise de padrão
        if "pattern_analysis" in detection_data:
            patterns = detection_data["pattern_analysis"].get("detected_patterns", [])
            if "evolutionary" in patterns or "phylogenetic" in patterns:
                confidence += 0.7
        
        return min(1.0, confidence)
    
    def _detect_adaptive_traits(self, detection_data: Dict[str, Any]) -> float:
        """Detecta traços adaptativos."""
        confidence = 0.0
        
        # Análise evolutiva
        if "evolutionary_analysis" in detection_data:
            evo_data = detection_data["evolutionary_analysis"]
            if evo_data.get("adaptive_trait", False):
                confidence += 0.8
        
        # Análise morfológica
        if "morphological_analysis" in detection_data:
            morph_data = detection_data["morphological_analysis"]
            if morph_data.get("adaptive_feature", False):
                confidence += 0.7
        
        return min(1.0, confidence)
    
    def _calculate_transfer_potential(self, analysis: Dict[str, Any]) -> float:
        """Calcula o potencial de transferência de conhecimento."""
        detected_features = analysis["universal_features_detected"]
        
        # Contar características detectadas por nível de universalidade
        level_counts = defaultdict(int)
        level_weights = {
            UniversalFeatureLevel.KINGDOM_LEVEL: 1.0,
            UniversalFeatureLevel.CLASS_LEVEL: 0.8,
            UniversalFeatureLevel.ORDER_LEVEL: 0.6,
            UniversalFeatureLevel.FAMILY_LEVEL: 0.4,
            UniversalFeatureLevel.GENUS_LEVEL: 0.2,
            UniversalFeatureLevel.SPECIES_LEVEL: 0.1
        }
        
        for feature_name, feature_data in detected_features.items():
            if feature_data["detected"]:
                level = UniversalFeatureLevel(feature_data["universal_level"])
                level_counts[level] += 1
        
        # Calcular potencial de transferência
        transfer_potential = 0.0
        total_weight = 0.0
        
        for level, count in level_counts.items():
            weight = level_weights[level]
            transfer_potential += count * weight
            total_weight += weight
        
        if total_weight > 0:
            return transfer_potential / total_weight
        else:
            return 0.0
    
    def _calculate_generalization_strength(self, analysis: Dict[str, Any]) -> float:
        """Calcula a força de generalização."""
        detected_features = analysis["universal_features_detected"]
        
        # Calcular força baseada no número e tipo de características detectadas
        strength = 0.0
        total_weight = 0.0
        
        for feature_name, feature_data in detected_features.items():
            if feature_data["detected"]:
                feature_weight = feature_data["weight"]
                confidence = feature_data["confidence"]
                
                strength += feature_weight * confidence
                total_weight += feature_weight
        
        if total_weight > 0:
            return strength / total_weight
        else:
            return 0.0
    
    def _identify_universal_patterns(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifica padrões universais."""
        patterns = []
        detected_features = analysis["universal_features_detected"]
        
        # Padrão de simetria bilateral
        if (detected_features.get("bilateral_symmetry", {}).get("detected", False) and
            detected_features.get("head_structure", {}).get("detected", False)):
            patterns.append({
                "pattern_name": "bilateral_cephalization",
                "features": ["bilateral_symmetry", "head_structure"],
                "pattern_type": "morphological_universal",
                "confidence": 0.8,
                "universality_level": "kingdom_level"
            })
        
        # Padrão de locomoção
        if (detected_features.get("limb_structure", {}).get("detected", False) and
            detected_features.get("movement_patterns", {}).get("detected", False)):
            patterns.append({
                "pattern_name": "locomotive_adaptation",
                "features": ["limb_structure", "movement_patterns"],
                "pattern_type": "behavioral_morphological",
                "confidence": 0.7,
                "universality_level": "class_level"
            })
        
        # Padrão de adaptação ecológica
        if (detected_features.get("habitat_adaptation", {}).get("detected", False) and
            detected_features.get("adaptive_traits", {}).get("detected", False)):
            patterns.append({
                "pattern_name": "ecological_adaptation",
                "features": ["habitat_adaptation", "adaptive_traits"],
                "pattern_type": "ecological_evolutionary",
                "confidence": 0.6,
                "universality_level": "family_level"
            })
        
        return patterns
    
    def _generate_transfer_suggestions(self, analysis: Dict[str, Any], species_context: str) -> List[str]:
        """Gera sugestões de transferência de conhecimento."""
        suggestions = []
        detected_features = analysis["universal_features_detected"]
        
        # Sugestões baseadas em características detectadas
        if detected_features.get("bilateral_symmetry", {}).get("detected", False):
            suggestions.append(
                "Simetria bilateral detectada - conhecimento transferível para outros animais bilaterais"
            )
        
        if detected_features.get("head_structure", {}).get("detected", False):
            suggestions.append(
                "Estrutura cefálica detectada - padrões de cabeça aplicáveis a outras espécies"
            )
        
        if detected_features.get("limb_structure", {}).get("detected", False):
            suggestions.append(
                "Estruturas locomotivas detectadas - padrões de locomoção transferíveis"
            )
        
        if detected_features.get("movement_patterns", {}).get("detected", False):
            suggestions.append(
                "Padrões de movimento detectados - comportamentos locomotivos universais"
            )
        
        # Sugestões baseadas no contexto da espécie
        if species_context:
            suggestions.append(
                f"Características universais de {species_context} podem ser aplicadas a espécies relacionadas"
            )
        
        return suggestions
    
    def learn_from_universal_example(self, detection_data: Dict[str, Any], species_label: str, feedback: str = ""):
        """Aprende com um exemplo de características universais."""
        analysis = self.analyze_universal_features(detection_data, species_label)
        
        # Atualizar scores de universalidade
        for feature_name, feature_data in analysis["universal_features_detected"].items():
            if feature_data["detected"]:
                feature = self.universal_features[feature_name]
                
                # Atualizar score de universalidade
                feature.universality_score = min(1.0, feature.universality_score + 0.01)
                
                # Atualizar poder de transferência
                feature.transfer_power = min(1.0, feature.transfer_power + 0.005)
                
                # Atualizar força de generalização
                feature.generalization_strength = min(1.0, feature.generalization_strength + 0.005)
        
        # Registrar exemplo
        example = {
            "timestamp": time.time(),
            "species_label": species_label,
            "detection_data": detection_data,
            "analysis": analysis,
            "feedback": feedback
        }
        
        self.transfer_history.append(example)
        
        # Manter apenas os últimos 1000 exemplos
        if len(self.transfer_history) > 1000:
            self.transfer_history = self.transfer_history[-1000:]
        
        logger.info(f"Exemplo universal aprendido: {species_label}")
    
    def get_universal_analysis(self) -> Dict[str, Any]:
        """Retorna análise das características universais."""
        if not self.transfer_history:
            return {"message": "Nenhum exemplo de aprendizado universal disponível"}
        
        # Estatísticas dos exemplos
        total_examples = len(self.transfer_history)
        species_counts = defaultdict(int)
        
        for example in self.transfer_history:
            species_counts[example["species_label"]] += 1
        
        # Análise de características universais
        feature_analysis = {}
        
        for feature_name, feature in self.universal_features.items():
            feature_analysis[feature_name] = {
                "name": feature.name,
                "type": feature.feature_type.value,
                "universal_level": feature.universal_level.value,
                "weight": feature.weight,
                "universality_score": feature.universality_score,
                "transfer_power": feature.transfer_power,
                "generalization_strength": feature.generalization_strength,
                "confidence_threshold": feature.confidence_threshold
            }
        
        return {
            "total_examples": total_examples,
            "species_distribution": dict(species_counts),
            "feature_analysis": feature_analysis,
            "transfer_history_count": len(self.transfer_history),
            "universal_maturity": min(1.0, total_examples / 100)
        }

class UniversalKnowledgeTransfer:
    """Sistema de transferência de conhecimento universal."""
    
    def __init__(self, config_file: str = "data/universal_features.json"):
        self.config_file = config_file
        self.analyzer = UniversalFeatureAnalyzer()
        self.knowledge_base: Dict[str, Dict[str, Any]] = {}
        self.transfer_rules: List[Dict[str, Any]] = []
        
        # Carregar conhecimento existente
        self._load_knowledge()
        
        logger.info("Sistema de transferência de conhecimento universal inicializado")
    
    def _load_knowledge(self):
        """Carrega conhecimento universal existente."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                
                self.knowledge_base = data.get("knowledge_base", {})
                self.transfer_rules = data.get("transfer_rules", [])
                
                logger.info(f"Conhecimento universal carregado: {len(self.knowledge_base)} espécies")
                
            except Exception as e:
                logger.error(f"Erro ao carregar conhecimento universal: {e}")
    
    def _save_knowledge(self):
        """Salva conhecimento universal."""
        try:
            data = {
                "knowledge_base": self.knowledge_base,
                "transfer_rules": self.transfer_rules,
                "timestamp": time.time()
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=4)
            
            logger.info(f"Conhecimento universal salvo em {self.config_file}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar conhecimento universal: {e}")
    
    def analyze_with_universal_knowledge(self, detection_data: Dict[str, Any], target_species: str = "") -> Dict[str, Any]:
        """
        Analisa objeto usando conhecimento universal.
        
        Args:
            detection_data: Dados de detecção
            target_species: Espécie alvo para transferência
            
        Returns:
            Análise com conhecimento universal
        """
        # Analisar características universais
        universal_analysis = self.analyzer.analyze_universal_features(detection_data, target_species)
        
        # Aplicar transferência de conhecimento
        transfer_results = self._apply_knowledge_transfer(universal_analysis, target_species)
        
        # Combinar análises
        combined_analysis = {
            "universal_analysis": universal_analysis,
            "knowledge_transfer": transfer_results,
            "transfer_suggestions": universal_analysis["transfer_suggestions"],
            "generalization_potential": universal_analysis["generalization_strength"],
            "universal_reasoning": universal_analysis["universal_reasoning"]
        }
        
        return combined_analysis
    
    def _apply_knowledge_transfer(self, universal_analysis: Dict[str, Any], target_species: str) -> Dict[str, Any]:
        """Aplica transferência de conhecimento."""
        transfer_results = {
            "applicable_knowledge": [],
            "transfer_confidence": 0.0,
            "generalization_suggestions": [],
            "universal_patterns": universal_analysis["feature_patterns"]
        }
        
        # Encontrar conhecimento aplicável
        for species, knowledge in self.knowledge_base.items():
            if species != target_species:  # Não aplicar conhecimento da mesma espécie
                similarity = self._calculate_knowledge_similarity(universal_analysis, knowledge)
                
                if similarity > 0.5:  # Threshold de similaridade
                    transfer_results["applicable_knowledge"].append({
                        "source_species": species,
                        "similarity": similarity,
                        "knowledge": knowledge
                    })
        
        # Calcular confiança de transferência
        if transfer_results["applicable_knowledge"]:
            similarities = [k["similarity"] for k in transfer_results["applicable_knowledge"]]
            transfer_results["transfer_confidence"] = np.mean(similarities)
        
        # Gerar sugestões de generalização
        transfer_results["generalization_suggestions"] = self._generate_generalization_suggestions(
            transfer_results["applicable_knowledge"]
        )
        
        return transfer_results
    
    def _calculate_knowledge_similarity(self, analysis: Dict[str, Any], knowledge: Dict[str, Any]) -> float:
        """Calcula similaridade entre análises."""
        detected_features = analysis["universal_features_detected"]
        known_features = knowledge.get("universal_features", {})
        
        similarity = 0.0
        total_weight = 0.0
        
        for feature_name, feature_data in detected_features.items():
            if feature_data["detected"]:
                feature_weight = feature_data["weight"]
                
                if feature_name in known_features:
                    known_confidence = known_features[feature_name].get("confidence", 0.0)
                    feature_similarity = min(feature_data["confidence"], known_confidence)
                else:
                    feature_similarity = 0.0
                
                similarity += feature_weight * feature_similarity
                total_weight += feature_weight
        
        if total_weight > 0:
            return similarity / total_weight
        else:
            return 0.0
    
    def _generate_generalization_suggestions(self, applicable_knowledge: List[Dict[str, Any]]) -> List[str]:
        """Gera sugestões de generalização."""
        suggestions = []
        
        for knowledge_item in applicable_knowledge:
            source_species = knowledge_item["source_species"]
            similarity = knowledge_item["similarity"]
            
            suggestions.append(
                f"Conhecimento de {source_species} aplicável (similaridade: {similarity:.3f})"
            )
        
        return suggestions
    
    def learn_universal_pattern(self, detection_data: Dict[str, Any], species_label: str, feedback: str = ""):
        """Aprende padrão universal de uma espécie."""
        # Analisar características universais
        universal_analysis = self.analyzer.analyze_universal_features(detection_data, species_label)
        
        # Armazenar conhecimento da espécie
        self.knowledge_base[species_label] = {
            "species": species_label,
            "universal_features": universal_analysis["universal_features_detected"],
            "universality_score": universal_analysis["universality_score"],
            "transfer_potential": universal_analysis["transfer_potential"],
            "generalization_strength": universal_analysis["generalization_strength"],
            "feature_patterns": universal_analysis["feature_patterns"],
            "timestamp": time.time(),
            "feedback": feedback
        }
        
        # Aprender com o analisador
        self.analyzer.learn_from_universal_example(detection_data, species_label, feedback)
        
        # Salvar conhecimento
        self._save_knowledge()
        
        logger.info(f"Padrão universal aprendido para {species_label}")
    
    def get_universal_knowledge_analysis(self) -> Dict[str, Any]:
        """Retorna análise do conhecimento universal."""
        analyzer_analysis = self.analyzer.get_universal_analysis()
        
        return {
            "knowledge_base_size": len(self.knowledge_base),
            "transfer_rules_count": len(self.transfer_rules),
            "species_in_knowledge_base": list(self.knowledge_base.keys()),
            "analyzer_analysis": analyzer_analysis,
            "universal_maturity": analyzer_analysis.get("universal_maturity", 0.0)
        }
