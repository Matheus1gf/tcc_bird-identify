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

# Importar sistema de características universais
try:
    from .universal_features import UniversalKnowledgeTransfer, UniversalFeatureAnalyzer, UniversalFeatureType, UniversalFeatureLevel
    UNIVERSAL_FEATURES_AVAILABLE = True
except ImportError:
    UNIVERSAL_FEATURES_AVAILABLE = False
    logger.warning("Sistema de características universais não disponível")

# Importar sistema de hierarquias de conceitos
try:
    from .concept_hierarchy import ConceptHierarchyManager, HierarchicalConcept, ConceptRelationshipType, ConceptAbstractionLevel, ConceptComplexity
    CONCEPT_HIERARCHY_AVAILABLE = True
except ImportError:
    CONCEPT_HIERARCHY_AVAILABLE = False
    logger.warning("Sistema de hierarquias de conceitos não disponível")

class ConceptType(Enum):
    """Tipos de conceitos abstratos."""
    ESSENTIAL = "essential"  # Características essenciais (bico, penas, etc.)
    BEHAVIORAL = "behavioral"  # Comportamentos característicos
    MORPHOLOGICAL = "morphological"  # Características morfológicas
    ECOLOGICAL = "ecological"  # Características ecológicas
    COMPOSITE = "composite"  # Conceitos compostos

class ConceptLevel(Enum):
    """Níveis de abstração dos conceitos."""
    BASIC = "basic"  # Conceitos básicos (bico, penas)
    INTERMEDIATE = "intermediate"  # Conceitos intermediários (passarinidade)
    ADVANCED = "advanced"  # Conceitos avançados (adaptações específicas)
    META = "meta"  # Meta-conceitos (evolução, nicho ecológico)

@dataclass
class AbstractConcept:
    """Representa um conceito abstrato."""
    name: str
    concept_type: ConceptType
    level: ConceptLevel
    definition: str
    essential_features: List[str]
    supporting_features: List[str]
    confidence_threshold: float
    learning_examples: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0
    generalization_power: float = 0.0
    abstraction_depth: int = 0

@dataclass
class ConceptHierarchy:
    """Hierarquia de conceitos abstratos."""
    root_concept: str
    sub_concepts: Dict[str, List[str]] = field(default_factory=dict)
    concept_relationships: Dict[str, Dict[str, float]] = field(default_factory=dict)
    abstraction_levels: Dict[str, ConceptLevel] = field(default_factory=dict)

class PassarinidadeAnalyzer:
    """Analisador específico para o conceito de 'passarinidade'."""
    
    def __init__(self):
        self.essential_features = {
            "bico": {
                "weight": 0.3,
                "description": "Estrutura córnea característica das aves",
                "detection_methods": ["shape_analysis", "texture_analysis", "color_analysis"]
            },
            "penas": {
                "weight": 0.25,
                "description": "Estruturas queratinosas que cobrem o corpo",
                "detection_methods": ["texture_analysis", "pattern_analysis", "color_analysis"]
            },
            "garras": {
                "weight": 0.2,
                "description": "Estruturas córneas nas extremidades dos dedos",
                "detection_methods": ["shape_analysis", "texture_analysis"]
            },
            "olhos": {
                "weight": 0.15,
                "description": "Órgãos visuais característicos",
                "detection_methods": ["shape_analysis", "color_analysis", "position_analysis"]
            },
            "proporcoes_corporais": {
                "weight": 0.1,
                "description": "Proporções características do corpo de aves",
                "detection_methods": ["shape_analysis", "size_analysis"]
            }
        }
        
        self.supporting_features = {
            "asas": {
                "weight": 0.15,
                "description": "Estruturas para voo",
                "detection_methods": ["shape_analysis", "texture_analysis"]
            },
            "cauda": {
                "weight": 0.1,
                "description": "Estrutura caudal",
                "detection_methods": ["shape_analysis", "texture_analysis"]
            },
            "patas": {
                "weight": 0.1,
                "description": "Estruturas locomotivas",
                "detection_methods": ["shape_analysis", "texture_analysis"]
            },
            "padroes_coloridos": {
                "weight": 0.1,
                "description": "Padrões de coloração característicos",
                "detection_methods": ["color_analysis", "pattern_analysis"]
            }
        }
        
        self.learning_examples = []
        self.confidence_history = deque(maxlen=100)
        
    def analyze_passarinidade(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa se um objeto possui características de passarinidade.
        
        Args:
            detection_data: Dados de detecção do objeto
            
        Returns:
            Análise de passarinidade
        """
        analysis = {
            "is_bird": False,
            "passarinidade_score": 0.0,
            "essential_features_detected": {},
            "supporting_features_detected": {},
            "confidence": 0.0,
            "reasoning": [],
            "missing_features": [],
            "concept_strength": 0.0
        }
        
        # Analisar características essenciais
        essential_score = 0.0
        essential_weight_sum = 0.0
        
        for feature, config in self.essential_features.items():
            feature_detected = self._detect_feature(feature, detection_data, config)
            analysis["essential_features_detected"][feature] = feature_detected
            
            if feature_detected["detected"]:
                essential_score += config["weight"] * feature_detected["confidence"]
                analysis["reasoning"].append(f"✅ {feature}: {feature_detected['confidence']:.3f}")
            else:
                analysis["missing_features"].append(feature)
                analysis["reasoning"].append(f"❌ {feature}: não detectado")
            
            essential_weight_sum += config["weight"]
        
        # Analisar características de apoio
        supporting_score = 0.0
        supporting_weight_sum = 0.0
        
        for feature, config in self.supporting_features.items():
            feature_detected = self._detect_feature(feature, detection_data, config)
            analysis["supporting_features_detected"][feature] = feature_detected
            
            if feature_detected["detected"]:
                supporting_score += config["weight"] * feature_detected["confidence"]
                analysis["reasoning"].append(f"➕ {feature}: {feature_detected['confidence']:.3f}")
            
            supporting_weight_sum += config["weight"]
        
        # Calcular score de passarinidade
        if essential_weight_sum > 0:
            essential_normalized = essential_score / essential_weight_sum
        else:
            essential_normalized = 0.0
            
        if supporting_weight_sum > 0:
            supporting_normalized = supporting_score / supporting_weight_sum
        else:
            supporting_normalized = 0.0
        
        # Score final (70% essenciais + 30% apoio)
        analysis["passarinidade_score"] = (essential_normalized * 0.7) + (supporting_normalized * 0.3)
        
        # Determinar se é um pássaro
        analysis["is_bird"] = analysis["passarinidade_score"] >= 0.6
        
        # Calcular confiança
        analysis["confidence"] = self._calculate_confidence(analysis)
        
        # Calcular força do conceito
        analysis["concept_strength"] = self._calculate_concept_strength(analysis)
        
        # Registrar exemplo para aprendizado
        self._record_learning_example(detection_data, analysis)
        
        return analysis
    
    def _detect_feature(self, feature_name: str, detection_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Detecta uma característica específica."""
        detection_methods = config["detection_methods"]
        confidence = 0.0
        detected = False
        
        # Simular detecção baseada nos métodos disponíveis
        for method in detection_methods:
            if method in detection_data:
                method_confidence = detection_data[method].get("confidence", 0.0)
                confidence = max(confidence, method_confidence)
        
        # Aplicar lógica específica para cada característica
        if feature_name == "bico":
            confidence = self._detect_bico(detection_data)
        elif feature_name == "penas":
            confidence = self._detect_penas(detection_data)
        elif feature_name == "garras":
            confidence = self._detect_garras(detection_data)
        elif feature_name == "olhos":
            confidence = self._detect_olhos(detection_data)
        elif feature_name == "proporcoes_corporais":
            confidence = self._detect_proporcoes(detection_data)
        elif feature_name == "asas":
            confidence = self._detect_asas(detection_data)
        elif feature_name == "cauda":
            confidence = self._detect_cauda(detection_data)
        elif feature_name == "patas":
            confidence = self._detect_patas(detection_data)
        elif feature_name == "padroes_coloridos":
            confidence = self._detect_padroes_coloridos(detection_data)
        
        detected = confidence >= 0.3  # Threshold mínimo para detecção
        
        return {
            "detected": detected,
            "confidence": confidence,
            "method": detection_methods,
            "description": config["description"]
        }
    
    def _detect_bico(self, detection_data: Dict[str, Any]) -> float:
        """Detecta características de bico."""
        confidence = 0.0
        
        # Análise de forma
        if "shape_analysis" in detection_data:
            shapes = detection_data["shape_analysis"].get("detected_shapes", [])
            if "beak" in shapes or "pointed" in shapes:
                confidence += 0.4
        
        # Análise de textura
        if "texture_analysis" in detection_data:
            textures = detection_data["texture_analysis"].get("detected_textures", [])
            if "smooth" in textures or "hard" in textures:
                confidence += 0.3
        
        # Análise de cor
        if "color_analysis" in detection_data:
            colors = detection_data["color_analysis"].get("detected_colors", [])
            if any(color in ["yellow", "orange", "red", "black", "brown"] for color in colors):
                confidence += 0.3
        
        return min(1.0, confidence)
    
    def _detect_penas(self, detection_data: Dict[str, Any]) -> float:
        """Detecta características de penas."""
        confidence = 0.0
        
        # Análise de textura
        if "texture_analysis" in detection_data:
            textures = detection_data["texture_analysis"].get("detected_textures", [])
            if "feathery" in textures or "soft" in textures:
                confidence += 0.5
        
        # Análise de padrão
        if "pattern_analysis" in detection_data:
            patterns = detection_data["pattern_analysis"].get("detected_patterns", [])
            if "feather_pattern" in patterns or "overlapping" in patterns:
                confidence += 0.4
        
        # Análise de cor
        if "color_analysis" in detection_data:
            colors = detection_data["color_analysis"].get("detected_colors", [])
            if len(colors) > 2:  # Penas geralmente têm múltiplas cores
                confidence += 0.3
        
        return min(1.0, confidence)
    
    def _detect_garras(self, detection_data: Dict[str, Any]) -> float:
        """Detecta características de garras."""
        confidence = 0.0
        
        # Análise de forma
        if "shape_analysis" in detection_data:
            shapes = detection_data["shape_analysis"].get("detected_shapes", [])
            if "claw" in shapes or "curved" in shapes:
                confidence += 0.6
        
        # Análise de textura
        if "texture_analysis" in detection_data:
            textures = detection_data["texture_analysis"].get("detected_textures", [])
            if "hard" in textures or "sharp" in textures:
                confidence += 0.4
        
        return min(1.0, confidence)
    
    def _detect_olhos(self, detection_data: Dict[str, Any]) -> float:
        """Detecta características de olhos."""
        confidence = 0.0
        
        # Análise de forma
        if "shape_analysis" in detection_data:
            shapes = detection_data["shape_analysis"].get("detected_shapes", [])
            if "circular" in shapes or "oval" in shapes:
                confidence += 0.4
        
        # Análise de cor
        if "color_analysis" in detection_data:
            colors = detection_data["color_analysis"].get("detected_colors", [])
            if "black" in colors or "brown" in colors or "yellow" in colors:
                confidence += 0.3
        
        # Análise de posição
        if "position_analysis" in detection_data:
            positions = detection_data["position_analysis"].get("detected_positions", [])
            if "head" in positions or "front" in positions:
                confidence += 0.3
        
        return min(1.0, confidence)
    
    def _detect_proporcoes(self, detection_data: Dict[str, Any]) -> float:
        """Detecta proporções corporais características."""
        confidence = 0.0
        
        # Análise de forma
        if "shape_analysis" in detection_data:
            shapes = detection_data["shape_analysis"].get("detected_shapes", [])
            if "compact" in shapes or "streamlined" in shapes:
                confidence += 0.5
        
        # Análise de tamanho
        if "size_analysis" in detection_data:
            size_data = detection_data["size_analysis"]
            if size_data.get("aspect_ratio", 0) > 1.2:  # Corpo mais longo que largo
                confidence += 0.5
        
        return min(1.0, confidence)
    
    def _detect_asas(self, detection_data: Dict[str, Any]) -> float:
        """Detecta características de asas."""
        confidence = 0.0
        
        # Análise de forma
        if "shape_analysis" in detection_data:
            shapes = detection_data["shape_analysis"].get("detected_shapes", [])
            if "wing" in shapes or "extended" in shapes:
                confidence += 0.6
        
        # Análise de textura
        if "texture_analysis" in detection_data:
            textures = detection_data["texture_analysis"].get("detected_textures", [])
            if "feathery" in textures:
                confidence += 0.4
        
        return min(1.0, confidence)
    
    def _detect_cauda(self, detection_data: Dict[str, Any]) -> float:
        """Detecta características de cauda."""
        confidence = 0.0
        
        # Análise de forma
        if "shape_analysis" in detection_data:
            shapes = detection_data["shape_analysis"].get("detected_shapes", [])
            if "tail" in shapes or "fan" in shapes:
                confidence += 0.6
        
        # Análise de textura
        if "texture_analysis" in detection_data:
            textures = detection_data["texture_analysis"].get("detected_textures", [])
            if "feathery" in textures:
                confidence += 0.4
        
        return min(1.0, confidence)
    
    def _detect_patas(self, detection_data: Dict[str, Any]) -> float:
        """Detecta características de patas."""
        confidence = 0.0
        
        # Análise de forma
        if "shape_analysis" in detection_data:
            shapes = detection_data["shape_analysis"].get("detected_shapes", [])
            if "leg" in shapes or "thin" in shapes:
                confidence += 0.5
        
        # Análise de textura
        if "texture_analysis" in detection_data:
            textures = detection_data["texture_analysis"].get("detected_textures", [])
            if "scaly" in textures or "rough" in textures:
                confidence += 0.5
        
        return min(1.0, confidence)
    
    def _detect_padroes_coloridos(self, detection_data: Dict[str, Any]) -> float:
        """Detecta padrões de coloração característicos."""
        confidence = 0.0
        
        # Análise de cor
        if "color_analysis" in detection_data:
            colors = detection_data["color_analysis"].get("detected_colors", [])
            if len(colors) >= 2:  # Múltiplas cores
                confidence += 0.4
        
        # Análise de padrão
        if "pattern_analysis" in detection_data:
            patterns = detection_data["pattern_analysis"].get("detected_patterns", [])
            if "striped" in patterns or "spotted" in patterns or "gradient" in patterns:
                confidence += 0.6
        
        return min(1.0, confidence)
    
    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calcula a confiança da análise."""
        passarinidade_score = analysis["passarinidade_score"]
        essential_count = len([f for f in analysis["essential_features_detected"].values() if f["detected"]])
        supporting_count = len([f for f in analysis["supporting_features_detected"].values() if f["detected"]])
        
        # Confiança baseada no score e número de características detectadas
        confidence = passarinidade_score * 0.7
        
        # Bonus por características essenciais detectadas
        if essential_count >= 3:
            confidence += 0.2
        elif essential_count >= 2:
            confidence += 0.1
        
        # Bonus por características de apoio
        if supporting_count >= 2:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _calculate_concept_strength(self, analysis: Dict[str, Any]) -> float:
        """Calcula a força do conceito de passarinidade."""
        essential_strength = 0.0
        supporting_strength = 0.0
        
        # Força das características essenciais
        for feature, detected in analysis["essential_features_detected"].items():
            if detected["detected"]:
                essential_strength += detected["confidence"] * self.essential_features[feature]["weight"]
        
        # Força das características de apoio
        for feature, detected in analysis["supporting_features_detected"].items():
            if detected["detected"]:
                supporting_strength += detected["confidence"] * self.supporting_features[feature]["weight"]
        
        # Força total do conceito
        total_strength = essential_strength + (supporting_strength * 0.5)
        
        return min(1.0, total_strength)
    
    def _record_learning_example(self, detection_data: Dict[str, Any], analysis: Dict[str, Any]):
        """Registra exemplo para aprendizado."""
        example = {
            "timestamp": time.time(),
            "detection_data": detection_data,
            "analysis": analysis,
            "passarinidade_score": analysis["passarinidade_score"],
            "is_bird": analysis["is_bird"],
            "confidence": analysis["confidence"]
        }
        
        self.learning_examples.append(example)
        
        # Manter apenas os últimos 1000 exemplos
        if len(self.learning_examples) > 1000:
            self.learning_examples = self.learning_examples[-1000:]
        
        # Atualizar histórico de confiança
        self.confidence_history.append(analysis["confidence"])
    
    def learn_from_feedback(self, example: Dict[str, Any], correct_label: bool):
        """Aprende com feedback humano."""
        if not self.learning_examples:
            return
        
        # Encontrar exemplo mais recente similar
        recent_example = self.learning_examples[-1]
        
        # Ajustar pesos baseado no feedback
        if correct_label != recent_example["is_bird"]:
            # Feedback contradiz a análise - ajustar pesos
            self._adjust_feature_weights(recent_example, correct_label)
    
    def _adjust_feature_weights(self, example: Dict[str, Any], correct_label: bool):
        """Ajusta pesos das características baseado no feedback."""
        adjustment_factor = 0.1 if correct_label else -0.1
        
        # Ajustar pesos das características essenciais
        for feature, config in self.essential_features.items():
            if feature in example["analysis"]["essential_features_detected"]:
                feature_data = example["analysis"]["essential_features_detected"][feature]
                if feature_data["detected"]:
                    # Aumentar peso se foi detectado e deveria ser pássaro
                    config["weight"] = max(0.01, min(0.5, config["weight"] + adjustment_factor))
        
        # Ajustar pesos das características de apoio
        for feature, config in self.supporting_features.items():
            if feature in example["analysis"]["supporting_features_detected"]:
                feature_data = example["analysis"]["supporting_features_detected"][feature]
                if feature_data["detected"]:
                    # Aumentar peso se foi detectado e deveria ser pássaro
                    config["weight"] = max(0.01, min(0.3, config["weight"] + adjustment_factor))
    
    def get_concept_analysis(self) -> Dict[str, Any]:
        """Retorna análise do conceito de passarinidade."""
        if not self.learning_examples:
            return {"message": "Nenhum exemplo de aprendizado disponível"}
        
        # Estatísticas dos exemplos
        total_examples = len(self.learning_examples)
        bird_examples = len([ex for ex in self.learning_examples if ex["is_bird"]])
        non_bird_examples = total_examples - bird_examples
        
        # Médias de confiança
        avg_confidence = np.mean([ex["confidence"] for ex in self.learning_examples])
        avg_passarinidade_score = np.mean([ex["passarinidade_score"] for ex in self.learning_examples])
        
        # Análise de características mais importantes
        feature_importance = {}
        
        for feature, config in self.essential_features.items():
            feature_examples = [ex for ex in self.learning_examples 
                              if feature in ex["analysis"]["essential_features_detected"] 
                              and ex["analysis"]["essential_features_detected"][feature]["detected"]]
            
            if feature_examples:
                avg_confidence_feature = np.mean([ex["confidence"] for ex in feature_examples])
                feature_importance[feature] = {
                    "weight": config["weight"],
                    "detection_rate": len(feature_examples) / total_examples,
                    "avg_confidence": avg_confidence_feature,
                    "importance_score": config["weight"] * avg_confidence_feature
                }
        
        return {
            "total_examples": total_examples,
            "bird_examples": bird_examples,
            "non_bird_examples": non_bird_examples,
            "avg_confidence": avg_confidence,
            "avg_passarinidade_score": avg_passarinidade_score,
            "feature_importance": feature_importance,
            "confidence_history": list(self.confidence_history),
            "concept_maturity": min(1.0, total_examples / 100)  # Maturidade baseada em exemplos
        }

class AbstractConceptLearner:
    """Sistema principal de aprendizado de conceitos abstratos."""
    
    def __init__(self, config_file: str = "data/abstract_concepts.json"):
        self.config_file = config_file
        self.concepts: Dict[str, AbstractConcept] = {}
        self.concept_hierarchy = ConceptHierarchy(root_concept="passarinidade")
        
        # Inicializar analisador de passarinidade
        self.passarinidade_analyzer = PassarinidadeAnalyzer()
        
        # Inicializar sistema de características universais
        self.universal_knowledge_transfer = None
        if UNIVERSAL_FEATURES_AVAILABLE:
            try:
                self.universal_knowledge_transfer = UniversalKnowledgeTransfer()
                logger.info("Sistema de características universais inicializado")
            except Exception as e:
                logger.error(f"Erro ao inicializar sistema de características universais: {e}")
                self.universal_knowledge_transfer = None
        
        # Inicializar sistema de hierarquias de conceitos
        self.concept_hierarchy_manager = None
        if CONCEPT_HIERARCHY_AVAILABLE:
            try:
                self.concept_hierarchy_manager = ConceptHierarchyManager()
                logger.info("Sistema de hierarquias de conceitos inicializado")
            except Exception as e:
                logger.error(f"Erro ao inicializar sistema de hierarquias de conceitos: {e}")
                self.concept_hierarchy_manager = None
        
        # Carregar conceitos existentes
        self._load_concepts()
        
        # Inicializar conceito de passarinidade se não existir
        if "passarinidade" not in self.concepts:
            self._initialize_passarinidade_concept()
        
        logger.info("Sistema de aprendizado de conceitos abstratos inicializado")
    
    def _load_concepts(self):
        """Carrega conceitos existentes."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                
                for concept_name, concept_data in data.get("concepts", {}).items():
                    self.concepts[concept_name] = AbstractConcept(
                        name=concept_name,
                        concept_type=ConceptType(concept_data["concept_type"]),
                        level=ConceptLevel(concept_data["level"]),
                        definition=concept_data["definition"],
                        essential_features=concept_data["essential_features"],
                        supporting_features=concept_data["supporting_features"],
                        confidence_threshold=concept_data["confidence_threshold"],
                        learning_examples=concept_data.get("learning_examples", []),
                        confidence_score=concept_data.get("confidence_score", 0.0),
                        generalization_power=concept_data.get("generalization_power", 0.0),
                        abstraction_depth=concept_data.get("abstraction_depth", 0)
                    )
                
                logger.info(f"Conceitos carregados: {len(self.concepts)} conceitos")
                
            except Exception as e:
                logger.error(f"Erro ao carregar conceitos: {e}")
    
    def _save_concepts(self):
        """Salva conceitos."""
        try:
            concepts_data = {}
            for name, concept in self.concepts.items():
                concept_dict = asdict(concept)
                # Converter enums para strings
                concept_dict["concept_type"] = concept.concept_type.value
                concept_dict["level"] = concept.level.value
                concepts_data[name] = concept_dict
            
            # Converter hierarquia também
            hierarchy_dict = asdict(self.concept_hierarchy)
            # Converter enums na hierarquia
            for concept_name, level in hierarchy_dict.get("abstraction_levels", {}).items():
                if hasattr(level, 'value'):
                    hierarchy_dict["abstraction_levels"][concept_name] = level.value
            
            data = {
                "concepts": concepts_data,
                "hierarchy": hierarchy_dict
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=4)
            
            logger.info(f"Conceitos salvos em {self.config_file}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar conceitos: {e}")
    
    def _initialize_passarinidade_concept(self):
        """Inicializa o conceito de passarinidade."""
        passarinidade = AbstractConcept(
            name="passarinidade",
            concept_type=ConceptType.ESSENTIAL,
            level=ConceptLevel.INTERMEDIATE,
            definition="Conjunto de características essenciais que definem um pássaro",
            essential_features=["bico", "penas", "garras", "olhos", "proporcoes_corporais"],
            supporting_features=["asas", "cauda", "patas", "padroes_coloridos"],
            confidence_threshold=0.6,
            confidence_score=0.0,
            generalization_power=0.0,
            abstraction_depth=2
        )
        
        self.concepts["passarinidade"] = passarinidade
        
        # Adicionar à hierarquia
        self.concept_hierarchy.abstraction_levels["passarinidade"] = ConceptLevel.INTERMEDIATE
        
        logger.info("Conceito de passarinidade inicializado")
    
    def analyze_object(self, detection_data: Dict[str, Any], species_context: str = "") -> Dict[str, Any]:
        """
        Analisa um objeto usando conceitos abstratos e características universais.
        
        Args:
            detection_data: Dados de detecção do objeto
            species_context: Contexto da espécie para análise
            
        Returns:
            Análise usando conceitos abstratos e características universais
        """
        analysis = {
            "object_analysis": {},
            "concept_applications": {},
            "universal_analysis": {},
            "abstract_reasoning": [],
            "generalization_suggestions": [],
            "universal_reasoning": []
        }
        
        # Aplicar conceito de passarinidade
        passarinidade_analysis = self.passarinidade_analyzer.analyze_passarinidade(detection_data)
        analysis["concept_applications"]["passarinidade"] = passarinidade_analysis
        
        # Aplicar análise de características universais
        if self.universal_knowledge_transfer:
            universal_analysis = self.universal_knowledge_transfer.analyze_with_universal_knowledge(
                detection_data, species_context
            )
            analysis["universal_analysis"] = universal_analysis
            
            # Adicionar raciocínio universal
            analysis["universal_reasoning"] = universal_analysis.get("universal_reasoning", [])
            
            # Adicionar sugestões de transferência
            transfer_suggestions = universal_analysis.get("transfer_suggestions", [])
            analysis["generalization_suggestions"].extend(transfer_suggestions)
        
        # Aplicar análise hierárquica de conceitos
        if self.concept_hierarchy_manager:
            hierarchical_analysis = self._analyze_with_hierarchy(detection_data, species_context)
            analysis["hierarchical_analysis"] = hierarchical_analysis
            
            # Adicionar raciocínio hierárquico
            analysis["hierarchical_reasoning"] = hierarchical_analysis.get("hierarchical_reasoning", [])
            
            # Adicionar sugestões hierárquicas
            hierarchical_suggestions = hierarchical_analysis.get("hierarchical_suggestions", [])
            analysis["generalization_suggestions"].extend(hierarchical_suggestions)
        
        # Raciocínio abstrato
        if passarinidade_analysis["is_bird"]:
            analysis["abstract_reasoning"].append(
                f"Objeto possui características essenciais de passarinidade "
                f"(score: {passarinidade_analysis['passarinidade_score']:.3f})"
            )
            
            # Sugestões de generalização
            if passarinidade_analysis["passarinidade_score"] > 0.8:
                analysis["generalization_suggestions"].append(
                    "Alta confiança em passarinidade - possível generalização para outras aves"
                )
        else:
            analysis["abstract_reasoning"].append(
                f"Objeto não possui características suficientes de passarinidade "
                f"(score: {passarinidade_analysis['passarinidade_score']:.3f})"
            )
            
            # Analisar características faltantes
            missing_features = passarinidade_analysis["missing_features"]
            if missing_features:
                analysis["abstract_reasoning"].append(
                    f"Características essenciais faltantes: {', '.join(missing_features)}"
                )
        
        # Análise geral do objeto
        analysis["object_analysis"] = {
            "detected_features": list(detection_data.keys()),
            "analysis_confidence": passarinidade_analysis["confidence"],
            "concept_strength": passarinidade_analysis["concept_strength"]
        }
        
        # Adicionar informações universais se disponíveis
        if self.universal_knowledge_transfer and "universal_analysis" in analysis:
            universal_data = analysis["universal_analysis"]
            analysis["object_analysis"]["universality_score"] = universal_data.get("universal_analysis", {}).get("universality_score", 0.0)
            analysis["object_analysis"]["transfer_potential"] = universal_data.get("universal_analysis", {}).get("transfer_potential", 0.0)
            analysis["object_analysis"]["generalization_strength"] = universal_data.get("universal_analysis", {}).get("generalization_strength", 0.0)
        
        return analysis
    
    def _analyze_with_hierarchy(self, detection_data: Dict[str, Any], species_context: str = "") -> Dict[str, Any]:
        """
        Analisa objeto usando hierarquia de conceitos.
        
        Args:
            detection_data: Dados de detecção
            species_context: Contexto da espécie
            
        Returns:
            Análise hierárquica
        """
        if not self.concept_hierarchy_manager:
            return {"error": "Sistema de hierarquias não disponível"}
        
        analysis = {
            "hierarchical_reasoning": [],
            "hierarchical_suggestions": [],
            "concept_matches": [],
            "inheritance_analysis": {},
            "similarity_analysis": {}
        }
        
        # Determinar conceito mais provável baseado nos dados
        most_likely_concept = self._determine_most_likely_concept(detection_data, species_context)
        
        if most_likely_concept:
            analysis["concept_matches"].append({
                "concept": most_likely_concept,
                "confidence": 0.8,
                "reasoning": f"Características detectadas sugerem {most_likely_concept}"
            })
            
            # Análise hierárquica do conceito
            hierarchy_analysis = self.concept_hierarchy_manager.analyze_concept_hierarchy(most_likely_concept)
            analysis["inheritance_analysis"] = hierarchy_analysis
            
            # Raciocínio hierárquico
            analysis["hierarchical_reasoning"].append(
                f"Conceito identificado: {most_likely_concept} (nível: {hierarchy_analysis.get('abstraction_level', 'unknown')})"
            )
            
            # Análise de similaridade
            similar_concepts = self.concept_hierarchy_manager.find_similar_concepts(most_likely_concept)
            analysis["similarity_analysis"] = {
                "similar_concepts": similar_concepts,
                "similarity_count": len(similar_concepts)
            }
            
            # Sugestões hierárquicas
            if hierarchy_analysis.get("ancestors"):
                analysis["hierarchical_suggestions"].append(
                    f"Conceito herda características de: {', '.join([a['name'] for a in hierarchy_analysis['ancestors'][:3]])}"
                )
            
            if similar_concepts:
                analysis["hierarchical_suggestions"].append(
                    f"Conceitos similares encontrados: {', '.join([s['name'] for s in similar_concepts[:3]])}"
                )
        
        return analysis
    
    def _determine_most_likely_concept(self, detection_data: Dict[str, Any], species_context: str = "") -> Optional[str]:
        """
        Determina o conceito mais provável baseado nos dados de detecção.
        
        Args:
            detection_data: Dados de detecção
            species_context: Contexto da espécie
            
        Returns:
            Nome do conceito mais provável
        """
        if not self.concept_hierarchy_manager:
            return None
        
        # Mapear características detectadas para conceitos
        concept_scores = {}
        
        # Análise de forma
        if "shape_analysis" in detection_data:
            shapes = detection_data["shape_analysis"].get("detected_shapes", [])
            
            if "beak" in shapes:
                concept_scores["bird"] = concept_scores.get("bird", 0) + 0.3
                concept_scores["passerine"] = concept_scores.get("passerine", 0) + 0.2
                concept_scores["raptor"] = concept_scores.get("raptor", 0) + 0.2
                concept_scores["waterfowl"] = concept_scores.get("waterfowl", 0) + 0.2
            
            if "wing" in shapes:
                concept_scores["bird"] = concept_scores.get("bird", 0) + 0.3
                concept_scores["eagle"] = concept_scores.get("eagle", 0) + 0.2
            
            if "compact" in shapes:
                concept_scores["passerine"] = concept_scores.get("passerine", 0) + 0.2
                concept_scores["blue_jay"] = concept_scores.get("blue_jay", 0) + 0.1
        
        # Análise de cor
        if "color_analysis" in detection_data:
            colors = detection_data["color_analysis"].get("detected_colors", [])
            
            if "blue" in colors:
                concept_scores["blue_jay"] = concept_scores.get("blue_jay", 0) + 0.4
            
            if "brown" in colors:
                concept_scores["eagle"] = concept_scores.get("eagle", 0) + 0.3
                concept_scores["raptor"] = concept_scores.get("raptor", 0) + 0.2
        
        # Análise de textura
        if "texture_analysis" in detection_data:
            textures = detection_data["texture_analysis"].get("detected_textures", [])
            
            if "feathery" in textures:
                concept_scores["bird"] = concept_scores.get("bird", 0) + 0.4
                concept_scores["passerine"] = concept_scores.get("passerine", 0) + 0.3
                concept_scores["raptor"] = concept_scores.get("raptor", 0) + 0.3
                concept_scores["waterfowl"] = concept_scores.get("waterfowl", 0) + 0.3
        
        # Usar contexto da espécie se disponível
        if species_context and species_context in self.concept_hierarchy_manager.concepts:
            concept_scores[species_context] = concept_scores.get(species_context, 0) + 0.5
        
        # Retornar conceito com maior score
        if concept_scores:
            return max(concept_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def learn_from_example(self, detection_data: Dict[str, Any], correct_label: str, feedback: str = "", species_context: str = ""):
        """
        Aprende com um exemplo fornecido, incluindo características universais.
        
        Args:
            detection_data: Dados de detecção
            correct_label: Rótulo correto ("bird" ou "not_bird")
            feedback: Feedback adicional
            species_context: Contexto da espécie para aprendizado universal
        """
        # Analisar objeto
        analysis = self.analyze_object(detection_data, species_context)
        
        # Determinar se é pássaro baseado no rótulo
        is_bird = correct_label.lower() in ["bird", "pássaro", "ave"]
        
        # Aprender com feedback
        self.passarinidade_analyzer.learn_from_feedback(analysis, is_bird)
        
        # Aprender características universais se disponível
        if self.universal_knowledge_transfer and species_context:
            self.universal_knowledge_transfer.learn_universal_pattern(
                detection_data, species_context, feedback
            )
        
        # Aprender com hierarquia de conceitos se disponível
        if self.concept_hierarchy_manager:
            # Determinar conceito mais provável
            most_likely_concept = self._determine_most_likely_concept(detection_data, species_context)
            
            if most_likely_concept:
                # Aprender com exemplo hierárquico
                is_positive = correct_label.lower() in ["bird", "pássaro", "ave"]
                self.concept_hierarchy_manager.learn_from_example(
                    most_likely_concept, 
                    f"Exemplo de {correct_label}", 
                    is_positive, 
                    feedback
                )
        
        # Atualizar conceito de passarinidade
        if "passarinidade" in self.concepts:
            concept = self.concepts["passarinidade"]
            
            # Atualizar score de confiança
            if is_bird:
                concept.confidence_score = min(1.0, concept.confidence_score + 0.01)
            else:
                concept.confidence_score = max(0.0, concept.confidence_score - 0.01)
            
            # Atualizar poder de generalização
            concept.generalization_power = min(1.0, concept.generalization_power + 0.005)
        
        # Salvar conceitos atualizados
        self._save_concepts()
        
        logger.info(f"Exemplo aprendido: {correct_label} (feedback: {feedback})")
    
    def get_concept_analysis(self, concept_name: str = "passarinidade") -> Dict[str, Any]:
        """
        Retorna análise de um conceito específico.
        
        Args:
            concept_name: Nome do conceito
            
        Returns:
            Análise do conceito
        """
        if concept_name not in self.concepts:
            return {"error": f"Conceito '{concept_name}' não encontrado"}
        
        concept = self.concepts[concept_name]
        
        # Obter análise específica do analisador
        if concept_name == "passarinidade":
            analyzer_analysis = self.passarinidade_analyzer.get_concept_analysis()
        else:
            analyzer_analysis = {"message": "Análise não disponível para este conceito"}
        
        return {
            "concept_name": concept.name,
            "concept_type": concept.concept_type.value,
            "level": concept.level.value,
            "definition": concept.definition,
            "confidence_score": concept.confidence_score,
            "generalization_power": concept.generalization_power,
            "abstraction_depth": concept.abstraction_depth,
            "essential_features": concept.essential_features,
            "supporting_features": concept.supporting_features,
            "confidence_threshold": concept.confidence_threshold,
            "learning_examples_count": len(concept.learning_examples),
            "analyzer_analysis": analyzer_analysis
        }
    
    def get_all_concepts(self) -> Dict[str, Any]:
        """Retorna todos os conceitos disponíveis."""
        result = {
            "concepts": {name: {
                "type": concept.concept_type.value,
                "level": concept.level.value,
                "definition": concept.definition,
                "confidence_score": concept.confidence_score,
                "generalization_power": concept.generalization_power
            } for name, concept in self.concepts.items()},
            "hierarchy": asdict(self.concept_hierarchy),
            "total_concepts": len(self.concepts)
        }
        
        # Adicionar análise de características universais se disponível
        if self.universal_knowledge_transfer:
            universal_analysis = self.universal_knowledge_transfer.get_universal_knowledge_analysis()
            result["universal_features"] = universal_analysis
        
        # Adicionar análise de hierarquias se disponível
        if self.concept_hierarchy_manager:
            hierarchy_stats = self.concept_hierarchy_manager.get_hierarchy_statistics()
            result["concept_hierarchy"] = hierarchy_stats
        
        return result
    
    def get_concept_hierarchy_analysis(self, concept_name: str = None) -> Dict[str, Any]:
        """
        Retorna análise da hierarquia de conceitos.
        
        Args:
            concept_name: Nome do conceito específico (opcional)
            
        Returns:
            Análise da hierarquia de conceitos
        """
        if not self.concept_hierarchy_manager:
            return {"error": "Sistema de hierarquias não disponível"}
        
        try:
            if concept_name:
                # Análise de conceito específico
                analysis = self.concept_hierarchy_manager.analyze_concept_hierarchy(concept_name)
                return analysis
            else:
                # Estatísticas gerais da hierarquia
                stats = self.concept_hierarchy_manager.get_hierarchy_statistics()
                return stats
                
        except Exception as e:
            error_msg = f"Erro ao obter análise de hierarquia: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def find_similar_concepts(self, concept_name: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Encontra conceitos similares na hierarquia.
        
        Args:
            concept_name: Nome do conceito base
            threshold: Threshold de similaridade
            
        Returns:
            Conceitos similares encontrados
        """
        if not self.concept_hierarchy_manager:
            return {"error": "Sistema de hierarquias não disponível"}
        
        try:
            similar_concepts = self.concept_hierarchy_manager.find_similar_concepts(concept_name, threshold)
            
            return {
                "base_concept": concept_name,
                "similar_concepts": similar_concepts,
                "similarity_count": len(similar_concepts),
                "threshold_used": threshold
            }
            
        except Exception as e:
            error_msg = f"Erro ao encontrar conceitos similares: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def infer_concept_properties(self, concept_name: str) -> Dict[str, Any]:
        """
        Infere propriedades de um conceito baseado na hierarquia.
        
        Args:
            concept_name: Nome do conceito
            
        Returns:
            Propriedades inferidas
        """
        if not self.concept_hierarchy_manager:
            return {"error": "Sistema de hierarquias não disponível"}
        
        try:
            properties = self.concept_hierarchy_manager.infer_concept_properties(concept_name)
            return properties
            
        except Exception as e:
            error_msg = f"Erro ao inferir propriedades: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_universal_features_analysis(self) -> Dict[str, Any]:
        """
        Retorna análise das características universais.
        
        Returns:
            Análise das características universais
        """
        if not self.universal_knowledge_transfer:
            return {"error": "Sistema de características universais não disponível"}
        
        try:
            analysis = self.universal_knowledge_transfer.get_universal_knowledge_analysis()
            return analysis
            
        except Exception as e:
            error_msg = f"Erro ao obter análise de características universais: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def learn_universal_pattern(self, detection_data: Dict[str, Any], species_label: str, feedback: str = "") -> Dict[str, Any]:
        """
        Aprende padrão universal de uma espécie.
        
        Args:
            detection_data: Dados de detecção
            species_label: Rótulo da espécie
            feedback: Feedback adicional
            
        Returns:
            Resultado do aprendizado
        """
        if not self.universal_knowledge_transfer:
            return {"error": "Sistema de características universais não disponível"}
        
        try:
            self.universal_knowledge_transfer.learn_universal_pattern(
                detection_data, species_label, feedback
            )
            
            logger.info(f"Padrão universal aprendido: {species_label}")
            
            return {
                "success": True,
                "message": f"Padrão universal aprendido: {species_label}",
                "feedback": feedback
            }
            
        except Exception as e:
            error_msg = f"Erro ao aprender padrão universal: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
