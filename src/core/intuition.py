#!/usr/bin/env python3
"""
Sistema de Intuição Neuro-Simbólica Simplificado
Funciona como uma criança descobrindo características fundamentais de pássaros
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
import json
import random
import time

# Configurar logging primeiro
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar sistema de auto-modificação
try:
    from .auto_modification import AutoModificationSystem
    AUTO_MODIFICATION_AVAILABLE = True
except ImportError:
    AUTO_MODIFICATION_AVAILABLE = False
    logger.warning("Sistema de auto-modificação não disponível")

# Importar sistema de evolução de algoritmos
try:
    from .algorithm_evolution import AlgorithmEvolutionSystem
    ALGORITHM_EVOLUTION_AVAILABLE = True
except ImportError:
    ALGORITHM_EVOLUTION_AVAILABLE = False
    logger.warning("Sistema de evolução de algoritmos não disponível")

# Importar sistema de auto-otimização
try:
    from .auto_optimization import AutoOptimizationSystem, ThresholdConfig
    AUTO_OPTIMIZATION_AVAILABLE = True
except ImportError:
    AUTO_OPTIMIZATION_AVAILABLE = False
    logger.warning("Sistema de auto-otimização não disponível")

# Importar sistema de otimização apurada dos pesos
try:
    from .advanced_weight_optimization import AdvancedWeightOptimizationSystem, WeightConfiguration
    ADVANCED_WEIGHT_OPTIMIZATION_AVAILABLE = True
except ImportError:
    ADVANCED_WEIGHT_OPTIMIZATION_AVAILABLE = False
    logger.warning("Sistema de otimização apurada dos pesos não disponível")

# Importar sistema de evolução de arquitetura
try:
    from .architecture_evolution import ArchitectureEvolutionSystem, ArchitectureConfig
    ARCHITECTURE_EVOLUTION_AVAILABLE = True
except ImportError:
    ARCHITECTURE_EVOLUTION_AVAILABLE = False
    logger.warning("Sistema de evolução de arquitetura não disponível")

# Importar sistema de aprendizado de conceitos abstratos
try:
    from .abstract_concept_learning import AbstractConceptLearner, AbstractConcept, ConceptType, ConceptLevel
    ABSTRACT_CONCEPT_LEARNING_AVAILABLE = True
except ImportError:
    ABSTRACT_CONCEPT_LEARNING_AVAILABLE = False
    logger.warning("Sistema de aprendizado de conceitos abstratos não disponível")

# Importar sistema de características universais
try:
    from .universal_features import UniversalKnowledgeTransfer, UniversalFeatureAnalyzer, UniversalFeatureType, UniversalFeatureLevel
    UNIVERSAL_FEATURES_AVAILABLE = True
except ImportError:
    UNIVERSAL_FEATURES_AVAILABLE = False
    logger.warning("Sistema de características universais não disponível")

# Importar sistema de few-shot learning
try:
    from .few_shot_learning import FewShotLearner, FewShotStrategy, LearningMode, ConceptType
    FEW_SHOT_LEARNING_AVAILABLE = True
except ImportError:
    FEW_SHOT_LEARNING_AVAILABLE = False
    logger.warning("Sistema de few-shot learning não disponível")

# Importar sistema de raciocínio conceitual
try:
    from .conceptual_reasoning import AbstractInferenceEngine, InferenceType, ConceptRelation, AbstractionLevel
    CONCEPTUAL_REASONING_AVAILABLE = True
except ImportError:
    CONCEPTUAL_REASONING_AVAILABLE = False
    logger.warning("Sistema de raciocínio conceitual não disponível")

# Importar sistema de raciocínio por analogia
try:
    from .analogical_reasoning import StructuralAnalogyEngine, AnalogyType, AnalogyStrength, MappingType
    ANALOGICAL_REASONING_AVAILABLE = True
except ImportError:
    ANALOGICAL_REASONING_AVAILABLE = False
    logger.warning("Sistema de raciocínio por analogia não disponível")

# Importar sistema de aprendizado por padrões universais
try:
    from .universal_pattern_learning import UniversalPatternLearner, PatternType, PatternLevel, PatternConfidence
    UNIVERSAL_PATTERN_LEARNING_AVAILABLE = True
except ImportError:
    UNIVERSAL_PATTERN_LEARNING_AVAILABLE = False
    logger.warning("Sistema de aprendizado por padrões universais não disponível")

# Importar sistema de memória episódica
try:
    from .episodic_memory import EpisodicMemorySystem, MemoryType, MemoryStrength, RetrievalType
    EPISODIC_MEMORY_AVAILABLE = True
except ImportError:
    EPISODIC_MEMORY_AVAILABLE = False
    logger.warning("Sistema de memória episódica não disponível")

# Importar sistema de raciocínio causal
try:
    from .causal_reasoning import CausalReasoningSystem, CausalRelationType, CausalStrength, EvidenceType
    CAUSAL_REASONING_AVAILABLE = True
except ImportError:
    CAUSAL_REASONING_AVAILABLE = False
    logger.warning("Sistema de raciocínio causal não disponível")

# Importar sistema de inferência abstrata
try:
    from .abstract_inference import AbstractInferenceEngine, InferenceType, AbstractionLevel, ConfidenceLevel
    ABSTRACT_INFERENCE_AVAILABLE = True
except ImportError:
    ABSTRACT_INFERENCE_AVAILABLE = False
    logger.warning("Sistema de inferência abstrata não disponível")

# Importar sistema de meta-aprendizado
try:
    from .meta_learning import MetaLearningSystem, LearningContext, MetaLearningStrategy, PerformanceMetric
    META_LEARNING_AVAILABLE = True
except ImportError:
    META_LEARNING_AVAILABLE = False
    logger.warning("Sistema de meta-aprendizado não disponível")

# Importar sistema de hierarquias de conceitos
try:
    from .concept_hierarchy import ConceptHierarchyManager, HierarchicalConcept, ConceptRelationshipType, ConceptAbstractionLevel, ConceptComplexity
    CONCEPT_HIERARCHY_AVAILABLE = True
except ImportError:
    CONCEPT_HIERARCHY_AVAILABLE = False
    logger.warning("Sistema de hierarquias de conceitos não disponível")

# Importar sistema de aprendizado de relações
try:
    from .relation_learning import RelationLearningSystem, RelationType, RelationStrength, LearningMode, EvidenceType, RelationEvidence
    RELATION_LEARNING_AVAILABLE = True
except ImportError:
    RELATION_LEARNING_AVAILABLE = False
    # Definir classes dummy para evitar erros
    class RelationType:
        TEMPORAL = "temporal"
        SPATIAL = "spatial"
        CAUSAL = "causal"
        FUNCTIONAL = "functional"
        STRUCTURAL = "structural"
        BEHAVIORAL = "behavioral"
        SIMILARITY = "similarity"
        OPPOSITION = "opposition"
        DEPENDENCY = "dependency"
        ASSOCIATION = "association"
    
    class RelationStrength:
        VERY_WEAK = "very_weak"
        WEAK = "weak"
        MODERATE = "moderate"
        STRONG = "strong"
        VERY_STRONG = "very_strong"
    
    class LearningMode:
        SUPERVISED = "supervised"
        UNSUPERVISED = "unsupervised"
        SEMI_SUPERVISED = "semi_supervised"
        REINFORCEMENT = "reinforcement"
        TRANSFER = "transfer"
    
    class EvidenceType:
        DIRECT_OBSERVATION = "direct_observation"
        STATISTICAL_CORRELATION = "statistical_correlation"
        EXPERT_KNOWLEDGE = "expert_knowledge"
        LOGICAL_INFERENCE = "logical_inference"
        PATTERN_RECOGNITION = "pattern_recognition"
        ANALOGICAL_REASONING = "analogical_reasoning"
    
    class RelationEvidence:
        def __init__(self, **kwargs):
            pass
    logger.warning("Sistema de aprendizado de relações não disponível")

# Importar sistema de transferência de conhecimento
try:
    from .knowledge_transfer import KnowledgeTransferSystem, TransferType, TransferStrategy, TransferConfidence, KnowledgeType
    KNOWLEDGE_TRANSFER_AVAILABLE = True
except ImportError:
    KNOWLEDGE_TRANSFER_AVAILABLE = False
    # Definir classes dummy para evitar erros
    class TransferType:
        DOMAIN_TO_DOMAIN = "domain_to_domain"
        CONCEPT_TO_CONCEPT = "concept_to_concept"
        SPECIES_TO_SPECIES = "species_to_species"
        CONTEXT_TO_CONTEXT = "context_to_context"
        SKILL_TO_SKILL = "skill_to_skill"
        PATTERN_TO_PATTERN = "pattern_to_pattern"
        FEATURE_TO_FEATURE = "feature_to_feature"
        STRATEGY_TO_STRATEGY = "strategy_to_strategy"
    
    class TransferStrategy:
        DIRECT = "direct"
        ADAPTIVE = "adaptive"
        SELECTIVE = "selective"
        HIERARCHICAL = "hierarchical"
        CONTEXTUAL = "contextual"
        ANALOGICAL = "analogical"
        META_LEARNING = "meta_learning"
        FEW_SHOT = "few_shot"
    
    class TransferConfidence:
        VERY_LOW = "very_low"
        LOW = "low"
        MODERATE = "moderate"
        HIGH = "high"
        VERY_HIGH = "very_high"
    
    class KnowledgeType:
        FACTUAL = "factual"
        PROCEDURAL = "procedural"
        CONCEPTUAL = "conceptual"
        METACOGNITIVE = "metacognitive"
        EXPERIENTIAL = "experiential"
        PATTERN_BASED = "pattern_based"
    logger.warning("Sistema de transferência de conhecimento não disponível")

# Importar sistema de meta-cognição
try:
    from .metacognition import MetaCognitiveSystem, MetaCognitiveProcess, ReflectionType, AdaptationStrategy, MetaCognitiveLevel
    METACOGNITION_AVAILABLE = True
except ImportError:
    METACOGNITION_AVAILABLE = False
    # Definir classes dummy para evitar erros
    class MetaCognitiveProcess:
        LEARNING_TO_LEARN = "learning_to_learn"
        SELF_REFLECTION = "self_reflection"
        STRATEGY_ADAPTATION = "strategy_adaptation"
        PERFORMANCE_MONITORING = "performance_monitoring"
        METACOGNITIVE_CONTROL = "metacognitive_control"
        KNOWLEDGE_MONITORING = "knowledge_monitoring"
    
    class ReflectionType:
        PERFORMANCE_REFLECTION = "performance_reflection"
        STRATEGY_REFLECTION = "strategy_reflection"
        LEARNING_REFLECTION = "learning_reflection"
        ERROR_REFLECTION = "error_reflection"
        SUCCESS_REFLECTION = "success_reflection"
        METACOGNITIVE_REFLECTION = "metacognitive_reflection"
    
    class AdaptationStrategy:
        PARAMETER_ADJUSTMENT = "parameter_adjustment"
        STRATEGY_SWITCHING = "strategy_switching"
        LEARNING_RATE_ADAPTATION = "learning_rate_adaptation"
        ARCHITECTURE_MODIFICATION = "architecture_modification"
        KNOWLEDGE_RESTRUCTURING = "knowledge_restructuring"
        METACOGNITIVE_ADAPTATION = "metacognitive_adaptation"
    
    class MetaCognitiveLevel:
        BASIC = "basic"
        INTERMEDIATE = "intermediate"
        ADVANCED = "advanced"
        EXPERT = "expert"
        MASTER = "master"
    logger.warning("Sistema de meta-cognição não disponível")

class LearningCandidateType(Enum):
    """Tipos de candidatos para aprendizado"""
    VISUAL_ANALYSIS = "visual_analysis"
    SPECIES_UNKNOWN = "species_unknown"
    CHARACTERISTIC_LEARNING = "characteristic_learning"

@dataclass
class LearningCandidate:
    """Candidato para aprendizado contínuo"""
    type: LearningCandidateType
    confidence: float
    characteristics: Dict[str, Any]
    reasoning: str
    image_path: str
    metadata: Dict[str, Any]

class IntuitionEngine:
    """Motor de Intuição Neuro-Simbólica Simplificado para Pássaros"""
    
    # Cache global para modelos YOLO
    _yolo_cache = {}
    _keras_cache = {}
    
    def __init__(self, yolo_model_path: str, keras_model_path: str, debug_logger):
        self.yolo_model_path = yolo_model_path
        self.keras_model_path = keras_model_path
        self.debug_logger = debug_logger
        self.yolo_model = None
        self.keras_model = None
        
        # Sistema de auto-modificação
        self.auto_modification_system = None
        if AUTO_MODIFICATION_AVAILABLE:
            try:
                self.auto_modification_system = AutoModificationSystem(auto_save=False)  # FIXED: Desabilitar auto_save durante init
                logger.info("Sistema de auto-modificação inicializado")
            except Exception as e:
                logger.error(f"Erro ao inicializar sistema de auto-modificação: {e}")
                self.auto_modification_system = None
        
        # Sistema de evolução de algoritmos
        self.algorithm_evolution_system = None
        if ALGORITHM_EVOLUTION_AVAILABLE:
            try:
                self.algorithm_evolution_system = AlgorithmEvolutionSystem(auto_save=False)  # FIXED: Desabilitar auto_save durante init
                logger.info("Sistema de evolução de algoritmos inicializado")
            except Exception as e:
                logger.error(f"Erro ao inicializar sistema de evolução de algoritmos: {e}")
                self.algorithm_evolution_system = None
        
        # Sistema de auto-otimização
        self.auto_optimization_system = None
        if AUTO_OPTIMIZATION_AVAILABLE:
            try:
                self.auto_optimization_system = AutoOptimizationSystem(auto_save=False)  # FIXED: Desabilitar auto_save durante init
                logger.info("Sistema de auto-otimização inicializado")
            except Exception as e:
                logger.error(f"Erro ao inicializar sistema de auto-otimização: {e}")
                self.auto_optimization_system = None
        
        # Sistema de otimização apurada dos pesos
        self.advanced_weight_optimization_system = None
        if ADVANCED_WEIGHT_OPTIMIZATION_AVAILABLE:
            try:
                self.advanced_weight_optimization_system = AdvancedWeightOptimizationSystem(auto_save=False)  # FIXED: Desabilitar auto_save durante init
                logger.info("Sistema de otimização apurada dos pesos inicializado")
            except Exception as e:
                logger.error(f"Erro ao inicializar sistema de otimização apurada dos pesos: {e}")
                self.advanced_weight_optimization_system = None
        
        # Sistema de evolução de arquitetura
        self.architecture_evolution_system = None
        if ARCHITECTURE_EVOLUTION_AVAILABLE:
            try:
                self.architecture_evolution_system = ArchitectureEvolutionSystem(auto_save=False)  # FIXED: Desabilitar auto_save durante init
                logger.info("Sistema de evolução de arquitetura inicializado")
            except Exception as e:
                logger.error(f"Erro ao inicializar sistema de evolução de arquitetura: {e}")
                self.architecture_evolution_system = None
        
        # Sistema de aprendizado de conceitos abstratos
        self.abstract_concept_learner = None
        if ABSTRACT_CONCEPT_LEARNING_AVAILABLE:
            try:
                self.abstract_concept_learner = AbstractConceptLearner()
                logger.info("Sistema de aprendizado de conceitos abstratos inicializado")
            except Exception as e:
                logger.error(f"Erro ao inicializar sistema de conceitos abstratos: {e}")
                self.abstract_concept_learner = None
        
        # Sistema de few-shot learning
        self.few_shot_learner = None
        if FEW_SHOT_LEARNING_AVAILABLE:
            try:
                self.few_shot_learner = FewShotLearner(auto_save=False)  # FIXED: Desabilitar auto_save durante init
                logger.info("Sistema de few-shot learning inicializado")
            except Exception as e:
                logger.error(f"Erro ao inicializar sistema de few-shot learning: {e}")
                self.few_shot_learner = None
        
        # Sistema de meta-aprendizado
        self.meta_learning_system = None
        if META_LEARNING_AVAILABLE:
            try:
                self.meta_learning_system = MetaLearningSystem(auto_save=False)  # FIXED: Desabilitar auto_save durante init
                logger.info("Sistema de meta-aprendizado inicializado")
            except Exception as e:
                logger.error(f"Erro ao inicializar sistema de meta-aprendizado: {e}")
                self.meta_learning_system = None
        
        # Sistema de raciocínio conceitual
        self.conceptual_reasoning_system = None
        if CONCEPTUAL_REASONING_AVAILABLE:
            try:
                self.conceptual_reasoning_system = AbstractInferenceEngine(auto_save=False)  # Desabilitar auto_save durante init
                logger.info("Sistema de raciocínio conceitual inicializado")
            except Exception as e:
                logger.error(f"Erro ao inicializar sistema de raciocínio conceitual: {e}")
                self.conceptual_reasoning_system = None
        
        # Sistema de raciocínio por analogia
        self.analogical_reasoning_system = None
        if ANALOGICAL_REASONING_AVAILABLE:
            try:
                self.analogical_reasoning_system = StructuralAnalogyEngine(auto_save=False)  # Desabilitar auto_save durante init
                logger.info("Sistema de raciocínio por analogia inicializado")
            except Exception as e:
                logger.error(f"Erro ao inicializar sistema de raciocínio por analogia: {e}")
                self.analogical_reasoning_system = None
        
        # Sistema de aprendizado por padrões universais
        self.universal_pattern_learner = None
        if UNIVERSAL_PATTERN_LEARNING_AVAILABLE:
            try:
                self.universal_pattern_learner = UniversalPatternLearner(auto_save=False)  # FIXED: Desabilitar auto_save durante init
                logger.info("Sistema de aprendizado por padrões universais inicializado")
            except Exception as e:
                logger.error(f"Erro ao inicializar sistema de aprendizado por padrões universais: {e}")
                self.universal_pattern_learner = None
        
        # Sistema de memória episódica
        self.episodic_memory_system = None
        if EPISODIC_MEMORY_AVAILABLE:
            try:
                self.episodic_memory_system = EpisodicMemorySystem(auto_save=False)  # Desabilitar auto_save durante init
                logger.info("Sistema de memória episódica inicializado")
            except Exception as e:
                logger.error(f"Erro ao inicializar sistema de memória episódica: {e}")
                self.episodic_memory_system = None
        
        # Sistema de raciocínio causal
        self.causal_reasoning_system = None
        if CAUSAL_REASONING_AVAILABLE:
            try:
                self.causal_reasoning_system = CausalReasoningSystem(auto_save=False)  # Desabilitar auto_save durante init
                logger.info("Sistema de raciocínio causal inicializado")
            except Exception as e:
                logger.error(f"Erro ao inicializar sistema de raciocínio causal: {e}")
                self.causal_reasoning_system = None
        
        # Sistema de hierarquias de conceitos
        self.concept_hierarchy_manager = None
        if CONCEPT_HIERARCHY_AVAILABLE:
            try:
                self.concept_hierarchy_manager = ConceptHierarchyManager(auto_save=False)  # FIXED: Desabilitar auto_save durante init
                logger.info("Sistema de hierarquias de conceitos inicializado")
            except Exception as e:
                logger.error(f"Erro ao inicializar sistema de hierarquias de conceitos: {e}")
                self.concept_hierarchy_manager = None
        
        # Sistema de aprendizado de relações
        self.relation_learning_system = None
        if RELATION_LEARNING_AVAILABLE:
            try:
                self.relation_learning_system = RelationLearningSystem(auto_save=False)  # FIXED: Desabilitar auto_save durante init
                logger.info("Sistema de aprendizado de relações inicializado")
            except Exception as e:
                logger.error(f"Erro ao inicializar sistema de aprendizado de relações: {e}")
                self.relation_learning_system = None
        
        # Sistema de transferência de conhecimento
        self.knowledge_transfer_system = None
        if KNOWLEDGE_TRANSFER_AVAILABLE:
            try:
                self.knowledge_transfer_system = KnowledgeTransferSystem(auto_save=False)  # FIXED: Desabilitar auto_save durante init
                logger.info("Sistema de transferência de conhecimento inicializado")
            except Exception as e:
                logger.error(f"Erro ao inicializar sistema de transferência de conhecimento: {e}")
                self.knowledge_transfer_system = None
        
        # Sistema de meta-cognição
        self.metacognitive_system = None
        if METACOGNITION_AVAILABLE:
            try:
                self.metacognitive_system = MetaCognitiveSystem(auto_save=False)  # FIXED: Desabilitar auto_save durante init
                logger.info("Sistema de meta-cognição inicializado")
            except Exception as e:
                logger.error(f"Erro ao inicializar sistema de meta-cognição: {e}")
                self.metacognitive_system = None
        
        # Conhecimento acumulado (como uma criança)
        self.learned_patterns = {
            'known_species': set(),
            'characteristic_patterns': {},
            'color_combinations': {},
            'shape_patterns': {}
        }
        
        # ============================================================================
        # SISTEMA DE CACHE DE PERFORMANCE - FASE 1.3.2
        # ============================================================================
        
        # Cache de análises de imagem
        self.analysis_cache = {}
        self.cache_max_size = 100  # Máximo de análises em cache
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # Cache de características visuais
        self.visual_features_cache = {}
        self.features_cache_max_size = 200
        
        # Cache de resultados YOLO
        self.yolo_results_cache = {}
        self.yolo_cache_max_size = 50
        
        # Estatísticas de performance
        self.performance_stats = {
            'total_analyses': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_analysis_time': 0.0,
            'total_analysis_time': 0.0
        }
        
        # ============================================================================
        # ALIASES PARA SISTEMAS COGNITIVOS - FASE 1.2.3
        # ============================================================================
        
        # Criar aliases para facilitar acesso aos sistemas cognitivos
        self.abstract_inference = self.conceptual_reasoning_system
        self.analogical_reasoning = self.analogical_reasoning_system
        self.causal_reasoning = self.causal_reasoning_system
        self.episodic_memory = self.episodic_memory_system
        self.few_shot_learning = self.few_shot_learner
        self.meta_learning = self.meta_learning_system
        self.metacognition = self.metacognitive_system
        self.universal_features = self.universal_pattern_learner
        self.concept_hierarchy = self.concept_hierarchy_manager
        self.relation_learning = self.relation_learning_system
        self.knowledge_transfer = self.knowledge_transfer_system
        
        # Log dos sistemas conectados
        connected_systems = []
        if self.abstract_inference: connected_systems.append("abstract_inference")
        if self.analogical_reasoning: connected_systems.append("analogical_reasoning")
        if self.causal_reasoning: connected_systems.append("causal_reasoning")
        if self.episodic_memory: connected_systems.append("episodic_memory")
        if self.few_shot_learning: connected_systems.append("few_shot_learning")
        if self.meta_learning: connected_systems.append("meta_learning")
        if self.metacognition: connected_systems.append("metacognition")
        if self.universal_features: connected_systems.append("universal_features")
        if self.concept_hierarchy: connected_systems.append("concept_hierarchy")
        if self.relation_learning: connected_systems.append("relation_learning")
        if self.knowledge_transfer: connected_systems.append("knowledge_transfer")
        
        logger.info(f"[CONEXÃO] Sistemas cognitivos conectados: {len(connected_systems)}")
        logger.info(f"[CONEXÃO] Sistemas: {', '.join(connected_systems)}")
        
        self._load_models()
        
    def _load_models(self):
        """Carrega modelos híbridos com múltiplas bibliotecas de detecção"""
        self.detection_models = {}
        
        # 1. Tentar carregar YOLO (múltiplas tentativas)
        self._load_yolo_models()
        
        # 2. Carregar OpenCV DNN (alternativa robusta)
        self._load_opencv_dnn()
        
        # 3. Carregar MediaPipe (detecção de objetos)
        # self._load_mediapipe()  # DESABILITADO temporariamente
        
        # 4. Carregar Keras
        self._load_keras()
        
        # 5. Log do status dos modelos
        self._log_model_status()
    
    def _load_yolo_models(self):
        """Carrega modelos YOLO com múltiplas versões e configurações avançadas"""
        try:
            # Verificar cache primeiro
            cache_key = f"yolo_{self.yolo_model_path}"
            if cache_key in self._yolo_cache:
                self.yolo_model = self._yolo_cache[cache_key]
                logger.info(f"[CACHE] Modelo YOLO carregado do cache: {cache_key}")
                return
            
            # Aplicar patch PyTorch ANTES de importar YOLO
            import sys
            import os
            # Aplicar patch de typing
            try:
                from src.utils.typing_patch import apply_typing_patch
                apply_typing_patch()
            except ImportError:
                pass
            
            from ultralytics import YOLO
            import torch
            
            # Lista de modelos YOLO para tentar (em ordem de prioridade)
            yolo_models = [
                # Modelos customizados treinados
                (self.yolo_model_path, 'YOLO customizado (melhor)'),
                ('runs/detect/train/weights/best.pt', 'YOLO customizado (melhor)'),
                ('runs/detect/train/weights/last.pt', 'YOLO customizado (último)'),
                
                # Modelos YOLOv8 (diferentes tamanhos)
                ('yolov8n.pt', 'YOLOv8 Nano'),
                ('yolov8s.pt', 'YOLOv8 Small'),
                ('yolov8m.pt', 'YOLOv8 Medium'),
                ('yolov8l.pt', 'YOLOv8 Large'),
                ('yolov8x.pt', 'YOLOv8 Extra Large'),
                
                # Modelos YOLOv11 (mais recentes)
                ('yolo11n.pt', 'YOLOv11 Nano'),
                ('yolo11s.pt', 'YOLOv11 Small'),
                ('yolo11m.pt', 'YOLOv11 Medium'),
                
                # Modelos YOLOv10
                ('yolo10n.pt', 'YOLOv10 Nano'),
                ('yolo10s.pt', 'YOLOv10 Small'),
                
                # Modelos YOLOv9
                ('yolo9c.pt', 'YOLOv9 Compact'),
                ('yolo9e.pt', 'YOLOv9 Efficient'),
            ]
            
            # Tentar carregar cada modelo
            for model_path, model_name in yolo_models:
                try:
                    if os.path.exists(model_path) or model_path.startswith('yolo'):
                        # Configurações avançadas para melhor detecção
                        model = YOLO(model_path)
                        
                        # Configurar parâmetros avançados
                        model.overrides = {
                            'conf': 0.15,  # Confiança mínima mais baixa
                            'iou': 0.35,   # Intersection over Union mais rigoroso
                            'agnostic_nms': False,  # NMS não agnóstico
                            'max_det': 1000,  # Máximo de detecções
                            'half': False,  # Precisão completa
                            'dnn': False,   # Usar PyTorch
                            'device': 'cpu'  # Forçar CPU para compatibilidade
                        }
                        
                        self.detection_models['yolo_advanced'] = model
                        # Adicionar ao cache
                        self._yolo_cache[cache_key] = model
                        logger.info(f"[SUCESSO] {model_name} carregado com configurações avançadas")
                        return
                        
                except Exception as e:
                    logger.warning(f"[ALERTA] {model_name} falhou: {e}")
                    continue
            
            # Tentativa final: YOLOv5 como fallback
            try:
                import torch.hub
                self.detection_models['yolo_v5'] = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                logger.info("[SUCESSO] YOLOv5 fallback carregado")
            except Exception as e:
                logger.warning(f"[ALERTA] YOLOv5 fallback falhou: {e}")
                        
        except Exception as e:
            logger.warning(f"[ALERTA] Erro geral ao carregar YOLO: {e}")
    
    def _load_opencv_dnn(self):
        """Carrega modelos OpenCV DNN para detecção alternativa"""
        try:
            # Tentar carregar modelos pré-treinados do OpenCV
            # YOLOv4, YOLOv3, MobileNet-SSD, etc.
            
            # YOLOv4 (se disponível)
            try:
                yolo_config = "yolov4.cfg"
                yolo_weights = "yolov4.weights"
                if os.path.exists(yolo_config) and os.path.exists(yolo_weights):
                    net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
                    self.detection_models['opencv_yolo'] = net
                    logger.info("[SUCESSO] OpenCV YOLO carregado")
            except Exception as e:
                logger.warning(f"[ALERTA] OpenCV YOLO não disponível: {e}")
            
            # MobileNet-SSD (mais leve e robusto)
            # OpenCV SSD desabilitado devido a problemas de compatibilidade
            logger.info("[INFO] OpenCV SSD desabilitado - usando apenas YOLO")
                
        except Exception as e:
            logger.warning(f"[ALERTA] Erro ao carregar OpenCV DNN: {e}")
    
    def _load_mediapipe(self):
        """Carrega MediaPipe para detecção de objetos"""
        try:
            import mediapipe as mp
            
            # Detecção de objetos com MediaPipe
            self.detection_models['mediapipe'] = mp.solutions.objectron.Objectron(
                static_image_mode=True,
                max_num_objects=5,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("[SUCESSO] MediaPipe carregado")
            
        except Exception as e:
            logger.warning(f"[ALERTA] MediaPipe não disponível: {e}")
    
    def _load_keras(self):
        """Carrega modelo Keras usando carregador alternativo"""
        try:
            from src.utils.keras_model_loader import load_keras_model
            self.keras_model = load_keras_model(self.keras_model_path)
            if self.keras_model is not None:
                logger.info(f"[SUCESSO] Modelo Keras carregado: {self.keras_model_path}")
            else:
                logger.warning(f"[ALERTA] Modelo Keras não carregado: {self.keras_model_path}")
        except Exception as e:
            logger.error(f"[ERRO] Falha ao carregar modelo Keras: {e}")
            self.keras_model = None

    def _log_model_status(self):
        """Log do status de todos os modelos carregados"""
        loaded_models = list(self.detection_models.keys())
        if loaded_models:
            logger.info(f"[ALVO] Modelos de detecção carregados: {', '.join(loaded_models)}")
        else:
            logger.warning("[ALERTA] Nenhum modelo de detecção carregado - usando apenas análise visual")
        
        if self.keras_model is not None:
            logger.info("[SUCESSO] Modelo Keras disponível")
        else:
            logger.info("[INFO] Modelo Keras não disponível")
    
    def analyze_image_intuition(self, image_input) -> Dict[str, Any]:
        """
        Análise principal de intuição - como uma criança descobrindo pássaros
        Aceita tanto caminho de arquivo quanto numpy array
        """
        try:
            # Importar logger em tempo real
            from src.utils.realtime_logger import log_info, log_error, log_warning
            from src.utils.terminal_logger import log_info as term_log_info, log_error as term_log_error, log_success as term_log_success
            
            # ============================================================================
            # CACHE DE PERFORMANCE - FASE 1.3.2
            # ============================================================================
            
            # Gerar chave de cache
            cache_key = self._generate_cache_key(image_input)
            
            # Verificar cache primeiro
            cached_result = self._get_cached_analysis(cache_key)
            if cached_result:
                logger.info(f"[CACHE] Análise recuperada do cache: {cache_key}")
                return cached_result['result']
            
            # Iniciar medição de tempo
            analysis_start_time = time.time()
            
            term_log_info("Iniciando análise de imagem", "IntuitionEngine", "analyze_image_intuition")
            log_info("Iniciando análise de imagem", "IntuitionEngine", "analyze_image_intuition")
            
            # Determinar tipo de entrada e processar adequadamente
            if isinstance(image_input, str):
                # Caminho de arquivo
                log_info(f"Processando arquivo: {image_input}", "IntuitionEngine", "analyze_image_intuition")
                image_path = image_input
                
                # Verificar se arquivo existe
                if not os.path.exists(image_path):
                    log_error(f"Arquivo não encontrado: {image_path}", "IntuitionEngine", "analyze_image_intuition")
                    return {'error': f'Arquivo não encontrado: {image_path}'}
                
                # Carregar imagem
                image = cv2.imread(image_path)
                if image is None:
                    log_error(f"Erro ao carregar imagem: {image_path}", "IntuitionEngine", "analyze_image_intuition")
                    return {'error': f'Erro ao carregar imagem: {image_path}'}
                
                log_info(f"Imagem carregada: {image.shape}", "IntuitionEngine", "analyze_image_intuition")
                
            elif isinstance(image_input, np.ndarray):
                # Numpy array - salvar temporariamente
                log_info(f"Processando numpy array: {image_input.shape}", "IntuitionEngine", "analyze_image_intuition")
                
                # Salvar array como imagem temporária
                temp_path = f"temp_analysis_{int(time.time())}.png"
                cv2.imwrite(temp_path, image_input)
                image_path = temp_path
                image = image_input
                
                log_info(f"Array salvo como: {temp_path}", "IntuitionEngine", "analyze_image_intuition")
                
            else:
                log_error(f"Tipo de entrada não suportado: {type(image_input)}", "IntuitionEngine", "analyze_image_intuition")
                return {'error': f'Tipo de entrada não suportado: {type(image_input)}'}
            
            # 1. Análise visual básica (como uma criança vê)
            term_log_info("Iniciando análise visual", "IntuitionEngine", "analyze_image_intuition")
            log_info("Iniciando análise visual", "IntuitionEngine", "analyze_image_intuition")
            visual_analysis = self._analyze_visual_characteristics(image_path)
            term_log_success("Análise visual concluída", "IntuitionEngine", "analyze_image_intuition")
            
            if 'error' in visual_analysis:
                log_error(f"Erro na análise visual: {visual_analysis['error']}", "IntuitionEngine", "analyze_image_intuition")
                return visual_analysis
            
            # 2. Detecção de características fundamentais
            term_log_info("Iniciando detecção de características", "IntuitionEngine", "analyze_image_intuition")
            log_info("Iniciando detecção de características", "IntuitionEngine", "analyze_image_intuition")
            fundamental_characteristics = self._detect_fundamental_characteristics(image_path)
            term_log_success("Detecção de características concluída", "IntuitionEngine", "analyze_image_intuition")
            
            if 'error' in fundamental_characteristics:
                log_error(f"Erro na detecção de características: {fundamental_characteristics['error']}", "IntuitionEngine", "analyze_image_intuition")
                return fundamental_characteristics
            
            # 3. Raciocínio lógico (neuro-simbólico)
            term_log_info("Iniciando raciocínio lógico", "IntuitionEngine", "analyze_image_intuition")
            log_info("Iniciando raciocínio lógico", "IntuitionEngine", "analyze_image_intuition")
            logical_reasoning = self._logical_reasoning(visual_analysis, fundamental_characteristics)
            term_log_success("Raciocínio lógico concluído", "IntuitionEngine", "analyze_image_intuition")
            
            # 4. Detecção de candidatos para aprendizado
            log_info("Iniciando detecção de candidatos", "IntuitionEngine", "analyze_image_intuition")
            learning_candidates = self._detect_learning_candidates(
                visual_analysis, fundamental_characteristics, logical_reasoning
            )
            
            # 5. Recomendação de ação
            log_info("Iniciando recomendação de ação", "IntuitionEngine", "analyze_image_intuition")
            recommendation = self._recommend_action(learning_candidates, logical_reasoning)
            
            # Limpar arquivo temporário se criado
            if isinstance(image_input, np.ndarray) and os.path.exists(temp_path):
                os.remove(temp_path)
                log_info(f"Arquivo temporário removido: {temp_path}", "IntuitionEngine", "analyze_image_intuition")
            
            result = {
                'is_bird': logical_reasoning.get('is_bird', False),
                'confidence': logical_reasoning.get('confidence', 0.0),
                'species': logical_reasoning.get('species', 'Desconhecida'),
                'intuition_level': logical_reasoning.get('intuition_level', 'Baixa'),
                'needs_manual_review': logical_reasoning.get('needs_manual_review', False),
                'reasoning_steps': logical_reasoning.get('reasoning_steps', []),
                'characteristics_found': logical_reasoning.get('characteristics_found', []),
                'color': visual_analysis.get('dominant_color', 'Indefinida'),
                'intuition_analysis': {
                    'visual_analysis': visual_analysis,
                    'fundamental_characteristics': fundamental_characteristics,
                    'logical_reasoning': logical_reasoning,
                    'learning_candidates': learning_candidates,
                    'recommendation': recommendation
                }
            }
            
            log_info("Análise concluída com sucesso", "IntuitionEngine", "analyze_image_intuition")
            
            # ============================================================================
            # ARMAZENAR NO CACHE E ATUALIZAR ESTATÍSTICAS - FASE 1.3.2
            # ============================================================================
            
            # Calcular tempo de análise
            analysis_time = time.time() - analysis_start_time
            
            # Atualizar estatísticas de performance
            self.performance_stats['total_analyses'] += 1
            self.performance_stats['total_analysis_time'] += analysis_time
            self.performance_stats['avg_analysis_time'] = (
                self.performance_stats['total_analysis_time'] / self.performance_stats['total_analyses']
            )
            
            # Armazenar resultado no cache
            self._store_cached_analysis(cache_key, result)
            
            logger.info(f"[PERFORMANCE] Análise concluída em {analysis_time:.2f}s")
            
            return result
            
        except Exception as e:
            log_error(f"Erro na análise de intuição: {str(e)}", "IntuitionEngine", "analyze_image_intuition")
            error_result = {
                'error': str(e),
                'is_bird': False,
                'confidence': 0.0,
                'species': 'Erro',
                'intuition_level': 'Erro',
                'needs_manual_review': True,
                'reasoning_steps': [],
                'characteristics_found': [],
                'color': 'Erro',
                'intuition_analysis': {
                    'error': str(e),
                    'visual_analysis': {},
                    'fundamental_characteristics': {},
                    'logical_reasoning': {},
                    'learning_candidates': {},
                    'recommendation': 'ERRO_ANALISE'
                }
            }
            
            # Armazenar resultado no cache mesmo em caso de erro
            self._store_cached_analysis(cache_key, error_result)
            
            return error_result
    
    def _analyze_visual_characteristics(self, image_path: str) -> Dict[str, Any]:
        """Análise visual básica - como uma criança vê cores e formas"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Imagem não carregada'}
            
            # Converter para HSV para análise de cores
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Análise de cores dominantes
            color_analysis = self._analyze_colors(hsv)
            
            # Análise de formas básicas
            shape_analysis = self._analyze_shapes(image)
            
            # Análise de texturas
            texture_analysis = self._analyze_textures(image)
            
            return {
                'dominant_color': color_analysis['dominant_color'],
                'color_distribution': color_analysis['distribution'],
                'shape_characteristics': shape_analysis,
                'texture_characteristics': texture_analysis,
                'bird_like_features': self._calculate_bird_like_score(
                    color_analysis, shape_analysis, texture_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"[ERRO] Erro na análise visual: {e}")
            return {'error': str(e)}
    
    def _analyze_colors(self, hsv_image: np.ndarray) -> Dict[str, Any]:
        """Analisa cores como uma criança reconheceria - mais sensível"""
        # Cores típicas de pássaros - ranges mais amplos
        bird_colors = {
            'brown': [(5, 30, 10), (25, 200, 255)],
            'black': [(0, 0, 0), (60, 60, 80)],
            'white': [(180, 180, 180), (255, 255, 255)],
            'red': [(0, 30, 50), (15, 200, 255)],
            'blue': [(90, 30, 0), (160, 200, 255)],
            'yellow': [(0, 80, 80), (60, 255, 255)],
            'green': [(40, 80, 0), (160, 255, 200)],
            'gray': [(0, 0, 50), (180, 30, 200)],
            'orange': [(5, 100, 100), (25, 255, 255)]
        }
        
        color_scores = {}
        
        for color_name, (lower, upper) in bird_colors.items():
            mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
            score = np.sum(mask) / (mask.shape[0] * mask.shape[1] * 255)
            color_scores[color_name] = score
        
        # Encontrar cor dominante
        dominant_color = max(color_scores, key=color_scores.get)
        
        # Calcular score de cores de pássaro - mais sensível
        bird_color_score = sum(color_scores.values()) / len(color_scores)
        
        # Bonus para cores muito comuns em pássaros
        common_bird_colors = ['brown', 'black', 'white', 'gray', 'green', 'blue']
        common_score = sum([color_scores[color] for color in common_bird_colors]) / len(common_bird_colors)
        
        # Penalty para cores raras em pássaros
        rare_bird_colors = ['purple', 'pink']
        rare_score = sum([color_scores.get(color, 0) for color in rare_bird_colors]) / len(rare_bird_colors)
        
        # Score final com bonus para cores comuns e penalty para cores raras
        final_score = (bird_color_score + common_score - rare_score * 0.3) / 2
        final_score = max(0, min(final_score, 1.0))  # Clamp entre 0 e 1
        
        # Análise de contraste e saturação
        contrast_score = self._analyze_color_contrast(hsv_image)
        saturation_score = self._analyze_color_saturation(hsv_image)
        
        # NOVO: Detecção de padrões complexos e iridescência
        has_complex_patterns = self._detect_complex_patterns(hsv_image)
        has_iridescence = self._detect_iridescence(hsv_image)
        
        # NOVO: Lista de cores dominantes para análise melhorada
        dominant_colors = [color for color, score in color_scores.items() if score > 0.1]
        
        return {
            'dominant_color': dominant_color,
            'dominant_colors': dominant_colors,  # NOVO
            'distribution': color_scores,
            'bird_color_score': final_score,
            'contrast_score': contrast_score,
            'saturation_score': saturation_score,
            'has_complex_patterns': has_complex_patterns,  # NOVO
            'has_iridescence': has_iridescence,  # NOVO
            'color_complexity': len([c for c in color_scores.values() if c > 0.1])
        }
    
    def _analyze_color_contrast(self, hsv_image: np.ndarray) -> float:
        """Analisa o contraste de cores na imagem"""
        try:
            # Converter para escala de cinza para análise de contraste
            gray = cv2.cvtColor(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
            
            # Calcular desvio padrão como medida de contraste
            contrast = np.std(gray) / 255.0
            
            return min(contrast, 1.0)
        except Exception:
            return 0.0
    
    def _analyze_color_saturation(self, hsv_image: np.ndarray) -> float:
        """Analisa a saturação média das cores"""
        try:
            # Extrair canal de saturação
            saturation = hsv_image[:, :, 1]
            
            # Calcular saturação média
            avg_saturation = np.mean(saturation) / 255.0
            
            return min(avg_saturation, 1.0)
        except Exception:
            return 0.0
    
    def _detect_complex_patterns(self, hsv_image: np.ndarray) -> bool:
        """Detecta padrões complexos de cor (indicam penas)"""
        try:
            # Converter para escala de cinza para análise de padrões
            gray = cv2.cvtColor(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
            
            # Usar transformada de Fourier para detectar padrões repetitivos
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Calcular energia em diferentes frequências
            h, w = magnitude_spectrum.shape
            center_h, center_w = h // 2, w // 2
            
            # Energia em frequências médias (padrões complexos)
            medium_freq_mask = np.zeros_like(magnitude_spectrum)
            cv2.circle(medium_freq_mask, (center_w, center_h), min(h, w) // 4, 1, -1)
            cv2.circle(medium_freq_mask, (center_w, center_h), min(h, w) // 8, 0, -1)
            
            medium_freq_energy = np.sum(magnitude_spectrum * medium_freq_mask)
            total_energy = np.sum(magnitude_spectrum)
            
            # Se há energia significativa em frequências médias, há padrões complexos
            return medium_freq_energy / total_energy > 0.3
            
        except Exception:
            return False
    
    def _detect_iridescence(self, hsv_image: np.ndarray) -> bool:
        """Detecta iridescência (característica de penas)"""
        try:
            # Iridescência aparece como variações rápidas de matiz
            hue = hsv_image[:, :, 0]
            
            # Calcular gradiente de matiz
            grad_x = cv2.Sobel(hue, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(hue, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Iridescência tem gradientes altos de matiz
            high_gradient_pixels = np.sum(gradient_magnitude > np.mean(gradient_magnitude) * 2)
            total_pixels = gradient_magnitude.size
            
            # Se mais de 15% dos pixels têm gradientes altos, há iridescência
            return high_gradient_pixels / total_pixels > 0.15
            
        except Exception:
            return False
    
    def _analyze_shapes(self, image: np.ndarray) -> Dict[str, Any]:
        """Analisa formas básicas como uma criança reconheceria"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar contornos
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'error': 'Nenhum contorno encontrado'}
        
        # Encontrar maior contorno (assumindo que é o objeto principal)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Análise de proporções
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Análise de compactness
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        return {
            'aspect_ratio': aspect_ratio,
            'compactness': compactness,
            'area_ratio': area / (image.shape[0] * image.shape[1]) if (image.shape[0] * image.shape[1]) > 0 else 0,
            'bird_shape_score': self._calculate_shape_score(aspect_ratio, compactness)
        }
    
    def _analyze_textures(self, image: np.ndarray) -> Dict[str, Any]:
        """Analisa texturas como uma criança reconheceria"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Análise de textura usando gradientes
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calcular magnitude do gradiente
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Análise de uniformidade da textura
        texture_variance = np.var(magnitude)
        texture_uniformity = 1.0 / (1.0 + texture_variance) if texture_variance > 0 else 0.5
            
        return {
            'texture_variance': texture_variance,
            'texture_uniformity': texture_uniformity,
            'feather_like_score': self._calculate_feather_score(texture_variance, texture_uniformity)
        }
    
    def _calculate_bird_like_score(self, color_analysis: Dict, shape_analysis: Dict, texture_analysis: Dict) -> float:
        """Calcula score geral de características de pássaro - mais sensível"""
        color_score = color_analysis.get('bird_color_score', 0)
        shape_score = shape_analysis.get('bird_shape_score', 0)
        texture_score = texture_analysis.get('feather_like_score', 0)
        
        # Score baseado em características individuais
        base_score = (color_score * 0.3 + 
                     shape_score * 0.4 + 
                     texture_score * 0.3)
        
        # Bonus para múltiplas características positivas
        positive_features = sum([1 for score in [color_score, shape_score, texture_score] if score > 0.3])
        
        if positive_features >= 2:
            bonus = 0.2  # Bonus de 20% para múltiplas características
        elif positive_features >= 1:
            bonus = 0.1  # Bonus de 10% para pelo menos uma característica
        else:
            bonus = 0.0
        
        # Score final com bonus
        final_score = min(base_score + bonus, 1.0)
        
        return final_score
    
    def _calculate_shape_score(self, aspect_ratio: float, compactness: float) -> float:
        """Calcula score de forma baseado em características de pássaro - mais sensível"""
        # Pássaros típicos têm aspect_ratio entre 0.3 e 3.0 (mais amplo)
        if 0.3 <= aspect_ratio <= 3.0:
            aspect_score = 1.0
        elif 0.2 <= aspect_ratio <= 4.0:
            aspect_score = 0.8
        else:
            aspect_score = max(0, 1.0 - abs(aspect_ratio - 1.0) / 2.0)
        
        # Compactness típica de pássaros (formas arredondadas) - mais flexível
        if compactness > 0.1:  # Qualquer forma não muito alongada
            compactness_score = min(1.0, compactness * 2)  # Mais sensível
        else:
            compactness_score = compactness
        
        return (aspect_score + compactness_score) / 2
    
    def _calculate_feather_score(self, variance: float, uniformity: float) -> float:
        """Calcula score de textura de penas"""
        # Penas têm textura variada mas não muito uniforme
        variance_score = min(1.0, variance / 1000.0)  # Normalizar
        uniformity_score = uniformity
        
        return (variance_score + uniformity_score) / 2
    
    def _detect_fundamental_characteristics(self, image_path: str) -> Dict[str, Any]:
        """Detecção híbrida multi-biblioteca de características fundamentais de pássaros"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Imagem não carregada'}
            
            characteristics = {
                'has_wings': False,
                'has_beak': False,
                'has_claws': False,
                'has_feathers': False,
                'has_eyes': False,
                'bird_body_shape': False,
                'yolo_detection': False,
                'hybrid_confidence': 0.0,
                'mammal_score': 0.0,
                'bird_score': 0.0,
                'detection_votes': {},
                'model_consensus': 0.0
            }
            
            # 1. Detecção multi-biblioteca (YOLO, OpenCV, MediaPipe)
            detection_results = self._multi_library_detection(image)
            characteristics.update(detection_results)
            
            # 2. Análise visual híbrida (sempre executada - mantém funcionamento atual)
            visual_characteristics = self._detect_visual_characteristics(image)
            characteristics.update(visual_characteristics)
            
            # 3. Análise de contornos ultra-avançada (mantém funcionamento atual)
            contour_analysis = self._analyze_contours_ultra_advanced(image)
            characteristics.update(contour_analysis)
            
            # 4. Análise de textura ultra-híbrida (mantém funcionamento atual)
            texture_analysis = self._analyze_texture_ultra_hybrid(image)
            characteristics.update(texture_analysis)
            
            # 5. Análise de forma ultra-rigorosa (mantém funcionamento atual)
            shape_analysis = self._analyze_shape_ultra_rigorous(image)
            characteristics.update(shape_analysis)
            
            # 6. Análise de padrões biométricos (mantém funcionamento atual)
            biometric_analysis = self._analyze_biometric_patterns(image)
            characteristics.update(biometric_analysis)
            
            # 7. Sistema de votação ponderada entre todas as técnicas
            consensus_result = self._calculate_model_consensus(characteristics, detection_results)
            characteristics.update(consensus_result)
            
            # 8. Calcular scores de mamífero vs pássaro (melhorado com votação)
            mammal_score = self._calculate_mammal_score_enhanced(characteristics)
            bird_score = self._calculate_bird_score_enhanced(characteristics)
            
            characteristics['mammal_score'] = mammal_score
            characteristics['bird_score'] = bird_score
            
            # 9. Calcular confiança híbrida final
            hybrid_confidence = self._calculate_final_hybrid_confidence(
                characteristics, mammal_score, bird_score
            )
            characteristics['hybrid_confidence'] = hybrid_confidence
            
            return characteristics
            
        except Exception as e:
            self.debug_logger.error(f"[ERRO] Erro na detecção de características: {str(e)}")
            return {'error': str(e)}
    
    def _detect_visual_characteristics(self, image: np.ndarray) -> Dict[str, Any]:
        """Detecção visual SIMPLIFICADA e EFICAZ"""
        try:
            # Análise básica de cores
            color_analysis = self._analyze_colors(image)
            
            # Análise básica de formas
            shape_analysis = self._analyze_shapes(image)
            
            # Análise básica de texturas
            texture_analysis = self._analyze_textures(image)
            
            # Detecção simples de características
            has_eyes = self._detect_simple_eyes(image)
            has_wings = self._detect_simple_wings(image)
            has_beak = self._detect_simple_beak(image)
            has_feathers = self._detect_simple_feathers(image)
            has_claws = self._detect_simple_claws(image)
            
            # NOVO: Detecção de características de répteis
            has_scales = self._detect_simple_scales(image)
            has_dorsal_crest = self._detect_simple_dorsal_crest(image)
            has_elongated_body = self._detect_simple_elongated_body(image)
            has_scaly_texture = self._detect_simple_scaly_texture(image)
            
            # Detecção simples de mamíferos
            has_mammal_features = self._detect_simple_mammal_features(image)
            
            # Calcular scores combinados
            bird_like_features = self._calculate_simple_bird_score(
                has_eyes, has_wings, has_beak, has_feathers, has_claws,
                color_analysis, shape_analysis, texture_analysis
            )
            
            bird_shape_score = self._calculate_simple_shape_score(shape_analysis)
            bird_color_score = color_analysis.get('bird_color_score', 0)
            
            return {
                'has_eyes': has_eyes,
                'has_wings': has_wings,
                'has_beak': has_beak,
                'has_feathers': has_feathers,
                'has_claws': has_claws,
                'has_mammal_features': has_mammal_features,
                # NOVO: Características de répteis
                'has_scales': has_scales,
                'has_dorsal_crest': has_dorsal_crest,
                'has_elongated_body': has_elongated_body,
                'has_scaly_texture': has_scaly_texture,
                'bird_like_features': bird_like_features,
                'bird_shape_score': bird_shape_score,
                'bird_color_score': bird_color_score,
                'color_analysis': color_analysis,
                'shape_analysis': shape_analysis,
                'texture_analysis': texture_analysis,
                # NOVO: Análise de cores melhorada
                'dominant_colors': color_analysis.get('dominant_colors', []),
                'has_complex_patterns': color_analysis.get('has_complex_patterns', False),
                'has_iridescence': color_analysis.get('has_iridescence', False)
            }
            
        except Exception as e:
            self.debug_logger.error(f"Erro na detecção visual: {str(e)}")
            return {
                'has_eyes': False,
                'has_wings': False,
                'has_beak': False,
                'has_feathers': False,
                'has_claws': False,
                'has_mammal_features': False,
                'bird_like_features': 0.0,
                'bird_shape_score': 0.0,
                'bird_color_score': 0.0,
                'error': str(e)
            }
    
    def _detect_simple_eyes(self, image: np.ndarray) -> bool:
        """Detecção RIGOROSA de olhos de pássaro"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Método 1: Detectar círculos pequenos e escuros (mais rigoroso)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 30, 
                                     param1=60, param2=40, minRadius=8, maxRadius=25)
            
            eye_count = 0
            if circles is not None:
                for circle in circles[0]:
                    x, y, r = circle
                    # Verificar se o círculo está em região escura (olho)
                    if y < image.shape[0] and x < image.shape[1]:
                        pixel_value = gray[int(y), int(x)]
                        if pixel_value < 80:  # Olhos são escuros
                            eye_count += 1
            
            # Método 2: Detectar regiões pequenas e escuras (mais rigoroso)
            _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            dark_regions = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 30 < area < 150:  # Tamanho mais restritivo
                    # Verificar se é circular
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.7:  # Mais circular
                            dark_regions += 1
            
            # Requer pelo menos 1 olho detectado por círculos OU 2 regiões escuras
            return eye_count >= 1 or dark_regions >= 2
            
        except:
            return False
    
    def _detect_simple_wings(self, image: np.ndarray) -> bool:
        """Detecção RIGOROSA de asas de pássaro"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Usar múltiplas técnicas de detecção de bordas
            edges1 = cv2.Canny(gray, 30, 100)
            edges2 = cv2.Canny(gray, 50, 150)
            combined_edges = cv2.bitwise_or(edges1, edges2)
            
            contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            wing_candidates = 0
            for contour in contours:
                if len(contour) > 10 and cv2.contourArea(contour) > 200:  # Mais rigoroso
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 1.0
                    
                    # Calcular características adicionais
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Asas são alongadas horizontalmente E têm forma específica
                    if (aspect_ratio > 2.0 and  # Mais alongadas
                        compactness > 0.1 and    # Não muito compactas
                        compactness < 0.6 and     # Mas não muito irregulares
                        area > 300):             # Área mínima maior
                        wing_candidates += 1
            
            # Requer pelo menos 1 candidato forte de asa
            return wing_candidates >= 1
            
        except:
            return False
    
    def _detect_simple_beak(self, image: np.ndarray) -> bool:
        """Detecção MELHORADA de bico de pássaro"""
        try:
            # VERIFICAÇÃO ESPECÍFICA PARA ANIMAIS AQUÁTICOS
            # Se a imagem tem características de ambiente aquático, ser mais rigoroso
            if self._is_aquatic_environment(image):
                return False  # Não detectar bico em ambiente aquático
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Método 1: Detecção por bordas (mais rigorosa)
            edges1 = cv2.Canny(gray, 30, 100)
            edges2 = cv2.Canny(gray, 50, 150)
            combined_edges = cv2.bitwise_or(edges1, edges2)
            
            contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            beak_candidates = 0
            for contour in contours:
                if len(contour) > 8 and cv2.contourArea(contour) > 150:  # MUITO mais rigoroso
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 1.0
                    
                    # Calcular características adicionais
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Bicos são alongados, pontiagudos e têm forma específica - CRITÉRIOS MUITO RIGOROSOS
                    if (aspect_ratio > 4.0 and   # MUITO mais alongados (evita nadadeiras)
                        compactness < 0.6 and   # Mais pontiagudos
                        area > 200 and         # Área mínima MUITO maior
                        area < 600 and         # Não muito grandes
                        y < h * 0.4):          # Deve estar na parte superior (evita nadadeiras inferiores)
                        beak_candidates += 1
            
            # Método 2: Detecção por forma triangular (bicos têm formato triangular)
            for contour in contours:
                if len(contour) > 6 and cv2.contourArea(contour) > 60:
                    # Aproximar contorno por polígono
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Se tem 3 vértices, pode ser triangular (bico)
                    if len(approx) == 3:
                        beak_candidates += 1
            
            # Método 3: Detecção por posição (bicos estão na parte superior/frontal)
            h, w = image.shape[:2]
            for contour in contours:
                if len(contour) > 5 and cv2.contourArea(contour) > 50:
                    x, y, contour_w, contour_h = cv2.boundingRect(contour)
                    aspect_ratio = contour_w / contour_h if contour_h > 0 else 1.0
                    
                    # Bicos estão na parte superior da imagem
                    if (aspect_ratio > 2.0 and 
                        y < h * 0.3 and  # Parte superior
                        contour_w < w * 0.3):  # Não muito largos
                        beak_candidates += 1
            
            # Requer pelo menos 2 candidatos fortes de bico (mais rigoroso)
            return beak_candidates >= 2
            
        except:
            return False
    
    def _detect_simple_feathers(self, image: np.ndarray) -> bool:
        """Detecção MELHORADA de penas para distinguir de pelo"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Método 1: Análise de textura usando gradientes (mais rigorosa)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            texture_variance = np.var(magnitude)
            texture_mean = np.mean(magnitude)
            
            # Método 2: Análise de padrões repetitivos (Fourier)
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Calcular energia em diferentes frequências
            h, w = magnitude_spectrum.shape
            center_y, center_x = h // 2, w // 2
            
            # Energia em frequências baixas (padrões grandes - pelo)
            low_freq_energy = np.sum(magnitude_spectrum[center_y-10:center_y+10, center_x-10:center_x+10])
            
            # Energia em frequências médias (padrões de penas)
            medium_freq_energy = np.sum(magnitude_spectrum[center_y-20:center_y+20, center_x-20:center_x+20]) - low_freq_energy
            
            # Energia total
            total_energy = np.sum(magnitude_spectrum)
            
            # Calcular ratios
            low_freq_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
            medium_freq_ratio = medium_freq_energy / total_energy if total_energy > 0 else 0
            
            # Método 3: Análise de densidade de bordas
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Método 4: Análise de contornos regulares (penas têm padrões mais regulares)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            regular_patterns = 0
            
            for contour in contours:
                if len(contour) > 10:
                    # Calcular regularidade do contorno
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    contour_area = cv2.contourArea(contour)
                    
                    if contour_area > 0:
                        solidity = contour_area / hull_area
                        # Penas têm contornos mais regulares que pelo
                        if 0.6 < solidity < 0.9:
                            regular_patterns += 1
            
            pattern_score = min(1.0, regular_patterns / 15.0)  # Normalizar
            
            # Critérios para distinguir penas de pelo
            feather_score = 0.0
            
            # Critério 1: Textura variável mas não muito (penas têm textura específica)
            if 150 < texture_variance < 500:  # Range mais específico
                feather_score += 0.3
            
            # Critério 2: Padrões de frequência média altos (penas)
            if medium_freq_ratio > 0.25:  # Mais rigoroso
                feather_score += 0.3
            
            # Critério 3: Densidade de bordas moderada (penas têm bordas definidas)
            if 0.08 < edge_density < 0.2:  # Range específico
                feather_score += 0.2
            
            # Critério 4: Padrões regulares (penas são mais regulares que pelo)
            if pattern_score > 0.3:  # Mais rigoroso
                feather_score += 0.2
            
            # Critério 5: Não deve ter muitos padrões de baixa frequência (pelo)
            if low_freq_ratio < 0.4:  # Evitar pelo
                feather_score += 0.1
            
            # Retornar True apenas se score for alto (penas) e não for pelo
            return feather_score > 0.6 and low_freq_ratio < 0.5
            
        except:
            return False
    
    def _is_aquatic_environment(self, image: np.ndarray) -> bool:
        """Detecta se a imagem tem características de ambiente aquático"""
        try:
            # Converter para HSV para análise de cores
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Definir ranges para cores aquáticas (azul, verde-azulado)
            # Azul do oceano
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            
            # Verde-azulado (água turva)
            lower_green_blue = np.array([80, 50, 50])
            upper_green_blue = np.array([100, 255, 255])
            
            # Criar máscaras
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            mask_green_blue = cv2.inRange(hsv, lower_green_blue, upper_green_blue)
            
            # Combinar máscaras
            aquatic_mask = cv2.bitwise_or(mask_blue, mask_green_blue)
            
            # Calcular porcentagem de pixels aquáticos
            total_pixels = image.shape[0] * image.shape[1]
            aquatic_pixels = np.sum(aquatic_mask > 0)
            aquatic_percentage = aquatic_pixels / total_pixels
            
            # Se mais de 30% da imagem tem cores aquáticas, é ambiente aquático
            return aquatic_percentage > 0.3
            
        except:
            return False
    
    def _detect_simple_claws(self, image: np.ndarray) -> bool:
        """Detecção MELHORADA de garras de pássaro"""
        try:
            # VERIFICAÇÃO ESPECÍFICA PARA ANIMAIS AQUÁTICOS
            # Se a imagem tem características de ambiente aquático, ser mais rigoroso
            if self._is_aquatic_environment(image):
                return False  # Não detectar garras em ambiente aquático
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Método 1: Detecção por bordas (mais rigorosa)
            edges1 = cv2.Canny(gray, 30, 100)
            edges2 = cv2.Canny(gray, 50, 150)
            combined_edges = cv2.bitwise_or(edges1, edges2)
            
            contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            claw_candidates = 0
            h, w = image.shape[:2]
            
            for contour in contours:
                if len(contour) > 6 and cv2.contourArea(contour) > 60:  # MUITO mais rigoroso
                    x, y, contour_w, contour_h = cv2.boundingRect(contour)
                    aspect_ratio = contour_w / contour_h if contour_h > 0 else 1.0
                    area = cv2.contourArea(contour)
                    
                    # Garras são pequenas, alongadas e estão na parte inferior - CRITÉRIOS MUITO RIGOROSOS
                    if (aspect_ratio > 2.5 and  # MUITO mais alongadas (evita nadadeiras)
                        area < 100 and         # Menores (evita nadadeiras grandes)
                        area > 60 and          # Mas não muito pequenas
                        y > h * 0.7 and        # MUITO mais na parte inferior (evita nadadeiras laterais)
                        contour_w < w * 0.1): # Largura pequena (evita nadadeiras largas)
                        claw_candidates += 1
            
            # Método 2: Detecção por forma pontiaguda (garras são pontiagudas)
            for contour in contours:
                if len(contour) > 5 and cv2.contourArea(contour) > 25:
                    # Calcular convexidade
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    area = cv2.contourArea(contour)
                    
                    if hull_area > 0:
                        solidity = area / hull_area
                        
                        # Garras são menos convexas (mais pontiagudas)
                        if solidity < 0.7:  # Pontiagudas
                            x, y, contour_w, contour_h = cv2.boundingRect(contour)
                            aspect_ratio = contour_w / contour_h if contour_h > 0 else 1.0
                            
                            if (aspect_ratio > 1.5 and 
                                cv2.contourArea(contour) < 120 and
                                y > h * 0.5):  # Parte inferior
                                claw_candidates += 1
            
            # Método 3: Detecção por múltiplas pequenas estruturas (pés têm múltiplas garras)
            small_structures = 0
            for contour in contours:
                if len(contour) > 4 and cv2.contourArea(contour) > 20:
                    x, y, contour_w, contour_h = cv2.boundingRect(contour)
                    aspect_ratio = contour_w / contour_h if contour_h > 0 else 1.0
                    
                    if (aspect_ratio > 1.3 and 
                        cv2.contourArea(contour) < 80 and
                        y > h * 0.7):  # Parte muito inferior
                        small_structures += 1
            
            # Se há múltiplas pequenas estruturas na parte inferior, podem ser garras
            if small_structures >= 2:
                claw_candidates += 1
            
            # Requer pelo menos 2 candidatos fortes de garra (mais rigoroso)
            return claw_candidates >= 2
            
        except:
            return False
    
    def _detect_simple_mammal_features(self, image: np.ndarray) -> bool:
        """Detecção ULTRA-RIGOROSA de características de mamíferos"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            mammal_features = 0
            
            # Detectar orelhas (formas arredondadas grandes) - EXTREMAMENTE RIGOROSO
            for contour in contours:
                if len(contour) > 15 and cv2.contourArea(contour) > 1000:  # EXTREMAMENTE maior
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    area = cv2.contourArea(contour)
                    
                    if hull_area > 0:
                        solidity = area / hull_area
                        
                        # Orelhas são EXTREMAMENTE convexas (solididade quase perfeita)
                        if solidity > 0.98:  # EXTREMAMENTE restritivo
                            mammal_features += 1
            
            # Detectar focinho (formas alongadas horizontalmente na parte inferior) - EXTREMAMENTE RIGOROSO
            h, w = image.shape[:2]
            for contour in contours:
                if len(contour) > 15 and cv2.contourArea(contour) > 500:  # EXTREMAMENTE maior
                    x, y, contour_w, contour_h = cv2.boundingRect(contour)
                    aspect_ratio = contour_w / contour_h if contour_h > 0 else 1.0
                    
                    # Focinho é EXTREMAMENTE alongado horizontalmente e está na parte inferior
                    if aspect_ratio > 4.0 and y > h * 0.85:  # EXTREMAMENTE restritivo
                        mammal_features += 1
            
            # Detectar nariz (pequenas formas circulares escuras) - EXTREMAMENTE RIGOROSO
            for contour in contours:
                if len(contour) > 10 and cv2.contourArea(contour) > 100:  # Mais rigoroso
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    area = cv2.contourArea(contour)
                    
                    if hull_area > 0:
                        solidity = area / hull_area
                        x, y, contour_w, contour_h = cv2.boundingRect(contour)
                        aspect_ratio = contour_w / contour_h if contour_h > 0 else 1.0
                        
                        # Nariz é EXTREMAMENTE circular e convexo
                        if 0.9 < aspect_ratio < 1.1 and solidity > 0.95:  # EXTREMAMENTE restritivo
                            mammal_features += 1
            
            # Retornar True apenas se encontrar características EXTREMAMENTE claras de mamífero
            return mammal_features >= 3  # Aumentado de 2 para 3
            
        except:
            return False
    
    def _detect_simple_scales(self, image: np.ndarray) -> bool:
        """Detecção de escamas (característica distintiva de répteis)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detectar padrões repetitivos pequenos (escamas)
            # Usar filtro de alta frequência para detectar texturas granulares
            kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
            filtered = cv2.filter2D(gray, -1, kernel)
            
            # Detectar bordas para encontrar contornos de escamas
            edges = cv2.Canny(filtered, 20, 60)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            scale_candidates = 0
            h, w = gray.shape
            
            for contour in contours:
                if len(contour) > 4 and cv2.contourArea(contour) > 10:
                    x, y, contour_w, contour_h = cv2.boundingRect(contour)
                    aspect_ratio = contour_w / contour_h if contour_h > 0 else 1.0
                    area = cv2.contourArea(contour)
                    
                    # Escamas são pequenas, aproximadamente circulares/ovais
                    if (0.5 < aspect_ratio < 2.0 and 
                        10 < area < 200 and
                        contour_w < 30 and contour_h < 30):
                        scale_candidates += 1
            
            # Se há muitas pequenas estruturas circulares, provavelmente são escamas
            return scale_candidates >= 15
            
        except:
            return False
    
    def _detect_simple_dorsal_crest(self, image: np.ndarray) -> bool:
        """Detecção de crista dorsal (comum em iguanas)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            h, w = gray.shape
            crest_candidates = 0
            
            for contour in contours:
                if len(contour) > 10 and cv2.contourArea(contour) > 100:
                    x, y, contour_w, contour_h = cv2.boundingRect(contour)
                    aspect_ratio = contour_w / contour_h if contour_h > 0 else 1.0
                    
                    # Crista dorsal é uma linha vertical alongada no centro superior
                    if (aspect_ratio < 0.5 and  # Muito mais alta que larga
                        contour_h > h * 0.3 and  # Pelo menos 30% da altura
                        contour_w < w * 0.2 and  # Não muito larga
                        y < h * 0.4):  # Na parte superior
                        crest_candidates += 1
            
            return crest_candidates >= 1
            
        except:
            return False
    
    def _detect_simple_elongated_body(self, image: np.ndarray) -> bool:
        """Detecção de corpo alongado (típico de répteis)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            h, w = gray.shape
            elongated_candidates = 0
            
            for contour in contours:
                if len(contour) > 15 and cv2.contourArea(contour) > 500:
                    x, y, contour_w, contour_h = cv2.boundingRect(contour)
                    aspect_ratio = contour_w / contour_h if contour_h > 0 else 1.0
                    
                    # Corpo alongado tem proporção específica
                    if (aspect_ratio > 1.5 and  # Mais largo que alto
                        contour_w > w * 0.4 and  # Pelo menos 40% da largura
                        contour_h > h * 0.2):  # Pelo menos 20% da altura
                        elongated_candidates += 1
            
            return elongated_candidates >= 1
            
        except:
            return False
    
    def _detect_simple_scaly_texture(self, image: np.ndarray) -> bool:
        """Detecção de textura escamosa"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calcular variância local para detectar texturas granulares
            kernel_size = 5
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
            variance = sqr_mean - mean**2
            
            # Texturas escamosas têm alta variância local
            high_variance_pixels = np.sum(variance > np.mean(variance) * 1.5)
            total_pixels = variance.size
            
            # Se mais de 20% dos pixels têm alta variância, é textura escamosa
            return high_variance_pixels / total_pixels > 0.2
            
        except:
            return False
    
    def _calculate_simple_bird_score(self, has_eyes: bool, has_wings: bool, has_beak: bool, 
                                   has_feathers: bool, has_claws: bool,
                                   color_analysis: Dict, shape_analysis: Dict, texture_analysis: Dict) -> float:
        """Calcula score simples de características de pássaro"""
        score = 0.0
        
        # Contar características básicas
        characteristics = [has_eyes, has_wings, has_beak, has_feathers, has_claws]
        char_count = sum(characteristics)
        
        # Score baseado no número de características
        score += char_count * 0.2
        
        # Bonus por múltiplas características
        if char_count >= 3:
            score += 0.3
        elif char_count >= 2:
            score += 0.2
        
        # Adicionar scores de análise visual
        score += color_analysis.get('bird_color_score', 0) * 0.3
        score += shape_analysis.get('bird_shape_score', 0) * 0.3
        score += texture_analysis.get('bird_texture_score', 0) * 0.2
        
        return min(score, 1.0)
    
    def _calculate_simple_shape_score(self, shape_analysis: Dict) -> float:
        """Calcula score simples de forma"""
        return shape_analysis.get('bird_shape_score', 0)
    
    def _analyze_body_structure(self, image: np.ndarray) -> Dict[str, bool]:
        """Análise da estrutura corporal para distinguir pássaros de mamíferos"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar contornos principais
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'has_bird_body': False, 'has_mammal_body': False}
        
        # Encontrar maior contorno (corpo principal)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calcular características estruturais
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Compactness (circularidade)
        compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Convexidade
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Critérios para pássaros vs mamíferos
        has_bird_body = (
            0.3 < aspect_ratio < 2.5 and  # Proporções típicas de pássaros
            0.2 < compactness < 0.7 and   # Não muito circular, não muito alongado
            0.6 < solidity < 0.9          # Estrutura moderadamente convexa
        )
        
        has_mammal_body = (
            0.8 < aspect_ratio < 1.2 and  # Proporções típicas de mamíferos (mais quadrados)
            compactness > 0.6 and         # Mais circulares
            solidity > 0.85              # Muito convexos (formas arredondadas)
        )
        
        return {
            'has_bird_body': has_bird_body,
            'has_mammal_body': has_mammal_body,
            'body_aspect_ratio': aspect_ratio,
            'body_compactness': compactness,
            'body_solidity': solidity
        }
    
    def _analyze_advanced_texture(self, image: np.ndarray) -> Dict[str, bool]:
        """Análise avançada de textura para distinguir penas de pelo"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Análise de gradientes locais
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calcular estatísticas de textura
        texture_variance = np.var(magnitude)
        
        # 2. Análise de padrões repetitivos (Fourier)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Calcular energia em diferentes frequências
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # Energia em frequências baixas (padrões grandes - pelo)
        low_freq_energy = np.sum(magnitude_spectrum[center_y-10:center_y+10, center_x-10:center_x+10])
        
        # Energia em frequências médias (padrões de penas)
        medium_freq_energy = np.sum(magnitude_spectrum[center_y-20:center_y+20, center_x-20:center_x+20]) - low_freq_energy
        
        # Energia total
        total_energy = np.sum(magnitude_spectrum)
        
        # Calcular ratios
        low_freq_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
        medium_freq_ratio = medium_freq_energy / total_energy if total_energy > 0 else 0
        
        # 3. Análise de densidade de bordas
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Critérios para penas vs pelo
        has_feather_texture = (
            medium_freq_ratio > 0.3 and      # Padrões de penas dominantes
            edge_density > 0.1 and           # Alta densidade de bordas
            texture_variance > 100 and       # Alta variabilidade de textura
            low_freq_ratio < 0.4             # Poucos padrões grandes (pelo)
        )
        
        has_fur_texture = (
            low_freq_ratio > 0.5 and         # Padrões grandes dominantes (pelo)
            edge_density < 0.08 and          # Baixa densidade de bordas
            texture_variance < 200 and       # Baixa variabilidade de textura
            medium_freq_ratio < 0.2          # Poucos padrões de penas
        )
        
        return {
            'has_feather_texture': has_feather_texture,
            'has_fur_texture': has_fur_texture,
            'texture_variance': texture_variance,
            'edge_density': edge_density,
            'low_freq_ratio': low_freq_ratio,
            'medium_freq_ratio': medium_freq_ratio
        }
    
    def _analyze_body_proportions(self, image: np.ndarray) -> Dict[str, bool]:
        """Análise de proporções corporais para distinguir pássaros de mamíferos"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar contornos
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'has_bird_proportions': False, 'has_mammal_proportions': False}
        
        # Analisar todos os contornos significativos
        bird_features = 0
        mammal_features = 0
        
        for contour in contours:
            if cv2.contourArea(contour) < 100:  # Ignorar contornos muito pequenos
                continue
                
            # Calcular características do contorno
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 1.0
            
            # Compactness
            compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Critérios para pássaros (formas alongadas, moderadamente compactas)
            if 1.5 < aspect_ratio < 4.0 and 0.2 < compactness < 0.6:
                bird_features += 1
            
            # Critérios para mamíferos (formas mais arredondadas, muito compactas)
            elif 0.8 < aspect_ratio < 1.5 and compactness > 0.6:
                mammal_features += 1
        
        # Decisão baseada na contagem de características
        has_bird_proportions = bird_features > mammal_features and bird_features > 0
        has_mammal_proportions = mammal_features > bird_features and mammal_features > 0
        
        return {
            'has_bird_proportions': has_bird_proportions,
            'has_mammal_proportions': has_mammal_proportions,
            'bird_feature_count': bird_features,
            'mammal_feature_count': mammal_features
        }
    
    def _detect_bird_eyes(self, image: np.ndarray) -> bool:
        """Detecção específica de olhos de pássaros"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar círculos pequenos (olhos)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, 
                                 param1=50, param2=30, minRadius=3, maxRadius=25)
        
        if circles is not None:
            # Verificar se os círculos têm características de olhos de pássaro
            for circle in circles[0]:
                x, y, r = circle
                # Verificar intensidade (olhos são escuros)
                roi = gray[int(y-r):int(y+r), int(x-r):int(x+r)]
                if roi.size > 0:
                    mean_intensity = np.mean(roi)
                    if mean_intensity < 100:  # Olhos são escuros
                        return True
        
        return False
    
    def _detect_bird_wings(self, image: np.ndarray) -> bool:
        """Detecção específica de asas de pássaros"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        wing_count = 0
        for contour in contours:
            if len(contour) > 10 and cv2.contourArea(contour) > 200:
                # Ajustar elipse
                ellipse = cv2.fitEllipse(contour)
                (center, axes, angle) = ellipse
                major_axis, minor_axis = axes
                
                if major_axis > 0 and minor_axis > 0:
                    aspect_ratio = major_axis / minor_axis
                    # Asas são alongadas
                    if aspect_ratio > 2.0:
                        wing_count += 1
        
        return wing_count >= 1
    
    def _detect_bird_beak(self, image: np.ndarray) -> bool:
        """Detecção específica de bico de pássaros"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) > 5 and cv2.contourArea(contour) > 50:
                # Calcular convexidade
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                area = cv2.contourArea(contour)
                
                if hull_area > 0:
                    solidity = area / hull_area
                    
                    # Aspect ratio
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 1.0
                    
                    # Bicos são pontiagudos (baixa convexidade) e alongados
                    if solidity < 0.7 and aspect_ratio > 2.0:
                        return True
        
        return False
    
    def _detect_bird_feathers(self, image: np.ndarray) -> bool:
        """Detecção específica de penas de pássaros"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Análise de textura usando gradientes
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calcular variabilidade da textura
        texture_variance = np.var(magnitude)
        
        # Análise de padrões repetitivos
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Energia em frequências médias (padrões de penas)
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        medium_freq_energy = np.sum(magnitude_spectrum[center_y-15:center_y+15, center_x-15:center_x+15])
        total_energy = np.sum(magnitude_spectrum)
        medium_freq_ratio = medium_freq_energy / total_energy if total_energy > 0 else 0
        
        # Critérios para penas
        return texture_variance > 150 and medium_freq_ratio > 0.25
    
    def _detect_bird_claws(self, image: np.ndarray) -> bool:
        """Detecção específica de garras de pássaros"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        claw_count = 0
        for contour in contours:
            if len(contour) > 5 and cv2.contourArea(contour) > 20:
                # Calcular características de forma
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 1.0
                
                # Garras são pequenas, alongadas e pontiagudas
                if compactness < 0.4 and aspect_ratio > 1.5:
                    claw_count += 1
        
        return claw_count >= 1
    
    def _detect_mammal_features(self, image: np.ndarray) -> bool:
        """Detecção específica de características de mamíferos"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Detectar orelhas (formas triangulares ou arredondadas)
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ear_count = 0
        for contour in contours:
            if len(contour) > 5 and cv2.contourArea(contour) > 100:
                # Calcular convexidade
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                area = cv2.contourArea(contour)
                
                if hull_area > 0:
                    solidity = area / hull_area
                    
                    # Aspect ratio
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 1.0
                    
                    # Orelhas são convexas e aproximadamente circulares
                    if solidity > 0.8 and 0.7 < aspect_ratio < 1.3:
                        ear_count += 1
        
        # 2. Detectar focinho (forma alongada na parte inferior)
        # Análise da região inferior da imagem
        h, w = gray.shape
        bottom_region = gray[int(h*0.6):h, :]
        
        if bottom_region.size > 0:
            # Detectar bordas na região inferior
            bottom_edges = cv2.Canny(bottom_region, 30, 100)
            bottom_contours, _ = cv2.findContours(bottom_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            snout_count = 0
            for contour in bottom_contours:
                if len(contour) > 5 and cv2.contourArea(contour) > 50:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 1.0
                    
                    # Focinho é alongado horizontalmente
                    if aspect_ratio > 2.0:
                        snout_count += 1
            
            # Se encontrou orelhas ou focinho, provavelmente é mamífero
            return ear_count >= 1 or snout_count >= 1
        
        return ear_count >= 1
    
    def _detect_beak_shape(self, image: np.ndarray) -> bool:
        """Detecta formas pontiagudas que podem ser bicos"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar bordas
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        beak_like_shapes = 0
        for contour in contours:
            if len(contour) > 5:
                # Calcular convexidade
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                contour_area = cv2.contourArea(contour)
                
                if contour_area > 0:
                    solidity = contour_area / hull_area
                    # Formas pontiagudas têm baixa solididade
                    if solidity < 0.7:
                        # Verificar se tem forma alongada
                        ellipse = cv2.fitEllipse(contour)
                        (center, axes, orientation) = ellipse
                        major_axis = max(axes)
                        minor_axis = min(axes)
                        if minor_axis > 0 and major_axis / minor_axis > 2.0:
                            beak_like_shapes += 1
        
        return beak_like_shapes > 0
    
    def _detect_feather_texture(self, image: np.ndarray) -> bool:
        """Detecta textura de penas usando análise de padrões - mais rigoroso"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Método 1: Análise de Fourier para padrões repetitivos
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
        medium_freq_energy = np.sum(magnitude_spectrum[center_y-20:center_y+20, center_x-20:center_x+20])
        total_energy = np.sum(magnitude_spectrum)
        
        feather_score_fourier = medium_freq_energy / total_energy if total_energy > 0 else 0
        
        # Método 2: Análise de textura usando gradientes
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calcular variância da textura
        texture_variance = np.var(magnitude)
        texture_uniformity = 1.0 / (1.0 + texture_variance) if texture_variance > 0 else 0.5
        
        # Método 3: Análise de padrões repetitivos específicos de penas
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Método 4: Detectar padrões de penas específicos
        # Penas têm padrões mais regulares que pelo
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regular_patterns = 0
        
        for contour in contours:
            if len(contour) > 10:
                # Calcular regularidade do contorno
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                contour_area = cv2.contourArea(contour)
                
                if contour_area > 0:
                    solidity = contour_area / hull_area
                    # Penas têm contornos mais regulares que pelo
                    if 0.6 < solidity < 0.9:
                        regular_patterns += 1
        
        pattern_score = min(1.0, regular_patterns / 10.0)  # Normalizar
        
        # Combinar os métodos com pesos ajustados
        feather_score = (feather_score_fourier * 0.3 + 
                        texture_uniformity * 0.2 + 
                        edge_density * 0.2 +
                        pattern_score * 0.3)
        
        return feather_score > 0.15  # Threshold mais alto para evitar falsos positivos
    
    def _analyze_contours_advanced(self, image: np.ndarray) -> Dict[str, bool]:
        """Análise avançada de contornos para detectar características de pássaro"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Múltiplas técnicas de detecção de bordas
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges3 = cv2.Canny(gray, 100, 200)
        
        # Combinar bordas
        combined_edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
        
        # Encontrar contornos
        contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        characteristics = {
            'has_wings_advanced': False,
            'has_beak_advanced': False,
            'has_body_advanced': False,
            'contour_count': len(contours)
        }
        
        if not contours:
            return characteristics
        
        # Analisar cada contorno
        wing_like = 0
        beak_like = 0
        body_like = 0
        
        for contour in contours:
            if len(contour) < 5:
                continue
                
            # Calcular características do contorno
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area < 100:  # Ignorar contornos muito pequenos
                continue
            
            # Calcular aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 1.0
            
            # Calcular compactness
            compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Calcular convexidade
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Detectar asas (formas alongadas)
            if 2.0 < aspect_ratio < 6.0 and compactness > 0.1:
                wing_like += 1
            
            # Detectar bico (formas pontiagudas e alongadas)
            if aspect_ratio > 3.0 and solidity < 0.8:
                beak_like += 1
            
            # Detectar corpo (formas arredondadas)
            if 0.5 < aspect_ratio < 2.0 and compactness > 0.3:
                body_like += 1
        
        characteristics['has_wings_advanced'] = wing_like > 0
        characteristics['has_beak_advanced'] = beak_like > 0
        characteristics['has_body_advanced'] = body_like > 0
        
        return characteristics
    
    def _analyze_texture_hybrid(self, image: np.ndarray) -> Dict[str, bool]:
        """Análise híbrida de textura para distinguir penas de pelo"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        characteristics = {
            'has_feathers_hybrid': False,
            'has_fur_texture': False,
            'texture_regularity': 0.0
        }
        
        # Método 1: Análise de gradientes locais
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calcular regularidade da textura
        texture_variance = np.var(magnitude)
        texture_mean = np.mean(magnitude)
        
        # Método 2: Análise de padrões repetitivos
        # Usar transformada de Fourier para detectar padrões
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Calcular energia em diferentes frequências
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # Energia em frequências baixas (padrões grandes)
        low_freq_energy = np.sum(magnitude_spectrum[center_y-10:center_y+10, center_x-10:center_x+10])
        
        # Energia em frequências médias (padrões de penas)
        medium_freq_energy = np.sum(magnitude_spectrum[center_y-20:center_y+20, center_x-20:center_x+20]) - low_freq_energy
        
        # Energia total
        total_energy = np.sum(magnitude_spectrum)
        
        # Calcular scores
        low_freq_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
        medium_freq_ratio = medium_freq_energy / total_energy if total_energy > 0 else 0
        
        # Método 3: Análise de bordas locais
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Critérios para distinguir penas de pelo
        # Penas: padrões mais regulares, frequências médias altas
        # Pelo: padrões menos regulares, frequências baixas altas
        
        feather_score = (
            medium_freq_ratio * 0.4 +  # Padrões de penas
            edge_density * 0.3 +       # Densidade de bordas
            (1.0 - low_freq_ratio) * 0.3  # Menos padrões grandes
        )
        
        fur_score = (
            low_freq_ratio * 0.5 +     # Padrões grandes (pelo)
            (1.0 - medium_freq_ratio) * 0.3 +  # Menos padrões de penas
            (1.0 - edge_density) * 0.2  # Menos bordas
        )
        
        characteristics['has_feathers_hybrid'] = feather_score > 0.3
        characteristics['has_fur_texture'] = fur_score > 0.4
        characteristics['texture_regularity'] = feather_score
        
        return characteristics
    
    def _calculate_hybrid_confidence(self, characteristics: Dict, yolo_detections: int, visual_characteristics: Dict) -> float:
        """Calcula confiança híbrida baseada em múltiplas técnicas"""
        confidence_factors = []
        
        # Fator 1: Detecção YOLO
        if characteristics.get('yolo_detection', False):
            confidence_factors.append(0.8)  # Alta confiança se YOLO detectou
        
        # Fator 2: Características visuais básicas
        visual_count = sum([
            characteristics.get('has_wings', False),
            characteristics.get('has_beak', False),
            characteristics.get('has_feathers', False),
            characteristics.get('has_eyes', False)
        ])
        
        if visual_count >= 3:
            confidence_factors.append(0.9)
        elif visual_count >= 2:
            confidence_factors.append(0.7)
        elif visual_count >= 1:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.2)
        
        # Fator 3: Análise avançada de contornos
        advanced_count = sum([
            characteristics.get('has_wings_advanced', False),
            characteristics.get('has_beak_advanced', False),
            characteristics.get('has_body_advanced', False)
        ])
        
        if advanced_count >= 2:
            confidence_factors.append(0.8)
        elif advanced_count >= 1:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
        
        # Fator 4: Análise de textura
        if characteristics.get('has_feathers_hybrid', False):
            confidence_factors.append(0.7)
        elif characteristics.get('has_fur_texture', False):
            confidence_factors.append(0.1)  # Baixa confiança para pelo
        else:
            confidence_factors.append(0.4)
        
        # Calcular confiança média ponderada
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.0
    
    def _logical_reasoning(self, visual_analysis: Dict, characteristics: Dict) -> Dict[str, Any]:
        """Raciocínio lógico neuro-simbólico SIMPLIFICADO e EFICAZ - FASE 1.6.3"""
        reasoning = {
            'is_bird': False,
            'confidence': 0.0,
            'species': 'Desconhecida',
            'reasoning_steps': [],
            'characteristics_found': [],
            'missing_characteristics': [],
            'intuition_level': 'Baixa',
            'needs_manual_review': False
        }
        
        # ============================================================================
        # PRIORIDADE MÁXIMA: VERIFICAR RESULTADOS DO YOLO - FASE 1.6.3
        # ============================================================================
        
        # Verificar se YOLO detectou pássaro
        yolo_detection = characteristics.get('yolo_detection', False)
        bird_confidence_avg = characteristics.get('bird_confidence_avg', 0.0)
        
        if yolo_detection and bird_confidence_avg > 0.5:
            reasoning['is_bird'] = True
            reasoning['confidence'] = bird_confidence_avg
            reasoning['intuition_level'] = 'Alta'
            reasoning['species'] = 'Pássaro de espécie desconhecida'
            reasoning['reasoning_steps'].append(f"[YOLO] Pássaro detectado com confiança {bird_confidence_avg:.2f}")
            logger.info(f"[LOGICAL_REASONING] YOLO detectou pássaro - ACEITO com confiança {bird_confidence_avg:.2f}")
            return reasoning
        
        # ============================================================================
        # ANÁLISE TRADICIONAL (MANTIDA COMO FALLBACK) - FASE 1.6.3
        # ============================================================================
        
        # Extrair dados básicos
        has_wings = characteristics.get('has_wings', False)
        has_beak = characteristics.get('has_beak', False)
        has_feathers = characteristics.get('has_feathers', False)
        has_eyes = characteristics.get('has_eyes', False)
        has_claws = characteristics.get('has_claws', False)
        
        bird_shape_score = visual_analysis.get('bird_shape_score', 0)
        bird_color_score = visual_analysis.get('bird_color_score', 0)
        bird_like_features = visual_analysis.get('bird_like_features', 0)
        
        # Detectar características de mamíferos
        has_mammal_features = characteristics.get('has_mammal_features', False)
        has_mammal_body = characteristics.get('has_mammal_body', False)
        has_fur_texture = characteristics.get('has_fur_texture', False)
        
        # Contar características de pássaro encontradas
        bird_characteristics = [has_wings, has_beak, has_feathers, has_eyes, has_claws]
        bird_count = sum(bird_characteristics)
        
        # Listar características encontradas
        if has_wings: reasoning['characteristics_found'].append('asas')
        if has_beak: reasoning['characteristics_found'].append('bico')
        if has_feathers: reasoning['characteristics_found'].append('penas')
        if has_eyes: reasoning['characteristics_found'].append('olhos')
        if has_claws: reasoning['characteristics_found'].append('garras')
        
        # LÓGICA SIMPLIFICADA E EFICAZ - PRIORIZANDO PÁSSAROS
        
        # 1. PRIMEIRO: Verificar características definitivas de pássaro (PRIORIDADE MÁXIMA)
        if bird_count >= 3:
            reasoning['is_bird'] = True
            reasoning['confidence'] = 0.9
            reasoning['intuition_level'] = 'Alta'
            reasoning['reasoning_steps'].append(f"[SUCESSO] {bird_count} características definitivas de pássaro detectadas")
            
        # 2. SEGUNDO: Verificar características moderadas + análise visual
        elif bird_count >= 2 and bird_like_features > 0.4:
            reasoning['is_bird'] = True
            reasoning['confidence'] = 0.8
            reasoning['intuition_level'] = 'Alta'
            reasoning['reasoning_steps'].append(f"[SUCESSO] {bird_count} características + análise visual positiva")
            
        # 3. TERCEIRO: Verificar características básicas + forma/cores adequadas
        elif bird_count >= 1 and (bird_shape_score > 0.4 or bird_color_score > 0.4):
            reasoning['is_bird'] = True
            reasoning['confidence'] = 0.7
            reasoning['intuition_level'] = 'Média'
            reasoning['reasoning_steps'].append("[SUCESSO] Características básicas + forma/cores adequadas")
            
        # 4. QUARTO: Verificar análise visual muito positiva
        elif bird_like_features > 0.5 and (has_eyes or has_wings):
            reasoning['is_bird'] = True
            reasoning['confidence'] = 0.6
            reasoning['intuition_level'] = 'Média'
            reasoning['reasoning_steps'].append("[SUCESSO] Análise visual muito positiva")
            
        # 5. QUINTO: Verificar análise visual moderada
        elif bird_like_features > 0.4 and (bird_shape_score > 0.3 or bird_color_score > 0.3):
            reasoning['is_bird'] = True
            reasoning['confidence'] = 0.5
            reasoning['intuition_level'] = 'Média'
            reasoning['reasoning_steps'].append("[SUCESSO] Análise visual moderada")
            
        # 5.5. QUINTO E MEIO: Casos com forma perfeita de pássaro (prioridade máxima)
        elif bird_shape_score >= 1.0 and bird_color_score > 0.2:
            reasoning['is_bird'] = True
            reasoning['confidence'] = 0.5
            reasoning['intuition_level'] = 'Média'
            reasoning['needs_manual_review'] = True
            reasoning['reasoning_steps'].append("[SUCESSO] Forma perfeita de pássaro detectada")
            
        # 6. SEXTO: Casos duvidosos - pode ser pássaro (MAIS RIGOROSO)
        elif bird_count >= 2 and bird_like_features > 0.4:
            reasoning['is_bird'] = True
            reasoning['confidence'] = 0.4
            reasoning['intuition_level'] = 'Baixa'
            reasoning['needs_manual_review'] = True
            reasoning['reasoning_steps'].append("❓ Caso duvidoso - recomenda análise manual")
            
        # 6.5. SEXTO E MEIO: Casos com 1 característica mas análise visual muito forte
        elif bird_count >= 1 and bird_like_features > 0.6 and (bird_shape_score > 0.7 or bird_color_score > 0.7):
            reasoning['is_bird'] = True
            reasoning['confidence'] = 0.4
            reasoning['intuition_level'] = 'Baixa'
            reasoning['needs_manual_review'] = True
            reasoning['reasoning_steps'].append("❓ Uma característica + análise visual muito forte")
            
        # 6.6. SEXTO E MEIO: Casos com análise visual extremamente forte (mesmo sem características específicas)
        elif bird_shape_score > 0.9 and bird_color_score > 0.2 and bird_like_features > 0.3:
            reasoning['is_bird'] = True
            reasoning['confidence'] = 0.4
            reasoning['intuition_level'] = 'Baixa'
            reasoning['needs_manual_review'] = True
            reasoning['reasoning_steps'].append("❓ Análise visual extremamente forte (forma perfeita)")
            
        # 7. SÉTIMO: Verificar se é definitivamente um mamífero (APENAS SE NÃO FOR PÁSSARO)
        elif has_mammal_features and not (has_wings or has_beak or has_feathers):
            reasoning['is_bird'] = False
            reasoning['confidence'] = 0.9
            reasoning['intuition_level'] = 'Alta'
            reasoning['reasoning_steps'].append("[ERRO] Detectadas características específicas de mamíferos")
        
        # 8. OITAVO: Provavelmente não é pássaro
        else:
            reasoning['is_bird'] = False
            reasoning['confidence'] = 0.2
            reasoning['intuition_level'] = 'Baixa'
            reasoning['reasoning_steps'].append("[ERRO] Poucas evidências de características de pássaro")
        
        # Determinar espécie (se for pássaro)
        if reasoning['is_bird']:
            species = self._determine_species(visual_analysis, characteristics)
            reasoning['species'] = species
            reasoning['reasoning_steps'].append(f"[PASSARO] Espécie identificada: {species}")
        else:
            reasoning['species'] = 'Não-Pássaro'
            reasoning['reasoning_steps'].append("🚫 Não é um pássaro")
        
        # NOVO: Verificação específica para pássaros azuis não detectados
        if not reasoning['is_bird']:
            logger.info("[BUSCA] Verificando se é um pássaro azul não detectado...")
            # Verificar se tem características de pássaro azul
            dominant_colors = visual_analysis.get('dominant_colors', [])
            has_blue_color = any('blue' in color.lower() for color in dominant_colors)
            has_complex_patterns = visual_analysis.get('has_complex_patterns', False)
            has_iridescence = visual_analysis.get('has_iridescence', False)
            
            logger.info(f"🔵 Cores dominantes: {dominant_colors}")
            logger.info(f"🔵 Tem cor azul: {has_blue_color}")
            logger.info(f"🔵 Tem bico: {has_beak}")
            logger.info(f"🔵 Tem padrões complexos: {has_complex_patterns}")
            logger.info(f"🔵 Tem iridescência: {has_iridescence}")
            logger.info(f"🔵 Tem penas: {has_feathers}")
            
            # Se tem cor azul + características de penas + bico, é provavelmente um pássaro
            if (has_blue_color and has_beak and 
                (has_complex_patterns or has_iridescence or has_feathers)):
                reasoning['is_bird'] = True
                reasoning['confidence'] = 0.7
                reasoning['intuition_level'] = 'Média'
                reasoning['reasoning_steps'].append("🔵 Pássaro azul detectado por características específicas")
                logger.info("🔵 PÁSSARO AZUL DETECTADO - Corrigindo falso negativo!")
            else:
                logger.info("🔵 Não é um pássaro azul - características insuficientes")
        
        # Calcular confiança geral
        reasoning['overall_confidence'] = reasoning['confidence']
        
        return reasoning
    
    def _advanced_cognitive_analysis(self, visual_analysis: Dict, characteristics: Dict, reasoning: Dict) -> Dict[str, float]:
        """Análise cognitiva avançada neuro-simbólica"""
        analysis = {
            'pattern_recognition': 0.0,
            'logical_inference': 0.0,
            'uncertainty_handling': 0.0,
            'adaptive_thinking': 0.0
        }
        
        try:
            # 1. Reconhecimento de Padrões
            pattern_score = 0.0
            
            # Padrões visuais
            if visual_analysis.get('bird_shape_score', 0) > 0.5:
                pattern_score += 0.3
            if visual_analysis.get('bird_color_score', 0) > 0.5:
                pattern_score += 0.3
            if visual_analysis.get('bird_like_features', 0) > 0.6:
                pattern_score += 0.4
            
            analysis['pattern_recognition'] = min(pattern_score, 1.0)
            
            # 2. Inferência Lógica
            logic_score = 0.0
            
            # Lógica baseada em características
            fundamental_count = len(reasoning.get('characteristics_found', []))
            if fundamental_count >= 3:
                logic_score += 0.4
            elif fundamental_count >= 2:
                logic_score += 0.3
            elif fundamental_count >= 1:
                logic_score += 0.2
            
            # Lógica baseada em confiança
            confidence = reasoning.get('confidence', 0)
            if confidence > 0.8:
                logic_score += 0.4
            elif confidence > 0.6:
                logic_score += 0.3
            elif confidence > 0.4:
                logic_score += 0.2
            
            analysis['logical_inference'] = min(logic_score, 1.0)
            
            # 3. Tratamento de Incerteza
            uncertainty_score = 0.0
            
            # Quanto mais características, menor a incerteza
            total_chars = len(reasoning.get('characteristics_found', [])) + len(reasoning.get('missing_characteristics', []))
            if total_chars > 0:
                certainty_ratio = len(reasoning.get('characteristics_found', [])) / total_chars
                uncertainty_score = 1.0 - abs(0.5 - certainty_ratio) * 2
            
            analysis['uncertainty_handling'] = uncertainty_score
            
            # 4. Pensamento Adaptativo
            adaptive_score = 0.0
            
            # Adaptabilidade baseada na análise visual
            if visual_analysis.get('bird_like_features', 0) > 0.7:
                adaptive_score += 0.3
            if reasoning.get('needs_manual_review', False):
                adaptive_score += 0.2  # Reconhece quando precisa de ajuda
            if reasoning.get('intuition_level') == 'Alta':
                adaptive_score += 0.3
            elif reasoning.get('intuition_level') == 'Média':
                adaptive_score += 0.2
            
            analysis['adaptive_thinking'] = min(adaptive_score, 1.0)
            
        except Exception as e:
            logger.warning(f"[ALERTA] Erro na análise cognitiva: {e}")
        
        return analysis
    
    def _calculate_neuro_symbolic_score(self, reasoning: Dict, cognitive_analysis: Dict) -> float:
        """Calcula score neuro-simbólico geral"""
        try:
            # Componentes do score
            confidence_component = reasoning.get('confidence', 0) * 0.3
            pattern_component = cognitive_analysis.get('pattern_recognition', 0) * 0.25
            logic_component = cognitive_analysis.get('logical_inference', 0) * 0.25
            uncertainty_component = cognitive_analysis.get('uncertainty_handling', 0) * 0.1
            adaptive_component = cognitive_analysis.get('adaptive_thinking', 0) * 0.1
            
            # Score final
            neuro_symbolic_score = (
                confidence_component + 
                pattern_component + 
                logic_component + 
                uncertainty_component + 
                adaptive_component
            )
            
            return min(neuro_symbolic_score, 1.0)
            
        except Exception as e:
            logger.warning(f"[ALERTA] Erro no cálculo do score neuro-simbólico: {e}")
            return 0.0
    
    def _assess_learning_potential(self, reasoning: Dict, characteristics: Dict) -> str:
        """Avalia o potencial de aprendizado da análise"""
        try:
            score = 0.0
            
            # Fatores que indicam potencial de aprendizado
            if reasoning.get('needs_manual_review', False):
                score += 0.3  # Casos que precisam de revisão são bons para aprender
            
            if reasoning.get('confidence', 0) < 0.6:
                score += 0.2  # Baixa confiança indica necessidade de aprendizado
            
            if len(reasoning.get('missing_characteristics', [])) > 0:
                score += 0.2  # Características faltantes indicam oportunidade de aprendizado
            
            if reasoning.get('intuition_level') == 'Baixa':
                score += 0.3  # Baixa intuição indica potencial de melhoria
            
            # Determinar nível
            if score >= 0.7:
                return 'Alto'
            elif score >= 0.4:
                return 'Médio'
            else:
                return 'Baixo'
                
        except Exception as e:
            logger.warning(f"[ALERTA] Erro na avaliação do potencial de aprendizado: {e}")
            return 'Baixo'
    
    def _assess_certainty_level(self, reasoning: Dict) -> str:
        """Avalia o nível de certeza da análise"""
        try:
            confidence = reasoning.get('confidence', 0)
            neuro_score = reasoning.get('neuro_symbolic_score', 0)
            
            # Calcular certeza combinada
            combined_certainty = (confidence + neuro_score) / 2
            
            if combined_certainty >= 0.8:
                return 'Alta Certeza'
            elif combined_certainty >= 0.6:
                return 'Certeza Moderada'
            elif combined_certainty >= 0.4:
                return 'Incerteza Moderada'
            else:
                return 'Alta Incerteza'
                
        except Exception as e:
            logger.warning(f"[ALERTA] Erro na avaliação do nível de certeza: {e}")
            return 'Incerteza'
    
    def _has_mammal_characteristics(self, visual_analysis: Dict, characteristics: Dict) -> bool:
        """Detecção híbrida de características típicas de mamíferos (cachorros, gatos, etc.)"""
        # Verificar características que indicam mamífero
        
        # 1. Ausência total de características de pássaro
        has_wings = characteristics.get('has_wings', False)
        has_beak = characteristics.get('has_beak', False)
        has_feathers = characteristics.get('has_feathers', False)
        has_wings_advanced = characteristics.get('has_wings_advanced', False)
        has_beak_advanced = characteristics.get('has_beak_advanced', False)
        
        bird_characteristics_count = sum([has_wings, has_beak, has_feathers, has_wings_advanced, has_beak_advanced])
        
        # 2. Análise de cores típicas de mamíferos
        dominant_color = visual_analysis.get('dominant_color', 'unknown')
        mammal_colors = ['brown', 'black', 'white', 'gray']
        
        # 3. Análise de forma - mamíferos têm formas mais arredondadas
        shape_score = visual_analysis.get('bird_shape_score', 0)
        
        # 4. Análise de textura híbrida - pelo vs penas
        has_feathers_hybrid = characteristics.get('has_feathers_hybrid', False)
        has_fur_texture = characteristics.get('has_fur_texture', False)
        texture_regularity = characteristics.get('texture_regularity', 0)
        
        # 5. Análise de contornos - mamíferos têm menos contornos complexos
        contour_count = characteristics.get('contour_count', 0)
        
        # Critérios rigorosos para identificar mamífero:
        mammal_score = 0
        
        # Critério 1: Nenhuma característica de pássaro detectada
        if bird_characteristics_count == 0:
            mammal_score += 2
        
        # Critério 2: Cor típica de mamífero
        if dominant_color in mammal_colors:
            mammal_score += 1
        
        # Critério 3: Forma muito arredondada (não típica de pássaro)
        if shape_score > 0.7:
            mammal_score += 1
        
        # Critério 4: Textura de pelo detectada
        if has_fur_texture:
            mammal_score += 2
        
        # Critério 5: Ausência de textura de penas
        if not has_feathers_hybrid:
            mammal_score += 1
        
        # Critério 6: Textura irregular (típica de pelo)
        if texture_regularity < 0.3:
            mammal_score += 1
        
        # Critério 7: Poucos contornos complexos (típico de mamíferos)
        if contour_count < 10:
            mammal_score += 1
        
        # Considerar mamífero se score >= 4 (de 7 possíveis)
        is_mammal = mammal_score >= 4
        
        return is_mammal
    
    def _analyze_contours_ultra_advanced(self, image: np.ndarray) -> Dict[str, Any]:
        """Análise ultra-avançada de contornos para detectar características de pássaro"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Múltiplas técnicas de detecção de bordas com diferentes parâmetros
        edges_configs = [
            (30, 100), (50, 150), (100, 200), (150, 300),
            (20, 80), (80, 160), (120, 240)
        ]
        
        all_edges = []
        for low, high in edges_configs:
            edges = cv2.Canny(gray, low, high)
            all_edges.append(edges)
        
        # Combinar todas as bordas
        combined_edges = all_edges[0]
        for edges in all_edges[1:]:
            combined_edges = cv2.bitwise_or(combined_edges, edges)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        characteristics = {
            'has_wings_ultra': False,
            'has_beak_ultra': False,
            'has_body_ultra': False,
            'contour_complexity': 0.0,
            'symmetry_score': 0.0,
            'edge_density': 0.0
        }
        
        if not contours:
            return characteristics
        
        # Calcular densidade de bordas
        edge_density = np.sum(combined_edges > 0) / (combined_edges.shape[0] * combined_edges.shape[1])
        characteristics['edge_density'] = edge_density
        
        # Analisar cada contorno com critérios ultra-rigorosos
        wing_like = 0
        beak_like = 0
        body_like = 0
        complexity_scores = []
        symmetry_scores = []
        
        for contour in contours:
            if len(contour) < 10:  # Ignorar contornos muito pequenos
                continue
                
            # Calcular características do contorno
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area < 200:  # Threshold mais alto para área mínima
                continue
            
            # Calcular aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 1.0
            
            # Calcular compactness
            compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Calcular convexidade
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Calcular complexidade (número de vértices vs área)
            complexity = len(contour) / area if area > 0 else 0
            complexity_scores.append(complexity)
            
            # Calcular simetria
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                
                # Verificar simetria horizontal
                left_points = [p for p in contour if p[0][0] < cx]
                right_points = [p for p in contour if p[0][0] > cx]
                
                if len(left_points) > 0 and len(right_points) > 0:
                    symmetry = min(len(left_points), len(right_points)) / max(len(left_points), len(right_points))
                    symmetry_scores.append(symmetry)
            
            # Critérios ultra-rigorosos para detectar características
            
            # Detectar asas (formas alongadas com complexidade moderada)
            if (2.5 < aspect_ratio < 8.0 and 
                0.1 < compactness < 0.4 and 
                0.3 < solidity < 0.8 and
                0.01 < complexity < 0.05):
                wing_like += 1
            
            # Detectar bico (formas pontiagudas e alongadas)
            if (aspect_ratio > 4.0 and 
                solidity < 0.7 and 
                compactness < 0.3 and
                complexity > 0.02):
                beak_like += 1
            
            # Detectar corpo (formas arredondadas com simetria)
            if (0.6 < aspect_ratio < 1.8 and 
                compactness > 0.4 and 
                solidity > 0.8 and
                complexity < 0.02):
                body_like += 1
        
        characteristics['has_wings_ultra'] = wing_like >= 2  # Pelo menos 2 formas de asa
        characteristics['has_beak_ultra'] = beak_like >= 1  # Pelo menos 1 forma de bico
        characteristics['has_body_ultra'] = body_like >= 1  # Pelo menos 1 forma de corpo
        
        # Calcular scores de complexidade e simetria
        if complexity_scores:
            characteristics['contour_complexity'] = np.mean(complexity_scores)
        if symmetry_scores:
            characteristics['symmetry_score'] = np.mean(symmetry_scores)
        
        return characteristics
    
    def _analyze_texture_ultra_hybrid(self, image: np.ndarray) -> Dict[str, Any]:
        """Análise ultra-híbrida de textura para distinguir penas de pelo"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        characteristics = {
            'has_feathers_ultra': False,
            'has_fur_ultra': False,
            'texture_regularity_ultra': 0.0,
            'pattern_frequency_score': 0.0,
            'texture_directionality': 0.0
        }
        
        # Método 1: Análise de gradientes locais avançada
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Calcular regularidade da textura
        texture_variance = np.var(magnitude)
        texture_mean = np.mean(magnitude)
        texture_std = np.std(magnitude)
        
        # Método 2: Análise de padrões repetitivos ultra-avançada
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Calcular energia em diferentes frequências
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # Energia em frequências baixas (padrões grandes - pelo)
        low_freq_energy = np.sum(magnitude_spectrum[center_y-15:center_y+15, center_x-15:center_x+15])
        
        # Energia em frequências médias (padrões de penas)
        medium_freq_energy = np.sum(magnitude_spectrum[center_y-30:center_y+30, center_x-30:center_x+30]) - low_freq_energy
        
        # Energia em frequências altas (detalhes finos)
        high_freq_energy = np.sum(magnitude_spectrum) - medium_freq_energy - low_freq_energy
        
        # Energia total
        total_energy = np.sum(magnitude_spectrum)
        
        # Calcular scores
        low_freq_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
        medium_freq_ratio = medium_freq_energy / total_energy if total_energy > 0 else 0
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # Método 3: Análise de direcionalidade da textura
        # Calcular histograma de direções
        hist, _ = np.histogram(direction.flatten(), bins=36, range=(-np.pi, np.pi))
        directionality = np.std(hist) / np.mean(hist) if np.mean(hist) > 0 else 0
        
        # Método 4: Análise de bordas locais ultra-avançada
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Método 5: Análise de padrões de repetição
        # Usar correlação para detectar padrões repetitivos
        correlation_scores = []
        for i in range(0, gray.shape[0] - 20, 20):
            for j in range(0, gray.shape[1] - 20, 20):
                patch = gray[i:i+20, j:j+20]
                if patch.size > 0:
                    # Calcular autocorrelação
                    corr = cv2.matchTemplate(patch, patch, cv2.TM_CCOEFF_NORMED)
                    correlation_scores.append(np.max(corr))
        
        pattern_frequency = np.mean(correlation_scores) if correlation_scores else 0
        
        # Critérios ultra-rigorosos para distinguir penas de pelo
        
        # Score para penas (padrões regulares, frequências médias altas)
        feather_score = (
            medium_freq_ratio * 0.3 +      # Padrões de penas
            edge_density * 0.2 +           # Densidade de bordas
            (1.0 - low_freq_ratio) * 0.2 + # Menos padrões grandes
            pattern_frequency * 0.2 +      # Padrões repetitivos
            directionality * 0.1           # Direcionalidade
        )
        
        # Score para pelo (padrões irregulares, frequências baixas altas)
        fur_score = (
            low_freq_ratio * 0.4 +         # Padrões grandes (pelo)
            (1.0 - medium_freq_ratio) * 0.3 + # Menos padrões de penas
            (1.0 - edge_density) * 0.2 +   # Menos bordas
            (1.0 - pattern_frequency) * 0.1 # Menos padrões repetitivos
        )
        
        characteristics['has_feathers_ultra'] = feather_score > 0.4  # Threshold mais alto
        characteristics['has_fur_ultra'] = fur_score > 0.5  # Threshold mais alto
        characteristics['texture_regularity_ultra'] = feather_score
        characteristics['pattern_frequency_score'] = pattern_frequency
        characteristics['texture_directionality'] = directionality
        
        return characteristics
    
    def _analyze_shape_ultra_rigorous(self, image: np.ndarray) -> Dict[str, Any]:
        """Análise ultra-rigorosa de formas para detectar características de pássaro"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar contornos principais
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        characteristics = {
            'bird_shape_score_ultra': 0.0,
            'mammal_shape_score_ultra': 0.0,
            'elongation_factor': 0.0,
            'roundness_factor': 0.0,
            'compactness_factor': 0.0
        }
        
        if not contours:
            return characteristics
        
        # Encontrar maior contorno (assumindo que é o objeto principal)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calcular características de forma
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Calcular aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Calcular compactness
        compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Calcular elongação
        elongation = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
        
        # Calcular redondez
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        roundness = area / hull_area if hull_area > 0 else 0
        
        characteristics['elongation_factor'] = elongation
        characteristics['roundness_factor'] = roundness
        characteristics['compactness_factor'] = compactness
        
        # Critérios ultra-rigorosos para pássaros vs mamíferos
        
        # Score para pássaros (formas alongadas, moderadamente compactas)
        bird_score = 0.0
        if 0.4 < aspect_ratio < 2.5:  # Aspect ratio típico de pássaros
            bird_score += 0.3
        if 0.2 < compactness < 0.6:  # Compactness típica de pássaros
            bird_score += 0.3
        if 0.7 < roundness < 0.95:  # Redondez típica de pássaros
            bird_score += 0.2
        if 1.2 < elongation < 3.0:  # Elongação típica de pássaros
            bird_score += 0.2
        
        # Score para mamíferos (formas muito arredondadas, muito compactas)
        mammal_score = 0.0
        if 0.8 < aspect_ratio < 1.2:  # Aspect ratio típico de mamíferos
            mammal_score += 0.4
        if compactness > 0.6:  # Compactness típica de mamíferos
            mammal_score += 0.3
        if roundness > 0.95:  # Redondez típica de mamíferos
            mammal_score += 0.3
        
        characteristics['bird_shape_score_ultra'] = bird_score
        characteristics['mammal_shape_score_ultra'] = mammal_score
        
        return characteristics
    
    def _analyze_biometric_patterns(self, image: np.ndarray) -> Dict[str, Any]:
        """Análise de padrões biométricos para distinguir pássaros de mamíferos"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        characteristics = {
            'biometric_bird_score': 0.0,
            'biometric_mammal_score': 0.0,
            'feature_density': 0.0,
            'structural_complexity': 0.0
        }
        
        # Método 1: Análise de densidade de características
        # Detectar cantos (Harris corner detection)
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        corner_count = np.sum(corners > 0.01 * corners.max())
        feature_density = corner_count / (gray.shape[0] * gray.shape[1])
        
        characteristics['feature_density'] = feature_density
        
        # Método 2: Análise de complexidade estrutural
        # Usar Laplaciano para detectar bordas
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        structural_complexity = np.var(laplacian)
        
        characteristics['structural_complexity'] = structural_complexity
        
        # Método 3: Análise de padrões de iluminação
        # Calcular histograma de intensidades
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist.ravel() / hist.sum()
        
        # Calcular entropia (medida de complexidade)
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
        
        # Método 4: Análise de textura usando LBP (Local Binary Patterns)
        # Implementação simplificada de LBP
        lbp_patterns = []
        for i in range(1, gray.shape[0] - 1):
            for j in range(1, gray.shape[1] - 1):
                center = gray[i, j]
                pattern = 0
                for k, (di, dj) in enumerate([(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]):
                    if gray[i + di, j + dj] >= center:
                        pattern |= (1 << k)
                lbp_patterns.append(pattern)
        
        # Calcular uniformidade dos padrões LBP
        lbp_uniformity = len(set(lbp_patterns)) / len(lbp_patterns) if lbp_patterns else 0
        
        # Critérios biométricos para pássaros vs mamíferos
        
        # Score para pássaros (alta complexidade estrutural, padrões variados)
        bird_score = 0.0
        if feature_density > 0.001:  # Alta densidade de características
            bird_score += 0.3
        if structural_complexity > 100:  # Alta complexidade estrutural
            bird_score += 0.3
        if entropy > 6.0:  # Alta entropia (padrões variados)
            bird_score += 0.2
        if lbp_uniformity < 0.8:  # Padrões não uniformes (penas)
            bird_score += 0.2
        
        # Score para mamíferos (baixa complexidade estrutural, padrões uniformes)
        mammal_score = 0.0
        if feature_density < 0.0005:  # Baixa densidade de características
            mammal_score += 0.3
        if structural_complexity < 50:  # Baixa complexidade estrutural
            mammal_score += 0.3
        if entropy < 5.0:  # Baixa entropia (padrões uniformes)
            mammal_score += 0.2
        if lbp_uniformity > 0.9:  # Padrões uniformes (pelo)
            mammal_score += 0.2
        
        characteristics['biometric_bird_score'] = bird_score
        characteristics['biometric_mammal_score'] = mammal_score
        
        return characteristics
    
    def _calculate_mammal_score(self, characteristics: Dict) -> float:
        """Calcula score de mamífero baseado em múltiplas características"""
        score = 0.0
        total_weight = 0.0
        
        # Peso 1: Ausência de características de pássaro
        bird_chars = [
            characteristics.get('has_wings', False),
            characteristics.get('has_beak', False),
            characteristics.get('has_feathers', False),
            characteristics.get('has_wings_ultra', False),
            characteristics.get('has_beak_ultra', False),
            characteristics.get('has_feathers_ultra', False)
        ]
        
        bird_count = sum(bird_chars)
        if bird_count == 0:
            score += 3.0  # Peso alto para ausência total
        elif bird_count <= 1:
            score += 1.0  # Peso médio para poucas características
        
        total_weight += 3.0
        
        # Peso 2: Detecção de textura de pelo
        if characteristics.get('has_fur_ultra', False):
            score += 2.0
        if characteristics.get('has_fur_texture', False):
            score += 1.0
        total_weight += 2.0
        
        # Peso 3: Score de forma de mamífero
        mammal_shape_score = characteristics.get('mammal_shape_score_ultra', 0)
        score += mammal_shape_score * 2.0
        total_weight += 2.0
        
        # Peso 4: Score biométrico de mamífero
        biometric_mammal_score = characteristics.get('biometric_mammal_score', 0)
        score += biometric_mammal_score * 1.5
        total_weight += 1.5
        
        # Peso 5: Ausência de textura de penas
        if not characteristics.get('has_feathers_ultra', False):
            score += 1.0
        if not characteristics.get('has_feathers_hybrid', False):
            score += 0.5
        total_weight += 1.5
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_bird_score(self, characteristics: Dict, yolo_confidence: float) -> float:
        """Calcula score de pássaro baseado em múltiplas características"""
        score = 0.0
        total_weight = 0.0
        
        # Peso 1: Detecção YOLO
        if characteristics.get('yolo_detection', False):
            score += yolo_confidence * 3.0  # Peso alto para YOLO
        total_weight += 3.0
        
        # Peso 2: Características visuais básicas
        visual_chars = [
            characteristics.get('has_wings', False),
            characteristics.get('has_beak', False),
            characteristics.get('has_feathers', False),
            characteristics.get('has_eyes', False)
        ]
        
        visual_count = sum(visual_chars)
        if visual_count >= 3:
            score += 2.0
        elif visual_count >= 2:
            score += 1.5
        elif visual_count >= 1:
            score += 1.0
        total_weight += 2.0
        
        # Peso 3: Características ultra-avançadas
        ultra_chars = [
            characteristics.get('has_wings_ultra', False),
            characteristics.get('has_beak_ultra', False),
            characteristics.get('has_feathers_ultra', False),
            characteristics.get('has_body_ultra', False)
        ]
        
        ultra_count = sum(ultra_chars)
        if ultra_count >= 3:
            score += 2.5
        elif ultra_count >= 2:
            score += 2.0
        elif ultra_count >= 1:
            score += 1.5
        total_weight += 2.5
        
        # Peso 4: Score de forma de pássaro
        bird_shape_score = characteristics.get('bird_shape_score_ultra', 0)
        score += bird_shape_score * 1.5
        total_weight += 1.5
        
        # Peso 5: NOVO - Detecção de características de réptil (penalização)
        reptile_score = self._calculate_reptile_score(characteristics)
        if reptile_score > 0.6:  # Se tem características fortes de réptil
            score -= 2.0  # Penalização significativa
        elif reptile_score > 0.3:  # Se tem características moderadas de réptil
            score -= 1.0  # Penalização moderada
        total_weight += 2.0
        
        # Peso 6: NOVO - Detecção melhorada de cores específicas
        color_bonus = self._calculate_color_bonus(characteristics)
        score += color_bonus * 0.5
        total_weight += 0.5
        
        # Peso 5: Score biométrico de pássaro
        biometric_bird_score = characteristics.get('biometric_bird_score', 0)
        score += biometric_bird_score * 1.0
        total_weight += 1.0
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_reptile_score(self, characteristics: Dict) -> float:
        """Calcula score de réptil baseado em características específicas"""
        score = 0.0
        total_weight = 0.0
        
        # Peso 1: Detecção de escamas (característica distintiva de répteis)
        if characteristics.get('has_scales', False):
            score += 3.0
        if characteristics.get('has_scales_ultra', False):
            score += 2.0
        total_weight += 3.0
        
        # Peso 2: Detecção de crista dorsal (comum em iguanas)
        if characteristics.get('has_dorsal_crest', False):
            score += 2.5
        if characteristics.get('has_spine_ridge', False):
            score += 2.0
        total_weight += 2.5
        
        # Peso 3: Forma alongada do corpo (típica de répteis)
        if characteristics.get('has_elongated_body', False):
            score += 2.0
        if characteristics.get('body_length_ratio', 0) > 2.5:  # Corpo muito alongado
            score += 1.5
        total_weight += 2.0
        
        # Peso 4: Ausência de penas (característica distintiva)
        if not characteristics.get('has_feathers', False):
            score += 1.0
        if not characteristics.get('has_feathers_ultra', False):
            score += 1.5
        total_weight += 2.0
        
        # Peso 5: Presença de garras (comum em répteis)
        if characteristics.get('has_claws', False):
            score += 1.5
        if characteristics.get('has_claws_ultra', False):
            score += 1.0
        total_weight += 1.5
        
        # Peso 6: Textura de pele escamosa
        if characteristics.get('has_scaly_texture', False):
            score += 2.0
        total_weight += 2.0
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_color_bonus(self, characteristics: Dict) -> float:
        """Calcula bônus para cores específicas que podem confundir o modelo"""
        score = 0.0
        
        # Peso 1: Cores vibrantes específicas (azul, verde, vermelho)
        dominant_colors = characteristics.get('dominant_colors', [])
        
        for color in dominant_colors:
            if 'blue' in color.lower() and characteristics.get('has_beak', False):
                score += 1.5  # Bônus para pássaros azuis com bico
            elif 'green' in color.lower() and characteristics.get('has_feathers', False):
                score += 1.0  # Bônus para pássaros verdes com penas
            elif 'red' in color.lower() and characteristics.get('has_wings', False):
                score += 1.0  # Bônus para pássaros vermelhos com asas
        
        # Peso 2: Padrões de cor complexos (indicam penas)
        if characteristics.get('has_complex_patterns', False):
            score += 1.0
        
        # Peso 3: Iridescência (característica de penas)
        if characteristics.get('has_iridescence', False):
            score += 1.5
        
        return min(score, 2.0)  # Limitar bônus máximo
    
    def _multi_library_detection(self, image: np.ndarray) -> Dict[str, Any]:
        """Detecção usando múltiplas bibliotecas (YOLO, OpenCV, MediaPipe)"""
        detection_results = {
            'yolo_detection': False,
            'opencv_detection': False,
            'mediapipe_detection': False,
            'detection_votes': {},
            'total_detections': 0,
            'bird_confidence_avg': 0.0
        }
        
        bird_detections = []
        all_detections = []
        
        # 1. Detecção YOLO (todas as versões disponíveis)
        logger.info(f"[BUSCA] Iniciando detecção YOLO com {len(self.detection_models)} modelos disponíveis")
        for model_name, model in self.detection_models.items():
            if 'yolo' in model_name.lower():
                logger.info(f"[BUSCA] Testando modelo YOLO: {model_name}")
                try:
                    yolo_result = self._detect_with_yolo(model, image, model_name)
                    logger.info(f"[BUSCA] Resultado YOLO {model_name}: detected={yolo_result['detected']}, confidence={yolo_result['confidence']}")
                    if yolo_result['detected']:
                        bird_detections.append(yolo_result['confidence'])
                        all_detections.append(f"YOLO_{model_name}")
                        detection_results['yolo_detection'] = True
                        detection_results['detection_votes'][f"yolo_{model_name}"] = yolo_result['confidence']
                        logger.info(f"[SUCESSO] YOLO {model_name} detectou pássaro com confiança {yolo_result['confidence']}")
                except Exception as e:
                    logger.warning(f"[ALERTA] Erro na detecção YOLO {model_name}: {e}")
        
        # 2. Detecção OpenCV DNN
        for model_name, model in self.detection_models.items():
            if 'opencv' in model_name.lower():
                try:
                    opencv_result = self._detect_with_opencv(model, image, model_name)
                    if opencv_result['detected']:
                        bird_detections.append(opencv_result['confidence'])
                        all_detections.append(f"OpenCV_{model_name}")
                        detection_results['opencv_detection'] = True
                        detection_results['detection_votes'][f"opencv_{model_name}"] = opencv_result['confidence']
                except Exception as e:
                    logger.warning(f"[ALERTA] Erro na detecção OpenCV {model_name}: {e}")
        
        # 3. Detecção MediaPipe
        if 'mediapipe' in self.detection_models:
            try:
                mediapipe_result = self._detect_with_mediapipe(self.detection_models['mediapipe'], image)
                if mediapipe_result['detected']:
                    bird_detections.append(mediapipe_result['confidence'])
                    all_detections.append("MediaPipe")
                    detection_results['mediapipe_detection'] = True
                    detection_results['detection_votes']['mediapipe'] = mediapipe_result['confidence']
            except Exception as e:
                logger.warning(f"[ALERTA] Erro na detecção MediaPipe: {e}")
        
        # 4. Calcular estatísticas finais
        detection_results['total_detections'] = len(all_detections)
        if bird_detections:
            detection_results['bird_confidence_avg'] = sum(bird_detections) / len(bird_detections)
        
        return detection_results
    
    def _detect_with_yolo(self, model, image: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Detecção usando modelo YOLO com configurações otimizadas - FASE 1.4.2"""
        try:
            # ============================================================================
            # CONFIGURAÇÕES OTIMIZADAS PARA DETECÇÃO DE PÁSSAROS - FASE 1.4.2
            # ============================================================================
            
            # Configurações otimizadas para melhor detecção de pássaros
            detection_params = {
                'conf': 0.05,  # Confiança mínima muito baixa para capturar mais pássaros
                'iou': 0.45,  # IoU threshold otimizado
                'agnostic_nms': False,
                'max_det': 100,  # Reduzido para melhor performance
                'half': False,
                'device': 'cpu',
                'verbose': False  # Reduzir logs
            }
            
            if 'v5' in model_name:
                # YOLOv5 tem interface diferente
                results = model(image, **detection_params)
                detections = results.pandas().xyxy[0]
                
                bird_detected = False
                max_confidence = 0.0
                bird_features = []
                all_detections = []
                
                for _, detection in detections.iterrows():
                    confidence = detection['confidence']
                    class_id = detection['class']
                    class_name = detection.get('name', f'class_{class_id}')
                    
                    all_detections.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence
                    })
                    
                    # Threshold mais baixo para pássaros
                    if confidence > 0.05:
                        # COCO classes expandidas para pássaros e características
                        bird_classes = [14]  # Bird class principal (corrigido de 15 para 14)
                        bird_feature_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20]  # Removido 14 pois é classe principal
                        
                        if class_id in bird_classes:
                            bird_detected = True
                            max_confidence = max(max_confidence, confidence)
                            logger.info(f"[YOLO] Pássaro detectado: {class_name} (conf: {confidence:.3f})")
                        elif class_id in bird_feature_classes and confidence > 0.15:
                            bird_features.append({
                                'class': class_id,
                                'confidence': confidence,
                                'name': class_name
                            })
                
                # Log de todas as detecções para debug
                if all_detections:
                    logger.info(f"[YOLO] Todas as detecções: {[(d['class_name'], d['confidence']) for d in all_detections]}")
                
                # NOVO: Detecção direta sem verificação pós-processamento problemática
                if bird_detected:
                    logger.info(f"[YOLO] Pássaro detectado com confiança {max_confidence:.3f} - ACEITO DIRETAMENTE")
                    # Aceitar detecção YOLO diretamente sem verificação adicional
                else:
                    logger.info(f"[YOLO] Nenhum pássaro detectado")
                
                return {
                    'detected': bird_detected,
                    'confidence': max_confidence,
                    'model': model_name,
                    'bird_features': bird_features,
                    'total_features': len(bird_features),
                    'all_detections': all_detections
                }
            else:
                # YOLOv8+ tem interface diferente
                results = model(image, **detection_params)
                
                bird_detected = False
                max_confidence = 0.0
                bird_features = []
                all_detections = []
                
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            class_name = result.names.get(class_id, f'class_{class_id}')
                            
                            all_detections.append({
                                'class_id': class_id,
                                'class_name': class_name,
                                'confidence': confidence
                            })
                            
                            # Threshold mais baixo para pássaros
                            if confidence > 0.05:
                                # COCO classes expandidas
                                bird_classes = [14]  # Bird class principal (corrigido de 15 para 14)
                                bird_feature_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20]  # Removido 14 pois é classe principal
                                
                                if class_id in bird_classes:
                                    bird_detected = True
                                    max_confidence = max(max_confidence, confidence)
                                    logger.info(f"[YOLO] Pássaro detectado: {class_name} (conf: {confidence:.3f})")
                                elif class_id in bird_feature_classes and confidence > 0.15:
                                    bird_features.append({
                                        'class': class_id,
                                        'confidence': confidence,
                                        'name': class_name
                                    })
                
                # Log de todas as detecções para debug
                if all_detections:
                    logger.info(f"[YOLO] Todas as detecções: {[(d['class_name'], d['confidence']) for d in all_detections]}")
                
                # NOVO: Detecção direta sem verificação pós-processamento problemática
                if bird_detected:
                    logger.info(f"[YOLO] Pássaro detectado com confiança {max_confidence:.3f} - ACEITO DIRETAMENTE")
                    # Aceitar detecção YOLO diretamente sem verificação adicional
                else:
                    logger.info(f"[YOLO] Nenhum pássaro detectado")
                
                return {
                    'detected': bird_detected,
                    'confidence': max_confidence,
                    'model': model_name,
                    'bird_features': bird_features,
                    'total_features': len(bird_features),
                    'all_detections': all_detections
                }
                
        except Exception as e:
            logger.warning(f"[YOLO] Erro na detecção {model_name}: {e}")
            return {
                'detected': False, 
                'confidence': 0.0, 
                'model': model_name,
                'bird_features': [],
                'total_features': 0,
                'all_detections': []
            }
    
    def _detect_with_opencv(self, model, image: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Detecção usando OpenCV DNN"""
        try:
            # Preparar imagem para OpenCV DNN
            blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
            model.setInput(blob)
            
            # Detectar objetos
            outputs = model.forward()
            
            bird_detected = False
            max_confidence = 0.0
            
            # Processar detecções (formato YOLO)
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Assumindo que 0 é pássaro ou classe de pássaro
                    if confidence > 0.3 and class_id == 0:
                        bird_detected = True
                        max_confidence = max(max_confidence, confidence)
            
            return {
                'detected': bird_detected,
                'confidence': max_confidence,
                'model': model_name
            }
            
        except Exception as e:
            logger.warning(f"[ALERTA] Erro na detecção OpenCV {model_name}: {e}")
            return {'detected': False, 'confidence': 0.0, 'model': model_name}
    
    def _detect_with_mediapipe(self, model, image: np.ndarray) -> Dict[str, Any]:
        """Detecção usando MediaPipe"""
        try:
            # Converter para RGB (MediaPipe requer RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detectar objetos
            results = model.process(rgb_image)
            
            bird_detected = False
            max_confidence = 0.0
            
            if results.detected_objects:
                for detected_object in results.detected_objects:
                    # MediaPipe Objectron detecta objetos 3D
                    # Assumir que qualquer detecção pode ser um pássaro
                    if detected_object.score > 0.3:
                        bird_detected = True
                        max_confidence = max(max_confidence, detected_object.score)
            
            return {
                'detected': bird_detected,
                'confidence': max_confidence,
                'model': 'mediapipe'
            }
            
        except Exception as e:
            logger.warning(f"[ALERTA] Erro na detecção MediaPipe: {e}")
            return {'detected': False, 'confidence': 0.0, 'model': 'mediapipe'}
    
    def _calculate_model_consensus(self, characteristics: Dict, detection_results: Dict) -> Dict[str, Any]:
        """Calcula consenso entre todos os modelos de detecção"""
        consensus_result = {
            'model_consensus': 0.0,
            'detection_agreement': 0.0,
            'confidence_consensus': 0.0,
            'final_decision': 'unknown'
        }
        
        # Contar votos
        votes = detection_results.get('detection_votes', {})
        total_models = len(votes)
        
        if total_models == 0:
            return consensus_result
        
        # Calcular consenso de detecção
        positive_votes = sum(1 for conf in votes.values() if conf > 0.3)
        detection_agreement = positive_votes / total_models
        
        # Calcular consenso de confiança
        if votes:
            avg_confidence = sum(votes.values()) / len(votes)
            confidence_consensus = avg_confidence
        else:
            confidence_consensus = 0.0
        
        # Consenso geral
        model_consensus = (detection_agreement + confidence_consensus) / 2
        
        # Decisão final baseada no consenso
        if model_consensus > 0.6:
            final_decision = 'bird_high_confidence'
        elif model_consensus > 0.4:
            final_decision = 'bird_medium_confidence'
        elif model_consensus > 0.2:
            final_decision = 'uncertain'
        else:
            final_decision = 'not_bird'
        
        consensus_result.update({
            'model_consensus': model_consensus,
            'detection_agreement': detection_agreement,
            'confidence_consensus': confidence_consensus,
            'final_decision': final_decision
        })
        
        return consensus_result
    
    def _calculate_mammal_score_enhanced(self, characteristics: Dict) -> float:
        """Calcula score de mamífero melhorado com votação de modelos"""
        score = 0.0
        total_weight = 0.0
        
        # Peso 1: Consenso de modelos (novo)
        model_consensus = characteristics.get('model_consensus', 0)
        final_decision = characteristics.get('final_decision', 'unknown')
        
        if final_decision == 'not_bird':
            score += 2.0  # Peso alto se modelos concordam que não é pássaro
        elif final_decision == 'uncertain':
            score += 0.5  # Peso baixo se incerto
        total_weight += 2.0
        
        # Peso 2: Ausência de características de pássaro (mantido)
        bird_chars = [
            characteristics.get('has_wings', False),
            characteristics.get('has_beak', False),
            characteristics.get('has_feathers', False),
            characteristics.get('has_wings_ultra', False),
            characteristics.get('has_beak_ultra', False),
            characteristics.get('has_feathers_ultra', False)
        ]
        
        bird_count = sum(bird_chars)
        if bird_count == 0:
            score += 2.0  # Peso alto para ausência total
        elif bird_count <= 1:
            score += 1.0  # Peso médio para poucas características
        total_weight += 2.0
        
        # Peso 3: Detecção de textura de pelo (mantido)
        if characteristics.get('has_fur_ultra', False):
            score += 2.0
        if characteristics.get('has_fur_texture', False):
            score += 1.0
        total_weight += 2.0
        
        # Peso 4: Score de forma de mamífero (mantido)
        mammal_shape_score = characteristics.get('mammal_shape_score_ultra', 0)
        score += mammal_shape_score * 1.5
        total_weight += 1.5
        
        # Peso 5: Score biométrico de mamífero (mantido)
        biometric_mammal_score = characteristics.get('biometric_mammal_score', 0)
        score += biometric_mammal_score * 1.0
        total_weight += 1.0
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _apply_generated_rules(self, characteristics: Dict, confidence: float) -> float:
        """
        Aplica regras geradas automaticamente pelo sistema de auto-modificação.
        
        Args:
            characteristics: Características detectadas na imagem
            confidence: Confiança atual da detecção
            
        Returns:
            Boost de confiança aplicado pelas regras
        """
        if not self.auto_modification_system:
            return 0.0
        
        try:
            # Carregar regras geradas
            generated_rules = self.auto_modification_system.rule_generator.load_generated_rules()
            
            total_boost = 0.0
            applied_rules = 0
            
            for rule in generated_rules:
                try:
                    # Verificar se a regra se aplica
                    if self._evaluate_rule_condition(rule, characteristics, confidence):
                        # Aplicar ação da regra
                        boost = self._apply_rule_action(rule)
                        total_boost += boost
                        applied_rules += 1
                        
                        logger.debug(f"Regra aplicada: {rule['description']} (boost: {boost})")
                        
                except Exception as e:
                    logger.warning(f"Erro ao aplicar regra {rule.get('description', 'unknown')}: {e}")
            
            if applied_rules > 0:
                logger.info(f"Aplicadas {applied_rules} regras geradas automaticamente (boost total: {total_boost:.3f})")
            
            return total_boost
            
        except Exception as e:
            logger.error(f"Erro ao aplicar regras geradas: {e}")
            return 0.0
    
    def _evaluate_rule_condition(self, rule: Dict[str, Any], characteristics: Dict, confidence: float) -> bool:
        """
        Avalia se uma regra se aplica às características atuais.
        
        Args:
            rule: Regra a ser avaliada
            characteristics: Características detectadas
            confidence: Confiança atual
            
        Returns:
            True se a regra se aplica, False caso contrário
        """
        try:
            rule_text = rule.get('rule', '')
            
            # Substituir variáveis na regra
            rule_text = rule_text.replace('confidence', str(confidence))
            
            # Verificar condições específicas
            if 'color_dominant' in rule_text:
                color = characteristics.get('dominant_color', '')
                if color:
                    rule_text = rule_text.replace("color_dominant == 'azul'", f"'{color}' == 'azul'")
                    rule_text = rule_text.replace("color_dominant == 'vermelho'", f"'{color}' == 'vermelho'")
            
            if 'species_detected' in rule_text:
                species = characteristics.get('detected_species', '')
                if species:
                    rule_text = rule_text.replace("species_detected == 'azulão'", f"'{species}' == 'azulão'")
                    rule_text = rule_text.replace("species_detected == 'canário'", f"'{species}' == 'canário'")
            
            if 'has_bico' in rule_text:
                has_beak = characteristics.get('has_beak', False) or characteristics.get('has_beak_ultra', False)
                rule_text = rule_text.replace('has_bico', str(has_beak))
            
            if 'has_penas' in rule_text:
                has_feathers = characteristics.get('has_feathers', False) or characteristics.get('has_feathers_ultra', False)
                rule_text = rule_text.replace('has_penas', str(has_feathers))
            
            if 'characteristics_match' in rule_text:
                # Verificar se características básicas estão presentes
                basic_chars = [
                    characteristics.get('has_beak', False),
                    characteristics.get('has_feathers', False),
                    characteristics.get('has_wings', False)
                ]
                chars_match = sum(basic_chars) >= 2
                rule_text = rule_text.replace('characteristics_match', str(chars_match))
            
            # Avaliar a regra (versão simplificada)
            # Em uma implementação real, seria necessário um parser mais robusto
            if 'and' in rule_text:
                parts = rule_text.split('and')
                for part in parts:
                    if not self._evaluate_simple_condition(part.strip()):
                        return False
                return True
            else:
                return self._evaluate_simple_condition(rule_text)
                
        except Exception as e:
            logger.warning(f"Erro ao avaliar condição da regra: {e}")
            return False
    
    def _evaluate_simple_condition(self, condition: str) -> bool:
        """
        Avalia uma condição simples (versão simplificada).
        
        Args:
            condition: Condição a ser avaliada
            
        Returns:
            True se a condição é verdadeira, False caso contrário
        """
        try:
            # Remover 'if' se presente
            condition = condition.replace('if ', '').replace(':', '')
            
            # Verificar operadores
            if '>' in condition:
                left, right = condition.split('>')
                return float(left.strip()) > float(right.strip())
            elif '<' in condition:
                left, right = condition.split('<')
                return float(left.strip()) < float(right.strip())
            elif '==' in condition:
                left, right = condition.split('==')
                return left.strip().strip("'\"") == right.strip().strip("'\"")
            elif 'True' in condition:
                return True
            elif 'False' in condition:
                return False
            
            return False
            
        except Exception as e:
            logger.warning(f"Erro ao avaliar condição simples: {e}")
            return False
    
    def _apply_rule_action(self, rule: Dict[str, Any]) -> float:
        """
        Aplica a ação de uma regra.
        
        Args:
            rule: Regra com ação a ser aplicada
            
        Returns:
            Valor do boost aplicado
        """
        try:
            action = rule.get('action', '')
            
            if 'confidence_boost' in action:
                # Extrair valor do boost
                import re
                boost_match = re.search(r'confidence_boost\s*\+\=\s*([0-9.]+)', action)
                if boost_match:
                    return float(boost_match.group(1))
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Erro ao aplicar ação da regra: {e}")
            return 0.0
    
    def run_bug_auto_correction_cycle(self, log_file: str = "logs/debug.log") -> Dict[str, Any]:
        """
        Executa um ciclo de auto-correção de bugs.
        
        Args:
            log_file: Caminho para o arquivo de log a ser analisado
        
        Returns:
            Relatório do ciclo de correção executado
        """
        if not self.bug_auto_correction_system:
            return {"error": "Sistema de auto-correção de bugs não disponível"}
        
        try:
            # Executar ciclo de auto-correção
            report = self.bug_auto_correction_system.run_auto_correction_cycle(log_file)
            
            logger.info(f"Ciclo de auto-correção de bugs executado: {report}")
            return report
            
        except Exception as e:
            error_msg = f"Erro ao executar ciclo de auto-correção de bugs: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def handle_exception_with_auto_correction(self, exception: Exception, traceback_str: str = None) -> Dict[str, Any]:
        """
        Trata uma exceção usando o sistema de auto-correção.
        
        Args:
            exception: Exceção capturada
            traceback_str: String do traceback
            
        Returns:
            Resultado do tratamento da exceção
        """
        if not self.bug_auto_correction_system:
            return {"error": "Sistema de auto-correção de bugs não disponível"}
        
        try:
            # Tratar exceção com auto-correção
            result = self.bug_auto_correction_system.handle_exception(exception, traceback_str)
            
            logger.info(f"Exceção tratada com auto-correção: {result}")
            return result
            
        except Exception as e:
            error_msg = f"Erro ao tratar exceção com auto-correção: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def initialize_algorithm_evolution(self) -> Dict[str, Any]:
        """
        Inicializa o sistema de evolução de algoritmos com a configuração atual.
        
        Returns:
            Resultado da inicialização
        """
        if not self.algorithm_evolution_system:
            return {"error": "Sistema de evolução de algoritmos não disponível"}
        
        try:
            # Criar configuração base a partir do estado atual
            base_config = {
                'parameters': {
                    'bird_threshold': 0.5,
                    'confidence_threshold': 0.6,
                    'boost_factor': 0.1,
                    'learning_rate': 0.01,
                    'color_weight': 0.3,
                    'shape_weight': 0.4,
                    'pattern_weight': 0.3
                },
                'architecture': {
                    'detection_config': {
                        'num_layers': 3,
                        'window_size': 640,
                        'confidence_threshold': 0.6
                    },
                    'learning_config': {
                        'optimizer': 'adam',
                        'batch_size': 32,
                        'epochs': 10
                    }
                },
                'strategy': {
                    'detection_order': ['yolo', 'color_analysis', 'shape_analysis', 'pattern_analysis'],
                    'fallback_strategy': 'balanced',
                    'learning_strategy': 'incremental'
                }
            }
            
            success = self.algorithm_evolution_system.initialize_with_base_config(base_config)
            
            if success:
                logger.info("Sistema de evolução de algoritmos inicializado com configuração base")
                return {"success": True, "message": "Sistema inicializado com sucesso"}
            else:
                return {"error": "Falha ao inicializar sistema de evolução"}
                
        except Exception as e:
            error_msg = f"Erro ao inicializar evolução de algoritmos: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def run_algorithm_evolution_cycle(self, performance_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Executa um ciclo de evolução de algoritmos.
        
        Args:
            performance_data: Dados de performance para avaliação
            
        Returns:
            Relatório do ciclo de evolução
        """
        if not self.algorithm_evolution_system:
            return {"error": "Sistema de evolução de algoritmos não disponível"}
        
        try:
            # Se não há dados de performance, criar dados simulados
            if performance_data is None:
                performance_data = {
                    'correct_detections': random.randint(80, 95),
                    'total_detections': 100,
                    'avg_processing_time': random.uniform(0.5, 2.0),
                    'performance_variance': random.uniform(0.1, 0.3),
                    'learning_rate': random.uniform(0.01, 0.1),
                    'convergence_time': random.randint(50, 100),
                    'memory_usage': random.uniform(0.3, 0.7),
                    'cpu_usage': random.uniform(0.2, 0.6)
                }
            
            # Executar ciclo de evolução
            report = self.algorithm_evolution_system.run_evolution_cycle(performance_data)
            
            logger.info(f"Ciclo de evolução de algoritmos executado: {report}")
            return report
            
        except Exception as e:
            error_msg = f"Erro ao executar ciclo de evolução de algoritmos: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_best_algorithm_config(self) -> Dict[str, Any]:
        """
        Retorna a configuração do melhor algoritmo evolucionado.
        
        Returns:
            Configuração do melhor algoritmo
        """
        if not self.algorithm_evolution_system:
            return {"error": "Sistema de evolução de algoritmos não disponível"}
        
        try:
            best_config = self.algorithm_evolution_system.get_best_algorithm_config()
            
            if best_config:
                logger.info("Configuração do melhor algoritmo obtida")
                return best_config
            else:
                return {"error": "Nenhuma configuração de algoritmo disponível"}
                
        except Exception as e:
            error_msg = f"Erro ao obter melhor configuração de algoritmo: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_algorithm_evolution_status(self) -> Dict[str, Any]:
        """
        Retorna o status do sistema de evolução de algoritmos.
        
        Returns:
            Status do sistema
        """
        if not self.algorithm_evolution_system:
            return {"error": "Sistema de evolução de algoritmos não disponível"}
        
        try:
            status = self.algorithm_evolution_system.get_evolution_status()
            return status
            
        except Exception as e:
            error_msg = f"Erro ao obter status da evolução de algoritmos: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def start_auto_optimization(self) -> Dict[str, Any]:
        """
        Inicia o sistema de auto-otimização.
        
        Returns:
            Resultado da inicialização
        """
        if not self.auto_optimization_system:
            return {"error": "Sistema de auto-otimização não disponível"}
        
        try:
            self.auto_optimization_system.start_optimization()
            logger.info("Sistema de auto-otimização iniciado")
            return {"success": True, "message": "Auto-otimização iniciada com sucesso"}
            
        except Exception as e:
            error_msg = f"Erro ao iniciar auto-otimização: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def stop_auto_optimization(self) -> Dict[str, Any]:
        """
        Para o sistema de auto-otimização.
        
        Returns:
            Resultado da parada
        """
        if not self.auto_optimization_system:
            return {"error": "Sistema de auto-otimização não disponível"}
        
        try:
            self.auto_optimization_system.stop_optimization()
            logger.info("Sistema de auto-otimização parado")
            return {"success": True, "message": "Auto-otimização parada com sucesso"}
            
        except Exception as e:
            error_msg = f"Erro ao parar auto-otimização: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_optimized_thresholds(self) -> Dict[str, Any]:
        """
        Retorna os thresholds otimizados atuais.
        
        Returns:
            Configuração de thresholds otimizada
        """
        if not self.auto_optimization_system:
            return {"error": "Sistema de auto-otimização não disponível"}
        
        try:
            config = self.auto_optimization_system.get_current_configuration()
            
            return {
                "bird_threshold": config.bird_threshold,
                "confidence_threshold": config.confidence_threshold,
                "boost_factor": config.boost_factor,
                "color_weight": config.color_weight,
                "shape_weight": config.shape_weight,
                "pattern_weight": config.pattern_weight,
                "detection_sensitivity": config.detection_sensitivity,
                "reasoning_threshold": config.reasoning_threshold,
                "adaptation_rate": config.adaptation_rate
            }
            
        except Exception as e:
            error_msg = f"Erro ao obter thresholds otimizados: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_optimization_performance_report(self) -> Dict[str, Any]:
        """
        Retorna relatório de performance da otimização.
        
        Returns:
            Relatório de performance
        """
        if not self.auto_optimization_system:
            return {"error": "Sistema de auto-otimização não disponível"}
        
        try:
            report = self.auto_optimization_system.get_performance_report()
            return report
            
        except Exception as e:
            error_msg = f"Erro ao obter relatório de performance: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def apply_optimized_thresholds(self) -> Dict[str, Any]:
        """
        Aplica os thresholds otimizados ao sistema.
        
        Returns:
            Resultado da aplicação
        """
        if not self.auto_optimization_system:
            return {"error": "Sistema de auto-otimização não disponível"}
        
        try:
            config = self.auto_optimization_system.get_current_configuration()
            
            # Aplicar thresholds otimizados
            self.bird_threshold = config.bird_threshold
            self.confidence_threshold = config.confidence_threshold
            self.boost_factor = config.boost_factor
            self.color_weight = config.color_weight
            self.shape_weight = config.shape_weight
            self.pattern_weight = config.pattern_weight
            self.detection_sensitivity = config.detection_sensitivity
            self.reasoning_threshold = config.reasoning_threshold
            self.adaptation_rate = config.adaptation_rate
            
            logger.info("Thresholds otimizados aplicados com sucesso")
            return {
                "success": True, 
                "message": "Thresholds otimizados aplicados",
                "thresholds": {
                    "bird_threshold": self.bird_threshold,
                    "confidence_threshold": self.confidence_threshold,
                    "boost_factor": self.boost_factor,
                    "color_weight": self.color_weight,
                    "shape_weight": self.shape_weight,
                    "pattern_weight": self.pattern_weight,
                    "detection_sensitivity": self.detection_sensitivity,
                    "reasoning_threshold": self.reasoning_threshold,
                    "adaptation_rate": self.adaptation_rate
                }
            }
            
        except Exception as e:
            error_msg = f"Erro ao aplicar thresholds otimizados: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def process_detection_for_optimization(self, 
                                         is_bird: bool, 
                                         predicted_bird: bool, 
                                         confidence: float, 
                                         processing_time: float) -> Dict[str, Any]:
        """
        Processa uma detecção para otimização automática.
        
        Args:
            is_bird: Se a imagem é realmente um pássaro
            predicted_bird: Se o sistema previu que é um pássaro
            confidence: Confiança da predição
            processing_time: Tempo de processamento
            
        Returns:
            Resultado do processamento
        """
        if not self.auto_optimization_system:
            return {"error": "Sistema de auto-otimização não disponível"}
        
        try:
            # Processar detecção para otimização
            new_config = self.auto_optimization_system.process_detection(
                is_bird, predicted_bird, confidence, processing_time
            )
            
            if new_config:
                # Aplicar nova configuração automaticamente
                self.bird_threshold = new_config.bird_threshold
                self.confidence_threshold = new_config.confidence_threshold
                self.boost_factor = new_config.boost_factor
                self.color_weight = new_config.color_weight
                self.shape_weight = new_config.shape_weight
                self.pattern_weight = new_config.pattern_weight
                self.detection_sensitivity = new_config.detection_sensitivity
                self.reasoning_threshold = new_config.reasoning_threshold
                self.adaptation_rate = new_config.adaptation_rate
                
                logger.info("Thresholds otimizados automaticamente aplicados")
                return {
                    "success": True,
                    "message": "Thresholds otimizados automaticamente",
                    "new_thresholds": {
                        "bird_threshold": new_config.bird_threshold,
                        "confidence_threshold": new_config.confidence_threshold,
                        "boost_factor": new_config.boost_factor,
                        "color_weight": new_config.color_weight,
                        "shape_weight": new_config.shape_weight,
                        "pattern_weight": new_config.pattern_weight,
                        "detection_sensitivity": new_config.detection_sensitivity,
                        "reasoning_threshold": new_config.reasoning_threshold,
                        "adaptation_rate": new_config.adaptation_rate
                    }
                }
            else:
                return {
                    "success": True,
                    "message": "Detecção processada, sem otimização necessária"
                }
                
        except Exception as e:
            error_msg = f"Erro ao processar detecção para otimização: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def start_advanced_weight_optimization(self) -> Dict[str, Any]:
        """
        Inicia o sistema de otimização apurada dos pesos.
        
        Returns:
            Resultado da inicialização
        """
        if not self.advanced_weight_optimization_system:
            return {"error": "Sistema de otimização apurada dos pesos não disponível"}
        
        try:
            self.advanced_weight_optimization_system.start_optimization()
            logger.info("Sistema de otimização apurada dos pesos iniciado")
            return {"success": True, "message": "Otimização apurada dos pesos iniciada com sucesso"}
            
        except Exception as e:
            error_msg = f"Erro ao iniciar otimização apurada dos pesos: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def stop_advanced_weight_optimization(self) -> Dict[str, Any]:
        """
        Para o sistema de otimização apurada dos pesos.
        
        Returns:
            Resultado da parada
        """
        if not self.advanced_weight_optimization_system:
            return {"error": "Sistema de otimização apurada dos pesos não disponível"}
        
        try:
            self.advanced_weight_optimization_system.stop_optimization()
            logger.info("Sistema de otimização apurada dos pesos parado")
            return {"success": True, "message": "Otimização apurada dos pesos parada com sucesso"}
            
        except Exception as e:
            error_msg = f"Erro ao parar otimização apurada dos pesos: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_optimized_weights(self) -> Dict[str, Any]:
        """
        Retorna os pesos otimizados atuais.
        
        Returns:
            Configuração de pesos otimizada
        """
        if not self.advanced_weight_optimization_system:
            return {"error": "Sistema de otimização apurada dos pesos não disponível"}
        
        try:
            weights = self.advanced_weight_optimization_system.get_current_weights()
            
            return {
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
                "optimization_score": weights.optimization_score
            }
            
        except Exception as e:
            error_msg = f"Erro ao obter pesos otimizados: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_weight_optimization_analysis(self) -> Dict[str, Any]:
        """
        Retorna análise de otimização dos pesos.
        
        Returns:
            Análise dos componentes e tendências
        """
        if not self.advanced_weight_optimization_system:
            return {"error": "Sistema de otimização apurada dos pesos não disponível"}
        
        try:
            analysis = self.advanced_weight_optimization_system.get_component_analysis()
            return analysis
            
        except Exception as e:
            error_msg = f"Erro ao obter análise de otimização dos pesos: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def apply_optimized_weights(self) -> Dict[str, Any]:
        """
        Aplica os pesos otimizados ao sistema.
        
        Returns:
            Resultado da aplicação
        """
        if not self.advanced_weight_optimization_system:
            return {"error": "Sistema de otimização apurada dos pesos não disponível"}
        
        try:
            weights = self.advanced_weight_optimization_system.get_current_weights()
            
            # Aplicar pesos otimizados aos componentes do sistema
            self.color_weight = weights.color_weight
            self.shape_weight = weights.shape_weight
            self.pattern_weight = weights.pattern_weight
            
            # Aplicar pesos de confiança
            self.yolo_confidence_weight = weights.yolo_confidence_weight
            self.color_confidence_weight = weights.color_confidence_weight
            self.shape_confidence_weight = weights.shape_confidence_weight
            self.pattern_confidence_weight = weights.pattern_confidence_weight
            
            # Aplicar pesos de características específicas
            self.beak_weight = weights.beak_weight
            self.wing_weight = weights.wing_weight
            self.tail_weight = weights.tail_weight
            self.eye_weight = weights.eye_weight
            
            # Aplicar pesos de contexto
            self.background_weight = weights.background_weight
            self.lighting_weight = weights.lighting_weight
            self.angle_weight = weights.angle_weight
            
            # Aplicar pesos de aprendizado
            self.learned_pattern_weight = weights.learned_pattern_weight
            self.species_boost_weight = weights.species_boost_weight
            self.characteristic_boost_weight = weights.characteristic_boost_weight
            
            logger.info("Pesos otimizados aplicados com sucesso")
            return {
                "success": True, 
                "message": "Pesos otimizados aplicados",
                "applied_weights": {
                    "color_weight": self.color_weight,
                    "shape_weight": self.shape_weight,
                    "pattern_weight": self.pattern_weight,
                    "texture_weight": weights.texture_weight,
                    "size_weight": weights.size_weight,
                    "optimization_score": weights.optimization_score
                }
            }
            
        except Exception as e:
            error_msg = f"Erro ao aplicar pesos otimizados: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def process_detection_batch_for_weight_optimization(self, 
                                                      detection_results: List[Dict[str, Any]], 
                                                      ground_truth: List[bool]) -> Dict[str, Any]:
        """
        Processa um lote de detecções para otimização de pesos.
        
        Args:
            detection_results: Lista de resultados de detecção
            ground_truth: Lista de valores verdadeiros
            
        Returns:
            Resultado do processamento
        """
        if not self.advanced_weight_optimization_system:
            return {"error": "Sistema de otimização apurada dos pesos não disponível"}
        
        try:
            # Processar lote para otimização de pesos
            new_weights = self.advanced_weight_optimization_system.process_detection_batch(
                detection_results, ground_truth
            )
            
            if new_weights:
                # Aplicar novos pesos automaticamente
                self.color_weight = new_weights.color_weight
                self.shape_weight = new_weights.shape_weight
                self.pattern_weight = new_weights.pattern_weight
                self.yolo_confidence_weight = new_weights.yolo_confidence_weight
                self.color_confidence_weight = new_weights.color_confidence_weight
                self.shape_confidence_weight = new_weights.shape_confidence_weight
                self.pattern_confidence_weight = new_weights.pattern_confidence_weight
                self.beak_weight = new_weights.beak_weight
                self.wing_weight = new_weights.wing_weight
                self.tail_weight = new_weights.tail_weight
                self.eye_weight = new_weights.eye_weight
                self.background_weight = new_weights.background_weight
                self.lighting_weight = new_weights.lighting_weight
                self.angle_weight = new_weights.angle_weight
                self.learned_pattern_weight = new_weights.learned_pattern_weight
                self.species_boost_weight = new_weights.species_boost_weight
                self.characteristic_boost_weight = new_weights.characteristic_boost_weight
                
                logger.info("Pesos otimizados automaticamente aplicados")
                return {
                    "success": True,
                    "message": "Pesos otimizados automaticamente",
                    "new_weights": {
                        "color_weight": new_weights.color_weight,
                        "shape_weight": new_weights.shape_weight,
                        "pattern_weight": new_weights.pattern_weight,
                        "texture_weight": new_weights.texture_weight,
                        "size_weight": new_weights.size_weight,
                        "optimization_score": new_weights.optimization_score
                    }
                }
            else:
                return {
                    "success": True,
                    "message": "Lote processado, sem otimização necessária"
                }
                
        except Exception as e:
            error_msg = f"Erro ao processar lote para otimização de pesos: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def start_architecture_evolution(self) -> Dict[str, Any]:
        """
        Inicia o sistema de evolução de arquitetura.
        
        Returns:
            Resultado da inicialização
        """
        if not self.architecture_evolution_system:
            return {"error": "Sistema de evolução de arquitetura não disponível"}
        
        try:
            self.architecture_evolution_system.start_evolution()
            logger.info("Sistema de evolução de arquitetura iniciado")
            return {"success": True, "message": "Evolução de arquitetura iniciada com sucesso"}
            
        except Exception as e:
            error_msg = f"Erro ao iniciar evolução de arquitetura: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def stop_architecture_evolution(self) -> Dict[str, Any]:
        """
        Para o sistema de evolução de arquitetura.
        
        Returns:
            Resultado da parada
        """
        if not self.architecture_evolution_system:
            return {"error": "Sistema de evolução de arquitetura não disponível"}
        
        try:
            self.architecture_evolution_system.stop_evolution()
            logger.info("Sistema de evolução de arquitetura parado")
            return {"success": True, "message": "Evolução de arquitetura parada com sucesso"}
            
        except Exception as e:
            error_msg = f"Erro ao parar evolução de arquitetura: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_current_architecture(self) -> Dict[str, Any]:
        """
        Retorna a arquitetura atual.
        
        Returns:
            Configuração da arquitetura atual
        """
        if not self.architecture_evolution_system:
            return {"error": "Sistema de evolução de arquitetura não disponível"}
        
        try:
            architecture = self.architecture_evolution_system.get_current_architecture()
            return architecture
            
        except Exception as e:
            error_msg = f"Erro ao obter arquitetura atual: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_architecture_analysis(self) -> Dict[str, Any]:
        """
        Retorna análise da arquitetura atual.
        
        Returns:
            Análise detalhada da arquitetura
        """
        if not self.architecture_evolution_system:
            return {"error": "Sistema de evolução de arquitetura não disponível"}
        
        try:
            analysis = self.architecture_evolution_system.get_architecture_analysis()
            return analysis
            
        except Exception as e:
            error_msg = f"Erro ao obter análise da arquitetura: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def run_architecture_evolution_cycle(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa um ciclo de evolução de arquitetura.
        
        Args:
            performance_data: Dados de performance do sistema
            
        Returns:
            Resultado do ciclo de evolução
        """
        if not self.architecture_evolution_system:
            return {"error": "Sistema de evolução de arquitetura não disponível"}
        
        try:
            result = self.architecture_evolution_system.run_evolution_cycle(performance_data)
            
            if result.get("success") and "improvement" in result:
                logger.info(f"Arquitetura evoluída: {result['improvement']:.3f} de melhoria")
            
            return result
            
        except Exception as e:
            error_msg = f"Erro ao executar ciclo de evolução de arquitetura: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def analyze_with_abstract_concepts(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa um objeto usando conceitos abstratos.
        
        Args:
            detection_data: Dados de detecção do objeto
            
        Returns:
            Análise usando conceitos abstratos
        """
        if not self.abstract_concept_learner:
            return {"error": "Sistema de conceitos abstratos não disponível"}
        
        try:
            analysis = self.abstract_concept_learner.analyze_object(detection_data)
            
            # Log da análise
            if analysis["concept_applications"].get("passarinidade", {}).get("is_bird"):
                logger.info("Objeto identificado como pássaro usando conceitos abstratos")
            else:
                logger.info("Objeto não identificado como pássaro usando conceitos abstratos")
            
            return analysis
            
        except Exception as e:
            error_msg = f"Erro ao analisar com conceitos abstratos: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def learn_from_concept_example(self, detection_data: Dict[str, Any], correct_label: str, feedback: str = "") -> Dict[str, Any]:
        """
        Aprende com um exemplo usando conceitos abstratos.
        
        Args:
            detection_data: Dados de detecção
            correct_label: Rótulo correto
            feedback: Feedback adicional
            
        Returns:
            Resultado do aprendizado
        """
        if not self.abstract_concept_learner:
            return {"error": "Sistema de conceitos abstratos não disponível"}
        
        try:
            self.abstract_concept_learner.learn_from_example(detection_data, correct_label, feedback)
            
            logger.info(f"Exemplo aprendido com conceitos abstratos: {correct_label}")
            
            return {
                "success": True,
                "message": f"Exemplo aprendido: {correct_label}",
                "feedback": feedback
            }
            
        except Exception as e:
            error_msg = f"Erro ao aprender com exemplo: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_concept_analysis(self, concept_name: str = "passarinidade") -> Dict[str, Any]:
        """
        Retorna análise de um conceito específico.
        
        Args:
            concept_name: Nome do conceito
            
        Returns:
            Análise do conceito
        """
        if not self.abstract_concept_learner:
            return {"error": "Sistema de conceitos abstratos não disponível"}
        
        try:
            analysis = self.abstract_concept_learner.get_concept_analysis(concept_name)
            return analysis
            
        except Exception as e:
            error_msg = f"Erro ao obter análise do conceito: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_all_concepts(self) -> Dict[str, Any]:
        """
        Retorna todos os conceitos disponíveis.
        
        Returns:
            Todos os conceitos
        """
        if not self.abstract_concept_learner:
            return {"error": "Sistema de conceitos abstratos não disponível"}
        
        try:
            concepts = self.abstract_concept_learner.get_all_concepts()
            return concepts
            
        except Exception as e:
            error_msg = f"Erro ao obter conceitos: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_universal_features_analysis(self) -> Dict[str, Any]:
        """
        Retorna análise das características universais.
        
        Returns:
            Análise das características universais
        """
        if not self.abstract_concept_learner:
            return {"error": "Sistema de conceitos abstratos não disponível"}
        
        try:
            analysis = self.abstract_concept_learner.get_universal_features_analysis()
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
        if not self.abstract_concept_learner:
            return {"error": "Sistema de conceitos abstratos não disponível"}
        
        try:
            result = self.abstract_concept_learner.learn_universal_pattern(
                detection_data, species_label, feedback
            )
            
            logger.info(f"Padrão universal aprendido: {species_label}")
            
            return result
            
        except Exception as e:
            error_msg = f"Erro ao aprender padrão universal: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def analyze_with_universal_features(self, detection_data: Dict[str, Any], species_context: str = "") -> Dict[str, Any]:
        """
        Analisa um objeto usando características universais.
        
        Args:
            detection_data: Dados de detecção do objeto
            species_context: Contexto da espécie
            
        Returns:
            Análise usando características universais
        """
        if not self.abstract_concept_learner:
            return {"error": "Sistema de conceitos abstratos não disponível"}
        
        try:
            analysis = self.abstract_concept_learner.analyze_object(detection_data, species_context)
            
            # Log da análise
            if analysis.get("universal_analysis"):
                logger.info("Análise com características universais realizada")
            
            return analysis
            
        except Exception as e:
            error_msg = f"Erro ao analisar com características universais: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_concept_hierarchy_analysis(self, concept_name: str = None) -> Dict[str, Any]:
        """
        Retorna análise da hierarquia de conceitos.
        
        Args:
            concept_name: Nome do conceito específico (opcional)
            
        Returns:
            Análise da hierarquia de conceitos
        """
        if not self.abstract_concept_learner:
            return {"error": "Sistema de conceitos abstratos não disponível"}
        
        try:
            analysis = self.abstract_concept_learner.get_concept_hierarchy_analysis(concept_name)
            return analysis
            
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
        if not self.abstract_concept_learner:
            return {"error": "Sistema de conceitos abstratos não disponível"}
        
        try:
            similar_concepts = self.abstract_concept_learner.find_similar_concepts(concept_name, threshold)
            return similar_concepts
            
        except Exception as e:
            error_msg = f"Erro ao encontrar conceitos similares: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def analyze_for_universal_generalization(self, detection_data: Dict[str, Any], species: str, context: str = "general") -> Dict[str, Any]:
        """
        Analisa dados para generalização universal.
        
        Args:
            detection_data: Dados de detecção
            species: Espécie identificada
            context: Contexto da análise
            
        Returns:
            Análise de generalização universal
        """
        if not self.universal_generalization_system:
            return {"error": "Sistema de generalização universal não disponível"}
        
        try:
            analysis = self.universal_generalization_system.analyze_for_universal_generalization(
                detection_data, species, context
            )
            return analysis
            
        except Exception as e:
            error_msg = f"Erro na análise de generalização universal: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def transfer_knowledge_between_species(self, source_species: str, target_species: str, 
                                         features: List[str], strategy: str = "adaptive") -> Dict[str, Any]:
        """
        Transfere conhecimento entre espécies.
        
        Args:
            source_species: Espécie fonte
            target_species: Espécie alvo
            features: Características a transferir
            strategy: Estratégia de transferência
            
        Returns:
            Resultado da transferência
        """
        if not self.universal_generalization_system:
            return {"error": "Sistema de generalização universal não disponível"}
        
        try:
            # Converter string para enum
            strategy_enum = TransferStrategy.ADAPTIVE
            if strategy == "direct":
                strategy_enum = TransferStrategy.DIRECT
            elif strategy == "selective":
                strategy_enum = TransferStrategy.SELECTIVE
            elif strategy == "hierarchical":
                strategy_enum = TransferStrategy.HIERARCHICAL
            elif strategy == "contextual":
                strategy_enum = TransferStrategy.CONTEXTUAL
            
            result = self.universal_generalization_system.transfer_knowledge_between_species(
                source_species, target_species, features, strategy_enum
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro na transferência de conhecimento: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def learn_universal_pattern(self, detection_data: Dict[str, Any], species: str, 
                               pattern_name: str, pattern_type: str = "morphological") -> Dict[str, Any]:
        """
        Aprende um novo padrão universal.
        
        Args:
            detection_data: Dados de detecção
            species: Espécie
            pattern_name: Nome do padrão
            pattern_type: Tipo do padrão
            
        Returns:
            Resultado do aprendizado
        """
        if not self.universal_generalization_system:
            return {"error": "Sistema de generalização universal não disponível"}
        
        try:
            # Converter string para enum
            pattern_type_enum = UniversalPatternType.MORPHOLOGICAL
            if pattern_type == "behavioral":
                pattern_type_enum = UniversalPatternType.BEHAVIORAL
            elif pattern_type == "ecological":
                pattern_type_enum = UniversalPatternType.ECOLOGICAL
            elif pattern_type == "physiological":
                pattern_type_enum = UniversalPatternType.PHYSIOLOGICAL
            elif pattern_type == "evolutionary":
                pattern_type_enum = UniversalPatternType.EVOLUTIONARY
            
            result = self.universal_generalization_system.learn_universal_pattern(
                detection_data, species, pattern_name, pattern_type_enum
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro ao aprender padrão universal: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def learn_concept_few_shot(self, concept_name: str, concept_type: str, 
                              examples: List[Dict[str, Any]], strategy: str = "prototype_based") -> Dict[str, Any]:
        """
        Aprende um conceito usando few-shot learning.
        
        Args:
            concept_name: Nome do conceito
            concept_type: Tipo do conceito
            examples: Lista de exemplos
            strategy: Estratégia de aprendizado
            
        Returns:
            Resultado do aprendizado few-shot
        """
        if not self.few_shot_learner:
            return {"error": "Sistema de few-shot learning não disponível"}
        
        try:
            # Converter string para enum
            concept_type_enum = ConceptType.SPECIES
            if concept_type == "object":
                concept_type_enum = ConceptType.OBJECT
            elif concept_type == "behavior":
                concept_type_enum = ConceptType.BEHAVIOR
            elif concept_type == "pattern":
                concept_type_enum = ConceptType.PATTERN
            elif concept_type == "feature":
                concept_type_enum = ConceptType.FEATURE
            
            strategy_enum = FewShotStrategy.PROTOTYPE_BASED
            if strategy == "similarity_based":
                strategy_enum = FewShotStrategy.SIMILARITY_BASED
            elif strategy == "meta_learning":
                strategy_enum = FewShotStrategy.META_LEARNING
            elif strategy == "transfer_learning":
                strategy_enum = FewShotStrategy.TRANSFER_LEARNING
            elif strategy == "adaptive_fusion":
                strategy_enum = FewShotStrategy.ADAPTIVE_FUSION
            
            result = self.few_shot_learner.learn_concept_few_shot(
                concept_name, concept_type_enum, examples, strategy_enum
            )
            
            return {
                "success": result.success,
                "confidence": result.confidence,
                "learning_mode": result.learning_mode.value,
                "strategy_used": result.strategy_used.value,
                "examples_used": result.examples_used,
                "transfer_sources": result.transfer_sources,
                "adaptation_required": result.adaptation_required,
                "performance_improvement": result.performance_improvement
            }
            
        except Exception as e:
            error_msg = f"Erro no aprendizado few-shot: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def classify_with_few_shot(self, detection_data: Dict[str, Any], 
                               concept_type: str = None) -> Dict[str, Any]:
        """
        Classifica dados usando few-shot learning.
        
        Args:
            detection_data: Dados de detecção
            concept_type: Tipo do conceito (opcional)
            
        Returns:
            Resultado da classificação few-shot
        """
        if not self.few_shot_learner:
            return {"error": "Sistema de few-shot learning não disponível"}
        
        try:
            # Converter string para enum
            concept_type_enum = None
            if concept_type:
                if concept_type == "species":
                    concept_type_enum = ConceptType.SPECIES
                elif concept_type == "object":
                    concept_type_enum = ConceptType.OBJECT
                elif concept_type == "behavior":
                    concept_type_enum = ConceptType.BEHAVIOR
                elif concept_type == "pattern":
                    concept_type_enum = ConceptType.PATTERN
                elif concept_type == "feature":
                    concept_type_enum = ConceptType.FEATURE
            
            result = self.few_shot_learner.classify_with_few_shot(detection_data, concept_type_enum)
            return result
            
        except Exception as e:
            error_msg = f"Erro na classificação few-shot: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_few_shot_analysis(self, concept_name: str = None) -> Dict[str, Any]:
        """
        Obtém análise do sistema few-shot.
        
        Args:
            concept_name: Nome do conceito específico (opcional)
            
        Returns:
            Análise do sistema few-shot
        """
        if not self.few_shot_learner:
            return {"error": "Sistema de few-shot learning não disponível"}
        
        try:
            analysis = self.few_shot_learner.get_few_shot_analysis(concept_name)
            return analysis
            
        except Exception as e:
            error_msg = f"Erro na análise few-shot: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def analyze_learning_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa uma sessão de aprendizado para meta-aprendizado.
        
        Args:
            session_data: Dados da sessão de aprendizado
            
        Returns:
            Análise da sessão de aprendizado
        """
        if not self.meta_learning_system:
            return {"error": "Sistema de meta-aprendizado não disponível"}
        
        try:
            # Converter dados para LearningSession
            from .meta_learning import LearningSession, LearningContext
            
            context_enum = LearningContext.NEW_CONCEPT
            if session_data.get("context") == "similar_concept":
                context_enum = LearningContext.SIMILAR_CONCEPT
            elif session_data.get("context") == "complex_concept":
                context_enum = LearningContext.COMPLEX_CONCEPT
            elif session_data.get("context") == "simple_concept":
                context_enum = LearningContext.SIMPLE_CONCEPT
            elif session_data.get("context") == "few_examples":
                context_enum = LearningContext.FEW_EXAMPLES
            elif session_data.get("context") == "many_examples":
                context_enum = LearningContext.MANY_EXAMPLES
            
            import time
            session = LearningSession(
                session_id=session_data.get("session_id", f"session_{int(time.time())}"),
                concept_name=session_data.get("concept_name", "unknown"),
                concept_type=session_data.get("concept_type", "species"),
                strategy_used=session_data.get("strategy_used", "prototype_based"),
                context=context_enum,
                examples_count=session_data.get("examples_count", 1),
                performance_metrics=session_data.get("performance_metrics", {}),
                success=session_data.get("success", False),
                learning_time=session_data.get("learning_time", 1.0),
                timestamp=time.time(),
                metadata=session_data.get("metadata", {})
            )
            
            analysis = self.meta_learning_system.analyze_learning_session(session)
            return analysis
            
        except Exception as e:
            error_msg = f"Erro na análise de sessão de aprendizado: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def recommend_learning_strategy(self, concept_name: str, concept_type: str, 
                                  examples_count: int, context: str = "new_concept") -> Dict[str, Any]:
        """
        Recomenda estratégia de aprendizado baseada em meta-aprendizado.
        
        Args:
            concept_name: Nome do conceito
            concept_type: Tipo do conceito
            examples_count: Número de exemplos
            context: Contexto de aprendizado
            
        Returns:
            Recomendação de estratégia de aprendizado
        """
        if not self.meta_learning_system:
            return {"error": "Sistema de meta-aprendizado não disponível"}
        
        try:
            # Converter string para enum
            from .meta_learning import LearningContext
            
            context_enum = LearningContext.NEW_CONCEPT
            if context == "similar_concept":
                context_enum = LearningContext.SIMILAR_CONCEPT
            elif context == "complex_concept":
                context_enum = LearningContext.COMPLEX_CONCEPT
            elif context == "simple_concept":
                context_enum = LearningContext.SIMPLE_CONCEPT
            elif context == "few_examples":
                context_enum = LearningContext.FEW_EXAMPLES
            elif context == "many_examples":
                context_enum = LearningContext.MANY_EXAMPLES
            
            recommendation = self.meta_learning_system.recommend_learning_strategy(
                concept_name, concept_type, examples_count, context_enum
            )
            return recommendation
            
        except Exception as e:
            error_msg = f"Erro na recomendação de estratégia: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_meta_learning_analysis(self) -> Dict[str, Any]:
        """
        Obtém análise geral do sistema de meta-aprendizado.
        
        Returns:
            Análise do sistema de meta-aprendizado
        """
        if not self.meta_learning_system:
            return {"error": "Sistema de meta-aprendizado não disponível"}
        
        try:
            analysis = self.meta_learning_system.get_meta_learning_analysis()
            return analysis
            
        except Exception as e:
            error_msg = f"Erro na análise de meta-aprendizado: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def infer_abstract_properties(self, concept_name: str, target_property: str = None) -> Dict[str, Any]:
        """
        Infere propriedades abstratas de um conceito.
        
        Args:
            concept_name: Nome do conceito
            target_property: Propriedade específica para inferir (opcional)
            
        Returns:
            Resultado da inferência abstrata
        """
        if not self.conceptual_reasoning_system:
            return {"error": "Sistema de raciocínio conceitual não disponível"}
        
        try:
            result = self.conceptual_reasoning_system.infer_abstract_properties(concept_name, target_property)
            return result
            
        except Exception as e:
            error_msg = f"Erro na inferência abstrata: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def add_concept(self, name: str, definition: str, properties: List[str], 
                   abstraction_level: str = "general", relations: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """
        Adiciona um conceito ao sistema de raciocínio conceitual.
        
        Args:
            name: Nome do conceito
            definition: Definição do conceito
            properties: Lista de propriedades
            abstraction_level: Nível de abstração
            relations: Relações com outros conceitos
            
        Returns:
            Resultado da adição do conceito
        """
        if not self.conceptual_reasoning_system:
            return {"error": "Sistema de raciocínio conceitual não disponível"}
        
        try:
            # Converter string para enum
            abstraction_enum = AbstractionLevel.GENERAL
            if abstraction_level == "concrete":
                abstraction_enum = AbstractionLevel.CONCRETE
            elif abstraction_level == "specific":
                abstraction_enum = AbstractionLevel.SPECIFIC
            elif abstraction_level == "abstract":
                abstraction_enum = AbstractionLevel.ABSTRACT
            elif abstraction_level == "universal":
                abstraction_enum = AbstractionLevel.UNIVERSAL
            
            success = self.conceptual_reasoning_system.add_concept(
                name, definition, properties, abstraction_enum, relations
            )
            
            return {
                "success": success,
                "concept_name": name,
                "message": f"Conceito '{name}' {'adicionado' if success else 'não pôde ser adicionado'} com sucesso"
            }
            
        except Exception as e:
            error_msg = f"Erro ao adicionar conceito: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_conceptual_analysis(self, concept_name: str = None) -> Dict[str, Any]:
        """
        Obtém análise conceitual.
        
        Args:
            concept_name: Nome do conceito específico (opcional)
            
        Returns:
            Análise conceitual
        """
        if not self.conceptual_reasoning_system:
            return {"error": "Sistema de raciocínio conceitual não disponível"}
        
        try:
            analysis = self.conceptual_reasoning_system.get_conceptual_analysis(concept_name)
            return analysis
            
        except Exception as e:
            error_msg = f"Erro na análise conceitual: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def find_structural_analogy(self, source_concept: str, target_concept: str) -> Dict[str, Any]:
        """
        Encontra analogia estrutural entre dois conceitos.
        
        Args:
            source_concept: Conceito fonte
            target_concept: Conceito alvo
            
        Returns:
            Resultado da analogia estrutural
        """
        if not self.analogical_reasoning_system:
            return {"error": "Sistema de raciocínio por analogia não disponível"}
        
        try:
            result = self.analogical_reasoning_system.find_structural_analogy(source_concept, target_concept)
            return result
            
        except Exception as e:
            error_msg = f"Erro na analogia estrutural: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def reason_by_analogy(self, source_concept: str, target_concept: str, 
                         property_to_transfer: str = None) -> Dict[str, Any]:
        """
        Raciocina por analogia entre dois conceitos.
        
        Args:
            source_concept: Conceito fonte
            target_concept: Conceito alvo
            property_to_transfer: Propriedade específica para transferir (opcional)
            
        Returns:
            Resultado do raciocínio por analogia
        """
        if not self.analogical_reasoning_system:
            return {"error": "Sistema de raciocínio por analogia não disponível"}
        
        try:
            result = self.analogical_reasoning_system.reason_by_analogy(
                source_concept, target_concept, property_to_transfer
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro no raciocínio por analogia: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def add_concept_structure(self, concept_name: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adiciona estrutura de um conceito ao sistema de analogia.
        
        Args:
            concept_name: Nome do conceito
            structure: Estrutura do conceito (componentes, relações, propriedades)
            
        Returns:
            Resultado da adição da estrutura
        """
        if not self.analogical_reasoning_system:
            return {"error": "Sistema de raciocínio por analogia não disponível"}
        
        try:
            success = self.analogical_reasoning_system.add_concept_structure(concept_name, structure)
            
            return {
                "success": success,
                "concept_name": concept_name,
                "message": f"Estrutura do conceito '{concept_name}' {'adicionada' if success else 'não pôde ser adicionada'} com sucesso"
            }
            
        except Exception as e:
            error_msg = f"Erro ao adicionar estrutura de conceito: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_analogy_analysis(self, concept_name: str = None) -> Dict[str, Any]:
        """
        Obtém análise de analogias.
        
        Args:
            concept_name: Nome do conceito específico (opcional)
            
        Returns:
            Análise de analogias
        """
        if not self.analogical_reasoning_system:
            return {"error": "Sistema de raciocínio por analogia não disponível"}
        
        try:
            analysis = self.analogical_reasoning_system.get_analogy_analysis(concept_name)
            return analysis
            
        except Exception as e:
            error_msg = f"Erro na análise de analogias: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def learn_universal_pattern(self, detection_data: Dict[str, Any], 
                              species: str, pattern_name: str = None) -> Dict[str, Any]:
        """
        Aprende um padrão universal a partir de dados de detecção.
        
        Args:
            detection_data: Dados de detecção
            species: Espécie observada
            pattern_name: Nome do padrão (opcional)
            
        Returns:
            Resultado do aprendizado de padrão universal
        """
        if not self.universal_pattern_learner:
            return {"error": "Sistema de aprendizado por padrões universais não disponível"}
        
        try:
            result = self.universal_pattern_learner.learn_universal_pattern(
                detection_data, species, pattern_name
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro no aprendizado de padrão universal: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def validate_pattern_universality(self, pattern_id: str, 
                                   new_species_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida se um padrão é universal aplicando-o a uma nova espécie.
        
        Args:
            pattern_id: ID do padrão
            new_species_data: Dados da nova espécie
            
        Returns:
            Resultado da validação de universalidade
        """
        if not self.universal_pattern_learner:
            return {"error": "Sistema de aprendizado por padrões universais não disponível"}
        
        try:
            result = self.universal_pattern_learner.validate_pattern_universality(
                pattern_id, new_species_data
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro na validação de universalidade: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def find_applicable_patterns(self, detection_data: Dict[str, Any], 
                               species: str = None) -> Dict[str, Any]:
        """
        Encontra padrões universais aplicáveis aos dados de detecção.
        
        Args:
            detection_data: Dados de detecção
            species: Espécie (opcional)
            
        Returns:
            Padrões aplicáveis encontrados
        """
        if not self.universal_pattern_learner:
            return {"error": "Sistema de aprendizado por padrões universais não disponível"}
        
        try:
            result = self.universal_pattern_learner.find_applicable_patterns(
                detection_data, species
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro na busca de padrões aplicáveis: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_universal_pattern_analysis(self, pattern_id: str = None) -> Dict[str, Any]:
        """
        Obtém análise de padrões universais.
        
        Args:
            pattern_id: ID do padrão específico (opcional)
            
        Returns:
            Análise de padrões universais
        """
        if not self.universal_pattern_learner:
            return {"error": "Sistema de aprendizado por padrões universais não disponível"}
        
        try:
            analysis = self.universal_pattern_learner.get_universal_pattern_analysis(pattern_id)
            return analysis
            
        except Exception as e:
            error_msg = f"Erro na análise de padrões universais: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def store_episodic_memory(self, memory_type: str, content: Dict[str, Any], 
                            context: Dict[str, Any] = None, location: str = None,
                            emotional_state: str = None) -> Dict[str, Any]:
        """
        Armazena uma memória episódica.
        
        Args:
            memory_type: Tipo da memória
            content: Conteúdo da memória
            context: Contexto da memória
            location: Localização (opcional)
            emotional_state: Estado emocional (opcional)
            
        Returns:
            Resultado do armazenamento
        """
        if not self.episodic_memory_system:
            return {"error": "Sistema de memória episódica não disponível"}
        
        try:
            result = self.episodic_memory_system.store_episodic_memory(
                memory_type, content, context, location, emotional_state
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro no armazenamento de memória episódica: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def retrieve_memories(self, query: Dict[str, Any], 
                         retrieval_type: str = "free_recall",
                         limit: int = 10) -> Dict[str, Any]:
        """
        Recupera memórias baseado em uma consulta.
        
        Args:
            query: Consulta para recuperação
            retrieval_type: Tipo de recuperação
            limit: Limite de resultados
            
        Returns:
            Memórias recuperadas
        """
        if not self.episodic_memory_system:
            return {"error": "Sistema de memória episódica não disponível"}
        
        try:
            result = self.episodic_memory_system.retrieve_memories(
                query, retrieval_type, limit
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro na recuperação de memórias: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def consolidate_memories(self) -> Dict[str, Any]:
        """
        Consolida memórias na fila de consolidação.
        
        Returns:
            Resultado da consolidação
        """
        if not self.episodic_memory_system:
            return {"error": "Sistema de memória episódica não disponível"}
        
        try:
            result = self.episodic_memory_system.consolidate_memories()
            return result
            
        except Exception as e:
            error_msg = f"Erro na consolidação de memórias: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def forget_old_memories(self) -> Dict[str, Any]:
        """
        Esquece memórias antigas baseado no agendamento.
        
        Returns:
            Resultado do esquecimento
        """
        if not self.episodic_memory_system:
            return {"error": "Sistema de memória episódica não disponível"}
        
        try:
            result = self.episodic_memory_system.forget_old_memories()
            return result
            
        except Exception as e:
            error_msg = f"Erro no esquecimento de memórias: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_memory_analysis(self, memory_id: str = None) -> Dict[str, Any]:
        """
        Obtém análise de memórias.
        
        Args:
            memory_id: ID da memória específica (opcional)
            
        Returns:
            Análise de memórias
        """
        if not self.episodic_memory_system:
            return {"error": "Sistema de memória episódica não disponível"}
        
        try:
            analysis = self.episodic_memory_system.get_memory_analysis(memory_id)
            return analysis
            
        except Exception as e:
            error_msg = f"Erro na análise de memórias: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def identify_causal_relation(self, cause: str, effect: str, 
                               evidence: List[Dict[str, Any]] = None,
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Identifica uma relação causal entre causa e efeito.
        
        Args:
            cause: Causa
            effect: Efeito
            evidence: Evidências (opcional)
            context: Contexto (opcional)
            
        Returns:
            Resultado da identificação de relação causal
        """
        if not self.causal_reasoning_system:
            return {"error": "Sistema de raciocínio causal não disponível"}
        
        try:
            result = self.causal_reasoning_system.identify_causal_relation(
                cause, effect, evidence, context
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro na identificação de relação causal: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def infer_causal_relations(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Faz inferências causais baseadas em uma consulta.
        
        Args:
            query: Consulta para inferência causal
            
        Returns:
            Inferências causais
        """
        if not self.causal_reasoning_system:
            return {"error": "Sistema de raciocínio causal não disponível"}
        
        try:
            result = self.causal_reasoning_system.infer_causal_relations(query)
            return result
            
        except Exception as e:
            error_msg = f"Erro na inferência causal: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def detect_correlation_vs_causation(self, event_a: str, event_b: str, 
                                      evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detecta se a relação é correlação ou causalidade.
        
        Args:
            event_a: Primeiro evento
            event_b: Segundo evento
            evidence: Evidências
            
        Returns:
            Análise de correlação vs causalidade
        """
        if not self.causal_reasoning_system:
            return {"error": "Sistema de raciocínio causal não disponível"}
        
        try:
            result = self.causal_reasoning_system.detect_correlation_vs_causation(
                event_a, event_b, evidence
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro na detecção de correlação vs causalidade: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_causal_analysis(self, relation_id: str = None) -> Dict[str, Any]:
        """
        Obtém análise de relações causais.
        
        Args:
            relation_id: ID da relação específica (opcional)
            
        Returns:
            Análise de relações causais
        """
        if not self.causal_reasoning_system:
            return {"error": "Sistema de raciocínio causal não disponível"}
        
        try:
            analysis = self.causal_reasoning_system.get_causal_analysis(relation_id)
            return analysis
            
        except Exception as e:
            error_msg = f"Erro na análise causal: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def add_abstract_concept(self, name: str, description: str, 
                           abstraction_level: AbstractionLevel,
                           properties: Dict[str, Any] = None,
                           examples: List[str] = None) -> Dict[str, Any]:
        """
        Adiciona um conceito abstrato.
        
        Args:
            name: Nome do conceito
            description: Descrição do conceito
            abstraction_level: Nível de abstração
            properties: Propriedades do conceito (opcional)
            examples: Exemplos do conceito (opcional)
            
        Returns:
            Resultado da adição do conceito
        """
        if not self.abstract_inference_engine:
            return {"error": "Sistema de inferência abstrata não disponível"}
        
        try:
            result = self.abstract_inference_engine.add_abstract_concept(
                name, description, abstraction_level, properties, examples
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro ao adicionar conceito abstrato: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def add_inference_rule(self, name: str, premise_patterns: List[str],
                          conclusion_pattern: str, inference_type: InferenceType,
                          abstraction_level: AbstractionLevel,
                          conditions: List[str] = None,
                          exceptions: List[str] = None) -> Dict[str, Any]:
        """
        Adiciona uma regra de inferência.
        
        Args:
            name: Nome da regra
            premise_patterns: Padrões de premissas
            conclusion_pattern: Padrão de conclusão
            inference_type: Tipo de inferência
            abstraction_level: Nível de abstração
            conditions: Condições (opcional)
            exceptions: Exceções (opcional)
            
        Returns:
            Resultado da adição da regra
        """
        if not self.abstract_inference_engine:
            return {"error": "Sistema de inferência abstrata não disponível"}
        
        try:
            result = self.abstract_inference_engine.add_inference_rule(
                name, premise_patterns, conclusion_pattern, inference_type,
                abstraction_level, conditions, exceptions
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro ao adicionar regra de inferência: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def perform_abstract_inference(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Realiza inferência abstrata.
        
        Args:
            query: Consulta para inferência
            
        Returns:
            Resultado da inferência abstrata
        """
        if not self.abstract_inference_engine:
            return {"error": "Sistema de inferência abstrata não disponível"}
        
        try:
            result = self.abstract_inference_engine.perform_abstract_inference(query)
            return result
            
        except Exception as e:
            error_msg = f"Erro na inferência abstrata: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def validate_abstract_inference(self, inference_id: str) -> Dict[str, Any]:
        """
        Valida uma inferência abstrata.
        
        Args:
            inference_id: ID da inferência
            
        Returns:
            Resultado da validação
        """
        if not self.abstract_inference_engine:
            return {"error": "Sistema de inferência abstrata não disponível"}
        
        try:
            result = self.abstract_inference_engine.validate_abstract_inference(inference_id)
            return result
            
        except Exception as e:
            error_msg = f"Erro na validação da inferência abstrata: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_abstract_inference_analysis(self, inference_id: str = None) -> Dict[str, Any]:
        """
        Obtém análise de inferências abstratas.
        
        Args:
            inference_id: ID da inferência específica (opcional)
            
        Returns:
            Análise de inferências abstratas
        """
        if not self.abstract_inference_engine:
            return {"error": "Sistema de inferência abstrata não disponível"}
        
        try:
            analysis = self.abstract_inference_engine.get_abstract_inference_analysis(inference_id)
            return analysis
            
        except Exception as e:
            error_msg = f"Erro na análise de inferência abstrata: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    # ===== MÉTODOS DE HIERARQUIAS DE CONCEITOS =====
    
    def add_hierarchical_concept(self, name: str, description: str, 
                                abstraction_level: str = "general",
                                complexity: str = "simple",
                                parent_concepts: List[str] = None,
                                properties: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Adiciona um conceito hierárquico.
        
        Args:
            name: Nome do conceito
            description: Descrição do conceito
            abstraction_level: Nível de abstração
            complexity: Nível de complexidade
            parent_concepts: Conceitos pais (opcional)
            properties: Propriedades do conceito (opcional)
            
        Returns:
            Resultado da adição do conceito
        """
        if not self.concept_hierarchy_manager:
            return {"error": "Sistema de hierarquias de conceitos não disponível"}
        
        try:
            # Converter strings para enums
            abstraction_enum = ConceptAbstractionLevel.GENERAL
            if abstraction_level == "concrete":
                abstraction_enum = ConceptAbstractionLevel.CONCRETE
            elif abstraction_level == "specific":
                abstraction_enum = ConceptAbstractionLevel.SPECIFIC
            elif abstraction_level == "abstract":
                abstraction_enum = ConceptAbstractionLevel.ABSTRACT
            elif abstraction_level == "universal":
                abstraction_enum = ConceptAbstractionLevel.UNIVERSAL
            elif abstraction_level == "metaphysical":
                abstraction_enum = ConceptAbstractionLevel.METAPHYSICAL
            
            complexity_enum = ConceptComplexity.SIMPLE
            if complexity == "compound":
                complexity_enum = ConceptComplexity.COMPOUND
            elif complexity == "complex":
                complexity_enum = ConceptComplexity.COMPLEX
            elif complexity == "meta":
                complexity_enum = ConceptComplexity.META
            
            result = self.concept_hierarchy_manager.add_concept(
                name, description, abstraction_enum, complexity_enum,
                parent_concepts, properties
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro ao adicionar conceito hierárquico: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def add_concept_relationship(self, source_concept: str, target_concept: str,
                                relationship_type: str, strength: float = 0.5,
                                confidence: float = 0.5) -> Dict[str, Any]:
        """
        Adiciona um relacionamento entre conceitos.
        
        Args:
            source_concept: Conceito origem
            target_concept: Conceito destino
            relationship_type: Tipo de relacionamento
            strength: Força do relacionamento
            confidence: Confiança no relacionamento
            
        Returns:
            Resultado da adição do relacionamento
        """
        if not self.concept_hierarchy_manager:
            return {"error": "Sistema de hierarquias de conceitos não disponível"}
        
        try:
            # Converter string para enum
            relationship_enum = ConceptRelationshipType.IS_A
            if relationship_type == "part_of":
                relationship_enum = ConceptRelationshipType.PART_OF
            elif relationship_type == "has_a":
                relationship_enum = ConceptRelationshipType.HAS_A
            elif relationship_type == "similar_to":
                relationship_enum = ConceptRelationshipType.SIMILAR_TO
            elif relationship_type == "opposite_of":
                relationship_enum = ConceptRelationshipType.OPPOSITE_OF
            elif relationship_type == "causes":
                relationship_enum = ConceptRelationshipType.CAUSES
            elif relationship_type == "requires":
                relationship_enum = ConceptRelationshipType.REQUIRES
            elif relationship_type == "inhibits":
                relationship_enum = ConceptRelationshipType.INHIBITS
            
            result = self.concept_hierarchy_manager.add_relationship(
                source_concept, target_concept, relationship_enum,
                strength, confidence
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro ao adicionar relacionamento: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def find_concept_relationships(self, concept_name: str, 
                                 relationship_type: str = None) -> Dict[str, Any]:
        """
        Encontra relacionamentos de um conceito.
        
        Args:
            concept_name: Nome do conceito
            relationship_type: Tipo de relacionamento (opcional)
            
        Returns:
            Relacionamentos encontrados
        """
        if not self.concept_hierarchy_manager:
            return {"error": "Sistema de hierarquias de conceitos não disponível"}
        
        try:
            result = self.concept_hierarchy_manager.find_relationships(
                concept_name, relationship_type
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro ao encontrar relacionamentos: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_concept_hierarchy(self, root_concept: str = None, 
                            max_depth: int = 3) -> Dict[str, Any]:
        """
        Obtém a hierarquia de conceitos.
        
        Args:
            root_concept: Conceito raiz (opcional)
            max_depth: Profundidade máxima
            
        Returns:
            Hierarquia de conceitos
        """
        if not self.concept_hierarchy_manager:
            return {"error": "Sistema de hierarquias de conceitos não disponível"}
        
        try:
            hierarchy = self.concept_hierarchy_manager.get_hierarchy(
                root_concept, max_depth
            )
            return hierarchy
            
        except Exception as e:
            error_msg = f"Erro ao obter hierarquia: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def analyze_concept_similarity(self, concept1: str, concept2: str) -> Dict[str, Any]:
        """
        Analisa similaridade entre dois conceitos.
        
        Args:
            concept1: Primeiro conceito
            concept2: Segundo conceito
            
        Returns:
            Análise de similaridade
        """
        if not self.concept_hierarchy_manager:
            return {"error": "Sistema de hierarquias de conceitos não disponível"}
        
        try:
            similarity = self.concept_hierarchy_manager.analyze_similarity(
                concept1, concept2
            )
            return similarity
            
        except Exception as e:
            error_msg = f"Erro na análise de similaridade: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_concept_hierarchy_analysis(self, concept_name: str = None) -> Dict[str, Any]:
        """
        Obtém análise da hierarquia de conceitos.
        
        Args:
            concept_name: Nome do conceito específico (opcional)
            
        Returns:
            Análise da hierarquia
        """
        if not self.concept_hierarchy_manager:
            return {"error": "Sistema de hierarquias de conceitos não disponível"}
        
        try:
            analysis = self.concept_hierarchy_manager.get_analysis(concept_name)
            return analysis
            
        except Exception as e:
            error_msg = f"Erro na análise da hierarquia: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    # ===== MÉTODOS DE APRENDIZADO DE RELAÇÕES =====
    
    def learn_relation(self, source_concept: str, target_concept: str,
                      relation_type: str, evidence: List[Dict[str, Any]] = None,
                      learning_mode: str = "unsupervised",
                      context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Aprende uma nova relação entre conceitos.
        
        Args:
            source_concept: Conceito origem
            target_concept: Conceito destino
            relation_type: Tipo da relação
            evidence: Evidências que suportam a relação
            learning_mode: Modo de aprendizado
            context: Contexto da relação
            
        Returns:
            Resultado do aprendizado
        """
        if not self.relation_learning_system:
            return {"error": "Sistema de aprendizado de relações não disponível"}
        
        try:
            # Converter strings para enums
            relation_enum = RelationType.STRUCTURAL
            if relation_type == "temporal":
                relation_enum = RelationType.TEMPORAL
            elif relation_type == "spatial":
                relation_enum = RelationType.SPATIAL
            elif relation_type == "causal":
                relation_enum = RelationType.CAUSAL
            elif relation_type == "functional":
                relation_enum = RelationType.FUNCTIONAL
            elif relation_type == "behavioral":
                relation_enum = RelationType.BEHAVIORAL
            elif relation_type == "similarity":
                relation_enum = RelationType.SIMILARITY
            elif relation_type == "opposition":
                relation_enum = RelationType.OPPOSITION
            elif relation_type == "dependency":
                relation_enum = RelationType.DEPENDENCY
            elif relation_type == "association":
                relation_enum = RelationType.ASSOCIATION
            
            learning_enum = LearningMode.UNSUPERVISED
            if learning_mode == "supervised":
                learning_enum = LearningMode.SUPERVISED
            elif learning_mode == "semi_supervised":
                learning_enum = LearningMode.SEMI_SUPERVISED
            elif learning_mode == "reinforcement":
                learning_enum = LearningMode.REINFORCEMENT
            elif learning_mode == "transfer":
                learning_enum = LearningMode.TRANSFER
            
            # Converter evidências para objetos RelationEvidence
            evidence_objects = []
            if evidence:
                for i, ev in enumerate(evidence):
                    evidence_type = EvidenceType.DIRECT_OBSERVATION
                    if ev.get("type") == "statistical_correlation":
                        evidence_type = EvidenceType.STATISTICAL_CORRELATION
                    elif ev.get("type") == "expert_knowledge":
                        evidence_type = EvidenceType.EXPERT_KNOWLEDGE
                    elif ev.get("type") == "logical_inference":
                        evidence_type = EvidenceType.LOGICAL_INFERENCE
                    elif ev.get("type") == "pattern_recognition":
                        evidence_type = EvidenceType.PATTERN_RECOGNITION
                    elif ev.get("type") == "analogical_reasoning":
                        evidence_type = EvidenceType.ANALOGICAL_REASONING
                    
                    import time
                    evidence_obj = RelationEvidence(
                        evidence_id=f"ev_{i}_{int(time.time())}",
                        evidence_type=evidence_type,
                        source=ev.get("source", "unknown"),
                        confidence=ev.get("confidence", 0.5),
                        context=ev.get("context", {}),
                        timestamp=time.time(),
                        metadata=ev.get("metadata", {})
                    )
                    evidence_objects.append(evidence_obj)
            
            result = self.relation_learning_system.learn_relation(
                source_concept, target_concept, relation_enum, evidence_objects,
                learning_enum, context
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro ao aprender relação: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def infer_relation(self, source_concept: str, target_concept: str,
                      relation_type: str = None,
                      context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Infere uma relação entre conceitos.
        
        Args:
            source_concept: Conceito origem
            target_concept: Conceito destino
            relation_type: Tipo de relação específico (opcional)
            context: Contexto da inferência
            
        Returns:
            Resultado da inferência
        """
        if not self.relation_learning_system:
            return {"error": "Sistema de aprendizado de relações não disponível"}
        
        try:
            # Converter string para enum se necessário
            relation_enum = None
            if relation_type:
                if relation_type == "temporal":
                    relation_enum = RelationType.TEMPORAL
                elif relation_type == "spatial":
                    relation_enum = RelationType.SPATIAL
                elif relation_type == "causal":
                    relation_enum = RelationType.CAUSAL
                elif relation_type == "functional":
                    relation_enum = RelationType.FUNCTIONAL
                elif relation_type == "behavioral":
                    relation_enum = RelationType.BEHAVIORAL
                elif relation_type == "similarity":
                    relation_enum = RelationType.SIMILARITY
                elif relation_type == "opposition":
                    relation_enum = RelationType.OPPOSITION
                elif relation_type == "dependency":
                    relation_enum = RelationType.DEPENDENCY
                elif relation_type == "association":
                    relation_enum = RelationType.ASSOCIATION
            
            result = self.relation_learning_system.infer_relation(
                source_concept, target_concept, relation_enum, context
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro na inferência de relação: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def validate_relation(self, relation_id: str, new_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Valida uma relação existente com novas evidências.
        
        Args:
            relation_id: ID da relação
            new_evidence: Novas evidências
            
        Returns:
            Resultado da validação
        """
        if not self.relation_learning_system:
            return {"error": "Sistema de aprendizado de relações não disponível"}
        
        try:
            # Converter evidências para objetos RelationEvidence
            evidence_objects = []
            for i, ev in enumerate(new_evidence):
                evidence_type = EvidenceType.DIRECT_OBSERVATION
                if ev.get("type") == "statistical_correlation":
                    evidence_type = EvidenceType.STATISTICAL_CORRELATION
                elif ev.get("type") == "expert_knowledge":
                    evidence_type = EvidenceType.EXPERT_KNOWLEDGE
                elif ev.get("type") == "logical_inference":
                    evidence_type = EvidenceType.LOGICAL_INFERENCE
                elif ev.get("type") == "pattern_recognition":
                    evidence_type = EvidenceType.PATTERN_RECOGNITION
                elif ev.get("type") == "analogical_reasoning":
                    evidence_type = EvidenceType.ANALOGICAL_REASONING
                
                import time
                evidence_obj = RelationEvidence(
                    evidence_id=f"ev_{i}_{int(time.time())}",
                    evidence_type=evidence_type,
                    source=ev.get("source", "unknown"),
                    confidence=ev.get("confidence", 0.5),
                    context=ev.get("context", {}),
                    timestamp=time.time(),
                    metadata=ev.get("metadata", {})
                )
                evidence_objects.append(evidence_obj)
            
            result = self.relation_learning_system.validate_relation(relation_id, evidence_objects)
            return result
            
        except Exception as e:
            error_msg = f"Erro na validação de relação: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def find_relation_patterns(self, concept: str = None, 
                             pattern_type: str = None) -> Dict[str, Any]:
        """
        Encontra padrões de relações.
        
        Args:
            concept: Conceito específico (opcional)
            pattern_type: Tipo de padrão (opcional)
            
        Returns:
            Padrões encontrados
        """
        if not self.relation_learning_system:
            return {"error": "Sistema de aprendizado de relações não disponível"}
        
        try:
            result = self.relation_learning_system.find_relation_patterns(concept, pattern_type)
            return result
            
        except Exception as e:
            error_msg = f"Erro ao encontrar padrões: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_relation_analysis(self, concept: str = None) -> Dict[str, Any]:
        """
        Obtém análise das relações aprendidas.
        
        Args:
            concept: Conceito específico (opcional)
            
        Returns:
            Análise das relações
        """
        if not self.relation_learning_system:
            return {"error": "Sistema de aprendizado de relações não disponível"}
        
        try:
            result = self.relation_learning_system.get_relation_analysis(concept)
            return result
            
        except Exception as e:
            error_msg = f"Erro na análise de relações: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    # ===== MÉTODOS DE TRANSFERÊNCIA DE CONHECIMENTO =====
    
    def transfer_knowledge(self, source_domain: str, target_domain: str,
                          source_knowledge: Dict[str, Any],
                          transfer_type: str = "concept_to_concept",
                          strategy: str = "adaptive",
                          knowledge_type: str = "conceptual",
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
        if not self.knowledge_transfer_system:
            return {"error": "Sistema de transferência de conhecimento não disponível"}
        
        try:
            # Converter strings para enums
            transfer_enum = TransferType.CONCEPT_TO_CONCEPT
            if transfer_type == "domain_to_domain":
                transfer_enum = TransferType.DOMAIN_TO_DOMAIN
            elif transfer_type == "species_to_species":
                transfer_enum = TransferType.SPECIES_TO_SPECIES
            elif transfer_type == "context_to_context":
                transfer_enum = TransferType.CONTEXT_TO_CONTEXT
            elif transfer_type == "skill_to_skill":
                transfer_enum = TransferType.SKILL_TO_SKILL
            elif transfer_type == "pattern_to_pattern":
                transfer_enum = TransferType.PATTERN_TO_PATTERN
            elif transfer_type == "feature_to_feature":
                transfer_enum = TransferType.FEATURE_TO_FEATURE
            elif transfer_type == "strategy_to_strategy":
                transfer_enum = TransferType.STRATEGY_TO_STRATEGY
            
            strategy_enum = TransferStrategy.ADAPTIVE
            if strategy == "direct":
                strategy_enum = TransferStrategy.DIRECT
            elif strategy == "selective":
                strategy_enum = TransferStrategy.SELECTIVE
            elif strategy == "hierarchical":
                strategy_enum = TransferStrategy.HIERARCHICAL
            elif strategy == "contextual":
                strategy_enum = TransferStrategy.CONTEXTUAL
            elif strategy == "analogical":
                strategy_enum = TransferStrategy.ANALOGICAL
            elif strategy == "meta_learning":
                strategy_enum = TransferStrategy.META_LEARNING
            elif strategy == "few_shot":
                strategy_enum = TransferStrategy.FEW_SHOT
            
            knowledge_enum = KnowledgeType.CONCEPTUAL
            if knowledge_type == "factual":
                knowledge_enum = KnowledgeType.FACTUAL
            elif knowledge_type == "procedural":
                knowledge_enum = KnowledgeType.PROCEDURAL
            elif knowledge_type == "metacognitive":
                knowledge_enum = KnowledgeType.METACOGNITIVE
            elif knowledge_type == "experiential":
                knowledge_enum = KnowledgeType.EXPERIENTIAL
            elif knowledge_type == "pattern_based":
                knowledge_enum = KnowledgeType.PATTERN_BASED
            
            result = self.knowledge_transfer_system.transfer_knowledge(
                source_domain, target_domain, source_knowledge,
                transfer_enum, strategy_enum, knowledge_enum, context
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro na transferência de conhecimento: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def find_similar_transfers(self, source_domain: str, target_domain: str,
                             transfer_type: str = None) -> Dict[str, Any]:
        """
        Encontra transferências similares.
        
        Args:
            source_domain: Domínio fonte
            target_domain: Domínio alvo
            transfer_type: Tipo de transferência (opcional)
            
        Returns:
            Transferências similares encontradas
        """
        if not self.knowledge_transfer_system:
            return {"error": "Sistema de transferência de conhecimento não disponível"}
        
        try:
            # Converter string para enum se necessário
            transfer_enum = None
            if transfer_type:
                if transfer_type == "domain_to_domain":
                    transfer_enum = TransferType.DOMAIN_TO_DOMAIN
                elif transfer_type == "concept_to_concept":
                    transfer_enum = TransferType.CONCEPT_TO_CONCEPT
                elif transfer_type == "species_to_species":
                    transfer_enum = TransferType.SPECIES_TO_SPECIES
                elif transfer_type == "context_to_context":
                    transfer_enum = TransferType.CONTEXT_TO_CONTEXT
                elif transfer_type == "skill_to_skill":
                    transfer_enum = TransferType.SKILL_TO_SKILL
                elif transfer_type == "pattern_to_pattern":
                    transfer_enum = TransferType.PATTERN_TO_PATTERN
                elif transfer_type == "feature_to_feature":
                    transfer_enum = TransferType.FEATURE_TO_FEATURE
                elif transfer_type == "strategy_to_strategy":
                    transfer_enum = TransferType.STRATEGY_TO_STRATEGY
            
            result = self.knowledge_transfer_system.find_similar_transfers(
                source_domain, target_domain, transfer_enum
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro ao encontrar transferências similares: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
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
        if not self.knowledge_transfer_system:
            return {"error": "Sistema de transferência de conhecimento não disponível"}
        
        try:
            result = self.knowledge_transfer_system.apply_transfer_rule(
                rule_id, source_data, target_context
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro ao aplicar regra de transferência: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
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
        if not self.knowledge_transfer_system:
            return {"error": "Sistema de transferência de conhecimento não disponível"}
        
        try:
            result = self.knowledge_transfer_system.learn_transfer_rule(
                source_pattern, target_pattern, examples, conditions
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro ao aprender regra de transferência: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_transfer_analysis(self, domain: str = None) -> Dict[str, Any]:
        """
        Obtém análise das transferências de conhecimento.
        
        Args:
            domain: Domínio específico (opcional)
            
        Returns:
            Análise das transferências
        """
        if not self.knowledge_transfer_system:
            return {"error": "Sistema de transferência de conhecimento não disponível"}
        
        try:
            result = self.knowledge_transfer_system.get_transfer_analysis(domain)
            return result
            
        except Exception as e:
            error_msg = f"Erro na análise de transferências: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    # ===== MÉTODOS DE META-COGNIÇÃO =====
    
    def perform_self_reflection(self, process_name: str, process_data: Dict[str, Any],
                               reflection_type: str = "performance_reflection") -> Dict[str, Any]:
        """
        Realiza auto-reflexão sobre um processo.
        
        Args:
            process_name: Nome do processo
            process_data: Dados do processo
            reflection_type: Tipo de reflexão
            
        Returns:
            Resultado da reflexão
        """
        if not self.metacognitive_system:
            return {"error": "Sistema de meta-cognição não disponível"}
        
        try:
            # Converter string para enum
            reflection_enum = ReflectionType.PERFORMANCE_REFLECTION
            if reflection_type == "strategy_reflection":
                reflection_enum = ReflectionType.STRATEGY_REFLECTION
            elif reflection_type == "learning_reflection":
                reflection_enum = ReflectionType.LEARNING_REFLECTION
            elif reflection_type == "error_reflection":
                reflection_enum = ReflectionType.ERROR_REFLECTION
            elif reflection_type == "success_reflection":
                reflection_enum = ReflectionType.SUCCESS_REFLECTION
            elif reflection_type == "metacognitive_reflection":
                reflection_enum = ReflectionType.METACOGNITIVE_REFLECTION
            
            result = self.metacognitive_system.perform_self_reflection(
                process_name, process_data, reflection_enum
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro na auto-reflexão: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def adapt_strategy(self, original_strategy: str, performance_data: Dict[str, Any],
                      adaptation_type: str = "parameter_adjustment") -> Dict[str, Any]:
        """
        Adapta uma estratégia baseada na performance.
        
        Args:
            original_strategy: Estratégia original
            performance_data: Dados de performance
            adaptation_type: Tipo de adaptação
            
        Returns:
            Resultado da adaptação
        """
        if not self.metacognitive_system:
            return {"error": "Sistema de meta-cognição não disponível"}
        
        try:
            # Converter string para enum
            adaptation_enum = AdaptationStrategy.PARAMETER_ADJUSTMENT
            if adaptation_type == "strategy_switching":
                adaptation_enum = AdaptationStrategy.STRATEGY_SWITCHING
            elif adaptation_type == "learning_rate_adaptation":
                adaptation_enum = AdaptationStrategy.LEARNING_RATE_ADAPTATION
            elif adaptation_type == "architecture_modification":
                adaptation_enum = AdaptationStrategy.ARCHITECTURE_MODIFICATION
            elif adaptation_type == "knowledge_restructuring":
                adaptation_enum = AdaptationStrategy.KNOWLEDGE_RESTRUCTURING
            elif adaptation_type == "metacognitive_adaptation":
                adaptation_enum = AdaptationStrategy.METACOGNITIVE_ADAPTATION
            
            result = self.metacognitive_system.adapt_strategy(
                original_strategy, performance_data, adaptation_enum
            )
            return result
            
        except Exception as e:
            error_msg = f"Erro na adaptação de estratégia: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def learn_to_learn(self, learning_context: str, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aprende como aprender melhor.
        
        Args:
            learning_context: Contexto de aprendizado
            learning_data: Dados de aprendizado
            
        Returns:
            Resultado do aprendizado
        """
        if not self.metacognitive_system:
            return {"error": "Sistema de meta-cognição não disponível"}
        
        try:
            result = self.metacognitive_system.learn_to_learn(learning_context, learning_data)
            return result
            
        except Exception as e:
            error_msg = f"Erro no aprendizado de como aprender: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_metacognitive_analysis(self) -> Dict[str, Any]:
        """
        Obtém análise meta-cognitiva do sistema.
        
        Returns:
            Análise meta-cognitiva
        """
        if not self.metacognitive_system:
            return {"error": "Sistema de meta-cognição não disponível"}
        
        try:
            result = self.metacognitive_system.get_metacognitive_analysis()
            return result
            
        except Exception as e:
            error_msg = f"Erro na análise meta-cognitiva: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _save_learned_patterns(self):
        """Salva padrões aprendidos para uso pelo sistema de auto-modificação."""
        try:
            # Converter padrões para formato compatível
            patterns_to_save = {
                "color_combinations": {},
                "known_species": {},
                "characteristic_patterns": {}
            }
            
            # Converter cores aprendidas
            if hasattr(self, 'learned_patterns') and 'color_combinations' in self.learned_patterns:
                for color, data in self.learned_patterns['color_combinations'].items():
                    if isinstance(data, dict):
                        patterns_to_save["color_combinations"][color] = {
                            "confidence": data.get("confidence", 0.5),
                            "count": data.get("count", 1),
                            "last_seen": data.get("last_seen", "2025-09-23T00:00:00")
                        }
            
            # Converter espécies aprendidas
            if hasattr(self, 'learned_patterns') and 'known_species' in self.learned_patterns:
                for species in self.learned_patterns['known_species']:
                    patterns_to_save["known_species"][species] = {
                        "confidence": 0.8,  # Confiança padrão
                        "count": 1,
                        "last_seen": "2025-09-23T00:00:00"
                    }
            
            # Converter características aprendidas
            if hasattr(self, 'learned_patterns') and 'characteristic_patterns' in self.learned_patterns:
                for char, data in self.learned_patterns['characteristic_patterns'].items():
                    if isinstance(data, dict):
                        patterns_to_save["characteristic_patterns"][char] = {
                            "confidence": data.get("confidence", 0.7),
                            "count": data.get("count", 1),
                            "last_seen": data.get("last_seen", "2025-09-23T00:00:00")
                        }
            
            # Salvar padrões
            os.makedirs("data", exist_ok=True)
            with open("data/learned_patterns.json", "w", encoding="utf-8") as f:
                json.dump(patterns_to_save, f, indent=2, ensure_ascii=False)
            
            logger.info("Padrões aprendidos salvos para auto-modificação")
            
        except Exception as e:
            logger.warning(f"Erro ao salvar padrões aprendidos: {e}")
    
    def _calculate_bird_score_enhanced(self, characteristics: Dict) -> float:
        """Calcula score de pássaro melhorado com votação de modelos"""
        score = 0.0
        total_weight = 0.0
        
        # Peso 1: Consenso de modelos (novo)
        model_consensus = characteristics.get('model_consensus', 0)
        final_decision = characteristics.get('final_decision', 'unknown')
        
        if final_decision == 'bird_high_confidence':
            score += 3.0  # Peso alto se modelos concordam que é pássaro
        elif final_decision == 'bird_medium_confidence':
            score += 2.0  # Peso médio
        elif final_decision == 'uncertain':
            score += 0.5  # Peso baixo se incerto
        total_weight += 3.0
        
        # Peso 2: Detecção multi-biblioteca (novo)
        detection_votes = characteristics.get('detection_votes', {})
        if detection_votes:
            avg_detection_confidence = sum(detection_votes.values()) / len(detection_votes)
            score += avg_detection_confidence * 2.0
        total_weight += 2.0
        
        # Peso 3: Características visuais básicas (mantido)
        visual_chars = [
            characteristics.get('has_wings', False),
            characteristics.get('has_beak', False),
            characteristics.get('has_feathers', False),
            characteristics.get('has_eyes', False)
        ]
        
        visual_count = sum(visual_chars)
        if visual_count >= 3:
            score += 2.0
        elif visual_count >= 2:
            score += 1.5
        elif visual_count >= 1:
            score += 1.0
        total_weight += 2.0
        
        # Peso 4: Características ultra-avançadas (mantido)
        ultra_chars = [
            characteristics.get('has_wings_ultra', False),
            characteristics.get('has_beak_ultra', False),
            characteristics.get('has_feathers_ultra', False),
            characteristics.get('has_body_ultra', False)
        ]
        
        ultra_count = sum(ultra_chars)
        if ultra_count >= 3:
            score += 2.5
        elif ultra_count >= 2:
            score += 2.0
        elif ultra_count >= 1:
            score += 1.5
        total_weight += 2.5
        
        # Peso 5: Score de forma de pássaro (mantido)
        bird_shape_score = characteristics.get('bird_shape_score_ultra', 0)
        score += bird_shape_score * 1.0
        total_weight += 1.0
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_final_hybrid_confidence(self, characteristics: Dict, mammal_score: float, bird_score: float) -> float:
        """Calcula confiança híbrida final considerando votação de modelos"""
        confidence_factors = []
        
        # Fator 1: Consenso de modelos
        model_consensus = characteristics.get('model_consensus', 0)
        confidence_factors.append(model_consensus)
        
        # Fator 2: Diferença entre scores de pássaro e mamífero
        score_difference = bird_score - mammal_score
        if score_difference > 0.3:
            confidence_factors.append(0.8)  # Alta confiança se pássaro claramente maior
        elif score_difference > 0.1:
            confidence_factors.append(0.6)  # Confiança média
        elif score_difference < -0.3:
            confidence_factors.append(0.1)  # Baixa confiança (provavelmente mamífero)
        else:
            confidence_factors.append(0.4)  # Confiança baixa (caso duvidoso)
        
        # Fator 3: Detecção multi-biblioteca
        detection_votes = characteristics.get('detection_votes', {})
        if detection_votes:
            avg_detection_confidence = sum(detection_votes.values()) / len(detection_votes)
            confidence_factors.append(avg_detection_confidence)
        else:
            confidence_factors.append(0.3)  # Baixa confiança sem detecção
        
        # Fator 4: Características ultra-avançadas
        ultra_count = sum([
            characteristics.get('has_wings_ultra', False),
            characteristics.get('has_beak_ultra', False),
            characteristics.get('has_feathers_ultra', False),
            characteristics.get('has_body_ultra', False)
        ])
        
        if ultra_count >= 3:
            confidence_factors.append(0.9)
        elif ultra_count >= 2:
            confidence_factors.append(0.7)
        elif ultra_count >= 1:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.2)
        
        # Calcular confiança média ponderada
        if confidence_factors:
            base_confidence = sum(confidence_factors) / len(confidence_factors)
        else:
            base_confidence = 0.0
        
        # Aplicar regras geradas automaticamente
        try:
            generated_rules_boost = self._apply_generated_rules(characteristics, base_confidence)
            if generated_rules_boost > 0:
                logger.info(f"Boost aplicado por regras geradas: {generated_rules_boost:.3f}")
                base_confidence += generated_rules_boost
                # Limitar confiança máxima a 1.0
                base_confidence = min(base_confidence, 1.0)
        except Exception as e:
            logger.warning(f"Erro ao aplicar regras geradas: {e}")
        
        return base_confidence
    
    def _calculate_ultra_hybrid_confidence(self, characteristics: Dict, yolo_detections: int, mammal_score: float, bird_score: float) -> float:
        """Calcula confiança híbrida ultra-rigorosa"""
        confidence_factors = []
        
        # Fator 1: Detecção YOLO
        if characteristics.get('yolo_detection', False):
            confidence_factors.append(0.9)  # Alta confiança se YOLO detectou
        
        # Fator 2: Diferença entre scores de pássaro e mamífero
        score_difference = bird_score - mammal_score
        if score_difference > 0.3:
            confidence_factors.append(0.8)  # Alta confiança se pássaro claramente maior
        elif score_difference > 0.1:
            confidence_factors.append(0.6)  # Confiança média
        elif score_difference < -0.3:
            confidence_factors.append(0.1)  # Baixa confiança (provavelmente mamífero)
        else:
            confidence_factors.append(0.4)  # Confiança baixa (caso duvidoso)
        
        # Fator 3: Características ultra-avançadas
        ultra_count = sum([
            characteristics.get('has_wings_ultra', False),
            characteristics.get('has_beak_ultra', False),
            characteristics.get('has_feathers_ultra', False),
            characteristics.get('has_body_ultra', False)
        ])
        
        if ultra_count >= 3:
            confidence_factors.append(0.9)
        elif ultra_count >= 2:
            confidence_factors.append(0.7)
        elif ultra_count >= 1:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.2)
        
        # Fator 4: Análise de textura ultra-híbrida
        if characteristics.get('has_feathers_ultra', False) and not characteristics.get('has_fur_ultra', False):
            confidence_factors.append(0.8)
        elif characteristics.get('has_fur_ultra', False) and not characteristics.get('has_feathers_ultra', False):
            confidence_factors.append(0.1)
        else:
            confidence_factors.append(0.4)
        
        # Calcular confiança média ponderada
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.0
    
    def _determine_species(self, visual_analysis: Dict, characteristics: Dict) -> str:
        """Determina espécie baseada em características visuais"""
        # Lógica simples baseada em cores e características
        dominant_color = visual_analysis.get('dominant_color', 'unknown')
        
        # Mapeamento básico de cores para espécies comuns
        color_species_map = {
            'brown': 'Pássaro marrom (possível rolinha, pardal)',
            'black': 'Pássaro preto (possível corvo, melro)',
            'white': 'Pássaro branco (possível pomba, gaivota)',
            'red': 'Pássaro vermelho (possível cardeal, beija-flor)',
            'blue': 'Pássaro azul (possível azulão, sabiá)',
            'yellow': 'Pássaro amarelo (possível canário, pintassilgo)',
            'green': 'Pássaro verde (possível papagaio, periquito)'
        }
        
        return color_species_map.get(dominant_color, 'Pássaro de espécie desconhecida')
    
    def _detect_learning_candidates(self, visual_analysis: Dict, characteristics: Dict, reasoning: Dict) -> List[LearningCandidate]:
        """Detecta candidatos para aprendizado contínuo"""
        candidates = []
        
        # Candidato 1: Análise visual interessante
        if visual_analysis.get('bird_like_features', 0) > 0.3:
            candidates.append(LearningCandidate(
                type=LearningCandidateType.VISUAL_ANALYSIS,
                confidence=visual_analysis['bird_like_features'],
                characteristics=visual_analysis,
                reasoning="Características visuais interessantes detectadas",
                image_path="",  # Será preenchido pelo chamador
                metadata={'analysis_type': 'visual'}
            ))
        
        # Candidato 2: Espécie desconhecida
        if reasoning.get('is_bird', False) and reasoning.get('species', '').startswith('Pássaro de espécie desconhecida'):
            candidates.append(LearningCandidate(
                type=LearningCandidateType.SPECIES_UNKNOWN,
                confidence=reasoning.get('confidence', 0),
                characteristics=characteristics,
                reasoning="Pássaro detectado mas espécie desconhecida",
                image_path="",
                metadata={'analysis_type': 'species_unknown'}
            ))
        
        # Candidato 3: Características para aprendizado
        if len(reasoning.get('characteristics_found', [])) > 0:
            candidates.append(LearningCandidate(
                type=LearningCandidateType.CHARACTERISTIC_LEARNING,
                confidence=reasoning.get('confidence', 0),
                characteristics={'found': reasoning['characteristics_found']},
                reasoning="Características específicas encontradas para aprendizado",
                image_path="",
                metadata={'analysis_type': 'characteristics'}
            ))
        
        return candidates
    
    def _recommend_action(self, candidates: List[LearningCandidate], reasoning: Dict) -> str:
        """Recomenda ação baseada na análise"""
        if not candidates:
            return "PROCESSAR_NORMALMENTE"
        
        # Priorizar candidatos com alta confiança
        high_confidence_candidates = [c for c in candidates if c.confidence > 0.7]
        
        if high_confidence_candidates:
            return "ANALISAR_MANUALMENTE"
        
        # Candidatos com confiança moderada
        medium_confidence_candidates = [c for c in candidates if c.confidence > 0.5]
        
        if medium_confidence_candidates:
            return "PROCESSAR_COM_CUIDADO"
        
        # Candidatos com baixa confiança
        return "REGISTRAR_PARA_ANALISE_FUTURA"
    
    def learn_from_feedback(self, image_path: str, human_feedback: Dict[str, Any]):
        """Aprende com feedback humano (como uma criança)"""
        try:
            # Extrair características da imagem
            visual_analysis = self._analyze_visual_characteristics(image_path)
            characteristics = self._detect_fundamental_characteristics(image_path)
            
            # Atualizar conhecimento baseado no feedback
            if human_feedback.get('is_bird', False):
                species = human_feedback.get('species', 'unknown')
                self.learned_patterns['known_species'].add(species)
                
                # Aprender padrões de características
                for char_name, char_value in characteristics.items():
                    if char_value and char_name != 'error':
                        if char_name not in self.learned_patterns['characteristic_patterns']:
                            self.learned_patterns['characteristic_patterns'][char_name] = []
                        
                        self.learned_patterns['characteristic_patterns'][char_name].append({
                            'visual_features': visual_analysis,
                            'species': species,
                            'confidence': human_feedback.get('confidence', 0.8)
                        })
                
                logger.info(f"[IA] Aprendizado: {species} adicionado ao conhecimento")
            
        except Exception as e:
            logger.error(f"[ERRO] Erro no aprendizado: {e}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas de aprendizado"""
        return {
            'known_species_count': len(self.learned_patterns['known_species']),
            'known_species': list(self.learned_patterns['known_species']),
            'characteristic_patterns': len(self.learned_patterns['characteristic_patterns']),
            'total_learning_events': sum(
                len(patterns) for patterns in self.learned_patterns['characteristic_patterns'].values()
            )
        }
    
    # ============================================================================
    # MÉTODOS BÁSICOS DE APRENDIZADO - FASE 1.2
    # ============================================================================
    
    def add_knowledge(self, knowledge_type: str, knowledge_data: Dict[str, Any]) -> bool:
        """
        Adiciona conhecimento ao sistema.
        
        Args:
            knowledge_type: Tipo de conhecimento (species, pattern, relation, etc.)
            knowledge_data: Dados do conhecimento
            
        Returns:
            bool: True se adicionado com sucesso
        """
        try:
            if knowledge_type == "species":
                species_name = knowledge_data.get("name", "unknown")
                characteristics = knowledge_data.get("characteristics", [])
                
                # Adicionar à memória episódica
                if hasattr(self, 'episodic_memory') and self.episodic_memory:
                    memory_content = {
                        "species": species_name,
                        "characteristics": characteristics,
                        "confidence": knowledge_data.get("confidence", 0.8)
                    }
                    memory_context = {
                        "learning_type": "species_knowledge",
                        "timestamp": time.time()
                    }
                    self.episodic_memory.store_episodic_memory(
                        memory_type="species_knowledge",
                        content=memory_content,
                        context=memory_context
                    )
                
                # Adicionar aos padrões aprendidos
                self.learned_patterns['known_species'].add(species_name)
                
                logger.info(f"[APRENDIZADO] Espécie adicionada: {species_name}")
                return True
                
            elif knowledge_type == "pattern":
                pattern_name = knowledge_data.get("name", "unknown")
                pattern_data = knowledge_data.get("data", {})
                
                # Adicionar aos padrões de características
                if pattern_name not in self.learned_patterns['characteristic_patterns']:
                    self.learned_patterns['characteristic_patterns'][pattern_name] = []
                
                self.learned_patterns['characteristic_patterns'][pattern_name].append(pattern_data)
                
                logger.info(f"[APRENDIZADO] Padrão adicionado: {pattern_name}")
                return True
                
            elif knowledge_type == "relation":
                # Adicionar relação causal
                if hasattr(self, 'causal_reasoning') and self.causal_reasoning:
                    source = knowledge_data.get("source", "")
                    target = knowledge_data.get("target", "")
                    relation_type = knowledge_data.get("type", "causes")
                    
                    self.causal_reasoning.add_relation(source, target, relation_type)
                    logger.info(f"[APRENDIZADO] Relação adicionada: {source} -> {target}")
                    return True
                
            return False
            
        except Exception as e:
            logger.error(f"[ERRO] Erro ao adicionar conhecimento: {e}")
            return False
    
    def learn_from_example(self, example_data: Dict[str, Any], correct_label: str = None) -> Dict[str, Any]:
        """
        Aprende com um exemplo específico.
        
        Args:
            example_data: Dados do exemplo (imagem, características, etc.)
            correct_label: Rótulo correto (se fornecido)
            
        Returns:
            Dict com resultado do aprendizado
        """
        try:
            result = {
                "success": False,
                "learned_concepts": [],
                "updated_patterns": [],
                "confidence": 0.0
            }
            
            # Extrair características do exemplo
            if "image_path" in example_data:
                visual_analysis = self._analyze_visual_characteristics(example_data["image_path"])
                characteristics = self._detect_fundamental_characteristics(example_data["image_path"])
            else:
                visual_analysis = example_data.get("visual_analysis", {})
                characteristics = example_data.get("characteristics", {})
            
            # Aprender conceitos abstratos
            if hasattr(self, 'abstract_inference') and self.abstract_inference:
                concept_result = self.abstract_inference.add_abstract_concept(
                    name=correct_label or "unknown_concept",
                    description=f"Conceito aprendido de {correct_label or 'exemplo'}",
                    abstraction_level="concrete",
                    properties=characteristics,
                    examples=[str(characteristics)]
                )
                if concept_result.get("success", False):
                    result["learned_concepts"].append("abstract_concept")
            
            # Aprender padrões universais
            if hasattr(self, 'universal_features') and self.universal_features:
                pattern_result = self.universal_features.learn_pattern(
                    pattern_data=characteristics,
                    species=correct_label or "unknown",
                    pattern_type="morphological"
                )
                if pattern_result:
                    result["learned_concepts"].append("universal_pattern")
            
            # Aprender few-shot
            if hasattr(self, 'few_shot_learning') and self.few_shot_learning:
                few_shot_result = self.few_shot_learning.learn_concept_few_shot(
                    concept_name=correct_label or "unknown_concept",
                    concept_type="bird_species",
                    examples=[example_data]
                )
                if few_shot_result.success:
                    result["learned_concepts"].append("few_shot_concept")
            
            # Atualizar padrões aprendidos
            for char_name, char_value in characteristics.items():
                if char_value and char_name != 'error':
                    if char_name not in self.learned_patterns['characteristic_patterns']:
                        self.learned_patterns['characteristic_patterns'][char_name] = []
                    
                    pattern_data = {
                        'visual_features': visual_analysis,
                        'species': correct_label or "unknown",
                        'confidence': example_data.get("confidence", 0.8),
                        'timestamp': time.time()
                    }
                    
                    self.learned_patterns['characteristic_patterns'][char_name].append(pattern_data)
                    result["updated_patterns"].append(char_name)
            
            result["success"] = len(result["learned_concepts"]) > 0
            result["confidence"] = example_data.get("confidence", 0.8)
            
            logger.info(f"[APRENDIZADO] Exemplo processado: {len(result['learned_concepts'])} conceitos aprendidos")
            return result
            
        except Exception as e:
            logger.error(f"[ERRO] Erro no aprendizado com exemplo: {e}")
            return {"success": False, "error": str(e)}
    
    def update_knowledge(self, knowledge_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Atualiza conhecimento existente.
        
        Args:
            knowledge_id: ID do conhecimento a ser atualizado
            update_data: Novos dados
            
        Returns:
            bool: True se atualizado com sucesso
        """
        try:
            # Atualizar memória episódica
            if hasattr(self, 'episodic_memory') and self.episodic_memory:
                # Buscar memória existente através da análise
                memory_analysis = self.episodic_memory.get_memory_analysis()
                if memory_analysis and "memories" in memory_analysis:
                    for memory_id, memory_data in memory_analysis["memories"].items():
                        if memory_data.get("content", {}).get("species") == knowledge_id:
                            # Atualizar memória (simplificado)
                            logger.info(f"[ATUALIZAÇÃO] Conhecimento encontrado na memória: {knowledge_id}")
                            return True
            
            # Atualizar padrões aprendidos
            if knowledge_id in self.learned_patterns['characteristic_patterns']:
                patterns = self.learned_patterns['characteristic_patterns'][knowledge_id]
                for pattern in patterns:
                    pattern.update(update_data)
                    pattern["last_updated"] = time.time()
                logger.info(f"[ATUALIZAÇÃO] Padrão atualizado: {knowledge_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"[ERRO] Erro ao atualizar conhecimento: {e}")
            return False
    
    def save_knowledge(self, file_path: str = None) -> bool:
        """
        Salva todo o conhecimento aprendido.
        
        Args:
            file_path: Caminho do arquivo (opcional)
            
        Returns:
            bool: True se salvo com sucesso
        """
        try:
            if file_path is None:
                file_path = "data/learned_knowledge.json"
            
            knowledge_data = {
                "learned_patterns": self.learned_patterns,
                "timestamp": time.time(),
                "version": "1.0"
            }
            
            # Adicionar dados dos sistemas cognitivos
            if hasattr(self, 'episodic_memory') and self.episodic_memory:
                knowledge_data["episodic_memory"] = self.episodic_memory.get_memory_analysis()
            
            if hasattr(self, 'abstract_inference') and self.abstract_inference:
                knowledge_data["abstract_inference"] = self.abstract_inference.get_stats()
            
            if hasattr(self, 'universal_features') and self.universal_features:
                knowledge_data["universal_features"] = self.universal_features.get_stats()
            
            # Salvar arquivo
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(knowledge_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[SALVAMENTO] Conhecimento salvo em: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"[ERRO] Erro ao salvar conhecimento: {e}")
            return False
    
    def load_knowledge(self, file_path: str = None) -> bool:
        """
        Carrega conhecimento salvo.
        
        Args:
            file_path: Caminho do arquivo (opcional)
            
        Returns:
            bool: True se carregado com sucesso
        """
        try:
            if file_path is None:
                file_path = "data/learned_knowledge.json"
            
            if not os.path.exists(file_path):
                logger.warning(f"[CARREGAMENTO] Arquivo não encontrado: {file_path}")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
            
            # Carregar padrões aprendidos
            if "learned_patterns" in knowledge_data:
                self.learned_patterns.update(knowledge_data["learned_patterns"])
            
            logger.info(f"[CARREGAMENTO] Conhecimento carregado de: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"[ERRO] Erro ao carregar conhecimento: {e}")
            return False
    
    # ============================================================================
    # MÉTODOS DE CACHE DE PERFORMANCE - FASE 1.3.2
    # ============================================================================
    
    def _generate_cache_key(self, image_input) -> str:
        """Gera chave única para cache baseada na imagem."""
        try:
            import hashlib
            
            if isinstance(image_input, str):
                # Para arquivos, usar hash do caminho + tamanho + data de modificação
                import os
                if os.path.exists(image_input):
                    stat = os.stat(image_input)
                    key_data = f"{image_input}_{stat.st_size}_{stat.st_mtime}"
                else:
                    key_data = image_input
            elif isinstance(image_input, np.ndarray):
                # Para arrays numpy, usar hash dos dados
                key_data = f"array_{image_input.shape}_{image_input.tobytes()[:1000]}"
            else:
                key_data = str(image_input)
            
            return hashlib.md5(key_data.encode()).hexdigest()[:16]
            
        except Exception as e:
            logger.error(f"[ERRO] Erro ao gerar chave de cache: {e}")
            return str(hash(str(image_input)))[:16]
    
    def _get_cached_analysis(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Recupera análise do cache."""
        try:
            if cache_key in self.analysis_cache:
                self.cache_hit_count += 1
                self.performance_stats['cache_hits'] += 1
                logger.debug(f"[CACHE] Hit: {cache_key}")
                return self.analysis_cache[cache_key]
            else:
                self.cache_miss_count += 1
                self.performance_stats['cache_misses'] += 1
                logger.debug(f"[CACHE] Miss: {cache_key}")
                return None
                
        except Exception as e:
            logger.error(f"[ERRO] Erro ao recuperar cache: {e}")
            return None
    
    def _store_cached_analysis(self, cache_key: str, analysis_result: Dict[str, Any]):
        """Armazena análise no cache."""
        try:
            # Limitar tamanho do cache
            if len(self.analysis_cache) >= self.cache_max_size:
                # Remover entrada mais antiga (FIFO)
                oldest_key = next(iter(self.analysis_cache))
                del self.analysis_cache[oldest_key]
            
            # Armazenar resultado
            self.analysis_cache[cache_key] = {
                'result': analysis_result,
                'timestamp': time.time(),
                'access_count': 1
            }
            
            logger.debug(f"[CACHE] Stored: {cache_key}")
            
        except Exception as e:
            logger.error(f"[ERRO] Erro ao armazenar cache: {e}")
    
    def _get_cached_visual_features(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Recupera características visuais do cache."""
        try:
            if image_path in self.visual_features_cache:
                logger.debug(f"[CACHE] Visual features hit: {image_path}")
                return self.visual_features_cache[image_path]
            return None
            
        except Exception as e:
            logger.error(f"[ERRO] Erro ao recuperar cache de características: {e}")
            return None
    
    def _store_cached_visual_features(self, image_path: str, features: Dict[str, Any]):
        """Armazena características visuais no cache."""
        try:
            # Limitar tamanho do cache
            if len(self.visual_features_cache) >= self.features_cache_max_size:
                # Remover entrada mais antiga
                oldest_key = next(iter(self.visual_features_cache))
                del self.visual_features_cache[oldest_key]
            
            self.visual_features_cache[image_path] = {
                'features': features,
                'timestamp': time.time()
            }
            
            logger.debug(f"[CACHE] Visual features stored: {image_path}")
            
        except Exception as e:
            logger.error(f"[ERRO] Erro ao armazenar cache de características: {e}")
    
    def _get_cached_yolo_result(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Recupera resultado YOLO do cache."""
        try:
            if image_path in self.yolo_results_cache:
                logger.debug(f"[CACHE] YOLO hit: {image_path}")
                return self.yolo_results_cache[image_path]
            return None
            
        except Exception as e:
            logger.error(f"[ERRO] Erro ao recuperar cache YOLO: {e}")
            return None
    
    def _store_cached_yolo_result(self, image_path: str, result: Dict[str, Any]):
        """Armazena resultado YOLO no cache."""
        try:
            # Limitar tamanho do cache
            if len(self.yolo_results_cache) >= self.yolo_cache_max_size:
                # Remover entrada mais antiga
                oldest_key = next(iter(self.yolo_results_cache))
                del self.yolo_results_cache[oldest_key]
            
            self.yolo_results_cache[image_path] = {
                'result': result,
                'timestamp': time.time()
            }
            
            logger.debug(f"[CACHE] YOLO result stored: {image_path}")
            
        except Exception as e:
            logger.error(f"[ERRO] Erro ao armazenar cache YOLO: {e}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache."""
        try:
            total_requests = self.cache_hit_count + self.cache_miss_count
            hit_rate = (self.cache_hit_count / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'analysis_cache_size': len(self.analysis_cache),
                'visual_features_cache_size': len(self.visual_features_cache),
                'yolo_cache_size': len(self.yolo_results_cache),
                'cache_hits': self.cache_hit_count,
                'cache_misses': self.cache_miss_count,
                'hit_rate_percent': round(hit_rate, 2),
                'performance_stats': self.performance_stats
            }
            
        except Exception as e:
            logger.error(f"[ERRO] Erro ao obter estatísticas de cache: {e}")
            return {}
    
    def clear_cache(self, cache_type: str = "all"):
        """Limpa cache especificado."""
        try:
            if cache_type in ["all", "analysis"]:
                self.analysis_cache.clear()
                logger.info("[CACHE] Analysis cache cleared")
            
            if cache_type in ["all", "visual"]:
                self.visual_features_cache.clear()
                logger.info("[CACHE] Visual features cache cleared")
            
            if cache_type in ["all", "yolo"]:
                self.yolo_results_cache.clear()
                logger.info("[CACHE] YOLO cache cleared")
            
            if cache_type == "all":
                self.cache_hit_count = 0
                self.cache_miss_count = 0
                logger.info("[CACHE] All caches cleared")
                
        except Exception as e:
            logger.error(f"[ERRO] Erro ao limpar cache: {e}")
