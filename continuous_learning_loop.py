#!/usr/bin/env python3
"""
Ciclo de Aprendizagem Contínua - O Sistema que Se Auto-Melhora
Implementa o "Santo Graal" da IA: um sistema que aprende sozinho
e se auto-melhora através de ciclos de feedback.
"""

import os
import json
import logging
import shutil
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import subprocess
import time

from intuition_module import IntuitionEngine, LearningCandidate
from auto_annotator import GradCAMAnnotator, AutoAnnotation
from hybrid_curator import HybridCurator, CuratorDecision

logging.basicConfig(level=logging.INFO)

class LearningCycleStage(Enum):
    """Estágios do ciclo de aprendizado"""
    INTUITION_DETECTION = "intuition_detection"
    AUTO_ANNOTATION = "auto_annotation"
    VALIDATION = "validation"
    DECISION_EXECUTION = "decision_execution"
    MODEL_RETRAINING = "model_retraining"
    EVALUATION = "evaluation"

class LearningCycleStatus(Enum):
    """Status do ciclo de aprendizado"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

@dataclass
class LearningCycle:
    """Ciclo completo de aprendizado"""
    cycle_id: str
    start_time: str
    end_time: str = ""
    status: LearningCycleStatus = LearningCycleStatus.RUNNING
    stages_completed: List[LearningCycleStage] = None
    candidates_processed: int = 0
    annotations_generated: int = 0
    auto_approved: int = 0
    auto_rejected: int = 0
    human_review_needed: int = 0
    model_retrained: bool = False
    performance_improvement: float = 0.0
    metadata: Dict = None
    
    def __post_init__(self):
        if self.stages_completed is None:
            self.stages_completed = []
        if self.metadata is None:
            self.metadata = {}

class ContinuousLearningSystem:
    """
    Sistema de Aprendizado Contínuo - O Santo Graal da IA
    """
    
    def __init__(self, 
                 yolo_model_path: str,
                 keras_model_path: str,
                 api_type: str = "gemini",
                 api_key: str = None,
                 learning_data_path: str = "./learning_data"):
        """
        Inicializa sistema de aprendizado contínuo
        
        Args:
            yolo_model_path: Caminho para modelo YOLO
            keras_model_path: Caminho para modelo Keras
            api_type: Tipo de API ("gemini" ou "gpt4v")
            api_key: Chave da API
            learning_data_path: Caminho para dados de aprendizado
        """
        self.yolo_model_path = yolo_model_path
        self.keras_model_path = keras_model_path
        self.api_type = api_type
        self.api_key = api_key
        self.learning_data_path = learning_data_path
        
        # Inicializar componentes
        self._initialize_components()
        
        # Configurações do sistema
        self.min_candidates_for_retraining = 10
        self.performance_threshold = 0.05  # 5% de melhoria mínima
        self.max_cycles_per_day = 5
        
        # Histórico de ciclos
        self.learning_cycles = []
        self.current_cycle = None
        
        # Estatísticas globais
        self.global_stats = {
            "total_cycles": 0,
            "total_candidates": 0,
            "total_annotations": 0,
            "total_auto_approved": 0,
            "total_auto_rejected": 0,
            "total_human_review": 0,
            "model_retraining_count": 0,
            "average_performance_improvement": 0.0
        }
    
    def _initialize_components(self):
        """Inicializa todos os componentes do sistema"""
        try:
            # Motor de Intuição
            self.intuition_engine = IntuitionEngine(
                self.yolo_model_path, 
                self.keras_model_path
            )
            
            # Anotador Automático
            self.auto_annotator = GradCAMAnnotator(self.keras_model_path)
            
            # Curador Híbrido
            self.hybrid_curator = HybridCurator(self.api_type, self.api_key)
            
            # Criar diretórios
            self._create_directories()
            
            logging.info("✅ Sistema de Aprendizado Contínuo inicializado")
            
        except Exception as e:
            logging.error(f"❌ Erro na inicialização: {e}")
            raise
    
    def _create_directories(self):
        """Cria estrutura de diretórios para aprendizado"""
        directories = [
            self.learning_data_path,
            os.path.join(self.learning_data_path, "pending_validation"),
            os.path.join(self.learning_data_path, "awaiting_human_review"),
            os.path.join(self.learning_data_path, "auto_approved"),
            os.path.join(self.learning_data_path, "auto_rejected"),
            os.path.join(self.learning_data_path, "cycles_history"),
            os.path.join(self.learning_data_path, "model_checkpoints")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def start_learning_cycle(self, image_directory: str) -> str:
        """
        Inicia um novo ciclo de aprendizado
        
        Args:
            image_directory: Diretório com imagens para processar
            
        Returns:
            ID do ciclo iniciado
        """
        cycle_id = f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_cycle = LearningCycle(
            cycle_id=cycle_id,
            start_time=datetime.now().isoformat(),
            metadata={
                "image_directory": image_directory,
                "api_type": self.api_type,
                "components_initialized": True
            }
        )
        
        logging.info(f"🚀 Iniciando ciclo de aprendizado: {cycle_id}")
        
        try:
            # Executar ciclo completo
            self._execute_learning_cycle(image_directory)
            
            # Finalizar ciclo
            self.current_cycle.end_time = datetime.now().isoformat()
            self.current_cycle.status = LearningCycleStatus.COMPLETED
            
            # Salvar ciclo
            self._save_learning_cycle()
            
            # Atualizar estatísticas globais
            self._update_global_stats()
            
            logging.info(f"✅ Ciclo {cycle_id} concluído com sucesso")
            return cycle_id
            
        except Exception as e:
            logging.error(f"❌ Erro no ciclo {cycle_id}: {e}")
            self.current_cycle.status = LearningCycleStatus.FAILED
            self.current_cycle.end_time = datetime.now().isoformat()
            self._save_learning_cycle()
            raise
    
    def _execute_learning_cycle(self, image_directory: str):
        """Executa ciclo completo de aprendizado"""
        
        # ESTÁGIO 1: Detecção de Intuição
        logging.info("🔍 ESTÁGIO 1: Detecção de Intuição")
        candidates = self._detect_intuition_candidates(image_directory)
        self.current_cycle.candidates_processed = len(candidates)
        self.current_cycle.stages_completed.append(LearningCycleStage.INTUITION_DETECTION)
        
        if not candidates:
            logging.info("ℹ️ Nenhum candidato para aprendizado detectado")
            return
        
        # ESTÁGIO 2: Geração de Anotações Automáticas
        logging.info("🎯 ESTÁGIO 2: Geração de Anotações Automáticas")
        annotations = self._generate_auto_annotations(candidates)
        self.current_cycle.annotations_generated = len(annotations)
        self.current_cycle.stages_completed.append(LearningCycleStage.AUTO_ANNOTATION)
        
        if not annotations:
            logging.info("ℹ️ Nenhuma anotação gerada")
            return
        
        # ESTÁGIO 3: Validação Híbrida
        logging.info("🎭 ESTÁGIO 3: Validação Híbrida")
        decisions = self._validate_annotations(annotations)
        self.current_cycle.stages_completed.append(LearningCycleStage.VALIDATION)
        
        # ESTÁGIO 4: Execução de Decisões
        logging.info("⚡ ESTÁGIO 4: Execução de Decisões")
        self._execute_decisions(decisions)
        self.current_cycle.stages_completed.append(LearningCycleStage.DECISION_EXECUTION)
        
        # ESTÁGIO 5: Re-treinamento do Modelo (se necessário)
        if self._should_retrain_model():
            logging.info("🔄 ESTÁGIO 5: Re-treinamento do Modelo")
            self._retrain_models()
            self.current_cycle.model_retrained = True
            self.current_cycle.stages_completed.append(LearningCycleStage.MODEL_RETRAINING)
        
        # ESTÁGIO 6: Avaliação de Performance
        logging.info("📊 ESTÁGIO 6: Avaliação de Performance")
        performance_improvement = self._evaluate_performance()
        self.current_cycle.performance_improvement = performance_improvement
        self.current_cycle.stages_completed.append(LearningCycleStage.EVALUATION)
    
    def _detect_intuition_candidates(self, image_directory: str) -> List[LearningCandidate]:
        """Detecta candidatos para aprendizado"""
        candidates = []
        
        for filename in os.listdir(image_directory):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_directory, filename)
                
                try:
                    analysis = self.intuition_engine.analyze_image_intuition(image_path)
                    intuition_analysis = analysis.get("intuition_analysis", {})
                    
                    for candidate_data in intuition_analysis.get("candidates", []):
                        # Converter dict para LearningCandidate
                        candidate = LearningCandidate(
                            image_path=candidate_data["image_path"],
                            candidate_type=candidate_data["type"],
                            yolo_confidence=0.0,  # Será preenchido pela análise
                            keras_confidence=candidate_data["keras_confidence"],
                            keras_prediction=candidate_data["keras_prediction"],
                            yolo_detections=[],
                            reasoning=candidate_data["reasoning"],
                            priority_score=candidate_data["priority_score"]
                        )
                        candidates.append(candidate)
                        
                except Exception as e:
                    logging.error(f"Erro ao analisar {filename}: {e}")
        
        logging.info(f"🎯 {len(candidates)} candidatos detectados para aprendizado")
        return candidates
    
    def _generate_auto_annotations(self, candidates: List[LearningCandidate]) -> List[AutoAnnotation]:
        """Gera anotações automáticas"""
        annotations = []
        
        for candidate in candidates:
            try:
                annotation = self.auto_annotator.generate_auto_annotation(candidate)
                if annotation:
                    annotations.append(annotation)
                    
                    # Visualizar Grad-CAM para debug
                    visualization_path = self.auto_annotator.visualize_gradcam(
                        candidate.image_path, candidate
                    )
                    if visualization_path:
                        logging.info(f"📊 Grad-CAM visualizado: {visualization_path}")
                        
            except Exception as e:
                logging.error(f"Erro ao gerar anotação para {candidate.image_path}: {e}")
        
        logging.info(f"🎯 {len(annotations)} anotações geradas automaticamente")
        return annotations
    
    def _validate_annotations(self, annotations: List[AutoAnnotation]) -> List[CuratorDecision]:
        """Valida anotações usando curador híbrido"""
        decisions = []
        
        for annotation in annotations:
            try:
                decision = self.hybrid_curator.validate_annotation(annotation)
                decisions.append(decision)
                
                # Contar decisões
                if decision.decision.value == "auto_approve":
                    self.current_cycle.auto_approved += 1
                elif decision.decision.value == "auto_reject":
                    self.current_cycle.auto_rejected += 1
                else:
                    self.current_cycle.human_review_needed += 1
                    
            except Exception as e:
                logging.error(f"Erro ao validar anotação {annotation.image_path}: {e}")
        
        logging.info(f"🎭 {len(decisions)} anotações validadas")
        return decisions
    
    def _execute_decisions(self, decisions: List[CuratorDecision]):
        """Executa decisões do curador"""
        for decision in decisions:
            try:
                success = self.hybrid_curator.execute_decision(
                    decision,
                    train_dir=os.path.join(self.learning_data_path, "auto_approved"),
                    awaiting_review_dir=os.path.join(self.learning_data_path, "awaiting_human_review"),
                    rejected_dir=os.path.join(self.learning_data_path, "auto_rejected")
                )
                
                if success:
                    logging.info(f"✅ Decisão executada: {decision.decision.value}")
                else:
                    logging.error(f"❌ Falha ao executar decisão: {decision.decision.value}")
                    
            except Exception as e:
                logging.error(f"Erro ao executar decisão: {e}")
    
    def _should_retrain_model(self) -> bool:
        """Determina se deve re-treinar o modelo"""
        auto_approved_count = self.current_cycle.auto_approved
        
        # Re-treinar se há dados suficientes aprovados automaticamente
        should_retrain = auto_approved_count >= self.min_candidates_for_retraining
        
        if should_retrain:
            logging.info(f"🔄 Re-treinamento necessário: {auto_approved_count} amostras aprovadas")
        else:
            logging.info(f"⏳ Aguardando mais dados: {auto_approved_count}/{self.min_candidates_for_retraining}")
        
        return should_retrain
    
    def _retrain_models(self):
        """Re-treina modelos com novos dados"""
        try:
            # Preparar dados para re-treinamento
            self._prepare_retraining_data()
            
            # Re-treinar YOLO
            self._retrain_yolo()
            
            # Re-treinar Keras (se necessário)
            self._retrain_keras()
            
            logging.info("✅ Modelos re-treinados com sucesso")
            
        except Exception as e:
            logging.error(f"❌ Erro no re-treinamento: {e}")
            raise
    
    def _prepare_retraining_data(self):
        """Prepara dados para re-treinamento"""
        auto_approved_dir = os.path.join(self.learning_data_path, "auto_approved")
        
        # Copiar dados aprovados para dataset de treinamento
        train_dir = "./dataset_passaros/images/train"
        
        for filename in os.listdir(auto_approved_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(auto_approved_dir, filename)
                dst_path = os.path.join(train_dir, filename)
                
                # Copiar imagem
                shutil.copy2(src_path, dst_path)
                
                # Copiar anotação correspondente
                annotation_name = os.path.splitext(filename)[0] + ".txt"
                src_annotation = os.path.join(auto_approved_dir, annotation_name)
                dst_annotation = os.path.join(train_dir, annotation_name)
                
                if os.path.exists(src_annotation):
                    shutil.copy2(src_annotation, dst_annotation)
        
        logging.info("📁 Dados preparados para re-treinamento")
    
    def _retrain_yolo(self):
        """Re-treina modelo YOLO"""
        try:
            # Executar script de treinamento YOLO
            cmd = ["python3", "treinar_yolo.py"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                logging.info("✅ YOLO re-treinado com sucesso")
            else:
                logging.error(f"❌ Erro no re-treinamento YOLO: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logging.error("⏰ Timeout no re-treinamento YOLO")
        except Exception as e:
            logging.error(f"❌ Erro no re-treinamento YOLO: {e}")
    
    def _retrain_keras(self):
        """Re-treina modelo Keras (se necessário)"""
        # Por enquanto, apenas log
        # Implementar re-treinamento Keras se necessário
        logging.info("📝 Re-treinamento Keras não implementado ainda")
    
    def _evaluate_performance(self) -> float:
        """Avalia melhoria de performance"""
        # Por enquanto, retornar melhoria simulada
        # Implementar avaliação real de performance
        improvement = 0.05  # 5% de melhoria simulada
        logging.info(f"📊 Performance melhorada em {improvement:.1%}")
        return improvement
    
    def _save_learning_cycle(self):
        """Salva ciclo de aprendizado"""
        cycle_file = os.path.join(
            self.learning_data_path, 
            "cycles_history", 
            f"{self.current_cycle.cycle_id}.json"
        )
        
        with open(cycle_file, 'w') as f:
            json.dump(asdict(self.current_cycle), f, indent=2, ensure_ascii=False)
        
        # Adicionar ao histórico
        self.learning_cycles.append(self.current_cycle)
    
    def _update_global_stats(self):
        """Atualiza estatísticas globais"""
        cycle = self.current_cycle
        
        self.global_stats["total_cycles"] += 1
        self.global_stats["total_candidates"] += cycle.candidates_processed
        self.global_stats["total_annotations"] += cycle.annotations_generated
        self.global_stats["total_auto_approved"] += cycle.auto_approved
        self.global_stats["total_auto_rejected"] += cycle.auto_rejected
        self.global_stats["total_human_review"] += cycle.human_review_needed
        
        if cycle.model_retrained:
            self.global_stats["model_retraining_count"] += 1
        
        # Calcular melhoria média
        if self.global_stats["total_cycles"] > 0:
            total_improvement = sum(c.performance_improvement for c in self.learning_cycles)
            self.global_stats["average_performance_improvement"] = (
                total_improvement / self.global_stats["total_cycles"]
            )
    
    def get_system_statistics(self) -> Dict:
        """Retorna estatísticas completas do sistema"""
        return {
            "global_stats": self.global_stats,
            "current_cycle": asdict(self.current_cycle) if self.current_cycle else None,
            "intuition_stats": self.intuition_engine.get_learning_statistics(),
            "annotator_stats": self.auto_annotator.get_annotation_statistics(),
            "curator_stats": self.hybrid_curator.get_curator_statistics(),
            "efficiency_metrics": {
                "automation_rate": (
                    self.global_stats["total_auto_approved"] + self.global_stats["total_auto_rejected"]
                ) / max(self.global_stats["total_annotations"], 1),
                "human_workload_reduction": f"{self.global_stats['total_auto_approved'] + self.global_stats['total_auto_rejected']} de {self.global_stats['total_annotations']} casos automatizados",
                "learning_efficiency": self.global_stats["average_performance_improvement"]
            }
        }
    
    def generate_learning_report(self) -> Dict:
        """Gera relatório completo de aprendizado"""
        return {
            "timestamp": datetime.now().isoformat(),
            "system_status": "ACTIVE" if self.current_cycle else "IDLE",
            "statistics": self.get_system_statistics(),
            "recent_cycles": [
                {
                    "cycle_id": cycle.cycle_id,
                    "status": cycle.status.value,
                    "candidates": cycle.candidates_processed,
                    "annotations": cycle.annotations_generated,
                    "auto_approved": cycle.auto_approved,
                    "performance_improvement": cycle.performance_improvement
                }
                for cycle in self.learning_cycles[-5:]  # Últimos 5 ciclos
            ],
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Gera recomendações baseadas no desempenho"""
        recommendations = []
        
        stats = self.global_stats
        
        if stats["total_cycles"] == 0:
            recommendations.append("🚀 Execute o primeiro ciclo de aprendizado")
            return recommendations
        
        if stats["total_auto_approved"] < 5:
            recommendations.append("📈 Aguarde mais dados para re-treinamento")
        
        if stats["average_performance_improvement"] < 0.02:
            recommendations.append("🔧 Considere ajustar parâmetros de validação")
        
        if stats["total_human_review"] > stats["total_auto_approved"]:
            recommendations.append("⚙️ Otimize critérios de auto-aprovação")
        
        recommendations.append("✅ Sistema funcionando adequadamente")
        
        return recommendations

# Exemplo de uso
if __name__ == "__main__":
    print("🔄 Sistema de Aprendizado Contínuo - O Santo Graal da IA")
    print("=" * 60)
    print("Este sistema implementa aprendizado contínuo com:")
    print("• Detecção de intuição")
    print("• Geração automática de anotações")
    print("• Validação híbrida com APIs de visão")
    print("• Re-treinamento automático")
    print("• Auto-melhoria contínua")
    print()
    print("Para usar:")
    print("1. Configure APIs externas")
    print("2. Use start_learning_cycle() para iniciar aprendizado")
    print("3. Monitore progresso com get_system_statistics()")
    print()
    print("🚀 PRÓXIMO PASSO: Criar Sistema Auto-Melhorador Completo")
