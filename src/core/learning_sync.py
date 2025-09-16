#!/usr/bin/env python3
"""
Sistema de Sincronização de Aprendizado
Conecta análise manual com aprendizado contínuo
"""

import os
import shutil
import json
import time
import threading
import logging
import subprocess
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LearningSyncSystem:
    """Sistema de sincronização entre análise manual e aprendizado contínuo"""
    
    def __init__(self, 
                 manual_approved_dir: str = "data/manual_analysis/approved",
                 learning_approved_dir: str = "data/learning_data/auto_approved",
                 dataset_train_dir: str = "data/datasets/dataset_passaros/images/train",
                 sync_interval: int = 30):
        self.manual_approved_dir = manual_approved_dir
        self.learning_approved_dir = learning_approved_dir
        self.dataset_train_dir = dataset_train_dir
        self.sync_interval = sync_interval
        self.sync_running = False
        self.sync_thread = None
        self.sync_stats = {
            "total_synced": 0,
            "retraining_triggered": 0,
            "last_sync": None
        }
        
        # Criar diretórios se não existirem
        os.makedirs(self.learning_approved_dir, exist_ok=True)
        os.makedirs(self.dataset_train_dir, exist_ok=True)
    
    def _sync_single_image(self, image_file: str) -> bool:
        """Sincroniza uma única imagem"""
        try:
            src_image = os.path.join(self.manual_approved_dir, image_file)
            dst_image = os.path.join(self.learning_approved_dir, image_file)
            
            # Copiar imagem
            shutil.copy2(src_image, dst_image)
            logger.debug(f"Imagem copiada: {image_file}")
            
            # Copiar anotação se existir
            annotation_file = os.path.splitext(image_file)[0] + ".txt"
            src_annotation = os.path.join(self.manual_approved_dir, annotation_file)
            dst_annotation = os.path.join(self.learning_approved_dir, annotation_file)
            
            if os.path.exists(src_annotation):
                shutil.copy2(src_annotation, dst_annotation)
                logger.debug(f"Anotação copiada: {annotation_file}")
            
            # Copiar JSON se existir
            json_file = os.path.splitext(image_file)[0] + ".json"
            src_json = os.path.join(self.manual_approved_dir, json_file)
            dst_json = os.path.join(self.learning_approved_dir, json_file)
            
            if os.path.exists(src_json):
                shutil.copy2(src_json, dst_json)
                logger.debug(f"JSON copiado: {json_file}")
            
            # Processar feedback detalhado se existir
            feedback_file = os.path.splitext(image_file)[0] + ".feedback.json"
            src_feedback = os.path.join(self.manual_approved_dir, feedback_file)
            dst_feedback = os.path.join(self.learning_approved_dir, feedback_file)

            if os.path.exists(src_feedback):
                shutil.copy2(src_feedback, dst_feedback)
                logger.debug(f"Feedback copiado: {feedback_file}")
                # Processar feedback para melhorar o aprendizado
                self._process_feedback(src_feedback)
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao sincronizar {image_file}: {e}")
            return False
    
    def _process_feedback(self, feedback_path: str):
        """Processa feedback detalhado para melhorar o aprendizado"""
        try:
            with open(feedback_path, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)

            logger.info(f"Processando feedback: {feedback_data.get('species', 'unknown')}")

            # Extrair informações do feedback
            species = feedback_data.get('species', 'unknown')
            decision_reason = feedback_data.get('decision_reason', '')
            visual_characteristics = feedback_data.get('visual_characteristics', [])
            additional_observations = feedback_data.get('additional_observations', '')

            # Criar arquivo de aprendizado estruturado
            learning_data = {
                "species": species,
                "feedback_type": "manual_detailed",
                "decision_reason": decision_reason,
                "visual_characteristics": visual_characteristics,
                "additional_observations": additional_observations,
                "timestamp": feedback_data.get('timestamp', datetime.now().isoformat()),
                "confidence": feedback_data.get('confidence', 0.0),
                "processed_at": datetime.now().isoformat()
            }

            # Salvar dados de aprendizado estruturados
            learning_file = feedback_path.replace('.feedback.json', '.learning.json')
            with open(learning_file, 'w', encoding='utf-8') as f:
                json.dump(learning_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Dados de aprendizado salvos: {learning_file}")

        except Exception as e:
            logger.error(f"Erro ao processar feedback {feedback_path}: {e}")
    
    def sync_approved_images(self) -> Dict:
        """Sincroniza imagens aprovadas manualmente"""
        logger.info("Iniciando sincronização de imagens aprovadas")
        
        if not os.path.exists(self.manual_approved_dir):
            logger.warning(f"Diretório de imagens aprovadas não existe: {self.manual_approved_dir}")
            return {"error": "Diretório não existe"}
        
        # Encontrar imagens aprovadas
        image_files = [f for f in os.listdir(self.manual_approved_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        logger.info(f"Encontradas {len(image_files)} imagens aprovadas")
        
        synced_count = 0
        for image_file in image_files:
            if self._sync_single_image(image_file):
                synced_count += 1
        
        # Verificar se deve re-treinar
        should_retrain = synced_count >= 5  # Threshold configurável
        
        sync_result = {
            "synced_count": synced_count,
            "total_images": len(image_files),
            "retraining_triggered": False
        }
        
        if should_retrain:
            logger.info("Condição de re-treinamento atendida: {synced_count} imagens")
            logger.info("Ativando re-treinamento automático")
            retrain_success = self._trigger_retraining()
            sync_result["retraining_triggered"] = retrain_success
            if retrain_success:
                self.sync_stats["retraining_triggered"] += 1
        
        self.sync_stats["total_synced"] += synced_count
        self.sync_stats["last_sync"] = datetime.now().isoformat()
        
        logger.info(f"Sincronização concluída: {synced_count} imagens")
        return sync_result
    
    def _trigger_retraining(self) -> bool:
        """Ativa re-treinamento dos modelos"""
        logger.info("Iniciando re-treinamento automático")
        
        try:
            # Preparar dados para re-treinamento
            self._prepare_retraining_data()
            
            # Re-treinar YOLO
            yolo_success = self._retrain_yolo()
            
            # Re-treinar Keras
            keras_success = self._retrain_keras()
            
            if yolo_success and keras_success:
                logger.info("Re-treinamento concluído com sucesso")
                return True
            else:
                logger.error("Re-treinamento falhou")
                return False
                
        except Exception as e:
            logger.error(f"Erro no re-treinamento: {e}")
            return False
    
    def _prepare_retraining_data(self):
        """Prepara dados para re-treinamento"""
        logger.info("Preparando dados para re-treinamento")
        
        # Copiar imagens aprovadas para dataset de treinamento
        for image_file in os.listdir(self.learning_approved_dir):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src = os.path.join(self.learning_approved_dir, image_file)
                dst = os.path.join(self.dataset_train_dir, image_file)
                shutil.copy2(src, dst)
        
        logger.info("Dados preparados para re-treinamento")
    
    def _retrain_yolo(self) -> bool:
        """Re-treina modelo YOLO"""
        logger.info("Re-treinando modelo YOLO")
        
        try:
            cmd = ["python3", "src/training/yolo_trainer.py"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                logger.info("YOLO re-treinado com sucesso")
                return True
            else:
                logger.error(f"Erro no re-treinamento YOLO: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Erro no re-treinamento YOLO: {e}")
            return False
    
    def _retrain_keras(self) -> bool:
        """Re-treina modelo Keras"""
        logger.info("Re-treinando modelo Keras")
        
        try:
            cmd = ["python3", "src/training/keras_trainer.py"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                logger.info("Keras re-treinado com sucesso")
                return True
            else:
                logger.error(f"Erro no re-treinamento Keras: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Erro no re-treinamento Keras: {e}")
            return False
    
    def start_continuous_sync(self):
        """Inicia sincronização contínua"""
        if self.sync_running:
            logger.warning("Sincronização contínua já está rodando")
            return
        
        self.sync_running = True
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()
        logger.info("Sincronização contínua iniciada")
    
    def stop_continuous_sync(self):
        """Para sincronização contínua"""
        self.sync_running = False
        if self.sync_thread:
            self.sync_thread.join(timeout=5)
        logger.info("Sincronização contínua parada")
    
    def _sync_loop(self):
        """Loop de sincronização contínua"""
        while self.sync_running:
            try:
                self.sync_approved_images()
                time.sleep(self.sync_interval)
            except Exception as e:
                logger.error(f"Erro no loop de sincronização: {e}")
                time.sleep(60)  # Esperar mais tempo em caso de erro

# Instância global
learning_sync = LearningSyncSystem()

def start_continuous_sync():
    """Inicia sincronização contínua"""
    learning_sync.start_continuous_sync()

def stop_continuous_sync():
    """Para sincronização contínua"""
    learning_sync.stop_continuous_sync()

def get_sync_stats():
    """Retorna estatísticas de sincronização"""
    return {
        "sync_running": learning_sync.sync_running,
        "manual_approved_count": len([f for f in os.listdir(learning_sync.manual_approved_dir) 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) 
                                if os.path.exists(learning_sync.manual_approved_dir) else 0,
        "learning_approved_count": len([f for f in os.listdir(learning_sync.learning_approved_dir) 
                                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) 
                                  if os.path.exists(learning_sync.learning_approved_dir) else 0,
        "sync_stats": learning_sync.sync_stats
    }
