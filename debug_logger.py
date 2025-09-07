#!/usr/bin/env python3
"""
Sistema de Logging Detalhado para Debug do Sistema de Raciocínio Lógico
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List

class DebugLogger:
    """Logger detalhado para debug do sistema"""
    
    def __init__(self, log_file: str = "debug_system.log"):
        self.log_file = log_file
        self.setup_logger()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def setup_logger(self):
        """Configura o logger com diferentes níveis"""
        # Criar logger
        self.logger = logging.getLogger('debug_system')
        self.logger.setLevel(logging.DEBUG)
        
        # Limpar handlers existentes
        self.logger.handlers.clear()
        
        # Handler para arquivo
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Handler para console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formato das mensagens
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def log_session_start(self, image_path: str):
        """Inicia uma nova sessão de análise"""
        self.logger.info("=" * 80)
        self.logger.info(f"INICIANDO ANÁLISE - Sessão: {self.session_id}")
        self.logger.info(f"Imagem: {image_path}")
        self.logger.info("=" * 80)
        
    def log_yolo_analysis(self, yolo_result: Dict[str, Any]):
        """Log detalhado da análise YOLO"""
        self.logger.info("--- ANÁLISE YOLO ---")
        
        if 'detections' in yolo_result:
            detections = yolo_result['detections']
            self.logger.info(f"Número de detecções: {len(detections)}")
            
            for i, detection in enumerate(detections):
                self.logger.info(f"  Detecção {i+1}:")
                self.logger.info(f"    Classe: {detection.get('class', 'N/A')}")
                self.logger.info(f"    Confiança: {detection.get('confidence', 'N/A'):.4f}")
                self.logger.info(f"    Bbox: {detection.get('bbox', 'N/A')}")
                
            # Análise de pássaros
            bird_detections = [d for d in detections if 'bird' in d.get('class', '').lower()]
            self.logger.info(f"Detecções de pássaro: {len(bird_detections)}")
            
            if bird_detections:
                avg_confidence = sum(d.get('confidence', 0) for d in bird_detections) / len(bird_detections)
                self.logger.info(f"Confiança média de pássaros: {avg_confidence:.4f}")
            else:
                self.logger.info("NENHUMA detecção de pássaro encontrada!")
                
        else:
            self.logger.warning("Resultado YOLO sem campo 'detections'")
            
        self.logger.info("--- FIM ANÁLISE YOLO ---")
        
    def log_keras_analysis(self, keras_result: Dict[str, Any]):
        """Log detalhado da análise Keras"""
        self.logger.info("--- ANÁLISE KERAS ---")
        
        if 'error' in keras_result:
            self.logger.warning(f"Erro na análise Keras: {keras_result['error']}")
        else:
            self.logger.info("Análise Keras executada com sucesso")
            if 'predictions' in keras_result:
                predictions = keras_result['predictions']
                self.logger.info(f"Número de predições: {len(predictions)}")
                for pred in predictions:
                    self.logger.info(f"  Classe: {pred.get('class', 'N/A')}, Confiança: {pred.get('confidence', 'N/A'):.4f}")
                    
        self.logger.info("--- FIM ANÁLISE KERAS ---")
        
    def log_intuition_analysis(self, intuition_result: Dict[str, Any]):
        """Log detalhado da análise de intuição"""
        self.logger.info("--- ANÁLISE DE INTUIÇÃO ---")
        
        if 'candidates_found' in intuition_result:
            candidates = intuition_result['candidates_found']
            self.logger.info(f"Candidatos de aprendizado encontrados: {candidates}")
            
        if 'reasoning' in intuition_result:
            reasoning = intuition_result['reasoning']
            self.logger.info(f"Raciocínio:")
            for reason in reasoning:
                self.logger.info(f"  - {reason}")
                
        if 'intuition_level' in intuition_result:
            level = intuition_result['intuition_level']
            self.logger.info(f"Nível de intuição: {level}")
            
        if 'recommendation' in intuition_result:
            recommendation = intuition_result['recommendation']
            self.logger.info(f"Recomendação: {recommendation}")
            
        self.logger.info("--- FIM ANÁLISE DE INTUIÇÃO ---")
        
    def log_frontend_analysis(self, frontend_data: Dict[str, Any]):
        """Log da análise no frontend"""
        self.logger.info("--- ANÁLISE FRONTEND ---")
        
        # Verificar se há detecções
        has_detections = frontend_data.get('has_detections', False)
        has_bird_detection = frontend_data.get('has_bird_detection', False)
        
        self.logger.info(f"Tem detecções: {has_detections}")
        self.logger.info(f"Tem detecção de pássaro: {has_bird_detection}")
        
        if 'detected_classes' in frontend_data:
            classes = frontend_data['detected_classes']
            self.logger.info(f"Classes detectadas: {classes}")
            
        self.logger.info("--- FIM ANÁLISE FRONTEND ---")
        
    def log_decision_logic(self, decision_data: Dict[str, Any]):
        """Log da lógica de decisão"""
        self.logger.info("--- LÓGICA DE DECISÃO ---")
        
        if 'action' in decision_data:
            action = decision_data['action']
            self.logger.info(f"Ação recomendada: {action}")
            
        if 'reasoning' in decision_data:
            reasoning = decision_data['reasoning']
            self.logger.info(f"Raciocínio da decisão: {reasoning}")
            
        self.logger.info("--- FIM LÓGICA DE DECISÃO ---")
        
    def log_error(self, error_msg: str, error_type: str = "GENERAL"):
        """Log de erros"""
        self.logger.error(f"ERRO [{error_type}]: {error_msg}")
        
    def log_warning(self, warning_msg: str):
        """Log de avisos"""
        self.logger.warning(f"AVISO: {warning_msg}")
        
    def log_info(self, info_msg: str):
        """Log de informações"""
        self.logger.info(f"INFO: {info_msg}")
        
    def log_session_end(self, final_result: Dict[str, Any]):
        """Finaliza a sessão de análise"""
        self.logger.info("--- RESULTADO FINAL ---")
        self.logger.info(f"Resultado: {json.dumps(final_result, indent=2, ensure_ascii=False)}")
        self.logger.info("=" * 80)
        self.logger.info(f"FIM DA ANÁLISE - Sessão: {self.session_id}")
        self.logger.info("=" * 80)

# Instância global do logger
debug_logger = DebugLogger()
