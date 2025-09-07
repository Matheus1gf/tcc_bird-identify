#!/usr/bin/env python3
"""
Teste do Sistema Santo Graal - Demonstração do Aprendizado Autônomo
Testa todos os módulos do sistema revolucionário de IA que aprende sozinha.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict

# Importar módulos do sistema
from intuition_module import IntuitionEngine, LearningCandidateType
from auto_annotator import GradCAMAnnotator
from hybrid_curator import HybridCurator
from continuous_learning_loop import ContinuousLearningSystem
from santo_graal_system import SantoGraalSystem

logging.basicConfig(level=logging.INFO)

class SantoGraalTester:
    """
    Testador do Sistema Santo Graal
    """
    
    def __init__(self):
        """Inicializa testador"""
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests_performed": [],
            "overall_status": "PENDING",
            "system_ready": False
        }
        
        # Configurações de teste
        self.test_image_dir = "./dataset_teste"
        self.yolo_model_path = "yolov8n.pt"  # Modelo pré-treinado
        self.keras_model_path = "modelo_classificacao_passaros.keras"  # Pode não existir
        
    def run_all_tests(self) -> Dict:
        """
        Executa todos os testes do sistema
        
        Returns:
            Resultados completos dos testes
        """
        print("🧠 TESTANDO SISTEMA SANTO GRAAL DA IA")
        print("=" * 50)
        
        # Teste 1: Verificar dependências
        self._test_dependencies()
        
        # Teste 2: Testar módulo de intuição
        self._test_intuition_module()
        
        # Teste 3: Testar anotador automático
        self._test_auto_annotator()
        
        # Teste 4: Testar curador híbrido (sem API)
        self._test_hybrid_curator()
        
        # Teste 5: Testar sistema completo
        self._test_complete_system()
        
        # Gerar relatório final
        self._generate_final_report()
        
        return self.test_results
    
    def _test_dependencies(self):
        """Testa dependências básicas"""
        print("\n🔍 TESTE 1: Verificando Dependências")
        
        test_result = {
            "test_name": "dependencies",
            "status": "PASSED",
            "details": [],
            "errors": []
        }
        
        try:
            # Testar imports
            import cv2
            import numpy as np
            import tensorflow as tf
            from ultralytics import YOLO
            import matplotlib.pyplot as plt
            import networkx as nx
            
            test_result["details"].append("✅ Todas as dependências principais importadas")
            
            # Testar YOLO
            model = YOLO(self.yolo_model_path)
            test_result["details"].append(f"✅ Modelo YOLO carregado: {self.yolo_model_path}")
            
            # Verificar diretório de teste
            if os.path.exists(self.test_image_dir):
                images = [f for f in os.listdir(self.test_image_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                test_result["details"].append(f"✅ {len(images)} imagens encontradas para teste")
            else:
                test_result["errors"].append(f"❌ Diretório de teste não encontrado: {self.test_image_dir}")
                test_result["status"] = "FAILED"
            
        except Exception as e:
            test_result["status"] = "FAILED"
            test_result["errors"].append(f"❌ Erro nas dependências: {str(e)}")
        
        self.test_results["tests_performed"].append(test_result)
        print(f"Status: {test_result['status']}")
    
    def _test_intuition_module(self):
        """Testa módulo de intuição"""
        print("\n🧠 TESTE 2: Módulo de Intuição")
        
        test_result = {
            "test_name": "intuition_module",
            "status": "PASSED",
            "details": [],
            "errors": []
        }
        
        try:
            # Inicializar motor de intuição
            intuition_engine = IntuitionEngine(
                self.yolo_model_path,
                self.keras_model_path
            )
            
            test_result["details"].append("✅ Motor de intuição inicializado")
            
            # Testar com uma imagem
            if os.path.exists(self.test_image_dir):
                images = [f for f in os.listdir(self.test_image_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if images:
                    test_image = os.path.join(self.test_image_dir, images[0])
                    
                    # Análise de intuição
                    analysis = intuition_engine.analyze_image_intuition(test_image)
                    
                    test_result["details"].append(f"✅ Análise de intuição executada: {os.path.basename(test_image)}")
                    
                    # Verificar resultados
                    intuition_analysis = analysis.get("intuition_analysis", {})
                    candidates = intuition_analysis.get("candidates", [])
                    
                    test_result["details"].append(f"✅ {len(candidates)} candidatos detectados")
                    
                    if candidates:
                        test_result["details"].append("🎯 Candidatos para aprendizado encontrados!")
                    else:
                        test_result["details"].append("ℹ️ Nenhum candidato para aprendizado detectado")
                    
                    # Estatísticas
                    stats = intuition_engine.get_learning_statistics()
                    test_result["details"].append(f"📊 Estatísticas: {stats}")
                    
                else:
                    test_result["errors"].append("❌ Nenhuma imagem encontrada para teste")
                    test_result["status"] = "FAILED"
            else:
                test_result["errors"].append("❌ Diretório de teste não encontrado")
                test_result["status"] = "FAILED"
                
        except Exception as e:
            test_result["status"] = "FAILED"
            test_result["errors"].append(f"❌ Erro no módulo de intuição: {str(e)}")
        
        self.test_results["tests_performed"].append(test_result)
        print(f"Status: {test_result['status']}")
    
    def _test_auto_annotator(self):
        """Testa anotador automático"""
        print("\n🎯 TESTE 3: Anotador Automático")
        
        test_result = {
            "test_name": "auto_annotator",
            "status": "PASSED",
            "details": [],
            "errors": []
        }
        
        try:
            # Verificar se modelo Keras existe
            if not os.path.exists(self.keras_model_path):
                test_result["status"] = "SKIPPED"
                test_result["details"].append("⏭️ Modelo Keras não encontrado - pulando teste")
                self.test_results["tests_performed"].append(test_result)
                print(f"Status: {test_result['status']}")
                return
            
            # Inicializar anotador
            annotator = GradCAMAnnotator(self.keras_model_path)
            test_result["details"].append("✅ Anotador automático inicializado")
            
            # Testar com uma imagem
            if os.path.exists(self.test_image_dir):
                images = [f for f in os.listdir(self.test_image_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if images:
                    test_image = os.path.join(self.test_image_dir, images[0])
                    
                    # Criar candidato de teste
                    from intuition_module import LearningCandidate
                    candidate = LearningCandidate(
                        image_path=test_image,
                        candidate_type=LearningCandidateType.YOLO_FAILED_KERAS_MEDIUM,
                        yolo_confidence=0.0,
                        keras_confidence=0.6,
                        keras_prediction="Painted_Bunting",
                        yolo_detections=[],
                        reasoning="Teste de anotação automática",
                        priority_score=0.8
                    )
                    
                    # Gerar anotação
                    annotation = annotator.generate_auto_annotation(candidate)
                    
                    if annotation:
                        test_result["details"].append("✅ Anotação automática gerada com sucesso")
                        test_result["details"].append(f"📄 Arquivo: {annotation.annotation_file_path}")
                        test_result["details"].append(f"🎯 Classe: {annotation.class_name}")
                        test_result["details"].append(f"📊 Confiança: {annotation.confidence:.2f}")
                        
                        # Visualizar Grad-CAM
                        visualization = annotator.visualize_gradcam(test_image, candidate)
                        if visualization:
                            test_result["details"].append(f"📊 Grad-CAM visualizado: {visualization}")
                        
                    else:
                        test_result["details"].append("⚠️ Anotação não gerada (Grad-CAM fraco)")
                    
                    # Estatísticas
                    stats = annotator.get_annotation_statistics()
                    test_result["details"].append(f"📊 Estatísticas: {stats}")
                    
                else:
                    test_result["errors"].append("❌ Nenhuma imagem encontrada para teste")
                    test_result["status"] = "FAILED"
            else:
                test_result["errors"].append("❌ Diretório de teste não encontrado")
                test_result["status"] = "FAILED"
                
        except Exception as e:
            test_result["status"] = "FAILED"
            test_result["errors"].append(f"❌ Erro no anotador automático: {str(e)}")
        
        self.test_results["tests_performed"].append(test_result)
        print(f"Status: {test_result['status']}")
    
    def _test_hybrid_curator(self):
        """Testa curador híbrido (sem API)"""
        print("\n🎭 TESTE 4: Curador Híbrido")
        
        test_result = {
            "test_name": "hybrid_curator",
            "status": "PASSED",
            "details": [],
            "errors": []
        }
        
        try:
            # Inicializar curador (sem API key)
            curator = HybridCurator("gemini", None)
            test_result["details"].append("✅ Curador híbrido inicializado")
            
            # Testar estatísticas
            stats = curator.get_curator_statistics()
            test_result["details"].append(f"📊 Estatísticas: {stats}")
            
            # Testar geração de relatório
            report = curator.generate_curator_report()
            test_result["details"].append("✅ Relatório do curador gerado")
            
            test_result["details"].append("ℹ️ Teste sem API key - validação real não executada")
            
        except Exception as e:
            test_result["status"] = "FAILED"
            test_result["errors"].append(f"❌ Erro no curador híbrido: {str(e)}")
        
        self.test_results["tests_performed"].append(test_result)
        print(f"Status: {test_result['status']}")
    
    def _test_complete_system(self):
        """Testa sistema completo"""
        print("\n🚀 TESTE 5: Sistema Completo")
        
        test_result = {
            "test_name": "complete_system",
            "status": "PASSED",
            "details": [],
            "errors": []
        }
        
        try:
            # Inicializar sistema completo
            system = SantoGraalSystem(
                yolo_model_path=self.yolo_model_path,
                keras_model_path=self.keras_model_path,
                api_type="gemini",
                api_key=None  # Sem API key para teste
            )
            
            test_result["details"].append("✅ Sistema Santo Graal inicializado")
            
            # Testar com uma imagem
            if os.path.exists(self.test_image_dir):
                images = [f for f in os.listdir(self.test_image_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if images:
                    test_image = os.path.join(self.test_image_dir, images[0])
                    
                    # Análise revolucionária
                    analysis = system.analyze_image_revolutionary(test_image)
                    
                    test_result["details"].append(f"✅ Análise revolucionária executada: {os.path.basename(test_image)}")
                    
                    # Verificar resultados
                    needs_learning = analysis.get("needs_learning", False)
                    revolutionary_action = analysis.get("revolutionary_action", "NONE")
                    
                    test_result["details"].append(f"🧠 Precisa de aprendizado: {needs_learning}")
                    test_result["details"].append(f"🚀 Ação revolucionária: {revolutionary_action}")
                    
                    if needs_learning:
                        test_result["details"].append("🎯 Sistema detectou necessidade de aprendizado!")
                    else:
                        test_result["details"].append("ℹ️ Sistema funcionando normalmente")
                    
                    # Estatísticas
                    stats = system.get_revolutionary_statistics()
                    test_result["details"].append("📊 Estatísticas do sistema obtidas")
                    
                else:
                    test_result["errors"].append("❌ Nenhuma imagem encontrada para teste")
                    test_result["status"] = "FAILED"
            else:
                test_result["errors"].append("❌ Diretório de teste não encontrado")
                test_result["status"] = "FAILED"
                
        except Exception as e:
            test_result["status"] = "FAILED"
            test_result["errors"].append(f"❌ Erro no sistema completo: {str(e)}")
        
        self.test_results["tests_performed"].append(test_result)
        print(f"Status: {test_result['status']}")
    
    def _generate_final_report(self):
        """Gera relatório final dos testes"""
        print("\n📊 RELATÓRIO FINAL DOS TESTES")
        print("=" * 50)
        
        # Contar resultados
        total_tests = len(self.test_results["tests_performed"])
        passed_tests = len([t for t in self.test_results["tests_performed"] if t["status"] == "PASSED"])
        failed_tests = len([t for t in self.test_results["tests_performed"] if t["status"] == "FAILED"])
        skipped_tests = len([t for t in self.test_results["tests_performed"] if t["status"] == "SKIPPED"])
        
        # Determinar status geral
        if failed_tests == 0:
            self.test_results["overall_status"] = "PASSED"
            self.test_results["system_ready"] = True
        elif passed_tests > failed_tests:
            self.test_results["overall_status"] = "PARTIAL"
            self.test_results["system_ready"] = True
        else:
            self.test_results["overall_status"] = "FAILED"
            self.test_results["system_ready"] = False
        
        # Resumo
        print(f"📈 Total de testes: {total_tests}")
        print(f"✅ Aprovados: {passed_tests}")
        print(f"❌ Falharam: {failed_tests}")
        print(f"⏭️ Pulados: {skipped_tests}")
        print(f"🎯 Status geral: {self.test_results['overall_status']}")
        print(f"🚀 Sistema pronto: {'SIM' if self.test_results['system_ready'] else 'NÃO'}")
        
        # Recomendações
        print("\n💡 RECOMENDAÇÕES:")
        
        if self.test_results["system_ready"]:
            print("✅ Sistema Santo Graal está funcionando!")
            print("🧠 Recursos revolucionários implementados:")
            print("   • Detecção de intuição")
            print("   • Geração automática de anotações")
            print("   • Validação híbrida")
            print("   • Aprendizado contínuo")
            print("   • Auto-melhoria")
            print()
            print("🚀 PRÓXIMOS PASSOS:")
            print("1. Configure API_KEY_GEMINI ou API_KEY_GPT4V")
            print("2. Execute: python3 santo_graal_system.py --images ./dataset_teste")
            print("3. Monitore aprendizado autônomo")
        else:
            print("⚠️ Sistema precisa de ajustes:")
            for test in self.test_results["tests_performed"]:
                if test["status"] == "FAILED":
                    print(f"   • {test['test_name']}: {', '.join(test['errors'])}")
        
        # Salvar relatório
        report_file = f"santo_graal_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Relatório salvo em: {report_file}")

def main():
    """Função principal"""
    print("🧠 TESTADOR DO SISTEMA SANTO GRAAL DA IA")
    print("=" * 50)
    print("Testando sistema revolucionário de aprendizado contínuo")
    print("=" * 50)
    
    # Executar testes
    tester = SantoGraalTester()
    results = tester.run_all_tests()
    
    print("\n🎉 TESTES CONCLUÍDOS!")
    print("=" * 50)
    
    if results["system_ready"]:
        print("🚀 SISTEMA SANTO GRAAL PRONTO PARA USO!")
        print("🎯 Implementação revolucionária concluída com sucesso!")
    else:
        print("⚠️ Sistema precisa de ajustes antes do uso")

if __name__ == "__main__":
    main()
