#!/usr/bin/env python3
"""
Teste do Novo Sistema de Intuição Neuro-Simbólica
"""

import unittest
import os
import sys
from PIL import Image
import numpy as np

# Adicionar src ao path
sys.path.insert(0, 'src')

from core.intuition import IntuitionEngine
from utils.debug_logger import DebugLogger

class TestNewIntuitionSystem(unittest.TestCase):
    """Teste do novo sistema de intuição"""
    
    @classmethod
    def setUpClass(cls):
        """Configuração inicial"""
        cls.yolo_model_path = "yolov8n.pt"
        cls.keras_model_path = "modelo_classificacao_passaros.keras"
        cls.debug_logger = DebugLogger()
        cls.intuition_engine = IntuitionEngine(cls.yolo_model_path, cls.keras_model_path, cls.debug_logger)
        
        # Criar imagem de teste
        cls.test_image_path = "test_bird_image.jpg"
        img = Image.new('RGB', (224, 224), color='brown')
        img.save(cls.test_image_path)
    
    @classmethod
    def tearDownClass(cls):
        """Limpeza final"""
        if os.path.exists(cls.test_image_path):
            os.remove(cls.test_image_path)
    
    def test_intuition_engine_initialization(self):
        """Testa inicialização do motor de intuição"""
        print("\n🔧 Testando inicialização do motor de intuição...")
        
        self.assertIsNotNone(self.intuition_engine)
        # self.assertIsNotNone(self.intuition_engine.bird_characteristics)  # Removido - não existe no novo sistema
        self.assertIsNotNone(self.intuition_engine.learned_patterns)
        
        print("✅ Motor de intuição inicializado com sucesso!")
    
    def test_visual_analysis(self):
        """Testa análise visual"""
        print("\n👁️ Testando análise visual...")
        
        visual_analysis = self.intuition_engine._analyze_visual_characteristics(self.test_image_path)
        
        self.assertIsNotNone(visual_analysis)
        self.assertIn('dominant_color', visual_analysis)
        self.assertIn('bird_like_features', visual_analysis)
        
        print(f"  - Cor dominante: {visual_analysis.get('dominant_color', 'N/A')}")
        print(f"  - Score de características: {visual_analysis.get('bird_like_features', 0):.2%}")
        
        print("✅ Análise visual funcionando!")
    
    def test_fundamental_characteristics(self):
        """Testa detecção de características fundamentais"""
        print("\n🔍 Testando detecção de características fundamentais...")
        
        characteristics = self.intuition_engine._detect_fundamental_characteristics(self.test_image_path)
        
        self.assertIsNotNone(characteristics)
        self.assertIn('has_eyes', characteristics)
        self.assertIn('has_wings', characteristics)
        self.assertIn('has_feathers', characteristics)
        
        print(f"  - Características encontradas: {sum(characteristics.values())}")
        print(f"  - Tem olhos: {characteristics.get('has_eyes', False)}")
        print(f"  - Tem asas: {characteristics.get('has_wings', False)}")
        print(f"  - Tem penas: {characteristics.get('has_feathers', False)}")
        
        print("✅ Detecção de características funcionando!")
    
    def test_logical_reasoning(self):
        """Testa raciocínio lógico"""
        print("\n🧠 Testando raciocínio lógico...")
        
        visual_analysis = self.intuition_engine._analyze_visual_characteristics(self.test_image_path)
        characteristics = self.intuition_engine._detect_fundamental_characteristics(self.test_image_path)
        
        reasoning = self.intuition_engine._logical_reasoning(visual_analysis, characteristics)
        
        self.assertIsNotNone(reasoning)
        self.assertIn('is_bird', reasoning)
        self.assertIn('confidence', reasoning)
        self.assertIn('species', reasoning)
        self.assertIn('reasoning_steps', reasoning)
        
        print(f"  - É pássaro: {reasoning.get('is_bird', False)}")
        print(f"  - Confiança: {reasoning.get('confidence', 0):.2%}")
        print(f"  - Espécie: {reasoning.get('species', 'N/A')}")
        print(f"  - Passos de raciocínio: {len(reasoning.get('reasoning_steps', []))}")
        
        print("✅ Raciocínio lógico funcionando!")
    
    def test_learning_candidates(self):
        """Testa detecção de candidatos para aprendizado"""
        print("\n📚 Testando detecção de candidatos para aprendizado...")
        
        visual_analysis = self.intuition_engine._analyze_visual_characteristics(self.test_image_path)
        characteristics = self.intuition_engine._detect_fundamental_characteristics(self.test_image_path)
        reasoning = self.intuition_engine._logical_reasoning(visual_analysis, characteristics)
        
        candidates = self.intuition_engine._detect_learning_candidates(visual_analysis, characteristics, reasoning)
        
        self.assertIsNotNone(candidates)
        self.assertIsInstance(candidates, list)
        
        print(f"  - Candidatos encontrados: {len(candidates)}")
        for i, candidate in enumerate(candidates):
            print(f"    {i+1}. {candidate.type.value}: {candidate.confidence:.2%}")
        
        print("✅ Detecção de candidatos funcionando!")
    
    def test_full_analysis(self):
        """Testa análise completa"""
        print("\n🎯 Testando análise completa...")
        
        results = self.intuition_engine.analyze_image_intuition(self.test_image_path)
        
        self.assertIsNotNone(results)
        self.assertIn('confidence', results)
        self.assertIn('species', results)
        self.assertIn('color', results)
        self.assertIn('intuition_analysis', results)
        
        intuition_data = results['intuition_analysis']
        self.assertIn('candidates_found', intuition_data)
        self.assertIn('recommendation', intuition_data)
        
        print(f"  - Confiança geral: {results['confidence']:.2%}")
        print(f"  - Espécie: {results['species']}")
        print(f"  - Cor: {results['color']}")
        print(f"  - Candidatos: {intuition_data['candidates_found']}")
        print(f"  - Recomendação: {intuition_data['recommendation']}")
        
        print("✅ Análise completa funcionando!")
    
    def test_learning_feedback(self):
        """Testa aprendizado com feedback"""
        print("\n🧠 Testando aprendizado com feedback...")
        
        # Simular feedback humano
        human_feedback = {
            'is_bird': True,
            'species': 'Pássaro de teste',
            'confidence': 0.9,
            'reasoning': 'Teste de aprendizado'
        }
        
        # Aplicar aprendizado
        self.intuition_engine.learn_from_feedback(self.test_image_path, human_feedback)
        
        # Verificar estatísticas
        stats = self.intuition_engine.get_learning_statistics()
        
        self.assertIsNotNone(stats)
        self.assertIn('known_species_count', stats)
        self.assertIn('total_learning_events', stats)
        
        print(f"  - Espécies conhecidas: {stats['known_species_count']}")
        print(f"  - Eventos de aprendizado: {stats['total_learning_events']}")
        
        print("✅ Aprendizado com feedback funcionando!")

if __name__ == '__main__':
    print("🧪 TESTE DO NOVO SISTEMA DE INTUIÇÃO NEURO-SIMBÓLICA")
    print("=" * 60)
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
