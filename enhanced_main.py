import cv2
import numpy as np
import os
import tensorflow as tf
import logging
import json
from ultralytics import YOLO
from typing import Dict, List, Optional

# Importar novos módulos
from knowledge_graph import KnowledgeGraph, NodeType, RelationType
from gradcam_module import AutoAnnotationSystem, GradCAM
from continuous_learning import ContinuousLearningSystem, ExternalValidator
from innovation_module import InnovationEngine, InnovationGoal, SpeciesBlueprint

logging.basicConfig(level=logging.INFO)

# --- CONFIGURAÇÃO ---
CLASSIFIER_MODEL_PATH = 'modelo_classificacao_passaros.keras'
YOLO_MODEL_PATH = 'runs/detect/train/weights/best.pt'
CLASSES_FILE_PATH = 'data/classes.txt'
KNOWLEDGE_BASE_PATH = 'base_conhecimento.json'
CONFIDENCE_THRESHOLD = 0.60

# Configurações para novos módulos
KNOWLEDGE_GRAPH_PATH = 'knowledge_graph.json'
LEARNING_DATA_PATH = './learning_data'
API_KEY_GEMINI = None  # Configure sua chave da API
API_KEY_GPT4V = None   # Configure sua chave da API

class EnhancedBirdIdentificationSystem:
    """
    Sistema aprimorado de identificação de pássaros com IA Neuro-Simbólica
    Integra todos os módulos desenvolvidos
    """
    
    def __init__(self):
        """Inicializa o sistema completo"""
        
        # Carregar modelos existentes
        self._load_models()
        
        # Inicializar novos módulos
        self._initialize_modules()
        
        # Carregar base de conhecimento existente
        self._load_existing_knowledge()
    
    def _load_models(self):
        """Carrega modelos de classificação e detecção"""
        
        logging.info("Carregando modelo de classificação de espécies...")
        self.classification_model = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH)
        
        logging.info(f"Carregando modelo de detecção de partes (YOLOv8) de '{YOLO_MODEL_PATH}'...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        
        logging.info("Modelos carregados com sucesso!")
    
    def _initialize_modules(self):
        """Inicializa novos módulos do sistema"""
        
        # Grafo de Conhecimento
        self.knowledge_graph = KnowledgeGraph()
        
        # Sistema de Auto-Anotação
        self.auto_annotation_system = AutoAnnotationSystem(
            self.classification_model,
            self.yolo_model,
            CONFIDENCE_THRESHOLD
        )
        
        # Validador Externo (configurar API keys)
        self.external_validator = None
        if API_KEY_GEMINI:
            self.external_validator = ExternalValidator("gemini", API_KEY_GEMINI)
        elif API_KEY_GPT4V:
            self.external_validator = ExternalValidator("gpt4v", API_KEY_GPT4V)
        
        # Sistema de Aprendizado Contínuo
        self.continuous_learning = ContinuousLearningSystem(
            self.auto_annotation_system,
            self.knowledge_graph,
            self.external_validator,
            LEARNING_DATA_PATH
        )
        
        # Motor de Inovação
        self.innovation_engine = InnovationEngine(self.knowledge_graph)
        
        logging.info("Módulos avançados inicializados!")
    
    def _load_existing_knowledge(self):
        """Carrega conhecimento existente da base atual"""
        
        if os.path.exists(KNOWLEDGE_BASE_PATH):
            with open(KNOWLEDGE_BASE_PATH, 'r') as f:
                existing_knowledge = json.load(f)
            
            # Migrar espécies conhecidas para o grafo
            for species_name, parts in existing_knowledge.get("especies_conhecidas", {}).items():
                self.knowledge_graph.add_species_from_analysis(
                    species_name, parts, confidence=0.8
                )
            
            logging.info("Conhecimento existente migrado para o grafo!")
    
    def analyze_image_enhanced(self, image_path: str, 
                              enable_continuous_learning: bool = True,
                              generate_innovation_suggestions: bool = False) -> Dict:
        """
        Análise aprimorada de imagem com todos os módulos integrados
        
        Args:
            image_path: Caminho para a imagem
            enable_continuous_learning: Se deve usar aprendizado contínuo
            generate_innovation_suggestions: Se deve gerar sugestões de inovação
            
        Returns:
            Análise completa da imagem
        """
        
        # Análise básica (método original)
        basic_analysis = self._analyze_image_basic(image_path)
        
        # Análise com Grad-CAM
        gradcam_analysis = self.auto_annotation_system.analyze_image(image_path)
        
        # Consultar grafo de conhecimento
        knowledge_insights = self._query_knowledge_graph(gradcam_analysis)
        
        # Aprendizado contínuo se habilitado
        learning_sample = None
        if enable_continuous_learning:
            learning_sample = self.continuous_learning.process_new_image(
                image_path, auto_validate=False  # Não validar automaticamente por padrão
            )
        
        # Sugestões de inovação se solicitadas
        innovation_suggestions = []
        if generate_innovation_suggestions:
            innovation_suggestions = self._generate_innovation_suggestions(
                gradcam_analysis
            )
        
        # Compilar análise completa
        enhanced_analysis = {
            "basic_analysis": basic_analysis,
            "gradcam_analysis": gradcam_analysis,
            "knowledge_insights": knowledge_insights,
            "learning_sample": learning_sample,
            "innovation_suggestions": innovation_suggestions,
            "system_recommendations": self._generate_system_recommendations(
                basic_analysis, gradcam_analysis, knowledge_insights
            )
        }
        
        return enhanced_analysis
    
    def _analyze_image_basic(self, image_path: str) -> Dict:
        """Análise básica usando o método original"""
        
        imagem = cv2.imread(image_path)
        if imagem is None:
            return {"error": f"Erro ao carregar a imagem: {image_path}"}
        
        # Detectar partes anatômicas
        fatos_visuais = self._extract_facts_with_yolo(imagem)
        
        # Verificar se é pássaro
        is_bird = 'bico' in fatos_visuais or ('corpo' in fatos_visuais and 'asa' in fatos_visuais)
        
        if not is_bird:
            return {
                "is_bird": False,
                "detected_facts": list(fatos_visuais),
                "conclusion": "Não foi possível confirmar que é um pássaro"
            }
        
        # Classificar espécie
        id_especie, confianca = self._identify_species(imagem)
        nomes_classes = self._load_class_names()
        nome_hipotese = nomes_classes[id_especie]
        
        return {
            "is_bird": True,
            "detected_facts": list(fatos_visuais),
            "species_hypothesis": nome_hipotese,
            "confidence": confianca,
            "is_known_species": confianca >= CONFIDENCE_THRESHOLD
        }
    
    def _extract_facts_with_yolo(self, imagem):
        """Extrai fatos visuais usando YOLO"""
        results = self.yolo_model(imagem, verbose=False)
        fatos_detectados = set()
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls)
                class_name = self.yolo_model.names[class_id]
                fatos_detectados.add(class_name)
        return fatos_detectados
    
    def _identify_species(self, imagem):
        """Identifica espécie usando modelo de classificação"""
        img_resized = cv2.resize(imagem, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0) / 255.0
        predicao = self.classification_model.predict(img_array, verbose=0)
        id_classe = np.argmax(predicao[0])
        confianca = np.max(predicao[0])
        return id_classe, confianca
    
    def _load_class_names(self):
        """Carrega nomes das classes"""
        with open(CLASSES_FILE_PATH, 'r') as f:
            nomes = [' '.join(line.strip().split()[1:]).replace('_', ' ') for line in f]
        return nomes
    
    def _query_knowledge_graph(self, gradcam_analysis: Dict) -> Dict:
        """Consulta grafo de conhecimento para insights"""
        
        detected_parts = []
        for detection in gradcam_analysis.get("yolo_detections", []):
            detected_parts.append(detection["class"])
        
        # Consultar espécies similares
        similar_species = self.knowledge_graph.query_similar_species(detected_parts)
        
        # Predizer partes faltantes
        predicted_species = gradcam_analysis.get("species_prediction", {}).get("species", "")
        missing_parts = self.knowledge_graph.predict_missing_parts(
            predicted_species, detected_parts
        )
        
        return {
            "similar_species": similar_species,
            "missing_parts": missing_parts,
            "knowledge_graph_stats": self.knowledge_graph.get_graph_statistics()
        }
    
    def _generate_innovation_suggestions(self, gradcam_analysis: Dict) -> List[Dict]:
        """Gera sugestões de inovação baseadas na análise"""
        
        suggestions = []
        
        # Analisar características detectadas
        detected_parts = []
        for detection in gradcam_analysis.get("yolo_detections", []):
            detected_parts.append(detection["class"])
        
        # Gerar blueprints para diferentes metas
        goals_to_try = [
            InnovationGoal.OPTIMIZE_FLIGHT,
            InnovationGoal.ENHANCE_HUNTING,
            InnovationGoal.IMPROVE_NAVIGATION
        ]
        
        for goal in goals_to_try:
            blueprint = self.innovation_engine.generate_blueprint(goal)
            
            # Verificar se blueprint é relevante para partes detectadas
            if any(part in detected_parts for part in blueprint.required_parts):
                suggestions.append({
                    "goal": goal.value,
                    "blueprint": blueprint,
                    "relevance_score": self._calculate_relevance_score(
                        blueprint, detected_parts
                    )
                })
        
        # Ordenar por relevância
        suggestions.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return suggestions[:3]  # Retornar top 3
    
    def _calculate_relevance_score(self, blueprint: SpeciesBlueprint, 
                                  detected_parts: List[str]) -> float:
        """Calcula score de relevância do blueprint"""
        
        overlap = len(set(blueprint.required_parts) & set(detected_parts))
        total_required = len(blueprint.required_parts)
        
        if total_required == 0:
            return 0.0
        
        return overlap / total_required
    
    def _generate_system_recommendations(self, basic_analysis: Dict, 
                                        gradcam_analysis: Dict,
                                        knowledge_insights: Dict) -> List[str]:
        """Gera recomendações do sistema"""
        
        recommendations = []
        
        # Recomendações baseadas na análise básica
        if not basic_analysis.get("is_bird", False):
            recommendations.append("Imagem não contém pássaro identificável")
            return recommendations
        
        # Recomendações baseadas em confiança
        confidence = basic_analysis.get("confidence", 0)
        if confidence < CONFIDENCE_THRESHOLD:
            recommendations.append("Baixa confiança na classificação - considerar validação humana")
        
        # Recomendações baseadas em espécies similares
        similar_species = knowledge_insights.get("similar_species", [])
        if similar_species:
            recommendations.append(f"Espécies similares encontradas: {[s[0] for s in similar_species[:3]]}")
        
        # Recomendações baseadas em partes faltantes
        missing_parts = knowledge_insights.get("missing_parts", [])
        if missing_parts:
            recommendations.append(f"Partes que podem estar presentes: {missing_parts}")
        
        return recommendations
    
    def process_directory_enhanced(self, directory: str, 
                                  enable_learning: bool = True,
                                  generate_innovation: bool = False) -> Dict:
        """
        Processa diretório completo com análise aprimorada
        
        Args:
            directory: Diretório com imagens
            enable_learning: Habilitar aprendizado contínuo
            generate_innovation: Gerar sugestões de inovação
            
        Returns:
            Relatório completo do processamento
        """
        
        results = []
        total_images = 0
        processed_images = 0
        
        for arquivo in os.listdir(directory):
            if arquivo.lower().endswith((".jpeg", ".jpg", ".png")):
                total_images += 1
                image_path = os.path.join(directory, arquivo)
                
                try:
                    analysis = self.analyze_image_enhanced(
                        image_path, 
                        enable_continuous_learning=enable_learning,
                        generate_innovation_suggestions=generate_innovation
                    )
                    
                    results.append({
                        "image": arquivo,
                        "analysis": analysis
                    })
                    processed_images += 1
                    
                except Exception as e:
                    logging.error(f"Erro ao processar {arquivo}: {e}")
                    results.append({
                        "image": arquivo,
                        "error": str(e)
                    })
        
        # Gerar relatório de aprendizado se habilitado
        learning_report = None
        if enable_learning:
            learning_report = self.continuous_learning.generate_learning_report()
        
        # Salvar grafo de conhecimento atualizado
        self.knowledge_graph.save_graph(KNOWLEDGE_GRAPH_PATH)
        
        return {
            "total_images": total_images,
            "processed_images": processed_images,
            "results": results,
            "learning_report": learning_report,
            "knowledge_graph_stats": self.knowledge_graph.get_graph_statistics()
        }
    
    def generate_innovation_blueprint(self, goal: str, 
                                     target_habitat: str = None) -> SpeciesBlueprint:
        """
        Gera blueprint de inovação para meta específica
        
        Args:
            goal: Meta de inovação (otimizar_voo, melhorar_caca, etc.)
            target_habitat: Habitat alvo
            
        Returns:
            Blueprint da nova espécie
        """
        
        goal_mapping = {
            "otimizar_voo": InnovationGoal.OPTIMIZE_FLIGHT,
            "melhorar_camuflagem": InnovationGoal.IMPROVE_CAMOUFLAGE,
            "melhorar_caca": InnovationGoal.ENHANCE_HUNTING,
            "adaptar_habitat": InnovationGoal.ADAPT_HABITAT,
            "aumentar_velocidade": InnovationGoal.INCREASE_SPEED,
            "melhorar_navegacao": InnovationGoal.IMPROVE_NAVIGATION
        }
        
        innovation_goal = goal_mapping.get(goal, InnovationGoal.OPTIMIZE_FLIGHT)
        
        return self.innovation_engine.generate_blueprint(innovation_goal, target_habitat=target_habitat)

# --- FLUXO DE EXECUÇÃO APRIMORADO ---
if __name__ == "__main__":
    # Inicializar sistema aprimorado
    system = EnhancedBirdIdentificationSystem()
    
    # Processar diretório de teste com todas as funcionalidades
    diretorio_de_teste = './dataset_teste'
    
    if not os.path.exists(diretorio_de_teste):
        print(f"\nAVISO: O diretório de teste '{diretorio_de_teste}' não foi encontrado.")
    else:
        print("Iniciando análise aprimorada...")
        
        # Processar com todas as funcionalidades habilitadas
        results = system.process_directory_enhanced(
            diretorio_de_teste,
            enable_learning=True,
            generate_innovation=True
        )
        
        print(f"\nProcessamento concluído!")
        print(f"Imagens processadas: {results['processed_images']}/{results['total_images']}")
        print(f"Estatísticas do grafo de conhecimento: {results['knowledge_graph_stats']}")
        
        # Salvar resultados
        with open('enhanced_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("Resultados salvos em 'enhanced_analysis_results.json'")
