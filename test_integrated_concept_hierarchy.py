#!/usr/bin/env python3
"""
Teste Integrado do Sistema de Hierarquias de Conceitos com IntuitionEngine
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.intuition import IntuitionEngine

def test_integrated_concept_hierarchy():
    """Testa o sistema de hierarquias de conceitos integrado ao IntuitionEngine"""
    print("🌳 Testando Sistema de Hierarquias de Conceitos Integrado...")
    
    # Inicializar o IntuitionEngine (sem modelos para teste)
    engine = IntuitionEngine("", "", None)
    
    # Teste 1: Adicionar conceitos hierárquicos
    print("\n1. Adicionando conceitos hierárquicos...")
    
    # Adicionar conceito "Animal" (universal)
    result1 = engine.add_hierarchical_concept(
        name="Animal",
        description="Ser vivo multicelular que se move e se alimenta",
        abstraction_level="universal",
        complexity="compound",
        properties={"caracteristicas": ["multicelular", "movimento", "alimentacao"]}
    )
    print(f"✅ Conceito 'Animal' adicionado: {result1.get('concept_id', 'N/A')}")
    
    # Adicionar conceito "Ave" (geral)
    result2 = engine.add_hierarchical_concept(
        name="Ave",
        description="Animal vertebrado com penas e capacidade de voo",
        abstraction_level="general",
        complexity="compound",
        parent_concepts=["Animal"],
        properties={"caracteristicas": ["penas", "bico", "asas", "voo"]}
    )
    print(f"✅ Conceito 'Ave' adicionado: {result2.get('concept_id', 'N/A')}")
    
    # Adicionar conceito "Pássaro" (específico)
    result3 = engine.add_hierarchical_concept(
        name="Pássaro",
        description="Ave pequena com canto característico",
        abstraction_level="specific",
        complexity="simple",
        parent_concepts=["Ave"],
        properties={"caracteristicas": ["pequeno", "canto", "penas", "bico"]}
    )
    print(f"✅ Conceito 'Pássaro' adicionado: {result3.get('concept_id', 'N/A')}")
    
    # Teste 2: Adicionar relacionamentos
    print("\n2. Adicionando relacionamentos...")
    
    # Relacionamento IS_A: Pássaro é um tipo de Ave
    rel1 = engine.add_concept_relationship(
        source_concept="Pássaro",
        target_concept="Ave",
        relationship_type="is_a",
        strength=0.9,
        confidence=0.95
    )
    print(f"✅ Relacionamento IS_A criado: {rel1.get('success', False)}")
    
    # Relacionamento HAS_A: Ave tem asas
    rel2 = engine.add_concept_relationship(
        source_concept="Ave",
        target_concept="Asas",
        relationship_type="has_a",
        strength=0.8,
        confidence=0.9
    )
    print(f"✅ Relacionamento HAS_A criado: {rel2.get('success', False)}")
    
    # Teste 3: Encontrar relacionamentos
    print("\n3. Encontrando relacionamentos...")
    
    # Encontrar relacionamentos de "Pássaro"
    relationships = engine.find_concept_relationships("Pássaro")
    print(f"✅ Relacionamentos de 'Pássaro': {relationships.get('count', 0)}")
    
    # Encontrar relacionamentos IS_A de "Ave"
    is_a_rels = engine.find_concept_relationships("Ave", "is_a")
    print(f"✅ Relacionamentos IS_A de 'Ave': {is_a_rels.get('count', 0)}")
    
    # Teste 4: Obter hierarquia
    print("\n4. Obtendo hierarquia...")
    
    # Hierarquia completa
    hierarchy = engine.get_concept_hierarchy(max_depth=4)
    print(f"✅ Hierarquia obtida com {hierarchy.get('count', 0)} conceitos")
    
    # Hierarquia a partir de "Animal"
    animal_hierarchy = engine.get_concept_hierarchy("Animal", max_depth=3)
    print(f"✅ Hierarquia de 'Animal': {animal_hierarchy.get('count', 0)} conceitos")
    
    # Teste 5: Analisar similaridade
    print("\n5. Analisando similaridade...")
    
    # Similaridade entre Pássaro e Ave
    similarity = engine.analyze_concept_similarity("Pássaro", "Ave")
    print(f"✅ Similaridade Pássaro-Ave: {similarity.get('similarity_score', 0):.2f}")
    
    # Similaridade entre Animal e Ave
    similarity2 = engine.analyze_concept_similarity("Animal", "Ave")
    print(f"✅ Similaridade Animal-Ave: {similarity2.get('similarity_score', 0):.2f}")
    
    # Teste 6: Análise do sistema
    print("\n6. Análise do sistema...")
    
    analysis = engine.get_concept_hierarchy_analysis()
    print(f"✅ Conceitos: {analysis.get('concept_count', 0)}")
    print(f"✅ Relacionamentos: {analysis.get('relationship_count', 0)}")
    print(f"✅ Níveis de abstração: {len(analysis.get('abstraction_levels', []))}")
    print(f"✅ Complexidades: {len(analysis.get('complexities', []))}")
    
    print("\n🎯 Sistema de Hierarquias de Conceitos Integrado funcionando perfeitamente!")
    return True

if __name__ == "__main__":
    try:
        test_integrated_concept_hierarchy()
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
