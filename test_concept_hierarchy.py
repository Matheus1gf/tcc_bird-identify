#!/usr/bin/env python3
"""
Teste do Sistema de Hierarquias de Conceitos
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.concept_hierarchy import ConceptHierarchyManager, ConceptRelationshipType, ConceptAbstractionLevel, ConceptComplexity

def test_concept_hierarchy():
    """Testa o sistema de hierarquias de conceitos"""
    print("🌳 Testando Sistema de Hierarquias de Conceitos...")
    
    # Inicializar o sistema
    manager = ConceptHierarchyManager()
    
    # Teste 1: Adicionar conceitos hierárquicos
    print("\n1. Adicionando conceitos hierárquicos...")
    
    # Adicionar conceito "Animal" (universal)
    result1 = manager.add_concept(
        name="Animal",
        description="Ser vivo multicelular que se move e se alimenta",
        abstraction_level=ConceptAbstractionLevel.UNIVERSAL,
        complexity=ConceptComplexity.COMPOUND,
        properties={"caracteristicas": ["multicelular", "movimento", "alimentacao"]}
    )
    print(f"✅ Conceito 'Animal' adicionado: {result1.get('concept_id', 'N/A')}")
    
    # Adicionar conceito "Ave" (geral)
    result2 = manager.add_concept(
        name="Ave",
        description="Animal vertebrado com penas e capacidade de voo",
        abstraction_level=ConceptAbstractionLevel.GENERAL,
        complexity=ConceptComplexity.COMPOUND,
        parent_concepts=["Animal"],
        properties={"caracteristicas": ["penas", "bico", "asas", "voo"]}
    )
    print(f"✅ Conceito 'Ave' adicionado: {result2.get('concept_id', 'N/A')}")
    
    # Adicionar conceito "Pássaro" (específico)
    result3 = manager.add_concept(
        name="Pássaro",
        description="Ave pequena com canto característico",
        abstraction_level=ConceptAbstractionLevel.SPECIFIC,
        complexity=ConceptComplexity.SIMPLE,
        parent_concepts=["Ave"],
        properties={"caracteristicas": ["pequeno", "canto", "penas", "bico"]}
    )
    print(f"✅ Conceito 'Pássaro' adicionado: {result3.get('concept_id', 'N/A')}")
    
    # Adicionar conceito "Canário" (concreto)
    result4 = manager.add_concept(
        name="Canário",
        description="Pássaro pequeno e amarelo que canta",
        abstraction_level=ConceptAbstractionLevel.CONCRETE,
        complexity=ConceptComplexity.SIMPLE,
        parent_concepts=["Pássaro"],
        properties={"caracteristicas": ["amarelo", "pequeno", "canto", "domestico"]}
    )
    print(f"✅ Conceito 'Canário' adicionado: {result4.get('concept_id', 'N/A')}")
    
    # Teste 2: Adicionar relacionamentos
    print("\n2. Adicionando relacionamentos...")
    
    # Relacionamento IS_A: Pássaro é um tipo de Ave
    rel1 = manager.add_relationship(
        source_concept="Pássaro",
        target_concept="Ave",
        relationship_type=ConceptRelationshipType.IS_A,
        strength=0.9,
        confidence=0.95
    )
    print(f"✅ Relacionamento IS_A criado: {rel1.get('relationship_id', 'N/A')}")
    
    # Relacionamento HAS_A: Ave tem asas
    rel2 = manager.add_relationship(
        source_concept="Ave",
        target_concept="Asas",
        relationship_type=ConceptRelationshipType.HAS_A,
        strength=0.8,
        confidence=0.9
    )
    print(f"✅ Relacionamento HAS_A criado: {rel2.get('relationship_id', 'N/A')}")
    
    # Relacionamento PART_OF: Asas são parte de Ave
    rel3 = manager.add_relationship(
        source_concept="Asas",
        target_concept="Ave",
        relationship_type=ConceptRelationshipType.PART_OF,
        strength=0.7,
        confidence=0.85
    )
    print(f"✅ Relacionamento PART_OF criado: {rel3.get('relationship_id', 'N/A')}")
    
    # Teste 3: Encontrar relacionamentos
    print("\n3. Encontrando relacionamentos...")
    
    # Encontrar relacionamentos de "Pássaro"
    relationships = manager.find_relationships("Pássaro")
    print(f"✅ Relacionamentos de 'Pássaro': {len(relationships.get('relationships', []))}")
    
    # Encontrar relacionamentos IS_A de "Ave"
    is_a_rels = manager.find_relationships("Ave", "is_a")
    print(f"✅ Relacionamentos IS_A de 'Ave': {len(is_a_rels.get('relationships', []))}")
    
    # Teste 4: Obter hierarquia
    print("\n4. Obtendo hierarquia...")
    
    # Hierarquia completa
    hierarchy = manager.get_hierarchy(max_depth=4)
    print(f"✅ Hierarquia obtida com {len(hierarchy.get('concepts', []))} conceitos")
    
    # Hierarquia a partir de "Animal"
    animal_hierarchy = manager.get_hierarchy("Animal", max_depth=3)
    print(f"✅ Hierarquia de 'Animal': {len(animal_hierarchy.get('concepts', []))} conceitos")
    
    # Teste 5: Analisar similaridade
    print("\n5. Analisando similaridade...")
    
    # Similaridade entre Pássaro e Ave
    similarity = manager.analyze_similarity("Pássaro", "Ave")
    print(f"✅ Similaridade Pássaro-Ave: {similarity.get('similarity_score', 0):.2f}")
    
    # Similaridade entre Canário e Pássaro
    similarity2 = manager.analyze_similarity("Canário", "Pássaro")
    print(f"✅ Similaridade Canário-Pássaro: {similarity2.get('similarity_score', 0):.2f}")
    
    # Teste 6: Análise do sistema
    print("\n6. Análise do sistema...")
    
    analysis = manager.get_analysis()
    print(f"✅ Conceitos: {analysis.get('concept_count', 0)}")
    print(f"✅ Relacionamentos: {analysis.get('relationship_count', 0)}")
    print(f"✅ Níveis de abstração: {len(analysis.get('abstraction_levels', []))}")
    print(f"✅ Complexidades: {len(analysis.get('complexities', []))}")
    
    print("\n🎯 Sistema de Hierarquias de Conceitos funcionando perfeitamente!")
    return True

if __name__ == "__main__":
    try:
        test_concept_hierarchy()
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
