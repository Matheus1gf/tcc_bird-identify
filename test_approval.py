#!/usr/bin/env python3
"""
Teste simples do sistema de aprovação
"""

import os
from manual_analysis_system import ManualAnalysisSystem

def test_approval_system():
    """Testa o sistema de aprovação"""
    print("=== TESTE DO SISTEMA DE APROVAÇÃO ===")
    
    # Inicializar sistema
    manual_analysis = ManualAnalysisSystem()
    
    # Verificar imagens pendentes
    pending_images = manual_analysis.get_pending_images()
    print(f"Imagens pendentes: {len(pending_images)}")
    
    if pending_images:
        # Pegar primeira imagem
        first_image = pending_images[0]
        print(f"Primeira imagem: {first_image['filename']}")
        
        # Tentar aprovar
        try:
            approved_path = manual_analysis.approve_image(
                first_image['filename'],
                "generic_bird",
                0.8,
                "Teste de aprovação"
            )
            print(f"✅ Aprovação bem-sucedida: {approved_path}")
            
            # Verificar se foi movida
            if os.path.exists(approved_path):
                print("✅ Arquivo existe na pasta approved")
            else:
                print("❌ Arquivo não encontrado na pasta approved")
                
        except Exception as e:
            print(f"❌ Erro na aprovação: {e}")
    else:
        print("❌ Nenhuma imagem pendente para testar")

if __name__ == "__main__":
    test_approval_system()
