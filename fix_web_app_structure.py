#!/usr/bin/env python3
"""
Script para corrigir automaticamente a estrutura do arquivo web_app.py
"""

import re
import ast

def fix_web_app_structure():
    print("🔧 CORRIGINDO ESTRUTURA DO WEB_APP.PY")
    print("=" * 60)
    
    # Ler o arquivo
    with open('src/interfaces/web_app.py', 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    print(f"📄 Total de linhas: {len(lines)}")
    
    # 1. Corrigir imports duplicados e problemáticos
    print("\n1️⃣ CORRIGINDO IMPORTS...")
    
    # Remover imports duplicados
    seen_imports = set()
    new_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            if stripped not in seen_imports:
                seen_imports.add(stripped)
                new_lines.append(line)
            else:
                print(f"   Removendo import duplicado: {stripped}")
        else:
            new_lines.append(line)
    
    lines = new_lines
    print(f"   ✅ Imports duplicados removidos")
    
    # 2. Corrigir estrutura de blocos
    print("\n2️⃣ CORRIGINDO ESTRUTURA DE BLOCOS...")
    
    # Encontrar onde está o problema principal
    problem_start = None
    for i, line in enumerate(lines):
        if 'BOTÃO PRINCIPAL - FORA DE QUALQUER LÓGICA CONDICIONAL' in line:
            problem_start = i
            break
    
    if problem_start:
        print(f"   📍 Problema encontrado na linha {problem_start + 1}")
        
        # Verificar indentação da linha anterior
        prev_line = lines[problem_start - 1] if problem_start > 0 else ""
        prev_indent = len(prev_line) - len(prev_line.lstrip())
        
        print(f"   📏 Indentação da linha anterior: {prev_indent}")
        
        # Corrigir indentação das linhas problemáticas
        for i in range(problem_start, min(problem_start + 20, len(lines))):
            line = lines[i]
            if line.strip() and not line.startswith('#'):
                # Se a linha não está vazia e não é comentário
                current_indent = len(line) - len(line.lstrip())
                if current_indent > 0:
                    # Reduzir indentação para 0 (nível base)
                    lines[i] = line.lstrip()
                    print(f"   🔧 Linha {i+1}: indentação corrigida")
    
    # 3. Corrigir blocos try/except mal estruturados
    print("\n3️⃣ CORRIGINDO BLOCOS TRY/EXCEPT...")
    
    # Encontrar blocos try sem except correspondente
    try_blocks = []
    except_blocks = []
    
    for i, line in enumerate(lines):
        if 'try:' in line:
            try_blocks.append(i)
        elif 'except' in line:
            except_blocks.append(i)
    
    print(f"   📊 Try blocks: {len(try_blocks)}")
    print(f"   📊 Except blocks: {len(except_blocks)}")
    
    # 4. Corrigir indentação geral
    print("\n4️⃣ CORRIGINDO INDENTAÇÃO GERAL...")
    
    fixed_lines = []
    indent_stack = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if not stripped or stripped.startswith('#'):
            # Linha vazia ou comentário - manter como está
            fixed_lines.append(line)
            continue
        
        # Calcular indentação esperada
        if stripped.endswith(':'):
            # Linha que abre bloco
            if 'if ' in stripped or 'for ' in stripped or 'while ' in stripped or 'try:' in stripped or 'except' in stripped or 'def ' in stripped or 'class ' in stripped:
                expected_indent = len(indent_stack) * 4
                indent_stack.append(True)
            else:
                expected_indent = len(indent_stack) * 4
        else:
            # Linha normal
            expected_indent = len(indent_stack) * 4
            
            # Verificar se deve fechar blocos
            if stripped.startswith('else:') or stripped.startswith('elif ') or stripped.startswith('except') or stripped.startswith('finally:'):
                if indent_stack:
                    indent_stack.pop()
                expected_indent = len(indent_stack) * 4
        
        # Aplicar indentação
        if expected_indent > 0:
            fixed_lines.append(' ' * expected_indent + stripped)
        else:
            fixed_lines.append(stripped)
    
    # 5. Salvar arquivo corrigido
    print("\n5️⃣ SALVANDO ARQUIVO CORRIGIDO...")
    
    with open('src/interfaces/web_app.py', 'w') as f:
        f.write('\n'.join(fixed_lines))
    
    print("   ✅ Arquivo salvo")
    
    # 6. Verificar se está correto
    print("\n6️⃣ VERIFICANDO CORREÇÃO...")
    
    try:
        with open('src/interfaces/web_app.py', 'r') as f:
            content = f.read()
        ast.parse(content)
        print("   ✅ Sintaxe corrigida com sucesso!")
        return True
    except SyntaxError as e:
        print(f"   ❌ Ainda há erro de sintaxe na linha {e.lineno}: {e.msg}")
        return False
    except IndentationError as e:
        print(f"   ❌ Ainda há erro de indentação na linha {e.lineno}: {e.msg}")
        return False

if __name__ == "__main__":
    success = fix_web_app_structure()
    if success:
        print("\n🎉 CORREÇÃO CONCLUÍDA COM SUCESSO!")
    else:
        print("\n❌ CORREÇÃO FALHOU - INTERVENÇÃO MANUAL NECESSÁRIA")
