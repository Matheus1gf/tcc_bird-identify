"""
Patch para resolver problemas de compatibilidade com typing_extensions
"""
import sys

def apply_typing_patch():
    """Aplica patch para resolver problemas de TypeIs"""
    try:
        # Tentar importar TypeIs
        from typing_extensions import TypeIs
        return True
    except ImportError:
        # Se não conseguir importar, criar um fallback
        try:
            from typing import TypeGuard
            # Criar um alias para TypeIs usando TypeGuard
            import typing_extensions
            typing_extensions.TypeIs = TypeGuard
            return True
        except ImportError:
            # Se nem TypeGuard estiver disponível, criar um stub
            import typing_extensions
            def stub_type_is(obj):
                return True
            typing_extensions.TypeIs = stub_type_is
            return True

if __name__ == "__main__":
    apply_typing_patch()
