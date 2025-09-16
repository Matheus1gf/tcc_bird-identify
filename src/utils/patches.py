#!/usr/bin/env python3
"""
Patch para resolver problema do PyTorch 2.6 com YOLO
"""

import torch
import ultralytics.nn.tasks
import ultralytics.nn.modules.conv

def apply_yolo_patch():
    """Aplica patch para resolver problema do PyTorch 2.6"""
    try:
        # Usar weights_only=False para resolver problema do PyTorch 2.6
        import functools
        
        # Salvar função original
        original_torch_load = torch.load
        
        # Criar nova função com weights_only=False
        def patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        # Aplicar patch
        torch.load = patched_torch_load
        
        print("✅ Patch YOLO aplicado com sucesso!")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao aplicar patch YOLO: {e}")
        return False

if __name__ == "__main__":
    apply_yolo_patch()
