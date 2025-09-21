#!/usr/bin/env python3
"""
Patch para compatibilidade entre PyTorch 2.6 e Ultralytics YOLO
Resolve o problema de weights_only=True no torch.load
"""

import torch
import warnings

def apply_pytorch_patch():
    """Aplica patch para compatibilidade PyTorch 2.6 + Ultralytics"""
    
    # Patch 1: Configurar globals seguros
    torch.serialization.add_safe_globals([
        'ultralytics.nn.tasks.DetectionModel',
        'ultralytics.nn.tasks.ClassificationModel', 
        'ultralytics.nn.tasks.SegmentationModel',
        'ultralytics.nn.tasks.PoseModel',
        'collections.OrderedDict',
        'torch.nn.modules.conv.Conv2d',
        'torch.nn.modules.batchnorm.BatchNorm2d',
        'torch.nn.modules.activation.ReLU',
        'torch.nn.modules.pooling.MaxPool2d',
        'torch.nn.modules.linear.Linear'
    ])
    
    # Patch 2: Monkey patch torch.load globalmente
    original_load = torch.load
    
    def patched_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
        """Versão patcheada do torch.load que força weights_only=False"""
        try:
            # Sempre usar weights_only=False para compatibilidade
            return original_load(f, map_location=map_location, pickle_module=pickle_module, 
                               weights_only=False, **kwargs)
        except Exception as e:
            warnings.warn(f"Erro no carregamento com weights_only=False: {e}")
            # Fallback para o comportamento original
            return original_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)
    
    torch.load = patched_load
    
    # Patch 3: Configurar variáveis de ambiente
    import os
    os.environ['TORCH_WEIGHTS_ONLY'] = 'False'
    
    print("✅ Patch PyTorch aplicado com sucesso!")

if __name__ == "__main__":
    apply_pytorch_patch()
