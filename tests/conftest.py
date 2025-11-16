import os
import sys
import types
from pathlib import Path
from typing import Tuple
from enum import Enum

from PIL import Image
import pytest


def create_temp_image(path: Path, size: Tuple[int, int] = (64, 64), color=(128, 128, 128)) -> str:
    """
    Cria e salva uma imagem simples para uso nos testes.

    Args:
        path: caminho completo do arquivo (incluindo nome).
        size: tamanho (largura, altura).
        color: cor RGB sólida.

    Returns:
        Caminho como string.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    image.save(path)
    return str(path)


@pytest.fixture
def temp_image(tmp_path):
    """Fixture que gera uma imagem temporária e retorna o caminho."""
    img_path = tmp_path / "imagem_teste.jpg"
    return create_temp_image(img_path)


def ensure_lightweight_intuition_stub():
    """
    Alguns módulos core importam src.core.intuition, que é muito pesado e possui
    dependências nem sempre necessárias nos testes. Este helper injeta um stub
    simples para permitir importar os módulos sem executar o código real.
    """
    if "src.core.intuition" in sys.modules:
        return

    stub_module = types.ModuleType("src.core.intuition")

    class LearningCandidateType(Enum):
        VISUAL_ANALYSIS = "visual_analysis"
        SPECIES_UNKNOWN = "species_unknown"
        CHARACTERISTIC_LEARNING = "characteristic_learning"

    class LearningCandidate:
        def __init__(self, *args, **kwargs):
            pass

    class IntuitionEngine:
        def __init__(self, *args, **kwargs):
            pass

        def analyze_image_intuition(self, *_args, **_kwargs):
            return {}

    stub_module.IntuitionEngine = IntuitionEngine
    stub_module.LearningCandidate = LearningCandidate
    stub_module.LearningCandidateType = LearningCandidateType

    sys.modules["src.core.intuition"] = stub_module
