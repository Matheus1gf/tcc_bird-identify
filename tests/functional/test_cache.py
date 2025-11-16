from pathlib import Path

from tests.conftest import create_temp_image
from src.core.cache import ImageRecognitionCache


def test_cache_add_and_retrieve_recognition(tmp_path):
    cache_file = tmp_path / "cache.json"
    image_file = tmp_path / "ave.jpg"
    image_path = create_temp_image(image_file)

    cache = ImageRecognitionCache(cache_file=str(cache_file))
    analysis_data = {
        "keras_analysis": {"species": "Ave Azul", "confidence": 0.87},
        "yolo_analysis": {"detections": []},
    }

    cache.add_recognized_image(
        image_path=image_path,
        species="Ave Azul",
        confidence=0.87,
        analysis_data=analysis_data,
        notes="Teste automatizado",
    )

    result = cache.is_image_recognized(image_path)
    assert result is not None
    assert result["species"] == "Ave Azul"
    assert result["confidence"] == 0.87

    stats = cache.get_species_statistics()
    assert "Ave Azul" in stats
    assert stats["Ave Azul"]["count"] == 1


def test_cache_records_rejection(tmp_path):
    cache_file = tmp_path / "cache.json"
    image_file = tmp_path / "nao_passaros.jpg"
    image_path = create_temp_image(image_file, color=(200, 0, 0))

    cache = ImageRecognitionCache(cache_file=str(cache_file))
    rejection_data = {"reason": "Não é pássaro", "timestamp": "2025-11-14T00:00:00"}

    cache.add_rejection_to_cache(image_path, rejection_data)

    rejection_entries = [
        info for key, info in cache.cache_data["images"].items() if key.startswith("rejected_")
    ]

    assert len(rejection_entries) == 1
    assert rejection_entries[0]["rejection_data"]["reason"] == "Não é pássaro"
