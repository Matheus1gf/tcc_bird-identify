from pathlib import Path

from tests.conftest import create_temp_image
from src.interfaces.manual_analysis import ManualAnalysisSystem


def test_manual_analysis_workflow(tmp_path):
    base_dir = tmp_path / "manual_system"
    system = ManualAnalysisSystem(base_dir=str(base_dir))

    image_source = tmp_path / "original.jpg"
    create_temp_image(image_source)

    detection_data = {"yolo": {"detections": []}}
    pending_path = system.add_image_for_analysis(str(image_source), detection_data)
    assert Path(pending_path).exists()

    pending_images = system.get_pending_images()
    assert len(pending_images) == 1
    filename = pending_images[0]["filename"]

    system.approve_image(
        filename=filename,
        species="Bem te vi",
        confidence=0.92,
        notes="Teste automatizado",
        decision_reason="boa iluminação",
        visual_characteristics=["bico longo"],
    )

    approved_file = base_dir / "approved" / filename
    assert approved_file.exists()

    annotation_file = base_dir / "annotations" / f"{Path(filename).stem}.txt"
    assert annotation_file.exists()

    stats = system.get_statistics()
    assert stats["approved"] == 1
    assert stats["pending"] == 0


def test_manual_analysis_rejection(tmp_path):
    base_dir = tmp_path / "manual_system"
    system = ManualAnalysisSystem(base_dir=str(base_dir))

    image_source = tmp_path / "rejeitado.jpg"
    create_temp_image(image_source, color=(0, 0, 0))

    detection_data = {"dummy": "data"}
    pending_path = system.add_image_for_analysis(str(image_source), detection_data)
    filename = Path(pending_path).name

    system.reject_image(
        filename=filename,
        reason="Não é pássaro",
        decision_reason="sem asas visíveis",
        visual_characteristics=["ausência de penas"],
    )

    rejected_file = base_dir / "rejected" / filename
    assert rejected_file.exists()

    stats = system.get_statistics()
    assert stats["rejected"] == 1
