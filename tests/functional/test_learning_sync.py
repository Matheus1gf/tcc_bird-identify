from pathlib import Path

from tests.conftest import create_temp_image
from src.interfaces.manual_analysis import ManualAnalysisSystem
from src.core.learning_sync import LearningSyncSystem


def test_learning_sync_copies_approved_images(tmp_path):
    manual_base = tmp_path / "manual"
    learning_dir = tmp_path / "learning"
    dataset_dir = tmp_path / "dataset"

    manual_system = ManualAnalysisSystem(base_dir=str(manual_base))
    source_image = tmp_path / "sync_image.jpg"
    create_temp_image(source_image, color=(50, 150, 200))

    detection_data = {"dummy": "data"}
    pending_path = manual_system.add_image_for_analysis(str(source_image), detection_data)
    filename = Path(pending_path).name

    manual_system.approve_image(
        filename=filename,
        species="Sabia",
        confidence=0.88,
        notes="teste sync",
        decision_reason="boa nitidez",
    )

    sync_system = LearningSyncSystem(
        manual_approved_dir=str(manual_base / "approved"),
        learning_approved_dir=str(learning_dir / "auto_approved"),
        dataset_train_dir=str(dataset_dir / "images" / "train"),
        sync_interval=999,
    )

    result = sync_system.sync_approved_images()
    assert result["synced_count"] == 1
    assert result["total_images"] == 1
    assert result["retraining_triggered"] is False

    synced_image = learning_dir / "auto_approved" / filename
    assert synced_image.exists()
