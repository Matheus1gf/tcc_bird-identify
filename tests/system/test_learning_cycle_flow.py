from unittest.mock import patch

from tests.conftest import ensure_lightweight_intuition_stub


def test_learning_cycle_completes_and_updates_stats(tmp_path):
    ensure_lightweight_intuition_stub()

    from src.core.learning import (
        ContinuousLearningSystem,
        LearningCycleStage,
        LearningCycleStatus,
    )

    images_dir = tmp_path / "imagens"
    images_dir.mkdir()
    (images_dir / "dummy.txt").write_text("imagem placeholder", encoding="utf-8")

    with patch("src.core.learning.IntuitionEngine"), patch(
        "src.core.learning.GradCAMAnnotator"
    ), patch("src.core.learning.HybridCurator"), patch.object(
        ContinuousLearningSystem, "_execute_learning_cycle", autospec=True
    ) as mock_execute, patch.object(
        ContinuousLearningSystem, "_save_learning_cycle", autospec=True
    ):

        def fake_execute(self, directory):
            self.current_cycle.stages_completed = [
                LearningCycleStage.INTUITION_DETECTION,
                LearningCycleStage.AUTO_ANNOTATION,
                LearningCycleStage.VALIDATION,
            ]
            self.current_cycle.candidates_processed = 2
            self.current_cycle.annotations_generated = 1

        mock_execute.side_effect = fake_execute

        learning_data = tmp_path / "learning_data"
        system = ContinuousLearningSystem(
            yolo_model_path="yolov8n.pt",
            keras_model_path="data/models/modelo_classificacao_passaros.keras",
            learning_data_path=str(learning_data),
        )

        cycle_id = system.start_learning_cycle(str(images_dir))

        assert cycle_id.startswith("cycle_")
        assert system.current_cycle.status == LearningCycleStatus.COMPLETED
        assert LearningCycleStage.VALIDATION in system.current_cycle.stages_completed
        assert system.global_stats["total_cycles"] == 1
        mock_execute.assert_called_once()
