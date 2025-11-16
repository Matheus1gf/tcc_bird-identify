from unittest.mock import MagicMock, patch

from tests.conftest import create_temp_image, ensure_lightweight_intuition_stub


def test_reasoning_pipeline_triggers_learning(tmp_path):
    ensure_lightweight_intuition_stub()

    from src.core.reasoning import LogicalAIReasoningSystem

    image_path = create_temp_image(tmp_path / "pipeline.jpg")

    with patch("src.core.reasoning.ContinuousLearningSystem") as MockCLS, patch(
        "src.core.cache.image_cache"
    ) as mock_cache:
        mock_cache.is_image_rejected.return_value = None
        mock_cache.is_image_recognized.return_value = None

        learning_system = MockCLS.return_value
        learning_system.intuition_engine.analyze_image_intuition.return_value = {
            "candidates": [
                {
                    "type": "species_unknown",
                    "priority_score": 0.8,
                    "keras_confidence": 0.65,
                    "keras_prediction": "Ave Misteriosa",
                    "reasoning": "YOLO sem detecções e Keras incerto",
                }
            ]
        }

        reasoning_system = LogicalAIReasoningSystem(
            yolo_model_path="yolov8n.pt", keras_model_path="data/models/modelo_classificacao_passaros.keras"
        )

        reasoning_system._perform_normal_analysis = MagicMock(return_value={"status": "mocked"})
        reasoning_system._activate_learning_cycle = MagicMock(return_value={"activated": True})

        result = reasoning_system.analyze_image_revolutionary(image_path)

        assert result["needs_learning"] is True
        assert result["revolutionary_action"] == "LEARNING_ACTIVATED"
        reasoning_system._activate_learning_cycle.assert_called_once()
        learning_system.intuition_engine.analyze_image_intuition.assert_called_once_with(image_path)
