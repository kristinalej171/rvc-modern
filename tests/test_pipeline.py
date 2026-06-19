"""
Интеграционные тесты для VC Pipeline.

Используют pytest fixtures и hypothesis для property-based тестирования.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from rvc.core.feature_extractor import (
    FeatureModelType,
    TransformerFeatureExtractor,
)
from rvc.core.indexer import FeatureIndexer, IndexStrategy
from rvc.core.pitch import (
    PitchMethod,
    PMPitchExtractor,
    f0_to_coarse,
    transpose_f0,
)


# ==== Fixtures ====


@pytest.fixture(scope="module")
def sample_audio() -> np.ndarray:
    """Синтетический аудио сигнал для тестов."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    # Синусоида 440 Гц (нота A4)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio


@pytest.fixture(scope="module")
def sample_features() -> np.ndarray:
    """Синтетические features shape (100, 256)."""
    np.random.seed(42)
    return np.random.randn(100, 256).astype(np.float32)


# ==== Unit Tests: Pitch ====


class TestPitchUtils:
    """Тесты утилит для работы с F0."""

    def test_transpose_f0_up(self) -> None:
        f0 = np.array([100.0, 200.0, 0.0, 300.0])
        result = transpose_f0(f0, key_shift=12)  # Октава вверх
        assert np.allclose(result[0], 200.0)
        assert np.allclose(result[1], 400.0)
        assert result[2] == 0.0  # Unvoiced не меняется
        assert np.allclose(result[3], 600.0)

    def test_transpose_f0_zero(self) -> None:
        f0 = np.array([100.0, 200.0, 300.0])
        result = transpose_f0(f0, key_shift=0)
        assert np.allclose(result, f0)

    def test_f0_to_coarse_basic(self) -> None:
        f0 = np.array([100.0, 200.0, 400.0])
        coarse = f0_to_coarse(f0)
        assert coarse.dtype == np.int32
        assert all(1 <= c <= 255 for c in coarse)

    def test_f0_to_coarse_unvoiced(self) -> None:
        f0 = np.array([100.0, 0.0, 400.0])
        coarse = f0_to_coarse(f0)
        assert coarse[1] == 0  # Unvoiced -> 0


class TestPMPitchExtractor:
    """Тесты Parselmouth pitch extractor."""

    def test_compute_f0_shape(
        self,
        sample_audio: np.ndarray,
        config,
    ) -> None:
        extractor = PMPitchExtractor(config)
        f0 = extractor.compute_f0(sample_audio)
        assert isinstance(f0, np.ndarray)
        assert f0.ndim == 1
        assert len(f0) > 0

    def test_compute_f0_with_plen(
        self,
        sample_audio: np.ndarray,
        config,
    ) -> None:
        extractor = PMPitchExtractor(config)
        target_len = 100
        f0 = extractor.compute_f0(sample_audio, p_len=target_len)
        assert len(f0) == target_len


# ==== Unit Tests: Feature Extractor ====


class TestFeatureExtractor:
    """Тесты feature extractor."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA недоступна",
    )
    def test_hubert_extraction(
        self,
        sample_audio: np.ndarray,
    ) -> None:
        extractor = TransformerFeatureExtractor(
            model_type=FeatureModelType.HUBERT_BASE,
            device="cuda:0",
            dtype=torch.float16,
            use_compile=False,
        )

        audio_tensor = torch.from_numpy(sample_audio)
        features = extractor.extract(audio_tensor, sample_rate=16000)

        assert features.ndim == 3
        assert features.shape[0] == 1
        assert features.shape[2] == extractor.get_feature_dim()

    def test_feature_dim(self) -> None:
        # Проверяем только без загрузки модели
        from rvc.core.feature_extractor import MODEL_REGISTRY

        for model_type, config in MODEL_REGISTRY.items():
            assert "hidden_dim" in config
            assert config["hidden_dim"] > 0


# ==== Unit Tests: Indexer ====


class TestFeatureIndexer:
    """Тесты FAISS indexer."""

    def test_build_and_search(
        self,
        sample_features: np.ndarray,
        config,
        tmp_path: Path,
    ) -> None:
        pytest.importorskip("faiss")

        from rvc.core.indexer import build_index

        index_path = tmp_path / "test.index"
        build_index(
            sample_features,
            index_path,
            strategy=IndexStrategy.IVF_FLAT,
            n_ivf=10,
        )
        assert index_path.exists()

        # Load и search
        indexer = FeatureIndexer.from_file(
            config,
            index_path,
            use_gpu=False,
        )
        assert indexer.is_loaded
        assert indexer.stats.total_vectors == len(sample_features)

        # Поиск
        query = sample_features[:5]
        distances, indices = indexer.search(query, k=3)
        assert distances.shape == (5, 3)
        assert indices.shape == (5, 3)

    def test_blend_features(
        self,
        sample_features: np.ndarray,
        config,
        tmp_path: Path,
    ) -> None:
        pytest.importorskip("faiss")

        from rvc.core.indexer import build_index

        index_path = tmp_path / "test_blend.index"
        build_index(sample_features, index_path, n_ivf=10)

        indexer = FeatureIndexer.from_file(
            config, index_path, use_gpu=False
        )

        # Blend
        features = torch.from_numpy(sample_features[:10]).unsqueeze(0)
        blended = indexer.blend_features(features, index_rate=0.5, k=3)

        assert blended.shape == features.shape
        assert not torch.allclose(blended, features)  # Должны отличаться


# ==== Integration Tests: Pipeline ====


class TestVCPipelineIntegration:
    """Интеграционные тесты pipeline (с моками для тяжёлых компонентов)."""

    def test_pipeline_init(self, config) -> None:
        from rvc.core.pipeline import VCPipeline

        pipeline = VCPipeline(config)
        assert pipeline._synthesizer is None
        assert pipeline._feature_extractor is None

    def test_inference_config_validation(self) -> None:
        from rvc.core.pipeline import InferenceConfig

        # Валидные параметры
        cfg = InferenceConfig(
            pitch_shift=12,
            index_rate=0.5,
            protect=0.3,
        )
        cfg.validate()

        # Невалидные параметры
        with pytest.raises(ValueError):
            InferenceConfig(pitch_shift=30).validate()

        with pytest.raises(ValueError):
            InferenceConfig(index_rate=1.5).validate()

    def test_progress_callback(self, config) -> None:
        from rvc.core.pipeline import ProgressEvent, VCPipeline

        events: list[ProgressEvent] = []

        def on_progress(event: ProgressEvent) -> None:
            events.append(event)

        pipeline = VCPipeline(config, on_progress=on_progress)
        pipeline._report_progress("test", 0.5, "Test message")

        assert len(events) == 1
        assert events[0].stage == "test"
        assert events[0].progress == 0.5
        assert events[0].message == "Test message"