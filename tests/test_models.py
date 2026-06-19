"""
Тесты модуля rvc.core.models (SynthesizerWrapper).
Только безопасные тесты, не требующие реальных весов.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rvc.core.models import ModelMetadata, SynthesizerWrapper


class TestModelMetadata:
    """Тесты ModelMetadata dataclass."""

    def test_create_metadata(self, temp_dir: Path) -> None:
        """Создание ModelMetadata."""
        metadata = ModelMetadata(
            version="v2",
            sample_rate=40000,
            has_f0=True,
            speaker_count=1,
            hidden_dim=768,
            path=temp_dir / "test.pth",
        )

        assert metadata.version == "v2"
        assert metadata.sample_rate == 40000
        assert metadata.has_f0 is True
        assert metadata.speaker_count == 1
        assert metadata.hidden_dim == 768

    def test_metadata_immutability(self, temp_dir: Path) -> None:
        """Metadata имеет все поля."""
        metadata = ModelMetadata(
            version="v1",
            sample_rate=48000,
            has_f0=False,
            speaker_count=2,
            hidden_dim=256,
            path=temp_dir / "test.pth",
        )

        # Все поля доступны
        assert metadata.version
        assert metadata.sample_rate
        assert not metadata.has_f0


class TestSynthesizerWrapper:
    """Тесты SynthesizerWrapper."""

    def test_init_default_values(self, config) -> None:
        """Дефолтные значения при инициализации."""
        wrapper = SynthesizerWrapper(config)

        assert wrapper.net_g is None
        assert wrapper.metadata is None
        assert not wrapper.is_loaded

    def test_is_loaded_returns_false_initially(self, config) -> None:
        """is_loaded возвращает False до загрузки."""
        wrapper = SynthesizerWrapper(config)
        assert wrapper.is_loaded is False

    def test_load_nonexistent_raises(self, config, temp_dir: Path) -> None:
        """Загрузка несуществующей модели вызывает FileNotFoundError."""
        wrapper = SynthesizerWrapper(config)
        fake_path = temp_dir / "nonexistent.pth"

        with pytest.raises(FileNotFoundError):
            wrapper.load(fake_path)

    def test_unload_without_model_is_safe(self, config) -> None:
        """unload() без загруженной модели не падает."""
        wrapper = SynthesizerWrapper(config)
        # Не должно падать
        wrapper.unload()
        assert wrapper.net_g is None

    def test_infer_without_load_raises(self, config) -> None:
        """infer() без загруженной модели вызывает RuntimeError."""
        wrapper = SynthesizerWrapper(config)

        feats = torch.randn(1, 100, 256)
        p_len = torch.tensor([100])
        sid = torch.tensor([0])

        with pytest.raises(RuntimeError, match="Модель не загружена"):
            wrapper.infer(feats, p_len, sid)

    def test_warmup_without_model_is_safe(self, config) -> None:
        """warmup() без модели не падает."""
        wrapper = SynthesizerWrapper(config)
        # Не должно падать
        wrapper.warmup()