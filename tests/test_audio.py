"""Тесты обработки аудио."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from rvc.core.audio import AudioProcessor


def test_normalize() -> None:
    """Проверка нормализации."""
    audio = np.array([0.5, -2.0, 1.5], dtype=np.float32)
    normalized = AudioProcessor.normalize(audio, max_value=0.95)
    assert np.isclose(np.abs(normalized).max(), 0.95, atol=1e-6)


def test_to_int16() -> None:
    """Проверка конвертации в int16."""
    audio = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
    int16 = AudioProcessor.to_int16(audio)
    assert int16.dtype == np.int16
    assert int16[1] == 32767
    assert int16[2] == -32767


def test_save_load_roundtrip(config) -> None:
    """Сохранение и загрузка аудио (roundtrip)."""
    processor = AudioProcessor(config)

    # Синусоида 440 Гц
    t = np.linspace(0, 1, config.audio.sample_rate, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.wav"
        processor.save_audio(path, audio, config.audio.sample_rate)

        assert path.exists()

        loaded = processor.load_audio(path, target_sr=config.audio.sample_rate)
        assert loaded.dtype == np.float32
        assert len(loaded) == len(audio)