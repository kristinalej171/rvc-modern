"""Общие фикстуры для тестов."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import torch

# Добавляем src в путь для корректного импорта
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def config():
    """RVC конфигурация для тестов (принудительно CPU)."""
    # Очистка env перед созданием конфига
    for key in list(os.environ.keys()):
        if key.startswith("RVC_"):
            del os.environ[key]

    os.environ["RVC_INFERENCE__DEVICE"] = "cpu"
    os.environ["RVC_INFERENCE__DTYPE"] = "float32"

    from rvc.config import RVCConfig

    return RVCConfig()


@pytest.fixture(scope="session")
def sample_audio() -> np.ndarray:
    """Синтетический аудио: синусоида 440 Гц, 2 сек, 16kHz."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio


@pytest.fixture(scope="session")
def short_audio() -> np.ndarray:
    """Очень короткий аудио для edge-case тестов (50мс)."""
    sr = 16000
    duration = 0.05
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture(scope="session")
def silent_audio() -> np.ndarray:
    """Тихий аудио (весь нулевой)."""
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture(scope="session")
def loud_audio() -> np.ndarray:
    """Громкий аудио (значения > 1.0)."""
    return np.array([2.0, -3.0, 4.5, -1.5], dtype=np.float32)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Временная директория для тестов."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_wav_file(sample_audio: np.ndarray) -> Generator[Path, None, None]:
    """Временный WAV-файл с тестовым аудио."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.wav"
        sf.write(str(path), sample_audio, 16000)
        yield path


@pytest.fixture
def temp_pth_checkpoint(temp_dir: Path) -> Path:
    """
    Мокированный .pth checkpoint (безопасный, weights_only=True совместимый).
    Используется для тестов загрузки моделей без реальных весов.
    """
    # Создаём минимальную структуру RVC-чекпоинта
    checkpoint = {
        "weight": {
            # Минимальный набор ключей для прохождения валидации
            "emb_g.weight": torch.randn(1, 256),
        },
        "config": [
            513,   # filter_length // 2 + 1
            32,
            192,   # inter_channels
            192,   # hidden_channels
            768,   # filter_channels
            2,     # n_heads
            6,     # n_layers
            3,     # kernel_size
            0,     # p_dropout
            "1",   # resblock
            [3, 7, 11],
            [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            [10, 10, 2, 2],
            512,
            [16, 16, 4, 4],
            109,   # spk_embed_dim
            256,   # gin_channels
            40000, # sampling_rate
        ],
        "version": "v1",
        "f0": 1,
        "info": "test_epoch",
    }
    path = temp_dir / "test_model.pth"
    torch.save(checkpoint, path)
    return path


@pytest.fixture
def sample_features() -> np.ndarray:
    """Синтетические features shape (100, 256)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 256)).astype(np.float32)


@pytest.fixture
def event_bus():
    """EventBus для тестов."""
    from rvc.core.events import EventBus
    return EventBus()