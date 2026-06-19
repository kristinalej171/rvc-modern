"""
Общие pytest fixtures для всего проекта.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))


@pytest.fixture(scope="session")
def config():
    """RVC конфигурация для тестов (CPU-only)."""
    os.environ["RVC_INFERENCE__DEVICE"] = "cpu"
    os.environ["RVC_INFERENCE__DTYPE"] = "float32"

    from rvc.config import RVCConfig

    return RVCConfig()


@pytest.fixture(scope="session")
def sample_wav_path(tmp_path_factory) -> Path:
    """Создать тестовый .wav файл."""
    import soundfile as sf

    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    path = tmp_path_factory.mktemp("audio") / "test.wav"
    sf.write(str(path), audio, sr)
    return path