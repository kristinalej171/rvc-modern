"""Тесты конфигурации."""

from __future__ import annotations

import os

from rvc.config import RVCConfig, detect_torch_dtype


def test_config_defaults() -> None:
    """Проверка значений по умолчанию."""
    # Очистка env для чистого теста
    for key in list(os.environ.keys()):
        if key.startswith("RVC_"):
            del os.environ[key]

    config = RVCConfig()
    assert config.audio.sample_rate == 40000
    assert config.inference.index_rate == 0.75
    assert config.training.batch_size == 4


def test_config_from_env() -> None:
    """Проверка загрузки из переменных окружения."""
    os.environ["RVC_AUDIO__SAMPLE_RATE"] = "48000"
    os.environ["RVC_INFERENCE__INDEX_RATE"] = "0.5"

    try:
        config = RVCConfig()
        assert config.audio.sample_rate == 48000
        assert config.inference.index_rate == 0.5
    finally:
        del os.environ["RVC_AUDIO__SAMPLE_RATE"]
        del os.environ["RVC_INFERENCE__INDEX_RATE"]


def test_detect_dtype_cpu() -> None:
    """На CPU всегда float32."""
    import torch

    assert detect_torch_dtype("cpu", "float16") == torch.float32