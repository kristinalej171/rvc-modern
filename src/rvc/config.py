"""Конфигурация RVC на основе Pydantic V2."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Корневая директория проекта
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


class AudioConfig(BaseSettings):
    """Настройки обработки аудио."""

    model_config = SettingsConfigDict(env_prefix="RVC_AUDIO_")

    sample_rate: Literal[32000, 40000, 48000] = 40000
    hop_length: int = Field(default=160, gt=0)
    win_length: int = Field(default=640, gt=0)
    filter_length: int = Field(default=2048, gt=0)
    n_mel_channels: int = Field(default=125, gt=0)
    f0_min: float = Field(default=50.0, gt=0)
    f0_max: float = Field(default=1100.0, gt=0)
    max_wav_value: float = Field(default=32768.0, gt=0)


class InferenceConfig(BaseSettings):
    """Настройки инференса."""

    model_config = SettingsConfigDict(env_prefix="RVC_INFERENCE_")

    device: str = Field(default="auto")
    dtype: Literal["float32", "float16", "bfloat16"] = "float16"
    index_rate: float = Field(default=0.75, ge=0.0, le=1.0)
    rms_mix_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    protect: float = Field(default=0.33, ge=0.0, le=0.5)
    filter_radius: int = Field(default=3, ge=0, le=7)
    resample_sr: int = Field(default=0, ge=0)
    pitch_method: Literal["pm", "harvest", "crepe", "rmvpe", "fcpe"] = "rmvpe"
    use_jit: bool = False

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Автоматический выбор устройства."""
        if v == "auto":
            if torch.cuda.is_available():
                return "cuda:0"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return v


class TrainingConfig(BaseSettings):
    """Настройки обучения."""

    model_config = SettingsConfigDict(env_prefix="RVC_TRAINING_")

    batch_size: int = Field(default=4, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    total_epoch: int = Field(default=20, gt=0, le=1000)
    save_every_epoch: int = Field(default=10, gt=0)
    seed: int = Field(default=1234, ge=0)
    fp16_run: bool = True
    preprocess_per: float = Field(default=3.7, gt=0)
    n_cpu: int = Field(default_factory=lambda: max(1, os.cpu_count() or 1))


class PathConfig(BaseSettings):
    """Пути к ресурсам."""

    model_config = SettingsConfigDict(env_prefix="RVC_PATH_")

    root: Path = PROJECT_ROOT
    assets: Path = PROJECT_ROOT / "assets"
    weights: Path = PROJECT_ROOT / "assets" / "weights"
    pretrained: Path = PROJECT_ROOT / "assets" / "pretrained"
    pretrained_v2: Path = PROJECT_ROOT / "assets" / "pretrained_v2"
    hubert: Path = PROJECT_ROOT / "assets" / "hubert"
    rmvpe: Path = PROJECT_ROOT / "assets" / "rmvpe"
    uvr5: Path = PROJECT_ROOT / "assets" / "uvr5_weights"
    configs: Path = PROJECT_ROOT / "configs"
    logs: Path = PROJECT_ROOT / "logs"
    index_root: Path = PROJECT_ROOT / "logs"
    output: Path = PROJECT_ROOT / "opt"

    @model_validator(mode="after")
    def create_directories(self) -> "PathConfig":
        """Создание необходимых директорий при инициализации."""
        for path in [
            self.weights,
            self.logs,
            self.output,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        return self


class RVCConfig(BaseSettings):
    """Главный конфиг RVC, агрегирующий все подсистемы."""

    model_config = SettingsConfigDict(
        env_prefix="RVC_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    audio: AudioConfig = Field(default_factory=AudioConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    paths: PathConfig = Field(default_factory=PathConfig)

    # Общие настройки
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    hubert_model: str = "facebook/hubert-base-ls960"

    def setup_logging(self) -> None:
        """Настройка логирования."""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Подавление шумных логгеров
        for name in ["numba", "httpx", "matplotlib", "filelock"]:
            logging.getLogger(name).setLevel(logging.WARNING)


@lru_cache(maxsize=1)
def get_config() -> RVCConfig:
    """Получение кэшированного экземпляра конфигурации."""
    config = RVCConfig()
    config.setup_logging()
    return config


def detect_torch_dtype(device: str, preferred: str = "float16") -> torch.dtype:
    """Определение оптимального dtype для устройства."""
    if device == "cpu":
        return torch.float32
    if preferred == "bfloat16" and device.startswith("cuda"):
        # Проверяем поддержку bf16
        try:
            if torch.cuda.get_device_capability()[0] >= 8:
                return torch.bfloat16
        except Exception:
            pass
    if preferred == "float16":
        return torch.float16
    return torch.float32