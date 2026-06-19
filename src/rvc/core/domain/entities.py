"""Доменные сущности RVC."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np


class ModelVersion(Enum):
    """Версии моделей RVC."""

    V1 = "v1"
    V2 = "v2"


class SampleRate(Enum):
    """Поддерживаемые частоты дискретизации."""

    SR_32K = 32000
    SR_40K = 40000
    SR_48K = 48000


class PitchMethod(Enum):
    """Методы извлечения питча."""

    PM = "pm"
    HARVEST = "harvest"
    CREPE = "crepe"
    RMVPE = "rmvpe"
    FCPE = "fcpe"


@dataclass
class ModelInfo:
    """Информация о модели."""

    name: str
    path: Path
    version: ModelVersion
    sample_rate: SampleRate
    has_pitch_guidance: bool
    created_at: datetime = field(default_factory=datetime.now)
    description: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        """Уникальный идентификатор модели."""
        return self.path.stem

    def __hash__(self) -> int:
        return hash(self.path)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelInfo):
            return False
        return self.path == other.path


@dataclass
class IndexInfo:
    """Информация об индексе."""

    name: str
    path: Path
    model_id: str
    vector_count: int
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def id(self) -> str:
        """Уникальный идентификатор индекса."""
        return self.path.stem

    def __hash__(self) -> int:
        return hash(self.path)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IndexInfo):
            return False
        return self.path == other.path


@dataclass
class AudioFile:
    """Аудио файл."""

    path: Path
    sample_rate: int
    duration: float
    channels: int
    format: str

    @property
    def name(self) -> str:
        """Имя файла."""
        return self.path.name

    @property
    def stem(self) -> str:
        """Имя без расширения."""
        return self.path.stem


@dataclass
class InferenceConfig:
    """Конфигурация инференса."""

    pitch_shift: int = 0
    index_rate: float = 0.75
    protect: float = 0.33
    filter_radius: int = 3
    resample_sr: int = 0
    rms_mix_rate: float = 1.0
    pitch_method: PitchMethod = PitchMethod.RMVPE

    def validate(self) -> None:
        """Валидация конфигурации."""
        if not -24 <= self.pitch_shift <= 24:
            raise ValueError("Pitch shift must be between -24 and 24")
        if not 0.0 <= self.index_rate <= 1.0:
            raise ValueError("Index rate must be between 0.0 and 1.0")
        if not 0.0 <= self.protect <= 0.5:
            raise ValueError("Protect must be between 0.0 and 0.5")
        if not 0 <= self.filter_radius <= 7:
            raise ValueError("Filter radius must be between 0 and 7")


@dataclass
class InferenceResult:
    """Результат инференса."""

    audio: np.ndarray
    sample_rate: int
    duration: float
    processing_time: float
    model_name: str
    config: InferenceConfig

    @property
    def samples_count(self) -> int:
        """Количество сэмплов."""
        return len(self.audio)


@dataclass
class BatchItem:
    """Элемент пакетной обработки."""

    input_path: Path
    output_path: Path
    status: str = "pending"  # pending, processing, completed, failed
    error: Optional[str] = None
    result: Optional[InferenceResult] = None


@dataclass
class BatchJob:
    """Пакетная задача."""

    items: list[BatchItem]
    config: InferenceConfig
    model_name: str
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def total_count(self) -> int:
        """Общее количество элементов."""
        return len(self.items)

    @property
    def completed_count(self) -> int:
        """Количество завершенных элементов."""
        return sum(1 for item in self.items if item.status == "completed")

    @property
    def failed_count(self) -> int:
        """Количество неудачных элементов."""
        return sum(1 for item in self.items if item.status == "failed")

    @property
    def progress(self) -> float:
        """Прогресс выполнения (0.0 - 1.0)."""
        if self.total_count == 0:
            return 0.0
        return (self.completed_count + self.failed_count) / self.total_count