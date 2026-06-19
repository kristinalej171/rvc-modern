"""Интерфейсы сервисов доменного слоя."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from rvc.core.domain.entities import (
    AudioFile,
    BatchJob,
    InferenceConfig,
    InferenceResult,
    ModelInfo,
)


class ModelService(ABC):
    """Абстрактный сервис управления моделями."""

    @abstractmethod
    def load_model(self, model_path: Path) -> ModelInfo:
        """Загрузить модель."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Выгрузить текущую модель."""
        pass

    @abstractmethod
    def get_current_model(self) -> Optional[ModelInfo]:
        """Получить текущую загруженную модель."""
        pass

    @abstractmethod
    def is_model_loaded(self) -> bool:
        """Проверить, загружена ли модель."""
        pass


class InferenceService(ABC):
    """Абстрактный сервис инференса."""

    @abstractmethod
    def infer_single(
        self,
        audio_path: Path,
        config: InferenceConfig,
        index_path: Optional[Path] = None,
    ) -> InferenceResult:
        """Выполнить инференс для одного файла."""
        pass

    @abstractmethod
    def infer_batch(
        self,
        audio_paths: list[Path],
        output_dir: Path,
        config: InferenceConfig,
        index_path: Optional[Path] = None,
    ) -> BatchJob:
        """Выполнить пакетный инференс."""
        pass

    @abstractmethod
    def cancel_batch(self, job_id: str) -> bool:
        """Отменить пакетную задачу."""
        pass


class AudioService(ABC):
    """Абстрактный сервис работы с аудио."""

    @abstractmethod
    def load_audio(
        self, path: Path, target_sr: Optional[int] = None
    ) -> AudioFile:
        """Загрузить аудио."""
        pass

    @abstractmethod
    def save_audio(
        self, path: Path, audio_data: any, sample_rate: int
    ) -> None:
        """Сохранить аудио."""
        pass

    @abstractmethod
    def convert_format(
        self, input_path: Path, output_path: Path, format: str
    ) -> None:
        """Конвертировать формат."""
        pass

    @abstractmethod
    def get_audio_info(self, path: Path) -> AudioFile:
        """Получить информацию об аудио."""
        pass