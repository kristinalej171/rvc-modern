"""Интерфейсы репозиториев для работы с данными."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from rvc.core.domain.entities import AudioFile, IndexInfo, ModelInfo


class ModelRepository(ABC):
    """Абстрактный репозиторий моделей."""

    @abstractmethod
    def get_all(self) -> list[ModelInfo]:
        """Получить все доступные модели."""
        pass

    @abstractmethod
    def get_by_name(self, name: str) -> Optional[ModelInfo]:
        """Получить модель по имени."""
        pass

    @abstractmethod
    def get_by_path(self, path: Path) -> Optional[ModelInfo]:
        """Получить модель по пути."""
        pass

    @abstractmethod
    def exists(self, name: str) -> bool:
        """Проверить существование модели."""
        pass

    @abstractmethod
    def refresh(self) -> None:
        """Обновить список моделей."""
        pass


class IndexRepository(ABC):
    """Абстрактный репозиторий индексов."""

    @abstractmethod
    def get_all(self) -> list[IndexInfo]:
        """Получить все доступные индексы."""
        pass

    @abstractmethod
    def get_by_name(self, name: str) -> Optional[IndexInfo]:
        """Получить индекс по имени."""
        pass

    @abstractmethod
    def get_by_model(self, model_id: str) -> list[IndexInfo]:
        """Получить индексы для модели."""
        pass

    @abstractmethod
    def exists(self, name: str) -> bool:
        """Проверить существование индекса."""
        pass

    @abstractmethod
    def refresh(self) -> None:
        """Обновить список индексов."""
        pass


class AudioRepository(ABC):
    """Абстрактный репозиторий аудио файлов."""

    @abstractmethod
    def load(self, path: Path, target_sr: Optional[int] = None) -> AudioFile:
        """Загрузить аудио файл."""
        pass

    @abstractmethod
    def save(
        self,
        path: Path,
        audio_data: any,  # np.ndarray or torch.Tensor
        sample_rate: int,
    ) -> None:
        """Сохранить аудио файл."""
        pass

    @abstractmethod
    def get_files_in_directory(self, directory: Path) -> list[AudioFile]:
        """Получить все аудио файлы в директории."""
        pass

    @abstractmethod
    def convert_format(
        self, input_path: Path, output_path: Path, format: str
    ) -> None:
        """Конвертировать формат аудио."""
        pass