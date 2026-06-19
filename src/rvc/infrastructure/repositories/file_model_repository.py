"""File-based реализация ModelRepository."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from rvc.core.domain.entities import ModelInfo, ModelVersion, SampleRate
from rvc.core.domain.repositories import ModelRepository
from rvc.core.exceptions import ModelNotFoundError

logger = logging.getLogger(__name__)


class FileModelRepository(ModelRepository):
    """Репозиторий моделей на основе файловой системы."""

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._models: dict[str, ModelInfo] = {}
        self._refresh_cache()

    def _refresh_cache(self) -> None:
        """Обновить кэш моделей."""
        self._models.clear()

        if not self._models_dir.exists():
            logger.warning("Models directory does not exist: %s", self._models_dir)
            return

        for model_file in self._models_dir.glob("*.pth"):
            try:
                model_info = self._parse_model_file(model_file)
                if model_info:
                    self._models[model_info.name] = model_info
                    logger.debug("Found model: %s", model_info.name)
            except Exception as e:
                logger.error("Failed to parse model %s: %s", model_file, e)

    def _parse_model_file(self, path: Path) -> Optional[ModelInfo]:
        """Парсинг файла модели для извлечения метаданных."""
        try:
            import torch

            # Загружаем только метаданные, не всю модель
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            # Извлекаем метаданные
            version = checkpoint.get("version", "v1")
            sample_rate = checkpoint.get("config", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40000])[-1]
            has_pitch = checkpoint.get("f0", 1) == 1

            # Определяем версию
            model_version = ModelVersion.V2 if version == "v2" else ModelVersion.V1

            # Определяем sample rate
            sr_enum = SampleRate.SR_40K  # default
            if sample_rate == 32000:
                sr_enum = SampleRate.SR_32K
            elif sample_rate == 48000:
                sr_enum = SampleRate.SR_48K

            return ModelInfo(
                name=path.stem,
                path=path,
                version=model_version,
                sample_rate=sr_enum,
                has_pitch_guidance=has_pitch,
                created_at=datetime.fromtimestamp(path.stat().st_mtime),
            )
        except Exception as e:
            logger.error("Failed to parse model file %s: %s", path, e)
            return None

    def get_all(self) -> list[ModelInfo]:
        """Получить все доступные модели."""
        return list(self._models.values())

    def get_by_name(self, name: str) -> Optional[ModelInfo]:
        """Получить модель по имени."""
        return self._models.get(name)

    def get_by_path(self, path: Path) -> Optional[ModelInfo]:
        """Получить модель по пути."""
        return self._models.get(path.stem)

    def exists(self, name: str) -> bool:
        """Проверить существование модели."""
        return name in self._models

    def refresh(self) -> None:
        """Обновить список моделей."""
        self._refresh_cache()
        logger.info("Refreshed model repository: %d models found", len(self._models))