"""File-based реализация IndexRepository."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from rvc.core.domain.entities import IndexInfo
from rvc.core.domain.repositories import IndexRepository

logger = logging.getLogger(__name__)


class FileIndexRepository(IndexRepository):
    """Репозиторий индексов на основе файловой системы."""

    def __init__(self, index_dir: Path) -> None:
        self._index_dir = index_dir
        self._indices: dict[str, IndexInfo] = {}
        self._refresh_cache()

    def _refresh_cache(self) -> None:
        """Обновить кэш индексов."""
        self._indices.clear()

        if not self._index_dir.exists():
            logger.warning("Index directory does not exist: %s", self._index_dir)
            return

        for index_file in self._index_dir.rglob("*.index"):
            # Пропускаем trained индексы
            if "trained" in index_file.name:
                continue

            try:
                index_info = self._parse_index_file(index_file)
                if index_info:
                    self._indices[index_info.name] = index_info
                    logger.debug("Found index: %s", index_info.name)
            except Exception as e:
                logger.error("Failed to parse index %s: %s", index_file, e)

    def _parse_index_file(self, path: Path) -> Optional[IndexInfo]:
        """Парсинг файла индекса для извлечения метаданных."""
        try:
            import faiss

            # Загружаем индекс
            index = faiss.read_index(str(path))

            # Извлекаем model_id из имени (предполагаем формат: model_name_IVF...)
            parts = path.stem.split("_")
            model_id = parts[0] if parts else "unknown"

            return IndexInfo(
                name=path.stem,
                path=path,
                model_id=model_id,
                vector_count=index.ntotal,
                created_at=datetime.fromtimestamp(path.stat().st_mtime),
            )
        except Exception as e:
            logger.error("Failed to parse index file %s: %s", path, e)
            return None

    def get_all(self) -> list[IndexInfo]:
        """Получить все доступные индексы."""
        return list(self._indices.values())

    def get_by_name(self, name: str) -> Optional[IndexInfo]:
        """Получить индекс по имени."""
        return self._indices.get(name)

    def get_by_model(self, model_id: str) -> list[IndexInfo]:
        """Получить индексы для модели."""
        return [
            idx for idx in self._indices.values() if idx.model_id == model_id
        ]

    def exists(self, name: str) -> bool:
        """Проверить существование индекса."""
        return name in self._indices

    def refresh(self) -> None:
        """Обновить список индексов."""
        self._refresh_cache()
        logger.info(
            "Refreshed index repository: %d indices found", len(self._indices)
        )