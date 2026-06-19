"""Реализация ModelService."""
import logging
from pathlib import Path
from typing import Optional

import torch

from rvc.config import RVCConfig
from rvc.core.domain.entities import ModelInfo
from rvc.core.domain.repositories import ModelRepository
from rvc.core.domain.services import ModelService
from rvc.core.events import Event, EventBus, EventType
from rvc.core.exceptions import ModelLoadError, ModelNotFoundError

logger = logging.getLogger(__name__)


class ModelServiceImpl(ModelService):
    """Реализация сервиса управления моделями."""

    def __init__(
        self,
        config: RVCConfig,
        model_repository: ModelRepository,
        event_bus: EventBus,
    ) -> None:
        self._config = config
        self._model_repository = model_repository
        self._event_bus = event_bus
        self._current_model: Optional[ModelInfo] = None
        self._model_instance: Optional[torch.nn.Module] = None

    def load_model(self, model_path: Path) -> ModelInfo:
        """
        Загрузить модель.

        🔧 BUG FIX: Перед загрузкой новой модели явная выгрузка предыдущей
        через self.unload_model() для предотвращения утечки VRAM.
        Ранее новая модель загружалась поверх старой, что приводило к
        удвоению потребления VRAM и CUDA OOM после 3-5 смен моделей.
        """
        # 🔧 BUG FIX: Явная выгрузка предыдущей модели ПЕРЕД загрузкой новой
        # Это критично для GPU с ограниченной VRAM (RTX 3060, 4060 и т.д.)
        if self._model_instance is not None or self._current_model is not None:
            logger.info(
                "Выгрузка предыдущей модели перед загрузкой новой: %s",
                self._current_model.name if self._current_model else "unknown",
            )
            self.unload_model()

        # Получаем информацию о модели
        model_info = self._model_repository.get_by_path(model_path)
        if not model_info:
            raise ModelNotFoundError(str(model_path))

        # Публикуем событие начала загрузки
        self._event_bus.publish(
            Event.create(
                EventType.MODEL_LOADING_STARTED,
                {"model_name": model_info.name, "model_path": str(model_path)},
            )
        )

        try:
            # 🔒 SECURITY: weights_only=True предотвращает RCE через pickle
            # (установлено в Фазе 1 Security Hardening)
            logger.info("Loading model: %s", model_path)
            checkpoint = torch.load(
                model_path,
                map_location=self._config.inference.device,
                weights_only=True,
            )

            # Создаем экземпляр модели
            from rvc.core.models import SynthesizerWrapper

            self._model_instance = SynthesizerWrapper(self._config)
            self._model_instance.load(model_path)

            self._current_model = model_info

            # Публикуем событие успешной загрузки
            self._event_bus.publish(
                Event.create(
                    EventType.MODEL_LOADING_COMPLETED,
                    {
                        "model_name": model_info.name,
                        "version": model_info.version.value,
                        "sample_rate": model_info.sample_rate.value,
                    },
                )
            )

            logger.info("Model loaded successfully: %s", model_info.name)
            return model_info

        except Exception as e:
            # Публикуем событие ошибки загрузки
            self._event_bus.publish(
                Event.create(
                    EventType.MODEL_LOADING_FAILED,
                    {
                        "model_name": model_info.name,
                        "error": str(e),
                    },
                )
            )
            # При ошибке загрузки — выгружаем частично загруженную модель
            self._cleanup_failed_load()
            raise ModelLoadError(str(model_path), str(e)) from e

    def _cleanup_failed_load(self) -> None:
        """
        Очистка после неудачной загрузки модели.
        Предотвращает утечки VRAM при ошибках.
        """
        if self._model_instance is not None:
            try:
                # SynthesizerWrapper имеет собственный unload()
                if hasattr(self._model_instance, "unload"):
                    self._model_instance.unload()
                del self._model_instance
            except Exception as cleanup_err:
                logger.warning("Ошибка при cleanup: %s", cleanup_err)
            self._model_instance = None

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass

    def unload_model(self) -> None:
        """
        Выгрузить текущую модель.

        🔧 BUG FIX: Добавлены явные шаги для гарантированного освобождения VRAM:
        1. Вызов model.unload() если модель это SynthesizerWrapper
        2. Удаление ссылок на объект
        3. torch.cuda.synchronize() — дождаться завершения всех CUDA-операций
        4. torch.cuda.empty_cache() — принудительное освобождение кэша VRAM
        5. gc.collect() — принудительный сбор мусора Python
        """
        if not self._current_model and not self._model_instance:
            logger.debug("Нет активной модели для выгрузки")
            return

        model_name = self._current_model.name if self._current_model else "unknown"
        logger.info("Выгрузка модели: %s", model_name)

        # Шаг 1: Вызов unload() если доступен
        if self._model_instance is not None:
            try:
                if hasattr(self._model_instance, "unload"):
                    self._model_instance.unload()
            except Exception as e:
                logger.warning("Ошибка при model.unload(): %s", e)

        # Шаг 2: Удаление ссылок
        self._model_instance = None
        self._current_model = None

        # Шаг 3-5: Принудительное освобождение VRAM
        if torch.cuda.is_available():
            try:
                # Дождаться завершения всех CUDA-операций
                torch.cuda.synchronize()
                # Освободить кэш аллокатора CUDA
                torch.cuda.empty_cache()
                # Принудительный сбор мусора Python для циклических ссылок
                import gc
                gc.collect()

                logger.debug(
                    "VRAM после выгрузки: %.2f MB allocated",
                    torch.cuda.memory_allocated() / 1024**2,
                )
            except Exception as e:
                logger.warning("Ошибка при очистке VRAM: %s", e)

        # Публикуем событие выгрузки
        self._event_bus.publish(
            Event.create(
                EventType.MODEL_UNLOADED,
                {"model_name": model_name},
            )
        )

        logger.info("Model unloaded: %s", model_name)

    def get_current_model(self) -> Optional[ModelInfo]:
        """Получить текущую загруженную модель."""
        return self._current_model

    def is_model_loaded(self) -> bool:
        """Проверить, загружена ли модель."""
        return self._current_model is not None and self._model_instance is not None

    def get_model_instance(self) -> Optional[torch.nn.Module]:
        """Получить экземпляр загруженной модели (для внутреннего использования)."""
        return self._model_instance