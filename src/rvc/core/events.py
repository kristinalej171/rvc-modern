"""Event-driven architecture для RVC."""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Типы событий в системе."""
    # Model events
    MODEL_LOADING_STARTED = "model.loading.started"
    MODEL_LOADING_COMPLETED = "model.loading.completed"
    MODEL_LOADING_FAILED = "model.loading.failed"
    MODEL_UNLOADED = "model.unloaded"

    # Inference events
    INFERENCE_STARTED = "inference.started"
    INFERENCE_PROGRESS = "inference.progress"
    INFERENCE_STAGE_STARTED = "inference.stage.started"
    INFERENCE_STAGE_COMPLETED = "inference.stage.completed"
    INFERENCE_COMPLETED = "inference.completed"
    INFERENCE_FAILED = "inference.failed"

    # Batch processing events
    BATCH_STARTED = "batch.started"
    BATCH_ITEM_STARTED = "batch.item.started"
    BATCH_ITEM_COMPLETED = "batch.item.completed"
    BATCH_ITEM_FAILED = "batch.item.failed"
    BATCH_COMPLETED = "batch.completed"

    # System events
    DEVICE_CHANGED = "system.device.changed"
    CONFIG_UPDATED = "system.config.updated"
    ERROR_OCCURRED = "system.error.occurred"


@dataclass
class Event:
    """Базовый класс события."""
    type: EventType
    data: dict[str, Any]
    timestamp: datetime
    event_id: str

    @classmethod
    def create(cls, event_type: EventType, data: dict[str, Any]) -> "Event":
        """Фабричный метод для создания события."""
        return cls(
            type=event_type,
            data=data,
            timestamp=datetime.now(),
            event_id=str(uuid4()),
        )


@dataclass
class InferenceProgressData:
    """Структурированные данные для события прогресса инференса."""
    stage: str
    progress: float  # 0.0 - 1.0
    message: str = ""
    model_name: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """Event bus для pub/sub паттерна."""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Callable[[Event], Any]]] = {}
        self._async_handlers: dict[EventType, list[Callable[[Event], Any]]] = {}

    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], Any],
        async_mode: bool = False,
    ) -> None:
        """Подписка на событие."""
        if async_mode:
            if event_type not in self._async_handlers:
                self._async_handlers[event_type] = []
            self._async_handlers[event_type].append(handler)
        else:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

        logger.debug(
            "Subscribed to event %s (async=%s)", event_type.value, async_mode
        )

    def unsubscribe(
        self, event_type: EventType, handler: Callable[[Event], Any]
    ) -> None:
        """Отписка от события."""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h != handler
            ]

        if event_type in self._async_handlers:
            self._async_handlers[event_type] = [
                h for h in self._async_handlers[event_type] if h != handler
            ]

    def publish(self, event: Event) -> None:
        """
        Публикация события (синхронная).
        Используется из синхронного ML-кода (например, VCPipeline).
        """
        logger.debug("Publishing event: %s", event.type.value)

        # Sync handlers
        if event.type in self._handlers:
            for handler in self._handlers[event.type]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(
                        "Error in event handler for %s: %s",
                        event.type.value,
                        e,
                        exc_info=True,
                    )

        # Async handlers (schedule in background if loop is running)
        if event.type in self._async_handlers:
            try:
                loop = asyncio.get_running_loop()
                for handler in self._async_handlers[event.type]:
                    try:
                        loop.create_task(self._run_async_handler(handler, event))
                    except Exception as e:
                        logger.error(
                            "Error scheduling async handler for %s: %s",
                            event.type.value,
                            e,
                            exc_info=True,
                        )
            except RuntimeError:
                # No running event loop - skip async handlers
                logger.debug(
                    "No running event loop, skipping async handlers for %s",
                    event.type.value,
                )

    async def publish_async(self, event: Event) -> None:
        """Публикация события (асинхронная)."""
        logger.debug("Publishing async event: %s", event.type.value)

        # Sync handlers
        if event.type in self._handlers:
            for handler in self._handlers[event.type]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(
                        "Error in sync handler for %s: %s",
                        event.type.value,
                        e,
                        exc_info=True,
                    )

        # Async handlers
        if event.type in self._async_handlers:
            tasks = [
                self._run_async_handler(handler, event)
                for handler in self._async_handlers[event.type]
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_async_handler(
        self, handler: Callable[[Event], Any], event: Event
    ) -> None:
        """Запуск асинхронного обработчика."""
        try:
            await handler(event)
        except Exception as e:
            logger.error(
                "Error in async handler for %s: %s",
                event.type.value,
                e,
                exc_info=True,
            )

    def clear(self) -> None:
        """Очистка всех подписок."""
        self._handlers.clear()
        self._async_handlers.clear()
        logger.debug("Cleared all event subscriptions")


# Singleton instance
event_bus = EventBus()