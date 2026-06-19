"""
Тесты модуля rvc.core.events (EventBus).
Покрывают:
- Event.create
- Subscribe/Unsubscribe/Publish
- Sync и async handlers
"""
from __future__ import annotations

import asyncio

import pytest

from rvc.core.events import Event, EventBus, EventType


class TestEvent:
    """Тесты класса Event."""

    def test_create_event(self) -> None:
        """Event.create возвращает корректный объект."""
        event = Event.create(
            EventType.MODEL_LOADING_STARTED,
            data={"model_name": "test"},
        )

        assert event.type == EventType.MODEL_LOADING_STARTED
        assert event.data["model_name"] == "test"
        assert event.timestamp is not None
        assert event.event_id  # UUID

    def test_unique_event_ids(self) -> None:
        """Каждый Event имеет уникальный ID."""
        e1 = Event.create(EventType.INFERENCE_STARTED, {})
        e2 = Event.create(EventType.INFERENCE_STARTED, {})
        assert e1.event_id != e2.event_id


class TestEventBus:
    """Тесты EventBus."""

    def test_subscribe_and_publish(self, event_bus: EventBus) -> None:
        """Подписка и публикация события."""
        received: list[Event] = []

        def handler(event: Event) -> None:
            received.append(event)

        event_bus.subscribe(EventType.INFERENCE_STARTED, handler)
        event_bus.publish(Event.create(EventType.INFERENCE_STARTED, {"test": 1}))

        assert len(received) == 1
        assert received[0].data["test"] == 1

    def test_multiple_handlers(self, event_bus: EventBus) -> None:
        """Несколько handlers получают событие."""
        received1: list[Event] = []
        received2: list[Event] = []

        event_bus.subscribe(
            EventType.INFERENCE_STARTED,
            lambda e: received1.append(e),
        )
        event_bus.subscribe(
            EventType.INFERENCE_STARTED,
            lambda e: received2.append(e),
        )

        event_bus.publish(Event.create(EventType.INFERENCE_STARTED, {}))

        assert len(received1) == 1
        assert len(received2) == 1

    def test_handler_receives_only_matching_event_type(
        self, event_bus: EventBus
    ) -> None:
        """Handler получает только события своего типа."""
        received: list[Event] = []

        event_bus.subscribe(
            EventType.INFERENCE_STARTED,
            lambda e: received.append(e),
        )

        # Публикуем другой тип
        event_bus.publish(Event.create(EventType.MODEL_LOADING_STARTED, {}))

        assert len(received) == 0

    def test_unsubscribe(self, event_bus: EventBus) -> None:
        """Unsubscribe прекращает получение событий."""
        received: list[Event] = []

        def handler(event: Event) -> None:
            received.append(event)

        event_bus.subscribe(EventType.INFERENCE_STARTED, handler)
        event_bus.unsubscribe(EventType.INFERENCE_STARTED, handler)

        event_bus.publish(Event.create(EventType.INFERENCE_STARTED, {}))

        assert len(received) == 0

    def test_handler_exception_does_not_break_others(
        self, event_bus: EventBus
    ) -> None:
        """Исключение в одном handler не ломает другие."""
        received: list[Event] = []

        def bad_handler(event: Event) -> None:
            raise RuntimeError("Intentional error")

        def good_handler(event: Event) -> None:
            received.append(event)

        event_bus.subscribe(EventType.INFERENCE_STARTED, bad_handler)
        event_bus.subscribe(EventType.INFERENCE_STARTED, good_handler)

        # Не должно падать
        event_bus.publish(Event.create(EventType.INFERENCE_STARTED, {}))

        assert len(received) == 1

    def test_clear_removes_all_subscriptions(
        self, event_bus: EventBus
    ) -> None:
        """clear() удаляет все подписки."""
        received: list[Event] = []
        event_bus.subscribe(
            EventType.INFERENCE_STARTED,
            lambda e: received.append(e),
        )

        event_bus.clear()
        event_bus.publish(Event.create(EventType.INFERENCE_STARTED, {}))

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_async_handler(self, event_bus: EventBus) -> None:
        """Async handlers выполняются через publish_async."""
        received: list[Event] = []

        async def async_handler(event: Event) -> None:
            await asyncio.sleep(0.01)
            received.append(event)

        event_bus.subscribe(
            EventType.INFERENCE_STARTED,
            async_handler,
            async_mode=True,
        )

        await event_bus.publish_async(
            Event.create(EventType.INFERENCE_STARTED, {})
        )

        assert len(received) == 1


class TestEventType:
    """Тесты EventType enum."""

    def test_all_event_types_have_string_values(self) -> None:
        """Все EventType имеют строковые значения."""
        for event_type in EventType:
            assert isinstance(event_type.value, str)

    def test_model_events_exist(self) -> None:
        """Существуют события загрузки модели."""
        assert EventType.MODEL_LOADING_STARTED
        assert EventType.MODEL_LOADING_COMPLETED
        assert EventType.MODEL_LOADING_FAILED
        assert EventType.MODEL_UNLOADED

    def test_inference_events_exist(self) -> None:
        """Существуют события инференса."""
        assert EventType.INFERENCE_STARTED
        assert EventType.INFERENCE_COMPLETED
        assert EventType.INFERENCE_FAILED
        assert EventType.INFERENCE_PROGRESS