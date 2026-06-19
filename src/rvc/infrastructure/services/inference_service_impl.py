"""Реализация InferenceService."""
import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator, Optional

from rvc.config import RVCConfig
from rvc.core.domain.entities import (
    BatchJob,
    BatchItem,
    InferenceConfig,
    InferenceResult,
)
from rvc.core.domain.repositories import AudioRepository, IndexRepository
from rvc.core.domain.services import InferenceService, ModelService
from rvc.core.events import Event, EventBus, EventType
from rvc.core.pipeline import VCPipeline

logger = logging.getLogger(__name__)


class InferenceServiceImpl(InferenceService):
    """
    Реализация сервиса инференса, связывающая домен и ML pipeline.
    Поддерживает streaming прогресса через EventBus.
    """

    def __init__(
        self,
        config: RVCConfig,
        model_service: ModelService,
        audio_repository: AudioRepository,
        index_repository: IndexRepository,
        event_bus: EventBus,
        pipeline: VCPipeline,
    ) -> None:
        self._config = config
        self._model_service = model_service
        self._audio_repo = audio_repository
        self._index_repo = index_repository
        self._event_bus = event_bus
        self._pipeline = pipeline

    def infer_single(
        self,
        audio_path: Path,
        config: InferenceConfig,
        index_path: Optional[Path] = None,
    ) -> InferenceResult:
        """
        Выполнить инференс для одного файла (синхронно).
        Для получения прогресса используйте infer_single_with_progress.
        """
        current_model = self._model_service.get_current_model()
        if not current_model:
            raise RuntimeError("Модель не загружена. Сначала загрузите модель через UI.")

        # Загружаем модель и индекс в пайплайн
        self._pipeline.load_model(current_model.path)
        if index_path:
            self._pipeline.load_index(index_path)

        # Вызов ML пайплайна
        pipeline_result = self._pipeline.convert(
            str(audio_path),
            inference_config=config,
        )

        # Адаптация результата пайплайна к доменной сущности
        return InferenceResult(
            audio=pipeline_result.audio,
            sample_rate=pipeline_result.sample_rate,
            duration=pipeline_result.duration,
            processing_time=pipeline_result.processing_time,
            model_name=current_model.name,
            config=config,
        )

    async def infer_single_with_progress(
        self,
        audio_path: Path,
        config: InferenceConfig,
        index_path: Optional[Path] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Выполнить инференс с streaming прогресса.
        Yields события прогресса в реальном времени.
        """
        current_model = self._model_service.get_current_model()
        if not current_model:
            raise RuntimeError("Модель не загружена.")

        # Создаём очередь для передачи событий из sync -> async
        progress_queue: asyncio.Queue[dict | None] = asyncio.Queue()

        # Подписываемся на события прогресса
        def on_progress(event: Event) -> None:
            try:
                progress_queue.put_nowait(event.data)
            except asyncio.QueueFull:
                pass

        self._event_bus.subscribe(EventType.INFERENCE_PROGRESS, on_progress)
        self._event_bus.subscribe(EventType.INFERENCE_STAGE_STARTED, on_progress)
        self._event_bus.subscribe(EventType.INFERENCE_STAGE_COMPLETED, on_progress)

        try:
            # Загружаем модель и индекс
            self._pipeline.load_model(current_model.path)
            if index_path:
                self._pipeline.load_index(index_path)

            # Запускаем инференс в отдельном потоке
            async def run_inference():
                result = await asyncio.to_thread(
                    self._pipeline.convert,
                    str(audio_path),
                    config,
                )
                # Сигнал завершения
                await progress_queue.put(None)
                return result

            inference_task = asyncio.create_task(run_inference())

            # Читаем события прогресса из очереди
            while True:
                try:
                    event_data = await asyncio.wait_for(
                        progress_queue.get(), timeout=0.1
                    )
                    if event_data is None:
                        break
                    yield {"type": "progress", "data": event_data}
                except asyncio.TimeoutError:
                    # Проверяем, завершилась ли задача
                    if inference_task.done():
                        break
                    continue

            # Получаем результат
            pipeline_result = await inference_task

            yield {
                "type": "completed",
                "data": {
                    "audio": pipeline_result.audio,
                    "sample_rate": pipeline_result.sample_rate,
                    "duration": pipeline_result.duration,
                    "processing_time": pipeline_result.processing_time,
                    "model_name": current_model.name,
                },
            }

        finally:
            # Отписываемся от событий
            self._event_bus.unsubscribe(EventType.INFERENCE_PROGRESS, on_progress)
            self._event_bus.unsubscribe(EventType.INFERENCE_STAGE_STARTED, on_progress)
            self._event_bus.unsubscribe(EventType.INFERENCE_STAGE_COMPLETED, on_progress)

    def infer_batch(
        self,
        audio_paths: list[Path],
        output_dir: Path,
        config: InferenceConfig,
        index_path: Optional[Path] = None,
    ) -> BatchJob:
        """Пакетный инференс (заготовка)."""
        current_model = self._model_service.get_current_model()
        model_name = current_model.name if current_model else "unknown"
        items = [
            BatchItem(
                input_path=p,
                output_path=output_dir / f"{p.stem}_converted.wav"
            )
            for p in audio_paths
        ]
        return BatchJob(items=items, config=config, model_name=model_name)

    def cancel_batch(self, job_id: str) -> bool:
        """Отмена пакетной задачи."""
        logger.warning("Отмена пакетной задачи пока не реализована.")
        return False