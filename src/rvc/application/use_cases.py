"""Use Cases (Интеракторы) для оркестрации бизнес-логики."""
import logging
import tempfile
from pathlib import Path
from typing import Optional

from rvc.application.dto import InferenceRequest, InferenceResponse
from rvc.core.domain.entities import InferenceConfig
from rvc.core.domain.services import AudioService, InferenceService, ModelService
from rvc.core.exceptions import RVCError
from rvc.core.utils import validate_path_within_root

logger = logging.getLogger(__name__)


class ModelManagementUseCase:
    """Use Case для управления моделями."""

    def __init__(self, model_service: ModelService) -> None:
        self._model_service = model_service

    async def load_model(self, model_path: Path, allowed_root: Path | None = None) -> str:
        """Загрузить модель в память."""
        import asyncio

        # 🔒 SECURITY: проверка path traversal если указан allowed_root
        if allowed_root is not None:
            try:
                model_path = validate_path_within_root(model_path, allowed_root)
            except ValueError as e:
                return f"✗ Ошибка безопасности: {e}"

        try:
            model_info = await asyncio.to_thread(
                self._model_service.load_model, model_path
            )
            return f"✓ Модель '{model_info.name}' успешно загружена."
        except RVCError as e:
            return f"✗ Ошибка: {e.message}"
        except Exception as e:
            logger.exception("Непредвиденная ошибка при загрузке модели")
            return f"✗ Критическая ошибка: {str(e)}"

    def unload_model(self) -> str:
        """Выгрузить модель из памяти."""
        self._model_service.unload_model()
        return "✓ Модели выгружены из памяти (VRAM очищена)."


class VoiceConversionUseCase:
    """Use Case для преобразования голоса."""

    def __init__(
        self,
        inference_service: InferenceService,
        audio_service: AudioService,
        project_root: Path | None = None,
    ) -> None:
        self._inference_service = inference_service
        self._audio_service = audio_service
        self._project_root = project_root

    async def convert_single(self, request: InferenceRequest) -> InferenceResponse:
        """Преобразовать один аудиофайл."""
        # 🔒 SECURITY: Валидация путей против Path Traversal
        if self._project_root is not None:
            try:
                validate_path_within_root(
                    request.input_audio_path, self._project_root
                )
                validate_path_within_root(
                    request.model_path, self._project_root
                )
                if request.index_path is not None:
                    validate_path_within_root(
                        request.index_path, self._project_root
                    )
            except ValueError as e:
                raise RVCError(
                    f"Path traversal attempt blocked: {e}",
                    details={"blocked_path": str(e)},
                ) from e

        config = InferenceConfig(
            pitch_shift=request.pitch_shift,
            index_rate=request.index_rate,
            protect=request.protect,
            resample_sr=request.resample_sr,
            rms_mix_rate=request.rms_mix_rate,
            filter_radius=request.filter_radius,
        )

        # Выполняем инференс
        result = await self._inference_service.infer_single(
            audio_path=request.input_audio_path,
            config=config,
            index_path=request.index_path,
        )

        # Сохраняем результат во временный файл
        suffix = f".{request.output_format}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            temp_path = Path(f.name)

        # Сохраняем и конвертируем при необходимости
        self._audio_service.save_audio(temp_path, result.audio, result.sample_rate)
        if request.output_format != "wav":
            final_path = temp_path.with_suffix(suffix)
            self._audio_service.convert_format(
                temp_path, final_path, request.output_format
            )
            temp_path.unlink(missing_ok=True)
        else:
            final_path = temp_path

        return InferenceResponse(
            output_audio_path=final_path,
            sample_rate=result.sample_rate,
            duration=result.duration,
            processing_time=result.processing_time,
            model_name=result.model_name,
        )