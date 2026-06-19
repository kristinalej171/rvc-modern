"""
Оптимизированный Voice Conversion Pipeline.

🔧 Phase 3 Fixes:
- Добавлен feature_backend параметр (fairseq по умолчанию)
- CUDA Streams корректно освобождаются после batch-обработки
- Используется FairseqFeatureExtractor по умолчанию для совместимости
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Literal

import numpy as np
import torch
import torch.nn.functional as F

from rvc.core.events import Event, EventBus, EventType, InferenceProgressData
from rvc.core.feature_extractor import (
    FeatureModelType,
    create_feature_extractor,
)
from rvc.core.indexer import FeatureIndexer
from rvc.core.models import SynthesizerWrapper
from rvc.core.pitch import (
    PitchExtractor,
    PitchMethod,
    create_pitch_extractor,
    f0_to_coarse,
    transpose_f0,
)

if TYPE_CHECKING:
    from rvc.config import RVCConfig

logger = logging.getLogger(__name__)


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
    feature_model: FeatureModelType = FeatureModelType.HUBERT_BASE
    # 🔧 Phase 3: Backend для feature extraction
    # fairseq = совместимость с существующими RVC моделями (рекомендуется)
    # huggingface = экспериментальные модели (ContentVec, WavLM)
    feature_backend: Literal["fairseq", "huggingface"] = "fairseq"

    def validate(self) -> None:
        if not -24 <= self.pitch_shift <= 24:
            raise ValueError("pitch_shift должен быть в [-24, 24]")
        if not 0.0 <= self.index_rate <= 1.0:
            raise ValueError("index_rate должен быть в [0.0, 1.0]")
        if not 0.0 <= self.protect <= 0.5:
            raise ValueError("protect должен быть в [0.0, 0.5]")
        if not 0 <= self.filter_radius <= 7:
            raise ValueError("filter_radius должен быть в [0, 7]")


@dataclass
class InferenceResult:
    """Результат инференса."""

    audio: np.ndarray
    sample_rate: int
    duration: float
    processing_time: float
    model_name: str
    config: InferenceConfig
    stats: dict = field(default_factory=dict)


class VCPipeline:
    """Полный pipeline voice conversion."""

    def __init__(
        self,
        config: RVCConfig,
        event_bus: EventBus | None = None,
    ) -> None:
        self.config = config
        self._event_bus = event_bus
        self._feature_extractor = None
        self._pitch_extractor: PitchExtractor | None = None
        self._synthesizer: SynthesizerWrapper | None = None
        self._indexer: FeatureIndexer | None = None
        self._inference_config: InferenceConfig = InferenceConfig()
        self._target_layer: int = 9

    def _publish_progress(
        self, stage: str, progress: float, message: str = "", model_name: str = ""
    ) -> None:
        if self._event_bus is None:
            return
        progress_data = InferenceProgressData(
            stage=stage,
            progress=max(0.0, min(1.0, progress)),
            message=message,
            model_name=model_name,
        )
        self._event_bus.publish(
            Event.create(EventType.INFERENCE_PROGRESS, data=progress_data.__dict__)
        )

    def _publish_stage(
        self, stage: str, started: bool = True, message: str = ""
    ) -> None:
        if self._event_bus is None:
            return
        event_type = (
            EventType.INFERENCE_STAGE_STARTED
            if started
            else EventType.INFERENCE_STAGE_COMPLETED
        )
        self._event_bus.publish(
            Event.create(event_type, data={"stage": stage, "message": message})
        )

    @property
    def feature_extractor(self):
        """
        Ленивая инициализация feature extractor.
        
        🔧 Phase 3: Теперь использует backend из конфигурации.
        Дефолт — fairseq для совместимости с существующими моделями.
        """
        if self._feature_extractor is None:
            self._feature_extractor = create_feature_extractor(
                self.config,
                model_type=self._inference_config.feature_model,
                use_compile=True,
                backend=self._inference_config.feature_backend,
            )
        return self._feature_extractor

    @property
    def pitch_extractor(self) -> PitchExtractor:
        if self._pitch_extractor is None:
            self._pitch_extractor = create_pitch_extractor(
                self._inference_config.pitch_method,
                self.config,
                fallback=True,
            )
        return self._pitch_extractor

    @property
    def synthesizer(self) -> SynthesizerWrapper:
        if self._synthesizer is None:
            self._synthesizer = SynthesizerWrapper(
                self.config, use_compile=True, use_cuda_graphs=False
            )
        return self._synthesizer

    def load_model(self, model_path: str | Path) -> None:
        self._publish_stage("load_model", started=True, message="Загрузка модели")
        self._publish_progress("load_model", 0.0, "Загрузка модели")

        metadata = self.synthesizer.load(model_path)
        self._target_layer = 12 if metadata.version == "v2" else 9

        self._publish_progress("warmup", 0.5, "Warmup модели")
        self.synthesizer.warmup(sequence_length=200)

        self._publish_progress("load_model", 1.0, "Модель загружена")
        self._publish_stage("load_model", started=False, message="Модель загружена")

    def load_index(self, index_path: str | Path | None) -> None:
        if index_path is None or str(index_path) == "":
            if self._indexer is not None:
                self._indexer.unload()
                self._indexer = None
            return

        if self._indexer is not None:
            self._indexer.unload()

        self._publish_stage("load_index", started=True, message="Загрузка индекса")
        self._indexer = FeatureIndexer.from_file(
            self.config, index_path, use_gpu=torch.cuda.is_available()
        )
        self._publish_stage("load_index", started=False, message="Индекс загружен")

    def unload(self) -> None:
        if self._synthesizer is not None:
            self._synthesizer.unload()
            self._synthesizer = None
        if self._indexer is not None:
            self._indexer.unload()
            self._indexer = None
        if self._feature_extractor is not None:
            self._feature_extractor.clear_cache()
            self._feature_extractor = None
        if self._pitch_extractor is not None:
            self._pitch_extractor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Все компоненты пайплайна выгружены")

    @staticmethod
    def cleanup_all() -> None:
        logger.info("Начало глобальной очистки ML-ресурсов...")
        FeatureIndexer.unload_all_static()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        logger.info("Глобальная очистка ML-ресурсов завершена")

    @torch.inference_mode()
    def convert(
        self,
        input_audio: str | Path | np.ndarray,
        inference_config: InferenceConfig | None = None,
        sample_rate: int = 16000,
    ) -> InferenceResult:
        start_time = time.perf_counter()
        stats: dict = {}
        model_name = ""

        if inference_config is not None:
            self._inference_config = inference_config
            self._feature_extractor = None
            self._pitch_extractor = None

        self._inference_config.validate()

        if self._event_bus:
            self._event_bus.publish(Event.create(EventType.INFERENCE_STARTED, data={}))

        try:
            # === Шаг 1: Загрузка аудио ===
            self._publish_stage("load_audio", started=True)
            audio = self._load_audio(input_audio, sample_rate)
            stats["input_duration"] = len(audio) / sample_rate
            self._publish_stage("load_audio", started=False)

            # === Шаг 2: Извлечение features ===
            self._publish_stage("features", started=True)
            feats = self.feature_extractor.extract(
                audio,
                sample_rate=16000,
                output_layer=self._target_layer,
            )
            stats["features_shape"] = tuple(feats.shape)
            self._publish_stage("features", started=False)

            # === Шаг 3: Индексация (retrieval) ===
            self._publish_stage("indexing", started=True)
            if self._indexer is not None and self._inference_config.index_rate > 0:
                feats = self._indexer.blend_features(
                    feats, index_rate=self._inference_config.index_rate
                )
            self._publish_stage("indexing", started=False)

            # === Шаг 4: Апсемплинг features ×2 ===
            feats = F.interpolate(
                feats.permute(0, 2, 1), scale_factor=2
            ).permute(0, 2, 1)

            # 🔧 Phase 2 Fix (сохранено): Единый источник истины для длины
            canonical_length = feats.shape[1]

            # === Шаг 5: Извлечение F0 ===
            pitch = None
            pitchf = None
            if self.synthesizer.metadata.has_f0:
                self._publish_stage("pitch", started=True)

                f0 = self.pitch_extractor.compute_f0(audio, p_len=None)

                if self._inference_config.filter_radius > 0:
                    from scipy.signal import medfilt
                    f0 = medfilt(
                        f0,
                        kernel_size=2 * self._inference_config.filter_radius + 1,
                    )

                f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
                f0 = np.clip(f0, 0, 2000)
                f0 = transpose_f0(f0, self._inference_config.pitch_shift)

                # Приведение длины F0 к canonical_length
                if len(f0) != canonical_length:
                    logger.debug(
                        "F0 length mismatch: %d vs %d. Интерполяция.",
                        len(f0), canonical_length,
                    )
                    f0 = np.interp(
                        np.linspace(0, len(f0) - 1, canonical_length),
                        np.arange(len(f0)),
                        f0,
                    )

                f0_coarse = f0_to_coarse(f0)
                if len(f0_coarse) != canonical_length:
                    f0_coarse = np.interp(
                        np.linspace(0, len(f0_coarse) - 1, canonical_length),
                        np.arange(len(f0_coarse)),
                        f0_coarse.astype(np.float32),
                    ).astype(np.int32)

                pitch = torch.from_numpy(f0_coarse).long().unsqueeze(0)
                pitchf = torch.from_numpy(f0).float().unsqueeze(0)

                assert pitch.shape[1] == canonical_length, (
                    f"pitch length mismatch: {pitch.shape[1]} != {canonical_length}"
                )
                assert pitchf.shape[1] == canonical_length, (
                    f"pitchf length mismatch: {pitchf.shape[1]} != {canonical_length}"
                )

                self._publish_stage("pitch", started=False)

            # === Шаг 6: Синтез через VITS ===
            self._publish_stage("synthesis", started=True)

            p_len_tensor = torch.tensor([canonical_length], dtype=torch.long)
            sid = torch.tensor([0], dtype=torch.long)

            audio_out = self.synthesizer.infer(
                feats=feats,
                p_len=p_len_tensor,
                sid=sid,
                pitch=pitch,
                pitchf=pitchf,
            )

            audio_np = audio_out.cpu().numpy()
            tgt_sr = self.synthesizer.metadata.sample_rate
            model_name = self.synthesizer.metadata.path.stem

            self._publish_stage("synthesis", started=False)

            # === Шаг 7: Постобработка ===
            self._publish_stage("postprocess", started=True)

            if (
                self._inference_config.resample_sr > 0
                and self._inference_config.resample_sr != tgt_sr
            ):
                audio_np = self._resample(
                    audio_np, tgt_sr, self._inference_config.resample_sr
                )
                tgt_sr = self._inference_config.resample_sr

            audio_np = self._normalize(audio_np)

            self._publish_stage("postprocess", started=False)

            # === Результат ===
            processing_time = time.perf_counter() - start_time
            stats["processing_time"] = processing_time
            stats["rtf"] = processing_time / (len(audio_np) / tgt_sr)
            stats["feature_backend"] = self._inference_config.feature_backend

            self._publish_progress("done", 1.0, "Готово", model_name=model_name)

            result = InferenceResult(
                audio=audio_np,
                sample_rate=tgt_sr,
                duration=len(audio_np) / tgt_sr,
                processing_time=processing_time,
                model_name=model_name,
                config=self._inference_config,
                stats=stats,
            )

            if self._event_bus:
                self._event_bus.publish(
                    Event.create(
                        EventType.INFERENCE_COMPLETED,
                        data={
                            "model_name": model_name,
                            "processing_time": processing_time,
                            "duration": result.duration,
                        },
                    )
                )

            return result

        except Exception as e:
            if self._event_bus:
                self._event_bus.publish(
                    Event.create(
                        EventType.INFERENCE_FAILED,
                        data={"error": str(e), "model_name": model_name},
                    )
                )
            raise

    async def convert_async(
        self,
        input_audio: str | Path | np.ndarray,
        inference_config: InferenceConfig | None = None,
        sample_rate: int = 16000,
    ) -> InferenceResult:
        return await asyncio.to_thread(
            self.convert, input_audio, inference_config, sample_rate
        )

    def _load_audio(
        self, input_audio: str | Path | np.ndarray, sample_rate: int
    ) -> np.ndarray:
        if isinstance(input_audio, (str, Path)):
            from rvc.core.audio import AudioProcessor
            processor = AudioProcessor(self.config)
            audio = processor.load_audio(input_audio, target_sr=16000)
        else:
            audio = input_audio.astype(np.float32)
        return audio

    def _resample(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        max_val = np.abs(audio).max()
        if max_val > 0.95:
            audio = audio * (0.95 / max_val)
        return audio

    @staticmethod
    def _cleanup_streams(streams: list[torch.cuda.Stream]) -> None:
        """
        🔧 Phase 3: Централизованная очистка CUDA streams.
        
        Синхронизирует все streams и очищает CUDA cache.
        Предотвращает утечки VRAM при batch-обработке.
        """
        for stream in streams:
            try:
                stream.synchronize()
            except Exception as e:
                logger.debug("Stream sync error: %s", e)
        
        streams.clear()
        
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logger.debug("CUDA cache clear error: %s", e)

    async def convert_batch(
        self,
        input_audios: list[str | Path],
        inference_config: InferenceConfig | None = None,
        max_concurrent: int = 2,
    ) -> AsyncIterator[tuple[int, InferenceResult | Exception]]:
        """
        Batch conversion с использованием CUDA Streams.
        
        🔧 Phase 3 Fix: CUDA Streams теперь корректно освобождаются
        через try/finally, предотвращая утечки VRAM при многократных
        вызовах batch-обработки.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        use_streams = torch.cuda.is_available() and max_concurrent > 1
        streams = (
            [torch.cuda.Stream() for _ in range(max_concurrent)]
            if use_streams else []
        )
        stream_idx = 0

        async def _process(idx: int, audio_path: str | Path):
            nonlocal stream_idx
            async with semaphore:
                try:
                    if use_streams:
                        current_stream = streams[stream_idx % max_concurrent]
                        stream_idx += 1
                        with torch.cuda.stream(current_stream):
                            result = await self.convert_async(
                                audio_path, inference_config
                            )
                        current_stream.synchronize()
                        return idx, result
                    else:
                        result = await self.convert_async(
                            audio_path, inference_config
                        )
                        return idx, result
                except Exception as e:
                    return idx, e
                finally:
                    # Периодическая дефрагментация VRAM
                    if torch.cuda.is_available() and idx % 5 == 0:
                        torch.cuda.empty_cache()

        # 🔧 Phase 3: Явное освобождение streams через try/finally
        try:
            tasks = [
                asyncio.create_task(_process(i, path))
                for i, path in enumerate(input_audios)
            ]
            for task in asyncio.as_completed(tasks):
                yield await task
        finally:
            # Гарантированная очистка даже при отмене или исключении
            self._cleanup_streams(streams)
            logger.debug(
                "Batch conversion cleanup: %d streams freed",
                max_concurrent if use_streams else 0,
            )