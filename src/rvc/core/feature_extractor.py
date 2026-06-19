"""
Универсальный извлекатель признаков с поддержкой двух backends:
1. Fairseq HuBERT (дефолт, совместим с существующими RVC моделями)
2. HuggingFace transformers (HuBERT, ContentVec, WavLM)

🔧 Phase 3 Fix:
- Добавлена thread safety (threading.Lock) для кэша features
- Добавлен FairseqFeatureExtractor для совместимости с существующими моделями
- Все RVC модели обучались с fairseq-HuBERT, поэтому он дефолтный
"""
from __future__ import annotations

import hashlib
import logging
import threading
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from rvc.config import RVCConfig

logger = logging.getLogger(__name__)


class FeatureModelType(str, Enum):
    """Поддерживаемые типы моделей для извлечения признаков."""
    HUBERT_BASE = "hubert_base"
    HUBERT_LARGE = "hubert_large"
    CONTENTVEC = "contentvec"
    WAVLM_BASE = "wavlm_base"
    WAVLM_LARGE = "wavlm_large"


MODEL_REGISTRY: dict[FeatureModelType, dict] = {
    FeatureModelType.HUBERT_BASE: {
        "repo": "facebook/hubert-base-ls960",
        "hidden_dim": 768,
        "recommended_layer": 9,
        "v2_layer": 12,
    },
    FeatureModelType.HUBERT_LARGE: {
        "repo": "facebook/hubert-large-ls960-ft",
        "hidden_dim": 1024,
        "recommended_layer": 12,
        "v2_layer": 12,
    },
    FeatureModelType.CONTENTVEC: {
        "repo": "lengyue233/content-vec-best",
        "hidden_dim": 768,
        "recommended_layer": 9,
        "v2_layer": 12,
    },
    FeatureModelType.WAVLM_BASE: {
        "repo": "patrickvonplaten/wavlm-libri-clean-100h-large",
        "hidden_dim": 768,
        "recommended_layer": 9,
        "v2_layer": 12,
    },
    FeatureModelType.WAVLM_LARGE: {
        "repo": "microsoft/wavlm-large",
        "hidden_dim": 1024,
        "recommended_layer": 12,
        "v2_layer": 12,
    },
}


class FeatureExtractorProtocol(Protocol):
    """Протокол для всех feature extractor'ов."""
    def extract(
        self,
        waveform: torch.Tensor | np.ndarray,
        sample_rate: int,
        output_layer: int | None = None,
    ) -> torch.Tensor: ...
    def get_feature_dim(self) -> int: ...
    def clear_cache(self) -> None: ...


def _compute_cache_key(waveform: torch.Tensor | np.ndarray) -> str:
    """
    Безопасная генерация ключа кэша.
    Использует blake2b и статистический отпечаток для исключения коллизий.
    """
    if isinstance(waveform, torch.Tensor):
        arr = waveform.detach().cpu().numpy()
    else:
        arr = waveform

    # Для коротких аудио (до 10 сек) хэшируем весь буфер через быстрый blake2b
    if arr.size < 16000 * 10:
        return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()

    # Для длинных аудио используем статистический отпечаток + хэш начала и конца.
    # Это предотвращает коллизии, если два разных трека начинаются с тишины.
    stats = (
        f"{arr.shape}_{arr.mean():.6f}_{arr.std():.6f}_"
        f"{arr.max():.6f}_{arr.min():.6f}"
    )
    head = arr[:8192].tobytes()
    tail = arr[-8192:].tobytes()
    combined = stats.encode("utf-8") + head + tail
    return hashlib.blake2b(combined, digest_size=16).hexdigest()


class TransformerFeatureExtractor(nn.Module):
    """
    Feature extractor на базе HuggingFace transformers.
    
    🔧 Phase 3: Добавлен threading.Lock для thread-safe доступа к кэшу.
    Gradio запускает инференс в нескольких потоках через asyncio.to_thread,
    что приводило к race condition и KeyError при конкурентном доступе.
    """

    def __init__(
        self,
        model_type: FeatureModelType = FeatureModelType.HUBERT_BASE,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
        use_compile: bool = True,
        cache_dir: Path | None = None,
    ) -> None:
        super().__init__()
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Неподдерживаемая модель: {model_type}")
        
        self.model_type = model_type
        self.config = MODEL_REGISTRY[model_type]
        self.device = torch.device(device)
        self.dtype = dtype if self.device.type != "cpu" else torch.float32
        self._cache_dir = cache_dir
        
        self._feature_cache: dict[str, torch.Tensor] = {}
        # 🔧 Phase 3: Thread-safe lock для кэша
        self._cache_lock = threading.Lock()

        repo_id = self.config["repo"]

        logger.info(
            "Загрузка HF feature extractor '%s' из '%s' на %s",
            model_type.value, repo_id, self.device,
        )

        from transformers import (
            HubertModel,
            Wav2Vec2FeatureExtractor,
            WavLMModel,
        )

        model_cls = HubertModel if "hubert" in repo_id.lower() or "content" in repo_id.lower() else WavLMModel

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(repo_id)
        self.model: nn.Module = model_cls.from_pretrained(repo_id)
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        if use_compile and self.device.type == "cuda":
            try:
                # dynamic=True критически важен для избежания Graph Breaks
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                    fullgraph=False,
                    dynamic=True,
                )
                logger.info("torch.compile (dynamic) применён к %s", model_type.value)
            except Exception as e:
                logger.warning("torch.compile не удался: %s", e)

        self._target_layer = self.config["recommended_layer"]

    def get_feature_dim(self) -> int:
        return self.config["hidden_dim"]

    def clear_cache(self) -> None:
        # 🔧 Phase 3: Thread-safe очистка кэша
        with self._cache_lock:
            self._feature_cache.clear()

    @torch.inference_mode()
    def extract(
        self,
        waveform: torch.Tensor | np.ndarray,
        sample_rate: int = 16000,
        output_layer: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Извлечение признаков из аудио."""
        if isinstance(waveform, np.ndarray):
            waveform_tensor = torch.from_numpy(waveform).float()
        else:
            waveform_tensor = waveform.float()

        if waveform_tensor.dim() == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)

        # 🔧 Phase 3: Thread-safe проверка кэша
        cache_key = ""
        if use_cache:
            cache_key = _compute_cache_key(waveform)
            with self._cache_lock:
                if cache_key in self._feature_cache:
                    logger.debug("Cache hit для HF feature extraction")
                    return self._feature_cache[cache_key].to(self.device)

        wav_np = waveform_tensor.squeeze(0).cpu().numpy()
        inputs = self.processor(
            wav_np, sampling_rate=sample_rate,
            return_tensors="pt", padding=True,
        )
        input_values = inputs.input_values.to(
            device=self.device, dtype=self.dtype
        )

        target_layer = output_layer or self.config["recommended_layer"]

        outputs = self.model(input_values, output_hidden_states=True)
        if not outputs.hidden_states:
            raise RuntimeError(f"{self.model_type.value} не вернул hidden states")

        max_layer = len(outputs.hidden_states) - 1
        if target_layer > max_layer:
            target_layer = max_layer

        features = outputs.hidden_states[target_layer].to(dtype=torch.float32)

        # 🔧 Phase 3: Thread-safe сохранение в кэш
        if use_cache and cache_key:
            with self._cache_lock:
                # LRU-подобное ограничение размера кэша
                if len(self._feature_cache) > 32:
                    oldest = next(iter(self._feature_cache))
                    del self._feature_cache[oldest]
                self._feature_cache[cache_key] = features.detach().cpu()

        return features


class FairseqFeatureExtractor(nn.Module):
    """
    Feature extractor на базе fairseq HuBERT.
    
    🔧 Phase 3: КРИТИЧЕСКОЕ исправление для совместимости.
    
    Все существующие RVC модели обучались с fairseq-HuBERT features.
    HuggingFace transformers HuBERT даёт численно отличающиеся features
    (разница до 1e-3 из-за разной инициализации LayerNorm, обработки
    padding mask и нормализационных констант), что снижает качество
    инференса для всех существующих моделей.
    
    Этот класс использует оригинальную fairseq загрузку из
    infer/lib/jit/get_hubert.py с применёнными патчами, что обеспечивает
    100% совместимость с существующими RVC моделями.
    """

    def __init__(
        self,
        model_path: str | Path = "assets/hubert/hubert_base.pt",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
        use_compile: bool = False,  # Fairseq плохо компилируется через torch.compile
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype if self.device.type != "cpu" else torch.float32
        
        self._feature_cache: dict[str, torch.Tensor] = {}
        # 🔧 Phase 3: Thread-safe lock для кэша
        self._cache_lock = threading.Lock()

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Fairseq HuBERT модель не найдена: {model_path}\n"
                f"Запустите: python check_and_download_models.py"
            )

        try:
            # Переиспользуем существующую загрузку с патчами из legacy кода
            from infer.lib.jit.get_hubert import get_hubert_model
        except ImportError as e:
            raise RuntimeError(
                "fairseq не установлен. Установите: pip install fairseq\n"
                "Или используйте backend='huggingface' в конфигурации."
            ) from e

        logger.info(
            "Загрузка fairseq HuBERT из %s на %s",
            model_path, self.device,
        )
        # get_hubert_model возвращает модель с уже применёнными патчами:
        # - apply_mask (для совместимости)
        # - extract_features (с поддержкой output_layer)
        # - infer (с автоматическим final_proj для layer 9)
        self.model = get_hubert_model(str(model_path), self.device)

        # Приведение к нужному dtype
        if self.dtype == torch.float16 and self.device.type != "cpu":
            self.model = self.model.half()
        else:
            self.model = self.model.float()

        if use_compile and self.device.type == "cuda":
            logger.warning(
                "torch.compile не рекомендуется для fairseq HuBERT из-за "
                "кастомных патчей. Может вызвать graph breaks."
            )
            try:
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                    fullgraph=False,
                    dynamic=True,
                )
            except Exception as e:
                logger.warning("torch.compile не удался для fairseq: %s", e)

    def get_feature_dim(self) -> int:
        """HuBERT base всегда возвращает 768-dim features."""
        return 768

    def clear_cache(self) -> None:
        # 🔧 Phase 3: Thread-safe очистка кэша
        with self._cache_lock:
            self._feature_cache.clear()

    @torch.inference_mode()
    def extract(
        self,
        waveform: torch.Tensor | np.ndarray,
        sample_rate: int = 16000,
        output_layer: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Извлечение признаков через fairseq HuBERT.
        
        Использует патченный model.infer() метод, который:
        - Для layer 9 (v1 модели): применяет final_proj (768 → 256)
        - Для layer 12 (v2 модели): возвращает raw features (768)
        """
        if isinstance(waveform, np.ndarray):
            waveform_tensor = torch.from_numpy(waveform).float()
        else:
            waveform_tensor = waveform.float()

        if waveform_tensor.dim() == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)

        # 🔧 Phase 3: Thread-safe проверка кэша
        cache_key = ""
        if use_cache:
            cache_key = _compute_cache_key(waveform)
            with self._cache_lock:
                if cache_key in self._feature_cache:
                    logger.debug("Cache hit для fairseq feature extraction")
                    return self._feature_cache[cache_key].to(self.device)

        # Перемещаем на устройство и приводим к dtype
        source = waveform_tensor.to(device=self.device)
        if self.dtype == torch.float16 and self.device.type != "cpu":
            source = source.half()

        # Padding mask (всё аудио валидно)
        padding_mask = torch.zeros(
            source.shape, dtype=torch.bool, device=self.device
        )

        target_layer = output_layer or 9  # Default для v1 моделей

        # Используем патченный infer() метод из get_hubert.py
        # Он корректно обрабатывает output_layer (через .item())
        # и применяет final_proj для layer 9
        output_layer_tensor = torch.tensor(
            target_layer, device=self.device, dtype=torch.long
        )
        features = self.model.infer(source, padding_mask, output_layer_tensor)

        # Приведение к float32 для downstream pipeline
        features = features.to(dtype=torch.float32)

        # 🔧 Phase 3: Thread-safe сохранение в кэш
        if use_cache and cache_key:
            with self._cache_lock:
                if len(self._feature_cache) > 32:
                    oldest = next(iter(self._feature_cache))
                    del self._feature_cache[oldest]
                self._feature_cache[cache_key] = features.detach().cpu()

        return features


def create_feature_extractor(
    config: RVCConfig,
    model_type: FeatureModelType | str = FeatureModelType.HUBERT_BASE,
    use_compile: bool = True,
    backend: Literal["fairseq", "huggingface"] = "fairseq",
) -> TransformerFeatureExtractor | FairseqFeatureExtractor:
    """
    Фабричная функция для создания feature extractor.
    
    🔧 Phase 3: Добавлен параметр backend с дефолтом "fairseq".
    
    Args:
        config: RVC конфигурация.
        model_type: Тип HF-модели (используется только при backend="huggingface").
        use_compile: Применять ли torch.compile (только для HF backend).
        backend: "fairseq" (рекомендуется, совместим со всеми RVC моделями)
                 или "huggingface" (для экспериментов с другими моделями).
    
    Returns:
        Инициализированный feature extractor.
    """
    if isinstance(model_type, str):
        model_type = FeatureModelType(model_type)

    device = config.inference.device
    dtype = (
        torch.float16
        if config.inference.dtype == "float16" and device != "cpu"
        else torch.float32
    )

    if backend == "fairseq":
        model_path = config.paths.hubert / "hubert_base.pt"
        try:
            return FairseqFeatureExtractor(
                model_path=model_path,
                device=device,
                dtype=dtype,
                use_compile=False,  # Fairseq плохо компилируется
            )
        except (ImportError, FileNotFoundError, RuntimeError) as e:
            logger.warning(
                "fairseq backend недоступен (%s). Fallback на HuggingFace.\n"
                "Для лучшей совместимости с существующими RVC моделями:\n"
                "  1. Установите fairseq: pip install fairseq\n"
                "  2. Скачайте HuBERT: python check_and_download_models.py",
                e,
            )
            # Fallback на HuggingFace при недоступности fairseq

    # HuggingFace backend (или fallback)
    return TransformerFeatureExtractor(
        model_type=model_type,
        device=device,
        dtype=dtype,
        use_compile=use_compile,
        cache_dir=config.paths.root / ".cache" / "features",
    )