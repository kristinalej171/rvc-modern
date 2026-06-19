"""
FAISS-based indexer для retrieval-based voice conversion.
Поддерживает:
- CPU и GPU индексы
- IVF и HNSW стратегии
- Адаптивное смешивание features
- Явное управление GPU-ресурсами (предотвращение утечек VRAM)
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

try:
    import faiss
    import faiss.contrib.torch_utils  # noqa: F401 - GPU support
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

if TYPE_CHECKING:
    from rvc.config import RVCConfig

logger = logging.getLogger(__name__)


class IndexStrategy(str, Enum):
    """Стратегии индексации."""
    IVF_FLAT = "ivf_flat"
    HNSW = "hnsw"
    AUTO = "auto"


@dataclass
class IndexStats:
    """Статистика индекса."""
    total_vectors: int
    dimension: int
    index_type: str
    is_gpu: bool
    file_path: Path | None


class GpuResourceManager:
    """
    Singleton-менеджер GPU-ресурсов FAISS.
    Предотвращает утечки VRAM при частой загрузке/выгрузке индексов
    через reference counting и явное освобождение.
    """
    _instance: GpuResourceManager | None = None
    _lock = threading.Lock()

    def __new__(cls) -> GpuResourceManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._resources: dict[int, faiss.StandardGpuResources] = {}
                cls._instance._ref_counts: dict[int, int] = {}
                cls._instance._resource_lock = threading.Lock()
        return cls._instance

    def acquire(self, gpu_id: int = 0) -> faiss.StandardGpuResources:
        """
        Получить GPU-ресурсы для указанного GPU.
        Если ресурсы уже созданы — увеличивает reference count.
        """
        with self._resource_lock:
            if gpu_id not in self._resources:
                logger.info("Создание FAISS GpuResources для GPU %d", gpu_id)
                res = faiss.StandardGpuResources()
                # Настраиваем временный memory allocator для GPU
                res.setTempMemory(gpu_id, 256 * 1024 * 1024)  # 256 MB temp memory
                self._resources[gpu_id] = res
                self._ref_counts[gpu_id] = 0
            self._ref_counts[gpu_id] += 1
            return self._resources[gpu_id]

    def release(self, gpu_id: int = 0) -> None:
        """
        Освободить ссылку на GPU-ресурсы.
        Когда reference count достигает 0 — ресурсы уничтожаются.
        """
        with self._resource_lock:
            if gpu_id not in self._ref_counts:
                return
            self._ref_counts[gpu_id] -= 1
            if self._ref_counts[gpu_id] <= 0:
                logger.info(
                    "Уничтожение FAISS GpuResources для GPU %d (ref_count=0)",
                    gpu_id,
                )
                # Явно удаляем ресурсы
                del self._resources[gpu_id]
                del self._ref_counts[gpu_id]

    def release_all(self) -> None:
        """Принудительно освободить все GPU-ресурсы (при shutdown)."""
        with self._resource_lock:
            released_gpus = list(self._resources.keys())
            self._resources.clear()
            self._ref_counts.clear()
            if released_gpus:
                logger.info(
                    "Принудительное освобождение FAISS GpuResources для GPU: %s",
                    released_gpus,
                )


# Singleton-инстанс
_gpu_resource_manager = GpuResourceManager()


class FeatureIndexer:
    """
    Индексер features для retrieval-based voice conversion.
    Выполняет approximate nearest neighbor search для нахождения
    похожих features в training set и их смешивания с входными.
    """

    def __init__(
        self,
        config: RVCConfig,
        strategy: IndexStrategy = IndexStrategy.AUTO,
        n_probe: int = 1,
        use_gpu: bool = True,
        gpu_id: int = 0,
    ) -> None:
        if not FAISS_AVAILABLE:
            raise RuntimeError(
                "FAISS не установлен. Установите: pip install faiss-cpu "
                "или pip install faiss-gpu"
            )
        self.config = config
        self.strategy = strategy
        self.n_probe = n_probe
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.gpu_id = gpu_id
        self.index: faiss.Index | None = None
        self._gpu_index: faiss.Index | None = None  # Отдельная ссылка на GPU-копию
        self.big_npy: np.ndarray | None = None
        self._stats: IndexStats | None = None

    @classmethod
    def from_file(
        cls,
        config: RVCConfig,
        index_path: Path | str,
        strategy: IndexStrategy = IndexStrategy.AUTO,
        use_gpu: bool = True,
        gpu_id: int = 0,
    ) -> "FeatureIndexer":
        """Загрузить indexer из .index файла."""
        indexer = cls(
            config=config,
            strategy=strategy,
            use_gpu=use_gpu,
            gpu_id=gpu_id,
        )
        indexer.load(index_path)
        return indexer

    def load(self, index_path: Path | str) -> None:
        """Загрузить индекс из файла."""
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"Index файл не найден: {index_path}")

        logger.info("Загрузка индекса: %s", index_path)
        cpu_index = faiss.read_index(str(index_path))

        # Установка nprobe для IVF индексов
        if hasattr(cpu_index, "nprobe"):
            cpu_index.nprobe = self.n_probe

        # Попытка перенести индекс на GPU через менеджер ресурсов
        if self.use_gpu:
            try:
                gpu_resources = _gpu_resource_manager.acquire(self.gpu_id)
                self._gpu_index = faiss.index_cpu_to_gpu(
                    gpu_resources, self.gpu_id, cpu_index
                )
                self.index = self._gpu_index
                logger.info("Индекс перенесён на GPU %d", self.gpu_id)
            except Exception as e:
                logger.warning("Не удалось перенести индекс на GPU: %s", e)
                self.index = cpu_index
                self._gpu_index = None
        else:
            self.index = cpu_index
            self._gpu_index = None

        # Восстановление big_npy для blending
        try:
            # Используем CPU-индекс для reconstruct (GPU может не поддерживать)
            source_index = cpu_index if self._gpu_index is not None else self.index
            self.big_npy = source_index.reconstruct_n(0, source_index.ntotal)
        except Exception as e:
            logger.warning(
                "Не удалось восстановить big_npy: %s. Blending будет недоступен.",
                e,
            )
            self.big_npy = None

        self._stats = IndexStats(
            total_vectors=self.index.ntotal,
            dimension=self.index.d,
            index_type=self.index.get_type_name(),
            is_gpu=self._gpu_index is not None,
            file_path=index_path,
        )
        logger.info(
            "Индекс загружен: %d vectors, dim=%d, type=%s, gpu=%s",
            self._stats.total_vectors,
            self._stats.dimension,
            self._stats.index_type,
            self._stats.is_gpu,
        )

    def unload(self) -> None:
        """
        Выгрузить индекс из памяти.
        Явно освобождает GPU-ресурсы через GpuResourceManager.
        """
        if self._gpu_index is not None:
            # Сначала удаляем GPU-копию индекса
            logger.debug("Удаление GPU-индекса")
            del self._gpu_index
            self._gpu_index = None

            # Освобождаем ссылку на GPU-ресурсы
            _gpu_resource_manager.release(self.gpu_id)

        if self.index is not None:
            logger.debug("Удаление CPU-индекса")
            del self.index
            self.index = None

        self.big_npy = None
        self._stats = None

        # Дополнительная очистка CUDA-кэша
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Индекс выгружен, ресурсы освобождены")

    @property
    def is_loaded(self) -> bool:
        """Проверка, загружен ли индекс."""
        return self.index is not None

    @property
    def stats(self) -> IndexStats | None:
        """Получить статистику индекса."""
        return self._stats

    def search(
        self,
        queries: np.ndarray | torch.Tensor,
        k: int = 8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Найти k ближайших соседей для каждого query vector."""
        if self.index is None:
            raise RuntimeError("Индекс не загружен")

        if isinstance(queries, torch.Tensor):
            queries = queries.cpu().numpy()

        queries = queries.astype(np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        distances, indices = self.index.search(queries, k)
        return distances, indices

    def blend_features(
        self,
        features: torch.Tensor,
        index_rate: float,
        k: int = 8,
    ) -> torch.Tensor:
        """Смешать входные features с retrieved из training set."""
        if not self.is_loaded:
            logger.warning("Индекс не загружен, используется только input")
            return features

        if index_rate <= 0.0:
            return features

        if self.big_npy is None:
            logger.warning("big_npy недоступен, blending невозможен")
            return features

        # Извлекаем features в numpy
        feat_np = features[0].detach().cpu().numpy().astype(np.float32)

        # Поиск соседей
        distances, indices = self.search(feat_np, k=k)

        # Валидация результатов
        if not (indices >= 0).all():
            logger.warning("Невалидные результаты поиска, blending пропущен")
            return features

        # Вычисление весов на основе расстояний
        weights = np.square(1.0 / (distances + 1e-8))
        weights /= weights.sum(axis=1, keepdims=True)

        # Weighted sum retrieved features
        retrieved = self.big_npy[indices]  # (T, k, D)
        weighted = np.sum(
            retrieved * weights[:, :, np.newaxis],
            axis=1,
        )  # (T, D)

        # Convert back to torch
        retrieved_torch = torch.from_numpy(weighted).to(
            device=features.device, dtype=features.dtype
        )
        retrieved_torch = retrieved_torch.unsqueeze(0)

        # Blend
        blended = (
            index_rate * retrieved_torch + (1 - index_rate) * features
        )
        return blended

    @staticmethod
    def unload_all_static() -> None:
        """
        Принудительно освободить все статические GPU-ресурсы FAISS.
        Вызывается при graceful shutdown приложения.
        """
        _gpu_resource_manager.release_all()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Все статические FAISS GPU-ресурсы освобождены")


def build_index(
    features: np.ndarray,
    output_path: Path,
    strategy: IndexStrategy = IndexStrategy.AUTO,
    n_ivf: int | None = None,
) -> faiss.Index:
    """Построить FAISS индекс из набора features."""
    if not FAISS_AVAILABLE:
        raise RuntimeError("FAISS не установлен")

    n_samples, dimension = features.shape
    features = features.astype(np.float32)

    # Автоматический выбор n_ivf
    if n_ivf is None:
        n_ivf = min(int(16 * np.sqrt(n_samples)), n_samples // 39)
        n_ivf = max(n_ivf, 1)

    # Выбор стратегии
    if strategy == IndexStrategy.AUTO:
        strategy = (
            IndexStrategy.HNSW
            if n_samples < 10000
            else IndexStrategy.IVF_FLAT
        )

    logger.info(
        "Построение индекса: N=%d, D=%d, strategy=%s",
        n_samples,
        dimension,
        strategy.value,
    )

    if strategy == IndexStrategy.HNSW:
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = 64
    else:  # IVF_FLAT
        factory_str = f"IVF{n_ivf},Flat"
        index = faiss.index_factory(dimension, factory_str)
        logger.info("Обучение IVF с %d кластерами", n_ivf)
        index.train(features)

    logger.info("Добавление %d vectors в индекс", n_samples)
    index.add(features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))
    logger.info("Индекс сохранён: %s", output_path)
    return index