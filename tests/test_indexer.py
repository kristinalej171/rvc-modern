"""
Тесты модуля rvc.core.indexer (FAISS).
Покрывают:
- build_index
- FeatureIndexer.load/search/blend
- GpuResourceManager (singleton + release)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

faiss = pytest.importorskip("faiss", reason="FAISS не установлен")

from rvc.core.indexer import (  # noqa: E402
    FeatureIndexer,
    GpuResourceManager,
    IndexStrategy,
    build_index,
)


class TestBuildIndex:
    """Тесты build_index."""

    def test_build_ivf_index(
        self, sample_features: np.ndarray, temp_dir: Path
    ) -> None:
        """Построение IVF индекса."""
        output = temp_dir / "test.index"
        index = build_index(
            sample_features,
            output,
            strategy=IndexStrategy.IVF_FLAT,
            n_ivf=10,
        )

        assert output.exists()
        assert index.ntotal == len(sample_features)
        assert index.d == sample_features.shape[1]

    def test_build_hnsw_index(
        self, sample_features: np.ndarray, temp_dir: Path
    ) -> None:
        """Построение HNSW индекса."""
        output = temp_dir / "test_hnsw.index"
        index = build_index(
            sample_features,
            output,
            strategy=IndexStrategy.HNSW,
        )

        assert output.exists()
        assert index.ntotal == len(sample_features)

    def test_auto_strategy_selects_hnsw_for_small(
        self, temp_dir: Path
    ) -> None:
        """AUTO стратегия выбирает HNSW для малых датасетов."""
        small_features = np.random.randn(500, 64).astype(np.float32)
        output = temp_dir / "small.index"
        index = build_index(small_features, output, strategy=IndexStrategy.AUTO)

        assert index.ntotal == 500
        # HNSW индекс не имеет nprobe
        assert not hasattr(index, "nprobe")

    def test_auto_strategy_selects_ivf_for_large(
        self, temp_dir: Path
    ) -> None:
        """AUTO стратегия выбирает IVF для больших датасетов."""
        large_features = np.random.randn(20000, 64).astype(np.float32)
        output = temp_dir / "large.index"
        index = build_index(large_features, output, strategy=IndexStrategy.AUTO)

        assert index.ntotal == 20000
        # IVF индекс имеет nprobe
        assert hasattr(index, "nprobe")

    def test_creates_parent_directories(
        self, sample_features: np.ndarray, temp_dir: Path
    ) -> None:
        """Создаёт родительские директории."""
        output = temp_dir / "nested" / "dir" / "test.index"
        build_index(sample_features, output, n_ivf=10)
        assert output.exists()


class TestFeatureIndexer:
    """Тесты FeatureIndexer."""

    @pytest.fixture
    def built_index_path(
        self, sample_features: np.ndarray, temp_dir: Path
    ) -> Path:
        """Готовый индекс для тестов."""
        output = temp_dir / "built.index"
        build_index(sample_features, output, n_ivf=10)
        return output

    def test_load_index(self, built_index_path: Path, config) -> None:
        """Загрузка индекса."""
        indexer = FeatureIndexer.from_file(
            config, built_index_path, use_gpu=False
        )

        assert indexer.is_loaded
        assert indexer.stats is not None
        assert indexer.stats.total_vectors == 100
        assert indexer.stats.dimension == 256

    def test_load_nonexistent_raises(self, temp_dir: Path, config) -> None:
        """Загрузка несуществующего файла вызывает FileNotFoundError."""
        fake_path = temp_dir / "nonexistent.index"

        with pytest.raises(FileNotFoundError):
            FeatureIndexer.from_file(config, fake_path, use_gpu=False)

    def test_search_returns_correct_shape(
        self, built_index_path: Path, config, sample_features: np.ndarray
    ) -> None:
        """Поиск возвращает корректные shapes."""
        indexer = FeatureIndexer.from_file(
            config, built_index_path, use_gpu=False
        )

        query = sample_features[:5]
        distances, indices = indexer.search(query, k=3)

        assert distances.shape == (5, 3)
        assert indices.shape == (5, 3)
        assert distances.dtype == np.float32
        assert indices.dtype == np.int64

    def test_search_single_query(
        self, built_index_path: Path, config, sample_features: np.ndarray
    ) -> None:
        """Поиск одного query вектора."""
        indexer = FeatureIndexer.from_file(
            config, built_index_path, use_gpu=False
        )

        query = sample_features[0]  # 1D array
        distances, indices = indexer.search(query, k=3)

        assert distances.shape == (1, 3)
        assert indices.shape == (1, 3)

    def test_blend_features(
        self, built_index_path: Path, config, sample_features: np.ndarray
    ) -> None:
        """Blend features смешивает с retrieved."""
        import torch

        indexer = FeatureIndexer.from_file(
            config, built_index_path, use_gpu=False
        )

        features = torch.from_numpy(sample_features[:10]).unsqueeze(0)
        blended = indexer.blend_features(features, index_rate=0.5, k=3)

        assert blended.shape == features.shape
        # При index_rate > 0 должны отличаться
        assert not torch.allclose(blended, features)

    def test_blend_with_zero_rate_returns_original(
        self, built_index_path: Path, config, sample_features: np.ndarray
    ) -> None:
        """index_rate=0 возвращает исходный features."""
        import torch

        indexer = FeatureIndexer.from_file(
            config, built_index_path, use_gpu=False
        )

        features = torch.from_numpy(sample_features[:10]).unsqueeze(0)
        blended = indexer.blend_features(features, index_rate=0.0, k=3)

        assert torch.allclose(blended, features)

    def test_blend_without_index_returns_original(
        self, config, sample_features: np.ndarray
    ) -> None:
        """Blend без загруженного индекса возвращает оригинал."""
        import torch

        indexer = FeatureIndexer(config, use_gpu=False)
        features = torch.from_numpy(sample_features[:10]).unsqueeze(0)

        blended = indexer.blend_features(features, index_rate=0.5, k=3)
        assert torch.allclose(blended, features)

    def test_unload_releases_resources(
        self, built_index_path: Path, config
    ) -> None:
        """Unload освобождает ресурсы."""
        indexer = FeatureIndexer.from_file(
            config, built_index_path, use_gpu=False
        )
        assert indexer.is_loaded

        indexer.unload()
        assert not indexer.is_loaded
        assert indexer.stats is None


class TestGpuResourceManager:
    """Тесты GpuResourceManager (singleton)."""

    def test_singleton_pattern(self) -> None:
        """Singleton возвращает один и тот же инстанс."""
        mgr1 = GpuResourceManager()
        mgr2 = GpuResourceManager()
        assert mgr1 is mgr2

    @pytest.mark.skipif(
        not faiss.get_num_gpus() > 0,
        reason="Нет доступных GPU для теста",
    )
    def test_acquire_and_release(self) -> None:
        """Acquire/release работают корректно."""
        mgr = GpuResourceManager()

        res = mgr.acquire(gpu_id=0)
        assert res is not None

        mgr.release(gpu_id=0)

    def test_release_nonexistent_no_error(self) -> None:
        """Release несуществующего GPU не падает."""
        mgr = GpuResourceManager()
        # Не должно падать
        mgr.release(gpu_id=999)

    def test_release_all_clears_everything(self) -> None:
        """release_all очищает все ресурсы."""
        mgr = GpuResourceManager()
        # release_all не должен падать даже без acquire
        mgr.release_all()