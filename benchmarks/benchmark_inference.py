"""
Benchmarking suite для отслеживания производительности pipeline.

Использование:
    python -m benchmarks.benchmark_inference --model path/to/model.pth
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from rvc.config import get_config
from rvc.core.pipeline import InferenceConfig, VCPipeline


def generate_test_audio(
    duration: float = 3.0,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Генерация синтетического аудио для бенчмарка."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # Смесь нескольких частот для реалистичности
    audio = (
        0.3 * np.sin(2 * np.pi * 220 * t)
        + 0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 880 * t)
        + 0.1 * np.random.randn(len(t)).astype(np.float32)
    )
    return audio


def benchmark_inference(
    pipeline: VCPipeline,
    audio: np.ndarray,
    config: InferenceConfig,
    n_runs: int = 10,
    warmup: int = 2,
) -> dict:
    """
    Запустить benchmark инференса.

    Args:
        pipeline: Инициализированный pipeline.
        audio: Тестовый аудио.
        config: Конфигурация инференса.
        n_runs: Количество запусков.
        warmup: Количество warmup запусков.

    Returns:
        Dict со статистикой.
    """
    # Warmup
    print(f"Warmup: {warmup} runs")
    for _ in range(warmup):
        pipeline.convert(audio, config)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Main benchmark
    print(f"Benchmark: {n_runs} runs")
    times: list[float] = []
    rtf_values: list[float] = []

    for i in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = pipeline.convert(audio, config)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        rtf_values.append(result.stats.get("rtf", 0.0))

        print(
            f"  Run {i+1}/{n_runs}: {elapsed*1000:.2f} ms, "
            f"RTF={result.stats.get('rtf', 0):.3f}"
        )

    # Статистика
    return {
        "n_runs": n_runs,
        "mean_ms": statistics.mean(times) * 1000,
        "median_ms": statistics.median(times) * 1000,
        "std_ms": statistics.stdev(times) * 1000 if len(times) > 1 else 0,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "mean_rtf": statistics.mean(rtf_values),
        "audio_duration_s": len(audio) / 16000,
        "device": str(pipeline.config.inference.device),
        "dtype": pipeline.config.inference.dtype,
        "pitch_method": config.pitch_method.value,
        "feature_model": config.feature_model.value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RVC Inference Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Путь к .pth модели",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Путь к .index файлу (опционально)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Длительность тестового аудио (сек)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Количество запусков",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Количество warmup запусков",
    )
    parser.add_argument(
        "--pitch-method",
        type=str,
        default="rmvpe",
        choices=["pm", "harvest", "rmvpe", "fcpe"],
    )
    parser.add_argument(
        "--feature-model",
        type=str,
        default="hubert_base",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Путь для сохранения результатов",
    )
    args = parser.parse_args()

    # Инициализация
    config = get_config()
    pipeline = VCPipeline(config)

    print(f"Загрузка модели: {args.model}")
    pipeline.load_model(args.model)

    if args.index:
        print(f"Загрузка индекса: {args.index}")
        pipeline.load_index(args.index)

    # Тестовый аудио
    audio = generate_test_audio(args.duration)

    # Конфигурация
    inference_config = InferenceConfig(
        pitch_method=args.pitch_method,
        feature_model=args.feature_model,
    )

    # Бенчмарк
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    results = benchmark_inference(
        pipeline,
        audio,
        inference_config,
        n_runs=args.n_runs,
        warmup=args.warmup,
    )

    # Вывод результатов
    print("\n" + "=" * 60)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.3f}")
        else:
            print(f"{key:20s}: {value}")

    # Сохранение в JSON
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nРезультаты сохранены в: {output_path}")

    # Cleanup
    pipeline.unload()


if __name__ == "__main__":
    main()