"""
Отдельный скрипт для тренировки FAISS-индекса.
Запускается subprocess-ом из TrainingUseCase.train_index().

Использование:
    python scripts/train_index_script.py \
        --exp-dir logs/my_model \
        --version v2 \
        --n-cpu 4

🔒 SECURITY: Этот скрипт заменяет небезопасный inline f-string в
training_use_case.py, который был уязвим к code injection через exp_name.
Теперь все параметры передаются через argparse (без shell-интерполяции).
"""
import argparse
import os
import sys
from pathlib import Path

import faiss
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FAISS index from extracted features"
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        required=True,
        help="Путь к директории эксперимента (logs/<exp_name>)",
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=["v1", "v2"],
        required=True,
        help="Версия модели (v1: 256 dim, v2: 768 dim)",
    )
    parser.add_argument(
        "--n-cpu",
        type=int,
        default=4,
        help="Количество CPU для MiniBatchKMeans",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Валидация входных путей (защита от traversal на уровне скрипта)
    exp_dir = args.exp_dir.resolve()
    if not exp_dir.exists():
        print(f"❌ Директория эксперимента не найдена: {exp_dir}", file=sys.stderr)
        return 1

    # Определяем директорию с features в зависимости от версии
    feature_dir_name = "3_feature256" if args.version == "v1" else "3_feature768"
    feature_dir = exp_dir / feature_dir_name

    if not feature_dir.exists():
        print(f"❌ Директория features не найдена: {feature_dir}", file=sys.stderr)
        return 1

    feature_dim = 256 if args.version == "v1" else 768

    # === Шаг 1: Загрузка features ===
    print(f"Загрузка features из {feature_dir}")
    npy_files = sorted(
        [f for f in feature_dir.iterdir() if f.suffix == ".npy"]
    )
    if not npy_files:
        print(f"❌ В {feature_dir} нет .npy файлов", file=sys.stderr)
        return 1

    npys = [np.load(f) for f in npy_files]
    big_npy = np.concatenate(npys, axis=0).astype(np.float32)
    print(f"Загружено {big_npy.shape[0]} vectors (dim={feature_dim})")

    # === Шаг 2: Опциональная кластеризация для больших датасетов ===
    if big_npy.shape[0] > 200_000:
        print("Датасет большой (>200k). Применяем MiniBatchKMeans...")
        kmeans = MiniBatchKMeans(
            n_clusters=10000,
            verbose=True,
            batch_size=256 * args.n_cpu,
            compute_labels=False,
            init="random",
        )
        big_npy = kmeans.fit(big_npy).cluster_centers_
        print(f"Кластеризация завершена: {big_npy.shape[0]} centers")

    # === Шаг 3: Построение FAISS индекса ===
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    n_ivf = max(n_ivf, 1)
    print(f"Создаём IVF-индекс: n_ivf={n_ivf}, dim={feature_dim}")

    index = faiss.index_factory(feature_dim, f"IVF{n_ivf},Flat")
    index.train(big_npy)

    total_vectors = big_npy.shape[0]
    batch_size = 8192
    for i in range(0, total_vectors, batch_size):
        batch_end = min(i + batch_size, total_vectors)
        index.add(big_npy[i:batch_end])
        print(f"Добавлено {batch_end}/{total_vectors} vectors")

    # === Шаг 4: Сохранение индекса ===
    exp_name = exp_dir.name
    index_filename = f"added_IVF{n_ivf}_Flat_nprobe_1_{exp_name}_{args.version}.index"
    index_path = exp_dir / index_filename

    faiss.write_index(index, str(index_path))
    print(f"✅ Индекс сохранён: {index_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())