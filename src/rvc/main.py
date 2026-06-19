"""Точка входа в RVC WebUI с graceful shutdown."""
import argparse
import asyncio
import atexit
import logging
import signal
import sys
from typing import Any

import gradio as gr
import torch

from rvc.config import get_config
from rvc.core.container import init_container
from rvc.core.pipeline import VCPipeline
from rvc.ui.webui import build_webui

# Настройка структурированного логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Глобальная ссылка на container для shutdown-хуков
_global_container: Any = None


def _graceful_shutdown(signum: int, frame: Any) -> None:
    """
    Обработчик сигналов SIGTERM/SIGINT.
    Корректно завершает все фоновые процессы обучения и освобождает GPU-ресурсы.
    """
    sig_name = signal.Signals(signum).name
    logger.warning("Получен сигнал %s. Начинаю graceful shutdown...", sig_name)

    # 1. Уничтожение всех фоновых процессов обучения
    if _global_container is not None:
        try:
            training_uc = _global_container.training_use_case
            loop = asyncio.new_event_loop()
            killed = loop.run_until_complete(
                training_uc.kill_all_training_processes()
            )
            loop.close()
            logger.info("Уничтожено %d фоновых процессов обучения", killed)
        except Exception as e:
            logger.error("Ошибка при уничтожении процессов обучения: %s", e)

    # 2. Глобальная очистка ML-ресурсов (FAISS GPU, CUDA cache)
    try:
        VCPipeline.cleanup_all()
        logger.info("ML-ресурсы освобождены")
    except Exception as e:
        logger.error("Ошибка при очистке ML-ресурсов: %s", e)

    logger.warning("Graceful shutdown завершён. Выход.")
    sys.exit(0)


def _atexit_handler() -> None:
    """Atexit-хук для финальной очистки ресурсов."""
    logger.debug("Atexit: финальная очистка ресурсов...")
    try:
        VCPipeline.cleanup_all()
    except Exception:
        pass


def _register_signal_handlers() -> None:
    """Зарегистрировать обработчики сигналов для graceful shutdown."""
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)
    atexit.register(_atexit_handler)
    logger.debug("Обработчики сигналов SIGTERM/SIGINT зарегистрированы")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RVC Modern WebUI")
    parser.add_argument("--host", default="127.0.0.1", help="Хост для запуска")
    parser.add_argument("--port", type=int, default=7860, help="Порт")
    parser.add_argument(
        "--share", action="store_true",
        help="Публичная ссылка (Gradio tunnel)",
    )
    parser.add_argument("--debug", action="store_true", help="Режим отладки")
    return parser.parse_args()


def main() -> int:
    global _global_container

    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Регистрируем обработчики сигналов ДО инициализации чего-либо
    _register_signal_handlers()

    # 1. Загрузка конфигурации
    config = get_config()
    logger.info("Конфигурация загружена. Устройство: %s", config.inference.device)
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info(
            "VRAM: %.2f GB",
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
        )

    # 2. Инициализация DI Container
    container = init_container(config)
    _global_container = container
    logger.info("DI Container и сервисы инициализированы.")

    # 3. Построение UI
    # 🔧 BUG FIX: build_webui() теперь внутри настраивает queue()
    # с rate limiting через concurrency_count=2 (Gradio 5.x API)
    demo = build_webui(container)

    # 4. Запуск сервера
    logger.info("Запуск Gradio сервера на %s:%d", args.host, args.port)
    # 🔧 BUG FIX: theme убран отсюда (в Gradio 5.x он передаётся в Blocks,
    # что и делается в build_webui()). queue() уже настроен внутри build_webui().
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        max_threads=40,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())