import sys
import os
import subprocess
import importlib
import logging
from pathlib import Path

# Добавляем корень проекта и src в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("environment_test.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Критически важные пакеты
REQUIRED_PACKAGES = [
    "torch", "torchaudio", "torchvision", "numpy", "scipy", "librosa",
    "soundfile", "fairseq", "faiss", "gradio", "pydub", "ffmpeg", "tqdm",
    "numba", "pyworld", "parselmouth", "tensorboardX", "sklearn",
    "setuptools", "pydantic", "pydantic_settings",
]

# Новые модули (Clean Architecture)
NEW_MODULES = [
    "rvc",
    "rvc.config",
    "rvc.core",
    "rvc.core.audio",
    "rvc.core.feature_extractor",
    "rvc.core.pitch",
    "rvc.core.models",
    "rvc.core.pipeline",
    "rvc.core.indexer",
    "rvc.ui.webui",
]

def check_python_version():
    logger.info("Проверка версии Python...")
    if sys.version_info < (3, 10):
        logger.error(f"Требуется Python 3.10+. Текущая: {sys.version}")
        return False
    logger.info(f"✅ Версия Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_system_dependencies():
    logger.info("Проверка системных зависимостей (ffmpeg)...")
    try:
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            version_line = result.stdout.decode('utf-8').split('\n')[0]
            logger.info(f"✅ ffmpeg установлен: {version_line}")
            return True
        else:
            logger.error("❌ ffmpeg работает некорректно.")
            return False
    except FileNotFoundError:
        logger.error("❌ ffmpeg не найден в PATH.")
        return False

def check_python_packages():
    logger.info("Проверка Python-пакетов...")
    all_ok = True
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            logger.info(f"✅ Пакет '{package}' успешно импортирован.")
        except ImportError as e:
            logger.error(f"❌ Ошибка импорта '{package}': {e}")
            all_ok = False
    return all_ok

def check_modules(module_list, label):
    logger.info(f"Проверка модулей: {label}...")
    all_ok = True
    for module in module_list:
        try:
            importlib.import_module(module)
            logger.info(f"✅ Модуль '{module}' успешно импортирован.")
        except Exception as e:
            logger.error(f"❌ Ошибка импорта '{module}': {e}")
            all_ok = False
    return all_ok

def check_pytorch_cuda():
    logger.info("Проверка CUDA в PyTorch...")
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA доступна. Версия: {torch.version.cuda}")
            logger.info(f"✅ GPU: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.warning("⚠️ CUDA недоступна — будет использоваться CPU.")
    except Exception as e:
        logger.error(f"❌ Ошибка проверки CUDA: {e}")

def check_required_files():
    logger.info("Проверка наличия весов моделей...")
    all_ok = True
    required_files = [
        "assets/hubert/hubert_base.pt",
        "assets/rmvpe/rmvpe.pt",
        "assets/pretrained_v2/f0G40k.pth",
        "assets/pretrained_v2/f0D40k.pth",
    ]
    for file_path in required_files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            logger.info(f"✅ Файл найден: {file_path}")
        else:
            logger.error(f"❌ Файл отсутствует: {file_path}")
            all_ok = False
    return all_ok

def main():
    logger.info("=" * 60)
    logger.info("Начало полного тестирования окружения RVC (Modern Edition)")
    logger.info("=" * 60)

    results = {
        "Python Version": check_python_version(),
        "System Dependencies": check_system_dependencies(),
        "Python Packages": check_python_packages(),
        "New Architecture (src/rvc)": check_modules(NEW_MODULES, "новая архитектура"),
        "Required Files": check_required_files(),
    }

    check_pytorch_cuda()

    logger.info("=" * 60)
    logger.info("Итоги тестирования:")
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ УСПЕХ" if passed else "❌ ОШИБКА"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    logger.info("=" * 60)
    if all_passed:
        logger.info("🎉 Все проверки пройдены! Окружение готово к работе.")
    else:
        logger.error("⚠️ Обнаружены ошибки. Исправьте их перед запуском.")
        logger.error("📄 Подробный лог: environment_test.log")

if __name__ == "__main__":
    main()
