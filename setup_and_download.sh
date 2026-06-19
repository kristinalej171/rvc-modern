#!/bin/bash

set -e  # Остановка при ошибке

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   RVC Modern Edition — Установка зависимостей   ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# Проверка наличия python3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Ошибка: Python 3 не найден.${NC}"
    echo -e "${YELLOW}  Установите Python 3.11+:${NC}"
    echo -e "  Ubuntu/Debian: ${GREEN}sudo apt install python3.11 python3.11-venv python3-pip${NC}"
    echo -e "  Arch:          ${GREEN}sudo pacman -S python${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "${GREEN}✓ Python $PYTHON_VERSION найден${NC}"

# Проверка минимальной версии Python (3.10+)
python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Требуется Python 3.10 или выше. Текущая версия: $PYTHON_VERSION${NC}"
    exit 1
fi

# Проверка ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${YELLOW}⚠ ffmpeg не найден. Попытка установки...${NC}"
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm ffmpeg
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y ffmpeg
    else
        echo -e "${RED}✗ Не удалось установить ffmpeg автоматически.${NC}"
        echo -e "${YELLOW}  Установите ffmpeg вручную и повторите запуск.${NC}"
        exit 1
    fi
fi

FFMPEG_VERSION=$(ffmpeg -version 2>/dev/null | head -n1)
echo -e "${GREEN}✓ $FFMPEG_VERSION${NC}"

# Определение директории скрипта
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Создание виртуального окружения
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}→ Создание виртуального окружения...${NC}"
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Виртуальное окружение создано${NC}"
else
    echo -e "${GREEN}✓ Виртуальное окружение уже существует${NC}"
fi

# Активация виртуального окружения
echo -e "${YELLOW}→ Активация виртуального окружения...${NC}"
source "$VENV_DIR/bin/activate"

# Обновление pip
echo -e "${YELLOW}→ Обновление pip...${NC}"
pip install --upgrade pip setuptools wheel 2>/dev/null

# Определение наличия GPU
HAS_NVIDIA_GPU=false
if command -v nvidia-smi &> /dev/null; then
    HAS_NVIDIA_GPU=true
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)
    echo -e "${GREEN}✓ NVIDIA GPU обнаружена: $GPU_NAME (driver: $CUDA_VERSION)${NC}"
fi

# Установка PyTorch
echo -e "${YELLOW}→ Установка PyTorch...${NC}"
if [ "$HAS_NVIDIA_GPU" = true ]; then
    echo -e "${BLUE}  Установка с поддержкой CUDA 12.4...${NC}"
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 2>/dev/null || \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    echo -e "${BLUE}  GPU не обнаружена, установка CPU-версии...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
echo -e "${GREEN}✓ PyTorch установлен${NC}"

# Установка зависимостей проекта
echo -e "${YELLOW}→ Установка зависимостей проекта...${NC}"

INSTALL_SUCCESS=false

# Вариант 1: Установка через pyproject.toml (предпочтительный)
if [ -f "pyproject.toml" ]; then
    echo -e "${BLUE}  Обнаружен pyproject.toml — установка через pip install -e .${NC}"
    if pip install -e . 2>/dev/null; then
        INSTALL_SUCCESS=true
        echo -e "${GREEN}✓ Зависимости установлены из pyproject.toml${NC}"
    fi
fi

# Вариант 2: Установка через requirements.txt (fallback)
if [ "$INSTALL_SUCCESS" = false ]; then
    if [ -f "requirements.txt" ]; then
        echo -e "${BLUE}  Установка через requirements.txt...${NC}"
        if pip install -r requirements.txt; then
            INSTALL_SUCCESS=true
            echo -e "${GREEN}✓ Зависимости установлены из requirements.txt${NC}"
        fi
    else
        echo -e "${RED}✗ Не найден ни pyproject.toml, ни requirements.txt!${NC}"
        echo -e "${YELLOW}  Убедитесь, что вы находитесь в корневой директории проекта.${NC}"
        exit 1
    fi
fi

if [ "$INSTALL_SUCCESS" = false ]; then
    echo -e "${RED}✗ Не удалось установить зависимости.${NC}"
    exit 1
fi

# Установка faiss-cpu (используется CPU-версия для избежания конфликтов CUDA с PyTorch)
echo -e "${YELLOW}→ Установка faiss-cpu (поиск индексов)...${NC}"
pip install faiss-cpu>=1.8.0 2>/dev/null && \
echo -e "${GREEN}✓ faiss-cpu установлен${NC}" || \
echo -e "${YELLOW}⚠ Ошибка установки faiss-cpu${NC}"

# Запуск скрипта проверки и скачивания моделей
echo ""
echo -e "${YELLOW}→ Проверка и загрузка моделей...${NC}"
if [ -f "check_and_download_models.py" ]; then
    python3 check_and_download_models.py
else
    echo -e "${RED}✗ Скрипт check_and_download_models.py не найден!${NC}"
    exit 1
fi

# Итоговая информация
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          ✓ Установка успешно завершена!          ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Для запуска WebUI выполните:${NC}"
echo -e "  ${GREEN}source .venv/bin/activate${NC}"
echo -e "  ${GREEN}python -m rvc.main${NC}"
echo ""
echo -e "${YELLOW}Или одной командой:${NC}"
echo -e "  ${GREEN}source .venv/bin/activate && python -m rvc.main${NC}"
