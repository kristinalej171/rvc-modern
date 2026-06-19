import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm

# Базовая директория проекта
BASE_DIR = Path(__file__).resolve().parent
RVC_DOWNLOAD_LINK = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"

# Словарь необходимых моделей: {локальный_путь: путь_на_huggingface}
REQUIRED_MODELS = {
    # Hubert and RMVPE
    "assets/hubert/hubert_base.pt": "hubert_base.pt",
    "assets/rmvpe/rmvpe.pt": "rmvpe.pt",
    
    # UVR5 ONNX
    "assets/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx": "uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx",
    
    # Pretrained v1
    "assets/pretrained/D32k.pth": "pretrained/D32k.pth",
    "assets/pretrained/D40k.pth": "pretrained/D40k.pth",
    "assets/pretrained/D48k.pth": "pretrained/D48k.pth",
    "assets/pretrained/G32k.pth": "pretrained/G32k.pth",
    "assets/pretrained/G40k.pth": "pretrained/G40k.pth",
    "assets/pretrained/G48k.pth": "pretrained/G48k.pth",
    "assets/pretrained/f0D32k.pth": "pretrained/f0D32k.pth",
    "assets/pretrained/f0D40k.pth": "pretrained/f0D40k.pth",
    "assets/pretrained/f0D48k.pth": "pretrained/f0D48k.pth",
    "assets/pretrained/f0G32k.pth": "pretrained/f0G32k.pth",
    "assets/pretrained/f0G40k.pth": "pretrained/f0G40k.pth",
    "assets/pretrained/f0G48k.pth": "pretrained/f0G48k.pth",
    
    # Pretrained v2
    "assets/pretrained_v2/D40k.pth": "pretrained_v2/D40k.pth",
    "assets/pretrained_v2/G40k.pth": "pretrained_v2/G40k.pth",
    "assets/pretrained_v2/f0D40k.pth": "pretrained_v2/f0D40k.pth",
    "assets/pretrained_v2/f0G40k.pth": "pretrained_v2/f0G40k.pth",
    
    # UVR5 Weights
    "assets/uvr5_weights/HP2_all_vocals.pth": "uvr5_weights/HP2_all_vocals.pth",
    "assets/uvr5_weights/HP3_all_vocals.pth": "uvr5_weights/HP3_all_vocals.pth",
    "assets/uvr5_weights/HP5_only_main_vocal.pth": "uvr5_weights/HP5_only_main_vocal.pth",
    "assets/uvr5_weights/VR-DeEchoAggressive.pth": "uvr5_weights/VR-DeEchoAggressive.pth",
    "assets/uvr5_weights/VR-DeEchoDeReverb.pth": "uvr5_weights/VR-DeEchoDeReverb.pth",
    "assets/uvr5_weights/VR-DeEchoNormal.pth": "uvr5_weights/VR-DeEchoNormal.pth",
}

def download_file(url: str, dest_path: Path):
    """Скачивает файл с отображением прогресса."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"⬇️ Скачивание {dest_path.name}...")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(dest_path, 'wb') as f, tqdm(
                desc=dest_path.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
        print(f"✅ Успешно скачано: {dest_path}")
    except Exception as e:
        print(f"❌ Ошибка при скачивании {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()

def check_and_download():
    """Проверяет наличие моделей и скачивает недостающие."""
    print("🔍 Проверка наличия необходимых моделей...")
    missing_models = []
    
    for local_path, remote_path in REQUIRED_MODELS.items():
        full_local_path = BASE_DIR / local_path
        if not full_local_path.exists():
            missing_models.append((local_path, remote_path))
            
    if not missing_models:
        print("✅ Все необходимые модели уже присутствуют!")
        return

    print(f"⚠️ Отсутствует {len(missing_models)} файлов. Начинаем загрузку...")
    for local_path, remote_path in missing_models:
        full_local_path = BASE_DIR / local_path
        url = f"{RVC_DOWNLOAD_LINK}{remote_path}"
        download_file(url, full_local_path)
        
    print("🎉 Проверка и загрузка моделей завершены.")

if __name__ == "__main__":
    check_and_download()