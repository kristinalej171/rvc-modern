"""Пресеты параметров для быстрого инференса."""

from dataclasses import dataclass
from typing import Dict

@dataclass
class InferencePreset:
    name: str
    description: str
    pitch_shift: int
    index_rate: float
    protect: float
    rms_mix_rate: float

PRESETS: Dict[str, InferencePreset] = {
    "🎤 Пение (Мужской -> Женский)": InferencePreset(
        name="🎤 Пение (Мужской -> Женский)",
        description="Оптимально для вокала. Сдвиг на +12 полутонов (октава вверх).",
        pitch_shift=12,
        index_rate=0.75,
        protect=0.33,
        rms_mix_rate=0.25,
    ),
    "🎤 Пение (Женский -> Мужской)": InferencePreset(
        name="🎤 Пение (Женский -> Мужской)",
        description="Оптимально для вокала. Сдвиг на -12 полутонов (октава вниз).",
        pitch_shift=-12,
        index_rate=0.75,
        protect=0.33,
        rms_mix_rate=0.25,
    ),
    "🗣️ Речь (Подкаст/Озвучка)": InferencePreset(
        name="🗣️ Речь (Подкаст/Озвучка)",
        description="Сохранение естественной интонации речи без сдвига тона.",
        pitch_shift=0,
        index_rate=0.85,
        protect=0.50,  # Максимальная защита согласных
        rms_mix_rate=0.50,
    ),
    "🎧 Кавер (Без сдвига)": InferencePreset(
        name="🎧 Кавер (Без сдвига)",
        description="Для случаев, когда тональности оригинала и модели совпадают.",
        pitch_shift=0,
        index_rate=0.60,
        protect=0.33,
        rms_mix_rate=0.20,
    ),
}

def get_preset_choices() -> list[str]:
    return ["⚙️ Ручная настройка"] + list(PRESETS.keys())