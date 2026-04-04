"""
Загрузка конфигурации приложения из params.yaml
"""

from pathlib import Path
import yaml

def get_params() -> dict:
    """
    Загружает конфигурацию приложения из YAML.

    Returns:
        dict: Конфигурация приложения.
    """
    # config_path = Path(__file__).resolve().parent / "params.yaml"
    config_path =  "params.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)