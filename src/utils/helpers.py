import yaml
from pathlib import Path
from typing import Dict, Any


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def load_config(config_path: str) -> Dict[str, Any]:
    config_file = get_project_root() / config_path
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_model_config() -> Dict[str, Any]:
    return load_config('config/model_config.yaml')


def load_lora_config() -> Dict[str, Any]:
    return load_config('config/lora_config.yaml')


def load_persona_config() -> Dict[str, Any]:
    return load_config('config/persona_config.yaml')