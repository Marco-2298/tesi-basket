"""Percorsi relativi alla radice: stessa configurazione su Windows e Colab."""
from pathlib import Path
import json
import os

ROOT = Path(os.environ.get('NBA_PROJECT_DIR', Path(__file__).resolve().parents[1])).expanduser().resolve()


def load_config():
    path = ROOT / 'configs' / 'project.json'
    if not path.is_file():
        raise FileNotFoundError(f'Configurazione mancante: {path}; eseguire scripts/prepare_workspace.py')
    data = json.loads(path.read_text(encoding='utf-8'))
    for key in ['clips_dir', 'detection_test_dir', 'weights', 'manifest', 'annotations', 'tasks']:
        value = Path(data[key])
        data[key] = value if value.is_absolute() else ROOT / value
    return data
