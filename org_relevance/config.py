import json
import os
from pathlib import Path
from dataclasses import dataclass

class Settings:
    def __init__(self, data: dict, project_root: Path):

        self.project_root = project_root

        self.DEBUG = data.get("debug", False)
        self.max_web_tries = data.get("max_web_tries", 1)
        self.web_provider = data.get("web_provider", "duckduckgo")

        # загрузка ключей в env
        for item in data.get("keys", []):
            name = item.get("name")
            key = item.get("key")
            if name and key:
                os.environ[name] = key
        
        # пути к данным
        self.data_dir = project_root / "data" / "clean"
        self.train_path = self.data_dir / "train.jsonl"
        self.eval_path = self.data_dir / "eval.jsonl"

        if not self.train_path.exists():
            print(f"[config] WARNING: train file not found: {self.train_path}")

        if not self.eval_path.exists():
            print(f"[config] WARNING: eval file not found: {self.eval_path}")

def _find_project_root() -> Path:
    """
    Ищет корень проекта 
    Поднимается вверх по папкам
    """
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "config.json").exists():
            return parent

    raise FileNotFoundError("config.json not found in project tree")


def load_config(path: Path | None = None) -> Settings:
    if path is None:
        root = _find_project_root()
        path = root / "config.json"
    else:
        root = path.parent


    data = json.loads(path.read_text(encoding="utf-8"))
    return Settings(data, root)


CONFIG = load_config()
if CONFIG.DEBUG:
    print("[config] Config is loaded!")