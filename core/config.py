# core/config.py
"""
App configuration utilities (Single Source of Truth).

- DEFAULT_SETTINGS
- load_settings / save_settings
- write_temp_camera_url_to_config / restore_config
"""
from __future__ import annotations
import json, os
from typing import Dict, Any

# --- project root (absolute) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CONFIG_PATH  = os.path.join(PROJECT_ROOT, "utils", "config.json")

DEFAULT_SETTINGS: Dict[str, Any] = {
    "confidence": 0.28,
    "nms": 0.45,
    "img_size": 640,
    "skip_frames": 8,
    "margin_px": 5,
    "model_path": "models/yolov5s.onnx",
    "crowd_level": 2,
    "accuracy_level": 2,
    "playback_speed": 1.0,
}

def _ensure_parent_dir(p: str):
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)

def _deep_merge(base: dict, extra: dict) -> dict:
    out = base.copy()
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_settings() -> Dict[str, Any]:
    s = DEFAULT_SETTINGS.copy()
    try:
        if os.path.isfile(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            s = _deep_merge(s, data)
    except Exception:
        pass
    return s

def save_settings(s: Dict[str, Any]) -> None:
    _ensure_parent_dir(CONFIG_PATH)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

def write_temp_camera_url_to_config(url_or_index) -> dict:
    orig = {}
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                orig = json.load(f)
        except Exception:
            orig = {}
    mod = orig.copy()
    mod["url"] = url_or_index
    _ensure_parent_dir(CONFIG_PATH)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(mod, f, ensure_ascii=False, indent=2)
    return orig

def restore_config(prev_obj: dict) -> None:
    try:
        _ensure_parent_dir(CONFIG_PATH)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(prev_obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
