import logging
from ui.ui_main import run_ui

logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")

if __name__ == "__main__":
    run_ui()