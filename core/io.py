"""
IO helpers:
- CameraSource: unified wrapper for webcam/IP/video file
- SessionExporter: save report to .xlsx (if openpyxl) or .csv
- CSV log appends (optional)
- ask_save_path: system dialog for 'Export' button
"""
from __future__ import annotations
import os
import csv
import time
import cv2
from datetime import datetime
from itertools import zip_longest
from imutils.video import VideoStream

# -------- CameraSource --------
class CameraSource:
    """
    Unified wrapper over local webcam (USB), IP camera (RTSP/HTTP), and video file.
    """
    def __init__(self, source_type: str, input_path=None, url_or_index=None):
        self.source_type = source_type
        self.input_path = input_path
        self.url_or_index = url_or_index
        self.mode = None   # "vs" (VideoStream) or "cv" (VideoCapture)
        self.vs = None     # imutils.VideoStream
        self.cap = None    # cv2.VideoCapture

    def start(self):
        if self.source_type == "file":
            self.cap = cv2.VideoCapture(self.input_path)
            self.mode = "cv"
        elif self.source_type == "webcam":
            is_local = isinstance(self.url_or_index, int) or (
                isinstance(self.url_or_index, str) and self.url_or_index.isdigit()
            )
            if is_local:
                idx = int(self.url_or_index) if isinstance(self.url_or_index, str) else self.url_or_index
                self.vs = VideoStream(idx).start()
                time.sleep(2.0)  # warm-up
                self.mode = "vs"
            else:
                self.cap = cv2.VideoCapture(self.url_or_index, cv2.CAP_FFMPEG)
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                self.mode = "cv"
        else:
            raise ValueError("source_type must be 'file' or 'webcam'.")
        return self

    def read(self):
        if self.mode == "vs":
            return self.vs.read()
        elif self.mode == "cv":
            ok, frame = self.cap.read()
            return frame if ok else None
        return None

    def is_opened(self):
        if self.mode == "vs":
            return True
        elif self.mode == "cv":
            return self.cap is not None and self.cap.isOpened()
        return False

    def get_fps(self, default=30):
        if self.mode == "cv" and self.cap is not None:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            return fps if fps and fps > 1 else default
        return default

    def release(self):
        try:
            if self.mode == "vs" and self.vs is not None:
                self.vs.stop()
            if self.mode == "cv" and self.cap is not None:
                self.cap.release()
        except Exception:
            pass

# -------- SessionExporter --------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

class SessionExporter:
    """
    Export session with columns:
    [Source, Media Name, Date, Start Time, End Time, Enter, Exit, Inside]
    """
    def __init__(self, out_dir: str = "data/exports"):
        self.out_dir = out_dir
        _ensure_dir(self.out_dir)
        self._started_at = None

    def start(self):
        self._started_at = datetime.now()

    def _build_row(self, enter_count: int, exit_count: int,
                   source_label: str, media_name: str | None):
        if self._started_at is None:
            self._started_at = datetime.now()

        ended_at = datetime.now()
        inside = max(0, int(enter_count) - int(exit_count))

        date_str   = self._started_at.strftime("%Y-%m-%d")
        start_time = self._started_at.strftime("%H:%M:%S")
        end_time   = ended_at.strftime("%H:%M:%S")

        row = [
            source_label,
            (media_name or "-"),
            date_str, start_time, end_time,
            int(enter_count), int(exit_count), inside
        ]
        headers = ["Source", "Media Name", "Date", "Start Time", "End Time", "Enter", "Exit", "Inside"]
        return headers, row

    def export_to(self, save_path: str, enter_count: int, exit_count: int,
                  source_label: str, media_name: str | None = None) -> str:
        """
        Save to a user-selected path.
        If extension is .xlsx and openpyxl is available -> Excel.
        Otherwise -> CSV (UTF-8-SIG).
        """
        headers, row = self._build_row(enter_count, exit_count, source_label, media_name)
        ext = os.path.splitext(save_path)[1].lower()

        if ext == ".xlsx":
            try:
                from openpyxl import Workbook
                from openpyxl.styles import Alignment, Font
                wb = Workbook()
                ws = wb.active
                ws.title = "Report"
                ws.append(headers)
                ws.append(row)

                bold = Font(bold=True)
                for cell in ws[1]:
                    cell.font = bold
                    cell.alignment = Alignment(horizontal="center")

                widths = [10, 28, 12, 12, 12, 10, 10, 10]
                for i, w in enumerate(widths, start=1):
                    ws.column_dimensions[chr(64 + i)].width = w

                _ensure_dir(os.path.dirname(save_path) or ".")
                wb.save(save_path)
                return save_path
            except Exception:
                save_path = os.path.splitext(save_path)[0] + ".csv"

        _ensure_dir(os.path.dirname(save_path) or ".")
        with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.writer(f)
            wr.writerow(headers)
            wr.writerow(row)
        return save_path

# -------- CSV append log (optional) --------
def csv_append_moves(move_in, in_time, move_out, out_time):
    """
    Append raw move data to data/logs/counting_data.csv (for debugging / audit).
    """
    export_data = zip_longest(move_in, in_time, move_out, out_time, fillvalue="")
    log_dir = "data/logs"
    _ensure_dir(log_dir)
    log_file = os.path.join(log_dir, "counting_data.csv")
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        if not file_exists:
            wr.writerow(("Move In", "In Time", "Move Out", "Out Time"))
        wr.writerows(export_data)

# -------- Save-As dialog --------
def ask_save_path(default_name: str) -> str | None:
    """Open a Save As dialog and return selected path, or None if canceled."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.asksaveasfilename(
            title="Save report",
            initialfile=default_name,
            defaultextension=".xlsx",
            filetypes=[("Excel Workbook", "*.xlsx"), ("CSV", "*.csv")]
        )
        root.destroy()
        return path if path else None
    except Exception:
        return None
