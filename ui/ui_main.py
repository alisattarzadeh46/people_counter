# ui/ui_main.py
from __future__ import annotations
import os
import json
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from core.direction import Flow
from core import config as CFG
from core.pipeline import run_people_counter, run_people_counter_batch
from core.tracking import set_margin_px
from ui.ui_settings import open_settings_panel
from ui.widgets import make_help_button, global_click_close_tip
from core.pipeline import run_people_counter_batch


BTN_H     = 2
BTN_W_LG  = 24
BTN_W_MD  = 18
BTN_W_SM  = 16

DISPLAY_OPTIONS = ["Top", "Bottom", "Left", "Right"]
OPPOSITE_DISPLAY = {"Top": "Bottom", "Bottom": "Top", "Left": "Right", "Right": "Left"}

CURRENT = CFG.load_settings()

def _selected_flow_from_display(label: str) -> Flow:
    mapping = {
        "Top": Flow.UP_TO_DOWN,
        "Bottom": Flow.DOWN_TO_UP,
        "Left": Flow.LEFT_TO_RIGHT,
        "Right": Flow.RIGHT_TO_LEFT,
    }
    return mapping[label]

def _check_models() -> bool:
    model_path = CURRENT.get("model_path", CFG.DEFAULT_SETTINGS["model_path"])
    if not os.path.isfile(model_path):
        messagebox.showerror("Model not found", f"Missing model file:\n\n{model_path}")
        return False
    return True

# --- webcam discovery helpers (unchanged) ---
# ... (همان کد قبلی _probe_webcams، _try_get_windows_camera_names، _make_unique_labels، _labelize_devices) ...

def _build_ipcam_url(protocol, ip, port, user, pwd, path):
    protocol = (protocol or "").strip().lower()
    ip = (ip or "").strip()
    port = str(port or "").strip()
    user = (user or "").strip()
    pwd  = (pwd or "").strip()
    path = (path or "").strip().lstrip("/")

    if protocol.startswith("local"):
        return int(ip) if ip.isdigit() else 0
    if not ip:
        raise ValueError("IP address is required for network camera.")

    auth = ""
    if user:
        auth = user + (f":{pwd}" if pwd else "") + "@"

    if protocol == "rtsp":
        port = port or "554"
        path = path or "Streaming/Channels/101"
        return f"rtsp://{auth}{ip}:{port}/{path}"
    elif "http" in protocol:
        port = port or "80"
        path = path or "video"
        return f"http://{auth}{ip}:{port}/{path}"
    else:
        raise ValueError("Unsupported protocol (choose RTSP or HTTP).")

def _inject_runtime_params():
    try:
        set_margin_px(int(CURRENT.get("margin_px", CFG.DEFAULT_SETTINGS["margin_px"])))
    except Exception:
        pass

def _runtime_tracker_knobs():
    return (
        int(CURRENT.get("_ct_max_disappeared", 50)),
        int(CURRENT.get("_ct_max_distance", 50))
    )

def _start_with_video(root, from_var):
    if not _check_models(): return
    ft = [("Video files","*.mp4;*.avi;*.mov;*.mkv;*.m4v"), ("All files","*.*")]
    path = filedialog.askopenfilename(title="Choose a video file", filetypes=ft)
    if not path: return
    if not os.path.isfile(path):
        messagebox.showerror("File not found", path); return

    _inject_runtime_params()
    ct_dis, ct_dist = _runtime_tracker_knobs()
    root.withdraw()
    try:
        run_people_counter(
            flow=_selected_flow_from_display(from_var.get()),
            source="file",
            input_path=path,
            onnx_model=CURRENT.get("model_path", CFG.DEFAULT_SETTINGS["model_path"]),
            confidence=float(CURRENT.get("confidence", CFG.DEFAULT_SETTINGS["confidence"])),
            nms_thres=float(CURRENT.get("nms", CFG.DEFAULT_SETTINGS["nms"])),
            skip_frames=int(CURRENT.get("skip_frames", CFG.DEFAULT_SETTINGS["skip_frames"])),
            img_size=int(CURRENT.get("img_size", CFG.DEFAULT_SETTINGS["img_size"])),
            ct_max_disappeared=ct_dis,
            ct_max_distance=ct_dist,
            playback_speed=float(CURRENT.get("playback_speed", CFG.DEFAULT_SETTINGS["playback_speed"]))
        )
    finally:
        root.deiconify()

# ... _start_analyze_headless همان قبلی (بدون playback_speed) ...

def _start_webcam_local(root, from_var, cam_dropdown_var, index_map):
    if not _check_models(): return
    label = cam_dropdown_var.get().strip()
    if not label or label not in index_map:
        messagebox.showerror("Webcam", "Please select a camera from the list."); return
    cam_index = index_map[label]

    prev_cfg = CFG.write_temp_camera_url_to_config(cam_index)
    _inject_runtime_params()
    ct_dis, ct_dist = _runtime_tracker_knobs()
    root.withdraw()
    try:
        run_people_counter(
            flow=_selected_flow_from_display(from_var.get()),
            source="webcam",
            input_path=None,
            onnx_model=CURRENT.get("model_path", CFG.DEFAULT_SETTINGS["model_path"]),
            confidence=float(CURRENT.get("confidence", CFG.DEFAULT_SETTINGS["confidence"])),
            nms_thres=float(CURRENT.get("nms", CFG.DEFAULT_SETTINGS["nms"])),
            skip_frames=int(CURRENT.get("skip_frames", CFG.DEFAULT_SETTINGS["skip_frames"])),
            img_size=int(CURRENT.get("img_size", CFG.DEFAULT_SETTINGS["img_size"])),
            ct_max_disappeared=ct_dis,
            ct_max_distance=ct_dist,
            playback_speed=float(CURRENT.get("playback_speed", CFG.DEFAULT_SETTINGS["playback_speed"]))
        )
    finally:
        CFG.restore_config(prev_cfg)
        root.deiconify()

def _start_ip_camera(root, from_var, cam_proto_var, cam_ip_var, cam_port_var, cam_user_var, cam_pass_var, cam_path_var):
    if not _check_models(): return
    try:
        cam_url = _build_ipcam_url(
            cam_proto_var.get(), cam_ip_var.get(), cam_port_var.get(),
            cam_user_var.get(), cam_pass_var.get(), cam_path_var.get()
        )
    except Exception as e:
        messagebox.showerror("Camera", f"Invalid camera configuration:\n{e}")
        return

    prev_cfg = CFG.write_temp_camera_url_to_config(cam_url)
    _inject_runtime_params()
    ct_dis, ct_dist = _runtime_tracker_knobs()
    root.withdraw()
    try:
        run_people_counter(
            flow=_selected_flow_from_display(from_var.get()),
            source="webcam",
            input_path=None,
            onnx_model=CURRENT.get("model_path", CFG.DEFAULT_SETTINGS["model_path"]),
            confidence=float(CURRENT.get("confidence", CFG.DEFAULT_SETTINGS["confidence"])),
            nms_thres=float(CURRENT.get("nms", CFG.DEFAULT_SETTINGS["nms"])),
            skip_frames=int(CURRENT.get("skip_frames", CFG.DEFAULT_SETTINGS["skip_frames"])),
            img_size=int(CURRENT.get("img_size", CFG.DEFAULT_SETTINGS["img_size"])),
            ct_max_disappeared=ct_dis,
            ct_max_distance=ct_dist,
            playback_speed=float(CURRENT.get("playback_speed", CFG.DEFAULT_SETTINGS["playback_speed"]))
        )
    finally:
        CFG.restore_config(prev_cfg)
        root.deiconify()

# ---------- Webcam discovery ----------
def _probe_webcams(max_index=8):
    import cv2
    indices = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                indices.append(i)
        cap.release()
    return indices

def _try_get_windows_camera_names():
    try:
        ps = r"Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image' } | Select-Object -ExpandProperty Name"
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=3)
        if out.returncode == 0:
            names = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
            return names
    except Exception:
        pass
    return []

def _make_unique_labels(names):
    seen = {}
    out = []
    for n in names:
        if n not in seen:
            seen[n] = 1
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n} #{seen[n]}")
    return out

def _labelize_devices(indices):
    names = _try_get_windows_camera_names()
    if names:
        names = _make_unique_labels(names)
    labeled = []
    for k, idx in enumerate(indices):
        if k < len(names):
            label = names[k]
        else:
            label = f"Unknown device #{k+1}"
        labeled.append((label, idx))
    return labeled

def run_ui():
    root = tk.Tk()
    root.title("People Counter")
    root.bind_all("<Button-1>", global_click_close_tip, add="+")

    outer = tk.Frame(root, padx=18, pady=16)
    outer.pack(fill="both", expand=True, pady=(16, 0))

    window_w, window_h = 550, 460
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{window_w}x{window_h}+{int((sw-window_w)/2)}+{int((sh-window_h)/2)}")
    root.resizable(False, False)

    # Direction row (centered)
    dir_row = tk.Frame(outer); dir_row.pack(pady=(4, 0))
    inner = tk.Frame(dir_row); inner.pack()
    tk.Label(inner, text="People enter from: ").pack(side="left")
    from_var = tk.StringVar(value="Top")
    ttk.Combobox(inner, state="readonly", width=10,
                 values=DISPLAY_OPTIONS, textvariable=from_var).pack(side="left", padx=(2, 8))
    tk.Label(inner, text=" →  and exit to: ").pack(side="left")
    exit_lbl = tk.Label(inner, text=OPPOSITE_DISPLAY[from_var.get()], font=("Segoe UI", 10, "bold"))
    exit_lbl.pack(side="left")
    from_var.trace_add("write", lambda *_: exit_lbl.config(text=OPPOSITE_DISPLAY[from_var.get()]))

    ttk.Separator(outer, orient="horizontal").pack(fill="x", pady=(8, 0))

    # Video actions
    row_vid = tk.Frame(outer); row_vid.pack(pady=(8, 0))
    tk.Button(row_vid, text="Open Video…", width=BTN_W_LG, height=BTN_H,
              command=lambda: _start_with_video(root, from_var)).pack(side="left", padx=10)
    tk.Button(row_vid, text="Analyze Video (No Preview)", width=BTN_W_LG, height=BTN_H,
              command=lambda: _start_analyze_headless(root, from_var)).pack(side="left", padx=10)

    ttk.Separator(outer, orient="horizontal").pack(fill="x", pady=(8, 0))

    # Webcam block
    webcam_block = tk.Frame(outer); webcam_block.pack(pady=(4, 0), fill="x")
    cam_var = tk.StringVar(value="")
    cam_dropdown = ttk.Combobox(webcam_block, state="readonly", width=48, textvariable=cam_var, values=[])
    cam_dropdown.pack(pady=(0, 6))

    index_map = {}
    def _refresh_cameras():
        indices = _probe_webcams(max_index=8)
        labeled = _labelize_devices(indices)
        labels = [lbl for (lbl, _) in labeled]
        index_map.clear()
        for lbl, idx in labeled: index_map[lbl] = idx
        if labels:
            cam_dropdown['values'] = labels; cam_var.set(labels[0]); cam_dropdown.state(["!disabled"])
        else:
            cam_dropdown['values'] = []; cam_var.set("— no camera detected —"); cam_dropdown.state(["disabled"])

    ttk.Button(webcam_block, text="Refresh cameras", command=_refresh_cameras).pack()
    tk.Button(webcam_block, text="Live Webcam", width=BTN_W_MD, height=BTN_H,
              command=lambda: _start_webcam_local(root, from_var, cam_var, index_map)).pack(pady=(6, 0))
    _refresh_cameras()

    ttk.Separator(outer, orient="horizontal").pack(fill="x", pady=(8, 0))

    # --- NEW layout for IP Camera (3 rows + centered button) ---
    ip_wrap = tk.Frame(outer)
    ip_wrap.pack(pady=(4, 0), fill="x")
    ip_inner = tk.Frame(ip_wrap)
    ip_inner.pack()  # centered

    # Row 1: Protocol + IP + Port
    row1 = tk.Frame(ip_inner); row1.pack(anchor="center", pady=2)
    tk.Label(row1, text="Protocol:").grid(row=0, column=0, sticky="w", padx=(0,4))
    cam_proto_var = tk.StringVar(value="RTSP")
    ttk.Combobox(row1, state="readonly", width=12,
                 values=["RTSP", "HTTP (MJPEG)"], textvariable=cam_proto_var).grid(row=0, column=1, sticky="w", padx=(0,12))
    tk.Label(row1, text="IP:").grid(row=0, column=2, sticky="w")
    cam_ip_var = tk.StringVar(value="")
    tk.Entry(row1, textvariable=cam_ip_var, width=16).grid(row=0, column=3, sticky="w", padx=(6,12))
    tk.Label(row1, text="Port:").grid(row=0, column=4, sticky="w")
    cam_port_var = tk.StringVar(value="554")
    tk.Entry(row1, textvariable=cam_port_var, width=8).grid(row=0, column=5, sticky="w", padx=(6,0))

    # Row 2: User + Pass
    row2 = tk.Frame(ip_inner); row2.pack(anchor="center", pady=2)
    tk.Label(row2, text="User:").grid(row=0, column=0, sticky="w")
    cam_user_var = tk.StringVar(value="")
    tk.Entry(row2, textvariable=cam_user_var, width=24).grid(row=0, column=1, sticky="w", padx=(6,12))
    tk.Label(row2, text="Pass:").grid(row=0, column=2, sticky="w")
    cam_pass_var = tk.StringVar(value="")
    tk.Entry(row2, textvariable=cam_pass_var, width=24, show="*").grid(row=0, column=3, sticky="w", padx=(6,0))

    # Row 3: Path + help
    row3 = tk.Frame(ip_inner); row3.pack(anchor="center", pady=2)
    tk.Label(row3, text="Path:").grid(row=0, column=0, sticky="w")
    cam_path_var = tk.StringVar(value="Streaming/Channels/101")
    tk.Entry(row3, textvariable=cam_path_var, width=28).grid(row=0, column=1, sticky="w", padx=(6,8))
    make_help_button(row3,
        "IP Camera help:\n"
        "RTSP: rtsp://user:pass@IP:port/Path (default 554, e.g., Streaming/Channels/101)\n"
        "HTTP (MJPEG): http://user:pass@IP:port/Path (default 80, e.g., mjpg/video.mjpg)\n"
        "Use substream for smoother analysis (e.g., .../102)."
    ).grid(row=0, column=2, sticky="w")

    # Button centered under the block
    tk.Button(outer, text="Start IP Camera", width=BTN_W_MD, height=BTN_H,
              command=lambda: _start_ip_camera(
                  root, from_var, cam_proto_var, cam_ip_var, cam_port_var, cam_user_var, cam_pass_var, cam_path_var
              )).pack(pady=(6, 0))

    ttk.Separator(outer, orient="horizontal").pack(fill="x", pady=(8, 0))

    tk.Button(outer, text="Settings…", width=BTN_W_SM, height=BTN_H,
              command=lambda: open_settings_panel(root, CURRENT)).pack(pady=(6, 0))

    root.mainloop()

def _start_analyze_headless(root, from_var):
    # --- imports needed here ---
    import os, threading, datetime
    from tkinter import filedialog, messagebox, ttk
    import tkinter as tk

    from core import config as CFG
    from core.pipeline import run_people_counter_batch

    # Load settings
    s = CFG.load_settings()
    ct_dis = int(s.get("_ct_max_disappeared", 50))
    ct_dist = int(s.get("_ct_max_distance", 50))

    # Pick video
    ft = [("Video files","*.mp4;*.avi;*.mov;*.mkv;*.m4v"), ("All files","*.*")]
    path = filedialog.askopenfilename(title="Choose a video for analysis", filetypes=ft)
    if not path:
        return
    if not os.path.isfile(path):
        messagebox.showerror("File not found", path)
        return

    # Progress window
    top = tk.Toplevel(root)
    top.title("Analyze Video (No Preview)")
    top.geometry("560x280")
    top.resizable(False, False)

    file_lbl = tk.Label(top, text=f"Analyzing: {os.path.basename(path)}", font=("Segoe UI", 10))
    file_lbl.pack(pady=(12, 8))

    style = ttk.Style(top)
    try: style.theme_use('clam')
    except Exception: pass
    style.configure('Analyze.Horizontal.TProgressbar', troughcolor='#3a3a3a', background='#2ecc71')

    pb = ttk.Progressbar(top, orient="horizontal", mode="determinate",
                         length=460, maximum=100, style='Analyze.Horizontal.TProgressbar')
    pb.pack(pady=(10, 4))
    percent_lbl = tk.Label(top, text="0%", font=("Segoe UI", 10))
    percent_lbl.pack(pady=(0, 8))

    hud = tk.Frame(top, bg="#1e1e1e", height=44)
    hud.pack(side="bottom", fill="x")
    def _mk(lbl, init):
        f = tk.Frame(hud, bg="#1e1e1e")
        t = tk.Label(f, text=lbl, fg="white", bg="#1e1e1e", font=("Segoe UI", 9, "bold"))
        v = tk.Label(f, text=init, fg="white", bg="#1e1e1e", font=("Segoe UI", 9))
        t.pack(side="left", padx=(0,4)); v.pack(side="left")
        f.pack(side="left", padx=24, pady=8)
        return v
    status_val = _mk("Status:", "Detecting")
    enter_val  = _mk("Enter:", "0")
    exit_val   = _mk("Exit:", "0")
    inside_val = _mk("Inside:", "0")

    btns = tk.Frame(top); btns.pack(pady=(6, 4))
    cancel_btn = tk.Button(btns, text="Cancel", width=18, height=2)
    cancel_btn.pack()

    cancel_event = threading.Event()

    def on_progress(pct, enter_c, exit_c, inside_c, status_text):
        def _apply():
            pb['value'] = pct
            percent_lbl.config(text=f"{int(pct)}%")
            status_val.config(text=status_text)
            enter_val.config(text=str(enter_c))
            exit_val.config(text=str(exit_c))
            inside_val.config(text=str(inside_c))
        try:
            top.after(0, _apply)
        except Exception:
            pass

    def worker():
        try:
            # Run batch analysis (no preview)
            enter_c, exit_c, exporter = run_people_counter_batch(
                flow=_selected_flow_from_display(from_var.get()),
                input_path=path,
                onnx_model=s.get("model_path"),
                confidence=float(s.get("confidence")),
                nms_thres=float(s.get("nms")),
                skip_frames=int(s.get("skip_frames")),
                img_size=int(s.get("img_size")),
                progress_cb=on_progress,
                cancel_event=cancel_event,
                ct_max_disappeared=ct_dis,
                ct_max_distance=ct_dist
            )

            if cancel_event.is_set():
                return

            # --- Auto open Save dialog right after analysis finishes ---
            def _ask_and_save():
                try:
                    from datetime import datetime
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    default_name = f"people_counter_video_{ts}.xlsx"
                    save_path = filedialog.asksaveasfilename(
                        title="Save report",
                        initialfile=default_name,
                        defaultextension=".xlsx",
                        filetypes=[("Excel Workbook", "*.xlsx"), ("CSV", "*.csv")]
                    )
                    if save_path:
                        media_name = os.path.basename(path)
                        saved = exporter.export_to(save_path, enter_c, exit_c, "Video (Analyze)", media_name)
                        messagebox.showinfo("Saved", f"Report saved to:\n{saved}")
                finally:
                    try:
                        top.destroy()
                    except Exception:
                        pass

            top.after(0, _ask_and_save)

        except Exception as e:
            # Show the error and close the progress window
            def _err():
                messagebox.showerror("Error", str(e))
                try:
                    top.destroy()
                except Exception:
                    pass
            try:
                top.after(0, _err)
            except Exception:
                pass

    def on_cancel():
        cancel_event.set()
        cancel_btn.config(state="disabled", text="Canceling...")

    cancel_btn.config(command=on_cancel)
    threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    run_ui()
