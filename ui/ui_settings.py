"""
Settings panel window (simplified, no overwrite of user img_size/skip_frames).

- Crowd level & Accuracy level DO NOT override img_size/skip_frames anymore.
- Confidence & NMS are sliders only.
- Playback speed is a slider (0.25x..2.0x).
- Saves/loads via core.config with absolute CONFIG_PATH.
"""
from __future__ import annotations
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from core import config as CFG
from ui.widgets import make_help_button

BTN_H     = 2
BTN_W_SM  = 16

def open_settings_panel(root, CURRENT: dict):
    s = CURRENT.copy()

    top = tk.Toplevel(root)
    top.title("Settings")
    top.geometry("760x500")
    top.resizable(False, False)

    frm = tk.Frame(top)
    frm.pack(padx=18, pady=14, fill="x")

    def _row_label(row, text):
        lbl = tk.Label(frm, text=text)
        lbl.grid(row=row, column=0, sticky="w", pady=6)
        return lbl

    # ---------------- Model path ----------------
    _row_label(0, "ONNX model path:")
    model_var = tk.StringVar(value=s.get("model_path", CFG.DEFAULT_SETTINGS["model_path"]))
    tk.Entry(frm, textvariable=model_var, width=52).grid(row=0, column=1, sticky="w", padx=(6,0))
    def _browse_model():
        p = filedialog.askopenfilename(title="Select ONNX model",
                                       filetypes=[("ONNX", "*.onnx"), ("All files","*.*")])
        if p: model_var.set(p)
    ttk.Button(frm, text="Browse…", command=_browse_model).grid(row=0, column=2, padx=6)
    make_help_button(frm,
        "Path to the YOLO ONNX weights (.onnx). Only 'person' class is used.\n"
        "Use a COCO-compatible model (e.g., yolov5s.onnx)."
    ).grid(row=0, column=3, padx=(6,0))

    # ---------------- Crowd level ----------------
    _row_label(1, "Crowd level:")
    crowd_var = tk.IntVar(value=int(s.get("crowd_level", CFG.DEFAULT_SETTINGS["crowd_level"])))
    tk.Scale(frm, from_=1, to=4, orient="horizontal", showvalue=True,
             variable=crowd_var, length=220, tickinterval=1, sliderlength=16)\
        .grid(row=1, column=1, sticky="w", padx=(6,0))
    make_help_button(frm,
        "How crowded the scene is (used to tune tracker strictness only)."
    ).grid(row=1, column=3, padx=(6,0))

    # ---------------- Accuracy level ----------------
    _row_label(2, "Accuracy level:")
    acc_var = tk.IntVar(value=int(s.get("accuracy_level", CFG.DEFAULT_SETTINGS["accuracy_level"])))
    tk.Scale(frm, from_=1, to=4, orient="horizontal", showvalue=True,
             variable=acc_var, length=220, tickinterval=1, sliderlength=16)\
        .grid(row=2, column=1, sticky="w", padx=(6,0))
    make_help_button(frm,
        "Overall accuracy vs speed (does NOT override your img_size/skip_frames)."
    ).grid(row=2, column=3, padx=(6,0))

    # ---------------- Playback speed (0.25x..2.0x) ----------------
    _row_label(3, "Playback speed (preview):")
    spd_var = tk.DoubleVar(value=float(s.get("playback_speed", CFG.DEFAULT_SETTINGS.get("playback_speed", 1.0))))
    tk.Scale(frm, from_=0.25, to=2.0, resolution=0.05, orient="horizontal",
             showvalue=True, variable=spd_var, length=220, sliderlength=16)\
        .grid(row=3, column=1, sticky="w", padx=(6,0))
    make_help_button(frm,
        "Preview playback speed only. 1.0x = normal; <1.0 slower; up to 2.0x to catch up."
    ).grid(row=3, column=3, padx=(6,0))

    # ---------------- Confidence (slider only) ----------------
    _row_label(4, "Confidence threshold:")
    conf_var = tk.DoubleVar(value=float(s.get("confidence", CFG.DEFAULT_SETTINGS["confidence"])))
    tk.Scale(frm, from_=0.10, to=0.60, resolution=0.01, orient="horizontal",
             showvalue=True, variable=conf_var, length=220, sliderlength=16)\
        .grid(row=4, column=1, sticky="w", padx=(6,0))
    make_help_button(frm,
        "Minimum detection score. Higher = fewer false positives, may miss small/distant persons."
    ).grid(row=4, column=3, padx=(6,0))

    # ---------------- NMS (slider only) ----------------
    _row_label(5, "NMS IoU threshold:")
    nms_var = tk.DoubleVar(value=float(s.get("nms", CFG.DEFAULT_SETTINGS["nms"])))
    tk.Scale(frm, from_=0.30, to=0.70, resolution=0.01, orient="horizontal",
             showvalue=True, variable=nms_var, length=220, sliderlength=16)\
        .grid(row=5, column=1, sticky="w", padx=(6,0))
    make_help_button(frm,
        "Non-Max Suppression IoU. Lower = merge duplicates more aggressively."
    ).grid(row=5, column=3, padx=(6,0))

    # ---------------- Advanced (user-controlled, WILL BE SAVED as-is) ----------------
    def _int_entry(row, key, default, label, help_text):
        _row_label(row, label)
        var = tk.StringVar(value=str(s.get(key, default)))
        tk.Entry(frm, textvariable=var, width=12).grid(row=row, column=1, sticky="w", padx=(6,0))
        make_help_button(frm, help_text).grid(row=row, column=3, padx=(6,0))
        return var

    img_var  = _int_entry(6, "img_size", CFG.DEFAULT_SETTINGS["img_size"], "Image size (px):",
                          "Inference input size. Multiples of 32 recommended.")
    skip_var = _int_entry(7, "skip_frames", CFG.DEFAULT_SETTINGS["skip_frames"], "Skip frames:",
                          "Run detector every N frames; tracking in between.")
    marg_var = _int_entry(8, "margin_px", CFG.DEFAULT_SETTINGS["margin_px"], "MARGIN_PX (full-cross):",
                          "Strictness margin around the midline.")

    ttk.Separator(frm).grid(row=9, column=0, columnspan=4, sticky="ew", pady=12)

    btn_row = tk.Frame(frm); btn_row.grid(row=10, column=0, columnspan=4, pady=(2,0), sticky="n")

    def _apply_derived_from_scales(sdict):
        """
        NOTE: No overrides of img_size or skip_frames here.
        We only tune tracker strictness based on crowd; accuracy is advisory only.
        """
        crowd = int(crowd_var.get())
        if crowd == 1:
            sdict["_ct_max_disappeared"] = 60
            sdict["_ct_max_distance"]   = 55
        elif crowd == 2:
            sdict["_ct_max_disappeared"] = 50
            sdict["_ct_max_distance"]   = 50
        elif crowd == 3:
            sdict["_ct_max_disappeared"] = 45
            sdict["_ct_max_distance"]   = 48
        else:
            sdict["_ct_max_disappeared"] = 40
            sdict["_ct_max_distance"]   = 46
        # accuracy_level kept for future use; no auto overrides.

    def _reset_to_defaults():
        model_var.set(CFG.DEFAULT_SETTINGS["model_path"])
        crowd_var.set(CFG.DEFAULT_SETTINGS["crowd_level"])
        acc_var.set(CFG.DEFAULT_SETTINGS["accuracy_level"])
        spd_var.set(CFG.DEFAULT_SETTINGS.get("playback_speed", 1.0))
        conf_var.set(CFG.DEFAULT_SETTINGS["confidence"])
        nms_var.set(CFG.DEFAULT_SETTINGS["nms"])
        img_var.set(str(CFG.DEFAULT_SETTINGS["img_size"]))
        skip_var.set(str(CFG.DEFAULT_SETTINGS["skip_frames"]))
        marg_var.set(str(CFG.DEFAULT_SETTINGS["margin_px"]))
        messagebox.showinfo("Settings", "Reset to defaults (not saved yet). Click 'Save & Close' to apply.")

    def _apply_and_close():
        try:
            # Base fields
            s["model_path"]     = model_var.get().strip() or CFG.DEFAULT_SETTINGS["model_path"]
            s["crowd_level"]    = int(crowd_var.get())
            s["accuracy_level"] = int(acc_var.get())
            s["playback_speed"] = float(spd_var.get())
            s["confidence"]     = float(conf_var.get())
            s["nms"]            = float(nms_var.get())

            # User-controlled numeric entries (WILL be respected)
            s["img_size"]       = int(img_var.get())
            s["skip_frames"]    = int(skip_var.get())
            s["margin_px"]      = int(marg_var.get())

            # Only tracker strictness from crowd; no overrides of user values
            _apply_derived_from_scales(s)

            CURRENT.update(s)
            CFG.save_settings(CURRENT)
            messagebox.showinfo("Settings", "Settings saved.")
            top.destroy()
        except Exception as e:
            messagebox.showerror("Settings", f"Invalid values:\n{e}")

    tk.Button(btn_row, text="Reset to defaults", width=BTN_W_SM, height=BTN_H,
              command=_reset_to_defaults).pack(side="left", padx=8)
    tk.Button(btn_row, text="Save & Close", width=BTN_W_SM, height=BTN_H,
              command=_apply_and_close).pack(side="left", padx=8)
