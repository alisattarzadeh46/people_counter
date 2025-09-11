"""
Small reusable UI widgets/helpers:
- sticky tooltips (click to open, click elsewhere to close)
- help button factory
"""
import tkinter as tk

_tip_win = None

def hide_tip():
    global _tip_win
    try:
        if _tip_win and _tip_win.winfo_exists():
            _tip_win.destroy()
    except Exception:
        pass
    _tip_win = None

def show_tip(anchor_widget, text: str):
    """Sticky tooltip; stays until clicking elsewhere."""
    global _tip_win
    hide_tip()
    _tip_win = tk.Toplevel(anchor_widget)
    _tip_win.wm_overrideredirect(True)
    _tip_win.attributes("-topmost", True)
    x = anchor_widget.winfo_rootx() + 20
    y = anchor_widget.winfo_rooty() + anchor_widget.winfo_height() + 6
    _tip_win.geometry(f"+{x}+{y}")
    frm = tk.Frame(_tip_win, bg="#ffffe0", bd=1, relief="solid")
    frm.pack()
    lbl = tk.Label(frm, text=text, bg="#ffffe0", justify="left", anchor="w")
    lbl.config(font=("Segoe UI", 9), wraplength=460, padx=10, pady=8)
    lbl.pack()
    _tip_win.bind("<Button-1>", lambda *_: hide_tip())

def make_help_button(parent, text: str):
    btn = tk.Button(parent, text="?", width=2, command=lambda: show_tip(btn, text))
    btn._is_help_button = True  # mark to avoid instant close
    return btn

def global_click_close_tip(event):
    w = event.widget
    if getattr(w, "_is_help_button", False):
        return
    hide_tip()
