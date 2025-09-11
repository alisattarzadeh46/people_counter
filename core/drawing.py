"""
Drawing helpers for video overlays + bottom HUD bar + Export button.
"""
from __future__ import annotations
import cv2
import numpy as np
from core.direction import Flow, is_vertical

BAR_H = 50  # bottom HUD bar height

def draw_border_and_labels(frame, flow: Flow):
    """Draw the midline and 'ENTRANCE/EXIT' labels on the VIDEO area (not the HUD bar)."""
    H, W = frame.shape[:2]
    color = (0, 0, 0)
    th = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    if is_vertical(flow):
        border_y = H // 2
        cv2.line(frame, (0, border_y), (W, border_y), color, th)

        if flow == Flow.UP_TO_DOWN:
            cv2.putText(frame, "EXIT",      (int(W*0.46), 30),   font, 0.8, color, 2, cv2.LINE_AA)
            cv2.putText(frame, "ENTRANCE",  (int(W*0.40), H-10), font, 0.8, color, 2, cv2.LINE_AA)
        else:  # DOWN_TO_UP
            cv2.putText(frame, "ENTRANCE",  (int(W*0.40), 30),   font, 0.8, color, 2, cv2.LINE_AA)
            cv2.putText(frame, "EXIT",      (int(W*0.46), H-10), font, 0.8, color, 2, cv2.LINE_AA)
    else:
        border_x = W // 2
        cv2.line(frame, (border_x, 0), (border_x, H), color, th)

        if flow == Flow.LEFT_TO_RIGHT:
            cv2.putText(frame, "EXIT",      (10, 30),     font, 0.8, color, 2, cv2.LINE_AA)
            cv2.putText(frame, "ENTRANCE",  (W-130, 30),  font, 0.8, color, 2, cv2.LINE_AA)
        else:  # RIGHT_TO_LEFT
            cv2.putText(frame, "ENTRANCE",  (10, 30),     font, 0.8, color, 2, cv2.LINE_AA)
            cv2.putText(frame, "EXIT",      (W-90, 30),   font, 0.8, color, 2, cv2.LINE_AA)

def compose_with_bottom_bar(frame):
    """Return a new image with an extra bottom bar area for HUD."""
    H, W = frame.shape[:2]
    out = np.zeros((H + BAR_H, W, 3), dtype=frame.dtype)
    out[:H] = frame
    out[H:] = (30, 30, 30)  # bar background
    return out

def draw_hud_bar(canvas, enter_count, exit_count, inside_count, status_text):
    """
    Draw stats with fixed positions and leave space on the right for the Export button.
    """
    H_total, W = canvas.shape[:2]
    y = H_total - BAR_H + 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 1
    color = (255, 255, 255)

    RESERVED_RIGHT = 260  # px space reserved for the Export button

    x_status = 20
    x_enter  = int(W * 0.30)
    x_exit   = int(W * 0.50)
    x_inside = min(int(W * 0.68), W - RESERVED_RIGHT)

    cv2.putText(canvas, "Status:", (x_status, y), font, scale, color, thick, cv2.LINE_AA)
    cv2.putText(canvas, status_text, (x_status + 80, y), font, scale, color, thick, cv2.LINE_AA)

    cv2.putText(canvas, f"Enter: {enter_count}",  (x_enter,  y), font, scale, color, thick, cv2.LINE_AA)
    cv2.putText(canvas, f"Exit: {exit_count}",    (x_exit,   y), font, scale, color, thick, cv2.LINE_AA)
    cv2.putText(canvas, f"Inside: {inside_count}",(x_inside, y), font, scale, color, thick, cv2.LINE_AA)

def compute_export_button_rect(W: int, H_total: int, label: str = "Export"):
    """
    Compute (x1,y1,x2,y2) for the Export button anchored to the right side of the HUD bar.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    icon_w = 18
    pad_x = 10
    pad_y = 8
    btn_w = icon_w + pad_x + tw + pad_x
    btn_h = th + pad_y + 6

    right_margin = 12
    x2 = W - right_margin
    x1 = x2 - btn_w

    bar_top = H_total - BAR_H
    cy = bar_top + BAR_H // 2
    y1 = cy - btn_h // 2
    y2 = y1 + btn_h
    return (x1, y1, x2, y2)

def draw_export_button(canvas, rect, hovered: bool = False, label: str = "Export"):
    """Render the Export button."""
    x1, y1, x2, y2 = rect
    bg = (60, 60, 60) if not hovered else (80, 80, 80)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, cv2.FILLED)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (160, 160, 160), 1)

    # small "save" arrow icon
    cx = x1 + 12
    cy = (y1 + y2) // 2
    cv2.line(canvas, (cx, cy-6), (cx, cy+6), (240, 240, 240), 2)
    cv2.line(canvas, (cx-5, cy+1), (cx, cy+7), (240, 240, 240), 2)
    cv2.line(canvas, (cx+5, cy+1), (cx, cy+7), (240, 240, 240), 2)

    cv2.putText(canvas, label, (x1+24, y2-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
