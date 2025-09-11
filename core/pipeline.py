"""
Main people counting pipeline (preview + headless modes).

- run_people_counter: live preview/record with HUD & Export button
- run_people_counter_batch: headless analysis with progress callback
"""
from __future__ import annotations
import os
import json
import logging
import datetime
import time  # <-- for accurate pacing
from typing import Optional, Callable, Tuple

import cv2
import imutils
import numpy as np
from imutils.video import FPS

from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject

from core.direction import Flow
from core.drawing import (
    draw_border_and_labels, compose_with_bottom_bar, draw_hud_bar,
    BAR_H, compute_export_button_rect, draw_export_button
)
from core.io import CameraSource, SessionExporter, csv_append_moves, ask_save_path
from core.tracking import (
    create_csrt_tracker, track_rects_template, match_bboxes_to_objects,
    decide_full_cross_with_state, which_side
)
from utils.helpers import letterbox  # your existing helper

logger = logging.getLogger(__name__)
WINDOW_NAME = "People Counter"

# Defaults for CentroidTracker (can be injected from UI)
CT_MAX_DISAPPEARED = 40
CT_MAX_DISTANCE = 50


# ---------------- DNN helpers ----------------
def _set_dnn_backend(net, prefer_cuda: bool = True) -> str:
    """
    Try CUDA if truly available; otherwise fall back to CPU.
    """
    use_cuda = False
    if prefer_cuda:
        try:
            buildinfo = cv2.getBuildInformation()
            has_cuda_build = ("CUDA: YES" in buildinfo)
            has_cuda_device = cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, "cuda") else False
            use_cuda = has_cuda_build and has_cuda_device
        except Exception:
            use_cuda = False

    if use_cuda:
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            logger.info("[INFO] DNN backend: CUDA")
            return "cuda"
        except Exception:
            pass

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    logger.info("[INFO] DNN backend: CPU")
    return "cpu"


def _forward_with_blob(net, img_size: int, frame):
    """
    Build blob for a given img_size, forward once, return (preds, ratio, dw, dh).
    ratio is a tuple (ry, rx) from letterbox.
    """
    img, r, (dw, dh) = letterbox(frame, new_shape=(img_size, img_size))
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (img_size, img_size), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds, r, dw, dh


def _detect_yolov5_onnx(net, frame, img_size=640, conf_thres=0.28, nms_thres=0.45):
    """
    Run ONNX YOLOv5 (person-only) via OpenCV-DNN and return rects [(sx,sy,ex,ey), ...].

    Robustness: if the model errors on a custom img_size (Concat shape mismatch),
    retry once with 640 to avoid crashing.
    """
    H, W = frame.shape[:2]

    # 1st attempt: requested size
    try:
        preds, r, dw, dh = _forward_with_blob(net, img_size, frame)
    except cv2.error:
        # fallback to CPU in case backend mismatch
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        try:
            preds, r, dw, dh = _forward_with_blob(net, img_size, frame)
        except cv2.error:
            # final fallback: force 640 (known-safe for many exports)
            preds, r, dw, dh = _forward_with_blob(net, 640, frame)

    if preds.ndim == 3:
        preds = preds[0]
    elif preds.ndim == 4:
        preds = preds[0, 0]

    boxes, confidences = [], []
    for row in preds:
        obj = float(row[4])
        if obj < 1e-6:
            continue
        class_scores = row[5:]
        cid = int(np.argmax(class_scores))
        if cid != 0:  # person class only
            continue
        cls_score = float(class_scores[cid])
        conf = obj * cls_score
        if conf < conf_thres:
            continue

        cx, cy, w, h = row[0], row[1], row[2], row[3]
        x1 = float(cx - w / 2)
        y1 = float(cy - h / 2)
        x2 = float(cx + w / 2)
        y2 = float(cy + h / 2)

        # de-letterbox (r is a tuple)
        x1 = (x1 - dw) / r[0]
        y1 = (y1 - dh) / r[1]
        x2 = (x2 - dw) / r[0]
        y2 = (y2 - dh) / r[1]

        sx, sy = max(0, int(round(x1))), max(0, int(round(y1)))
        ex, ey = min(W - 1, int(round(x2))), min(H - 1, int(round(y2)))
        if ex <= sx or ey <= sy:
            continue

        boxes.append([sx, sy, ex - sx, ey - sy])  # x, y, w, h
        confidences.append(float(conf))

    rects = []
    if boxes:
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, nms_thres)
        idx_iter = (idxs.flatten() if hasattr(idxs, "flatten") else ([idxs] if idxs is not None else []))
        for i in idx_iter:
            x, y, w, h = boxes[i]
            rects.append((x, y, x + w, y + h))
    return rects


# ---------------- Preview mode ----------------
def run_people_counter(
    flow: Flow,
    source: str,
    input_path: Optional[str] = None,
    onnx_model: Optional[str] = None,
    confidence: float = 0.30,
    skip_frames: int = 12,
    nms_thres: float = 0.45,
    img_size: int = 640,
    ct_max_disappeared: int = CT_MAX_DISAPPEARED,
    ct_max_distance: int = CT_MAX_DISTANCE,
    # playback speed (0.1x..2.0x)
    playback_speed: float = 1.0,
):
    """
    Preview runner with HUD + Export button.

    `source`: "webcam" | "file"
      - for "webcam": URL or index is read from `utils/config.json["url"]`
      - for "file": supply `input_path`
    """
    # Read the camera/url from config if webcam
    with open("utils/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    if onnx_model is None or not os.path.isfile(onnx_model):
        raise FileNotFoundError(f"ONNX model not found: {onnx_model}")
    net = cv2.dnn.readNetFromONNX(onnx_model)
    _ = _set_dnn_backend(net, prefer_cuda=True)

    # Build the camera source
    if source == "webcam":
        cam_src = config.get("url", 0)
        src = CameraSource("webcam", url_or_index=cam_src).start()
    elif source == "file":
        if not input_path or not os.path.isfile(input_path):
            raise FileNotFoundError(f"Video file not found: {input_path}")
        src = CameraSource("file", input_path=input_path).start()
    else:
        raise ValueError("source must be 'webcam' or 'file'.")

    exporter = SessionExporter()
    exporter.start()

    cv2.namedWindow(WINDOW_NAME)
    export_requested = False
    mouse_pos = (0, 0)
    export_btn_rect = None

    def _on_mouse(event, x, y, flags, param):
        nonlocal export_requested, mouse_pos
        mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and export_btn_rect is not None:
            x1, y1, x2, y2 = export_btn_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                export_requested = True

    cv2.setMouseCallback(WINDOW_NAME, _on_mouse)

    W = H = None

    ct = CentroidTracker(maxDisappeared=ct_max_disappeared, maxDistance=ct_max_distance)
    opencv_trackers = []
    trackableObjects: dict[int, TrackableObject] = {}

    TRACKER_AVAILABLE = create_csrt_tracker() is not None

    prev_gray = None
    last_rects = []

    totalFrames = 0
    enter_count = 0
    exit_count = 0

    move_out, move_in = [], []
    out_time, in_time = [], []

    fps = FPS().start()

    # ---- Accurate pacing setup ----
    # allow up to 2.0x (speed up) and down to 0.1x (slow-mo)
    try:
        playback_speed = float(playback_speed)
    except Exception:
        playback_speed = 1.0
    playback_speed = max(0.1, min(2.0, playback_speed))

    fps_video = src.get_fps(default=30)  # target fps from source (fallback 30)
    target_period = 1.0 / max(1, fps_video)
    t_prev = time.perf_counter()

    while True:
        t_frame_start = time.perf_counter()

        frame = src.read()
        if frame is None:
            break

        frame = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        status = "Detecting" if (not TRACKER_AVAILABLE or totalFrames % int(skip_frames) == 0) else "Tracking"
        rects = []

        if status == "Detecting":
            opencv_trackers = []
            rects = _detect_yolov5_onnx(net, frame, img_size=img_size,
                                        conf_thres=float(confidence), nms_thres=float(nms_thres))
            last_rects = rects.copy()
            prev_gray = gray.copy()
            if TRACKER_AVAILABLE:
                for (sx, sy, ex, ey) in rects:
                    t = create_csrt_tracker()
                    if t is not None:
                        t.init(frame, (sx, sy, ex - sx, ey - sy))
                        opencv_trackers.append(t)
        else:
            if TRACKER_AVAILABLE and len(opencv_trackers) > 0:
                alive = []
                for t in opencv_trackers:
                    ok, bbox = t.update(frame)
                    if ok:
                        x, y, w, h = [int(v) for v in bbox]
                        sx, sy, ex, ey = x, y, x + w, y + h
                        sx, sy = max(0, sx), max(0, sy)
                        ex, ey = min(W - 1, ex), min(H - 1, ey)
                        if ex > sx and ey > sy:
                            rects.append((sx, sy, ex, ey))
                            alive.append(t)
                opencv_trackers = alive
                last_rects = rects.copy() if rects else last_rects
                prev_gray = gray.copy()
            else:
                rects = track_rects_template(prev_gray, gray, last_rects, search_radius=24)
                last_rects = rects.copy() if rects else last_rects
                prev_gray = gray.copy()

        # Draw midline & labels on video area
        draw_border_and_labels(frame, flow)

        # Update centroid tracker and bind ids to bboxes
        objects = ct.update(rects)
        id2bbox = match_bboxes_to_objects(objects, rects)

        # Per-object state & counting (full-cross rule)
        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)
            if to is None:
                to = TrackableObject(objectID, centroid)
                to.frames_seen = 1
                to.last_side = which_side(flow, centroid, W, H, use_bbox=False)
                to.color_state = "pre"
            else:
                to.centroids.append(centroid)
                to.frames_seen = getattr(to, "frames_seen", 0) + 1

                if not getattr(to, "counted", False) and objectID in id2bbox:
                    decision = decide_full_cross_with_state(flow, to, id2bbox[objectID], W, H)
                    if decision == "enter":
                        enter_count += 1
                        move_in.append(enter_count)
                        in_time.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
                        to.counted = True
                        to.color_state = "enter"
                    elif decision == "exit":
                        exit_count += 1
                        move_out.append(exit_count)
                        out_time.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
                        to.counted = True
                        to.color_state = "exit"
            trackableObjects[objectID] = to

        # Draw tracked boxes with stateful colors
        for (objectID, centroid) in objects.items():
            if objectID not in id2bbox:
                continue
            (sx, sy, ex, ey) = id2bbox[objectID]
            to = trackableObjects.get(objectID)
            color = (255, 255, 0)  # pre
            if to is not None:
                if getattr(to, "color_state", "pre") == "enter":
                    color = (0, 255, 0)
                elif getattr(to, "color_state", "pre") == "exit":
                    color = (0, 0, 255)
            cv2.rectangle(frame, (sx, sy), (ex, ey), color, 2)

        # Compose HUD bar and draw stats
        inside_count = max(0, enter_count - exit_count)
        canvas = compose_with_bottom_bar(frame.copy())
        draw_hud_bar(canvas, enter_count, exit_count, inside_count, status)

        # Export button on the HUD bar
        export_btn_rect = compute_export_button_rect(W, canvas.shape[0], label="Export")
        mx, my = mouse_pos
        x1, y1, x2, y2 = export_btn_rect
        hover = (x1 <= mx <= x2) and (y1 <= my <= y2)
        draw_export_button(canvas, export_btn_rect, hovered=hover, label="Export")

        # Append CSV log (optional)
        if move_in or move_out:
            csv_append_moves(move_in, in_time, move_out, out_time)
            move_in.clear()
            in_time.clear()
            move_out.clear()
            out_time.clear()

        # Show composed frame
        cv2.imshow(WINDOW_NAME, canvas)

        # ---- Accurate pacing (sleep to hit target fps, then minimal waitKey) ----
        desired = target_period / playback_speed
        elapsed = time.perf_counter() - t_frame_start
        sleep = desired - elapsed
        if sleep > 0:
            time.sleep(sleep)

        key = cv2.waitKey(1) & 0xFF  # minimal wait to pump events
        if key == ord('e'):
            export_requested = True
        if key == ord("q"):
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        # Export handling
        if export_requested:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = "webcam" if (source == "webcam") else "video"
            default_name = f"people_counter_{prefix}_{stamp}.xlsx"
            save_path = ask_save_path(default_name)
            if save_path:
                source_label = "Webcam" if (source == "webcam") else "Video"
                media_name = os.path.basename(input_path) if (source == "file" and input_path) else None
                exporter.export_to(save_path, enter_count, exit_count, source_label, media_name)
                if source == "webcam":
                    break
                else:
                    # Pause after export for video files (optional UX)
                    cv2.putText(canvas, "PAUSED (press any key to continue)", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 220), 2, cv2.LINE_AA)
                    cv2.imshow(WINDOW_NAME, canvas)
                    cv2.waitKey(0)
                    export_requested = False

        totalFrames += 1
        fps.update()

    fps.stop()
    logger.info(f"Elapsed time: {fps.elapsed():.2f}")
    logger.info(f"Approx. FPS: {fps.fps():.2f}")

    try:
        src.release()
    except Exception:
        pass
    cv2.destroyAllWindows()


# ---------------- Headless (Analyze Video) ----------------
def run_people_counter_batch(
    flow: Flow,
    input_path: str,
    onnx_model: str,
    confidence: float = 0.30,
    skip_frames: int = 8,
    nms_thres: float = 0.45,
    img_size: int = 640,
    progress_cb: Optional[Callable[[float, int, int, int, str], None]] = None,
    cancel_event=None,
    ct_max_disappeared: int = CT_MAX_DISAPPEARED,
    ct_max_distance: int = CT_MAX_DISTANCE,
) -> Tuple[int, int, SessionExporter]:
    """
    Headless processing for 'Analyze Video' (no preview window).
    Returns (enter_count, exit_count, exporter).
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)
    if onnx_model is None or not os.path.isfile(onnx_model):
        raise FileNotFoundError(onnx_model)

    net = cv2.dnn.readNetFromONNX(onnx_model)
    _ = _set_dnn_backend(net, prefer_cuda=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    ct = CentroidTracker(maxDisappeared=ct_max_disappeared, maxDistance=ct_max_distance)
    opencv_trackers = []
    TRACKER_AVAILABLE = create_csrt_tracker() is not None

    prev_gray = None
    last_rects = []

    W = H = None
    totalFrames = 0
    enter_count = 0
    exit_count = 0

    trackableObjects: dict[int, TrackableObject] = {}

    exporter = SessionExporter()
    exporter.start()

    while True:
        if cancel_event is not None and cancel_event.is_set():
            break
        ok, frame = cap.read()
        if not ok:
            break

        frame = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        status = "Detecting" if (not TRACKER_AVAILABLE or totalFrames % int(skip_frames) == 0) else "Tracking"
        rects = []

        if status == "Detecting":
            opencv_trackers = []
            # same robust detect with fallback as preview:
            rects = _detect_yolov5_onnx(net, frame, img_size=img_size,
                                        conf_thres=float(confidence), nms_thres=float(nms_thres))
            last_rects = rects.copy()
            prev_gray = gray.copy()
            if TRACKER_AVAILABLE:
                for (sx, sy, ex, ey) in rects:
                    t = create_csrt_tracker()
                    if t is not None:
                        t.init(frame, (sx, sy, ex - sx, ey - sy))
                        opencv_trackers.append(t)
        else:
            if TRACKER_AVAILABLE and len(opencv_trackers) > 0:
                alive = []
                for t in opencv_trackers:
                    ok, bbox = t.update(frame)
                    if ok:
                        x, y, w, h = [int(v) for v in bbox]
                        sx, sy, ex, ey = x, y, x + w, y + h
                        sx, sy = max(0, sx), max(0, sy)
                        ex, ey = min(W - 1, ex), min(H - 1, ey)
                        if ex > sx and ey > sy:
                            rects.append((sx, sy, ex, ey))
                            alive.append(t)
                opencv_trackers = alive
                last_rects = rects.copy() if rects else last_rects
                prev_gray = gray.copy()
            else:
                rects = track_rects_template(prev_gray, gray, last_rects, search_radius=24)
                last_rects = rects.copy() if rects else last_rects
                prev_gray = gray.copy()

        # Update centroid tracker and bind ids to bboxes
        objects = ct.update(rects)
        id2bbox = match_bboxes_to_objects(objects, rects)

        # Per-object state & counting (full-cross rule)
        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)
            if to is None:
                to = TrackableObject(objectID, centroid)
                to.frames_seen = 1
                to.last_side = which_side(flow, centroid, W, H, use_bbox=False)
            else:
                to.centroids.append(centroid)
                to.frames_seen = getattr(to, "frames_seen", 0) + 1
                if not getattr(to, "counted", False) and objectID in id2bbox:
                    decision = decide_full_cross_with_state(flow, to, id2bbox[objectID], W, H)
                    if decision == "enter":
                        enter_count += 1
                        to.counted = True
                    elif decision == "exit":
                        exit_count += 1
                        to.counted = True
            trackableObjects[objectID] = to

        totalFrames += 1
        if progress_cb is not None:
            pct = (totalFrames / total) * 100.0
            inside = max(0, enter_count - exit_count)
            progress_cb(min(100.0, pct), enter_count, exit_count, inside, status)

    cap.release()
    return enter_count, exit_count, exporter
