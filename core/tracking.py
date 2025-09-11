"""
Tracking helpers:
- OpenCV CSRT tracker factory
- Lightweight template-based box tracking (fallback)
- Match bboxes to tracked object IDs
- Side and crossing decision with robust "armed-after-start" logic

New behavior:
- Objects are not counted until they first establish a clear starting side.
  If an object appears straddling the midline at start, it must first move to
  one side (become 'armed') and only then a full crossing will be counted.
"""
from __future__ import annotations
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np

from core.direction import Flow
from core import config as CFG

# -----------------------------------------------------------------------------
# OpenCV Tracker (CSRT) factory
# -----------------------------------------------------------------------------
def create_csrt_tracker():
    """Create an OpenCV CSRT tracker if available, else return None."""
    try:
        if hasattr(cv2, "legacy"):
            return cv2.legacy.TrackerCSRT_create()
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
    except Exception:
        return None
    return None

# --- Runtime-tunable margin (for midline tolerance) ---
_RUNTIME_MARGIN_PX = None  # type: Optional[int]

def set_margin_px(v):
    """Set a runtime override for margin_px; pass None to clear."""
    global _RUNTIME_MARGIN_PX
    try:
        _RUNTIME_MARGIN_PX = None if v is None else int(v)
    except Exception:
        _RUNTIME_MARGIN_PX = None

# -----------------------------------------------------------------------------
# Simple template-based propagation (fallback when OpenCV trackers not used)
# -----------------------------------------------------------------------------
def _crop_safe(img, x, y, w, h):
    H, W = img.shape[:2]
    x1 = max(0, int(x)); y1 = max(0, int(y))
    x2 = min(W, int(x + w)); y2 = min(H, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1, x1, y1)
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def track_rects_template(prev_gray, gray, last_rects, search_radius: int = 24):
    """
    Track rectangles between two grayscale frames using template matching.
    Returns propagated rects in (sx, sy, ex, ey) format.
    """
    if prev_gray is None or gray is None or not last_rects:
        return []

    out = []
    for (sx, sy, ex, ey) in last_rects:
        w = ex - sx; h = ey - sy
        tmpl, _ = _crop_safe(prev_gray, sx, sy, w, h)
        if tmpl is None or tmpl.size == 0:
            continue

        cx = (sx + ex) // 2
        cy = (sy + ey) // 2
        sr = int(search_radius)

        cand_x1 = max(0, cx - sr); cand_y1 = max(0, cy - sr)
        cand_x2 = min(gray.shape[1], cx + sr); cand_y2 = min(gray.shape[0], cy + sr)
        cand = gray[cand_y1:cand_y2, cand_x1:cand_x2]
        if cand.size == 0:
            continue

        try:
            res = cv2.matchTemplate(cand, tmpl, cv2.TM_CCOEFF_NORMED)
            _minVal, maxVal, _minLoc, maxLoc = cv2.minMaxLoc(res)
        except Exception:
            continue

        if maxVal < 0.25:
            # Too weak, drop
            continue

        nx = cand_x1 + maxLoc[0]
        ny = cand_y1 + maxLoc[1]
        nsx = nx
        nsy = ny
        nex = nx + w
        ney = ny + h

        nsx = max(0, nsx); nsy = max(0, nsy)
        nex = min(gray.shape[1] - 1, nex); ney = min(gray.shape[0] - 1, ney)
        if nex > nsx and ney > nsy:
            out.append((nsx, nsy, nex, ney))
    return out


# -----------------------------------------------------------------------------
# Utilities: bind object IDs to bboxes
# -----------------------------------------------------------------------------
def match_bboxes_to_objects(objects: Dict[int, np.ndarray],
                            rects: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int, int, int]]:
    """
    Try to assign each tracked objectID the closest bbox by centroid distance.
    """
    out: Dict[int, Tuple[int, int, int, int]] = {}
    if not objects or not rects:
        return out

    rect_centroids = []
    for (sx, sy, ex, ey) in rects:
        rect_centroids.append(((sx + ex) // 2, (sy + ey) // 2))

    for oid, c in objects.items():
        cx, cy = int(c[0]), int(c[1])
        best_i = -1
        best_d = 1e9
        for i, rc in enumerate(rect_centroids):
            dx = rc[0] - cx; dy = rc[1] - cy
            d = dx * dx + dy * dy
            if d < best_d:
                best_d = d; best_i = i
        if best_i >= 0:
            out[oid] = rects[best_i]
    return out


# -----------------------------------------------------------------------------
# Side-of-line and crossing decision
# -----------------------------------------------------------------------------
def _midline(flow: Flow, W: int, H: int) -> Tuple[str, int]:
    """
    Return midline type ('v' vertical | 'h' horizontal) and its coordinate.
    """
    if flow in (Flow.LEFT_TO_RIGHT, Flow.RIGHT_TO_LEFT):
        return 'v', W // 2
    return 'h', H // 2


def _sides_for_flow(flow: Flow) -> Tuple[str, str]:
    """
    Return canonical names for the two sides depending on flow.
    """
    if flow == Flow.LEFT_TO_RIGHT:
        return "left", "right"
    if flow == Flow.RIGHT_TO_LEFT:
        return "right", "left"
    if flow == Flow.UP_TO_DOWN:
        return "top", "bottom"
    return "bottom", "top"  # DOWN_TO_UP


def which_side(flow: Flow, point_xy: Tuple[int, int], W: int, H: int,
               use_bbox: bool = False, bbox: Optional[Tuple[int, int, int, int]] = None,
               margin_px: Optional[int] = None) -> str:
    """
    Decide which side ('left'/'right' or 'top'/'bottom') the point/bbox is on.
    Returns 'on_line' if within margin around midline.

    If `use_bbox=True`, bbox center is used.
    """
    if margin_px is None:
        if _RUNTIME_MARGIN_PX is not None:
            margin_px = _RUNTIME_MARGIN_PX
        else:
            s = CFG.load_settings()
            margin_px = int(s.get("margin_px", 5))

    if use_bbox and bbox is not None:
        sx, sy, ex, ey = bbox
        cx = (sx + ex) // 2; cy = (sy + ey) // 2
    else:
        cx, cy = point_xy

    t, m = _midline(flow, W, H)
    if t == 'v':
        if abs(cx - m) <= margin_px:
            return "on_line"
        return "left" if cx < m else "right"
    else:
        if abs(cy - m) <= margin_px:
            return "on_line"
        return "top" if cy < m else "bottom"


def _direction_of_change(flow: Flow, from_side: str, to_side: str) -> Optional[str]:
    """
    Map a side-change to 'enter' or 'exit' according to flow.
    Returns None if no valid mapping.
    """
    s_from, s_to = _sides_for_flow(flow)
    if from_side == s_from and to_side == s_to:
        return "enter"   # moving with the flow
    if from_side == s_to and to_side == s_from:
        return "exit"    # moving opposite to the flow
    return None


def decide_full_cross_with_state(flow: Flow,
                                 to,  # TrackableObject with mutable attrs
                                 bbox: Tuple[int, int, int, int],
                                 W: int, H: int,
                                 min_frames_seen: int = 3,
                                 margin_px: Optional[int] = None) -> Optional[str]:
    """
    Decide whether a tracked object has performed a full cross AFTER the run started.

    Rules:
    - At first sightings, object must establish a clear starting side (not 'on_line').
      Until then, it is 'unarmed' and will NOT be counted.
    - Once 'armed', if it changes side (ignoring 'on_line' states) and its center
      is beyond the margin on the new side, we emit 'enter' or 'exit' once.
    - We also require the object to be seen a minimum number of frames
      (min_frames_seen) to reduce flicker at spawn time.
    """
    if margin_px is None:
        s = CFG.load_settings()
        margin_px = int(s.get("margin_px", 5))

    # Ensure state fields exist
    if not hasattr(to, "frames_seen"):
        to.frames_seen = 0
    if not hasattr(to, "armed"):
        to.armed = False
    if not hasattr(to, "last_side"):
        # Establish initial side from centroid position (or bbox center)
        # NOTE: if starts 'on_line', we keep it and won't arm yet.
        centroid = to.centroids[-1] if getattr(to, "centroids", None) else None
        if centroid is not None:
            to.last_side = which_side(flow, centroid, W, H, use_bbox=False, bbox=None, margin_px=margin_px)
        else:
            # fallback to bbox center
            to.last_side = which_side(flow, (0, 0), W, H, use_bbox=True, bbox=bbox, margin_px=margin_px)

    # Update current side
    centroid = to.centroids[-1] if getattr(to, "centroids", None) else None
    cur_side = which_side(flow, centroid if centroid is not None else (0, 0),
                          W, H, use_bbox=(centroid is None), bbox=bbox, margin_px=margin_px)

    # Wait for minimum frames to stabilize
    if to.frames_seen < min_frames_seen:
        # Try to arm if we already have a clear side
        if not to.armed and cur_side != "on_line":
            to.armed = True
            to.last_side = cur_side
        return None

    # If not armed yet, arm only when clearly off the line
    if not to.armed:
        if cur_side != "on_line":
            to.armed = True
            to.last_side = cur_side
        return None

    # If armed: count only when we have a decisive side change (ignore 'on_line')
    if cur_side == "on_line":
        return None

    if cur_side != to.last_side:
        # Validate we crossed to the other side (beyond margin)
        # and map to enter/exit based on flow.
        decision = _direction_of_change(flow, to.last_side, cur_side)
        to.last_side = cur_side
        return decision

    return None
