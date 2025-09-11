# utils/helpers.py
"""
Utility helpers.

Currently provides:
- letterbox: resize + pad image to a target rectangle while keeping aspect ratio.
  Returns (img, ratio, (dw, dh)) where:
    - ratio is the scaling factor applied to (h, w)
    - (dw, dh) are the left/top paddings applied after resize
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple

def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] | int = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scaleFill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
):
    """
    Resize and pad image to meet stride-multiple constraints.

    Args:
        img: BGR image (H, W, 3)
        new_shape: (h, w) target or single int
        color: padding color (B, G, R)
        auto: if True, minimize padding to be multiple of stride
        scaleFill: if True, stretch to fill new_shape (no aspect)
        scaleup: if False, only scale down (avoid upsampling)
        stride: pad to multiples of this

    Returns:
        img_out: letterboxed image
        r: scaling ratio (h_ratio, w_ratio) OR single float if uniform
        (dw, dh): left/top padding (pixels)
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better accuracy)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # (w, h) padding
    if auto:  # minimum rectangle; make padding multiples of stride
        dw %= stride
        dh %= stride

    if scaleFill:  # stretch
        new_unpad = (new_shape[1], new_shape[0])
        r = (new_shape[1] / shape[1], new_shape[0] / shape[0])
        dw, dh = 0, 0
    else:
        r = (r, r)

    # resize
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # pad (dw, dh) are for both sides; split evenly
    dw /= 2
    dh /= 2
    top, bottom = int(round(dh)), int(round(dh))
    left, right = int(round(dw)), int(round(dw))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, r, (left, top)
