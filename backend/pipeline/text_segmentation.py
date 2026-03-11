"""Pixel-level text segmentation for manga pages.

Uses Manga-Text-Segmentation-2025 (Unet++ / EfficientNetV2) to produce
a binary mask of text regions. Used for mask-based inpainting and
deriving text boxes (Whalefishin-style).
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_smp = None
_albumentations = None


def _ensure_deps():
    global _smp, _albumentations
    if _smp is None:
        try:
            import segmentation_models_pytorch as smp
            _smp = smp
        except ImportError as e:
            raise ImportError(
                "segmentation_models_pytorch required. pip install segmentation-models-pytorch albumentations timm"
            ) from e
    if _albumentations is None:
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            _albumentations = (A, ToTensorV2)
        except ImportError as e:
            raise ImportError("albumentations required. pip install albumentations") from e
    return _smp, _albumentations


def _convert_batchnorm_to_groupnorm(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            num_groups = 8
            if num_channels < num_groups or num_channels % num_groups != 0:
                for i in range(min(num_channels, 8), 1, -1):
                    if num_channels % i == 0:
                        num_groups = i
                        break
                else:
                    num_groups = 1
            setattr(module, name, nn.GroupNorm(num_groups=num_groups, num_channels=num_channels))
        else:
            _convert_batchnorm_to_groupnorm(child)


_MODEL_CACHE = None
_MODEL_PATH: Optional[Path] = None


def _get_model_path() -> Path:
    global _MODEL_PATH
    if _MODEL_PATH is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="a-b-c-x-y-z/Manga-Text-Segmentation-2025",
            filename="model.pth",
            local_dir=None,
            local_dir_use_symlinks=True,
        )
        _MODEL_PATH = Path(path)
    return _MODEL_PATH


def load_segmentation_model(device: Optional[str] = None):
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    smp, (A, ToTensorV2) = _ensure_deps()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = _get_model_path()

    model = smp.UnetPlusPlus(
        encoder_name="tu-efficientnetv2_rw_m",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
        decoder_attention_type="scse",
    )
    _convert_batchnorm_to_groupnorm(model.decoder)
    state_dict = torch.load(model_path, map_location=dev)
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()

    _MODEL_CACHE = (model, dev)
    logger.info("Loaded Manga-Text-Segmentation model on %s", dev)
    return _MODEL_CACHE


def segment_page(
    image: np.ndarray,
    threshold: float = 0.5,
    dilation_radius: int = 2,
) -> np.ndarray:
    """Produce binary mask of text regions (255 = text).

    Args:
        image: BGR image (H, W, 3).
        threshold: Probability threshold (0–1).
        dilation_radius: Pixels to dilate mask.

    Returns:
        Binary mask (H, W) uint8, 255 where text detected.
    """
    model, device = load_segmentation_model()
    A, ToTensorV2 = _albumentations

    h, w = image.shape[:2]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32

    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    augmented = transform(image=image)
    tensor = augmented["image"].unsqueeze(0).to(device)

    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)

    with torch.no_grad():
        if device == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(tensor)
                probs = logits.sigmoid()
        else:
            logits = model(tensor)
            probs = logits.sigmoid()

    prob_map = probs[0, 0, :h, :w].cpu().numpy()
    binary_mask = (prob_map > threshold).astype(np.uint8) * 255

    if dilation_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_radius * 2 + 1, dilation_radius * 2 + 1),
        )
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    return binary_mask


def mask_to_bubbles(
    text_mask: np.ndarray,
    contour_size_ratio: float = 0.02,
    min_area_ratio: float = 0.0005,
) -> List[dict]:
    """Derive text box bubbles from segmentation mask (Whalefishin-style).

    Uses Sobel + morphology + contours to find text blocks.

    Args:
        text_mask: Binary mask (255 = text).
        contour_size_ratio: Element size as fraction of image height (default 0.02).
        min_area_ratio: Min contour area as fraction of image area.

    Returns:
        List of bubble dicts with bbox, center, area, contour.
    """
    if len(text_mask.shape) == 3:
        img = cv2.cvtColor(text_mask, cv2.COLOR_BGR2GRAY)
    else:
        img = text_mask

    h, w = img.shape[:2]
    img_area = h * w
    ele_size = max(3, int(h * contour_size_ratio))
    noise_param = max(2, ele_size // 3)

    img_sobel = cv2.Sobel(img, cv2.CV_8U, 1, 0)
    _, img_thresh = cv2.threshold(img_sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (ele_size, ele_size))
    img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, element)

    contours, _ = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_points = noise_param ** 2
    contours = [c for c in contours if c.shape[0] > min_points]

    bubbles: List[dict] = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < img_area * min_area_ratio:
            continue

        x1 = max(x - noise_param, 0)
        y1 = max(y - noise_param, 0)
        x2 = min(x + cw + noise_param, w)
        y2 = min(y + ch + noise_param, h)
        bbox = [x1, y1, x2 - x1, y2 - y1]
        cx, cy = x1 + bbox[2] // 2, y1 + bbox[3] // 2

        contour_rect = np.array([
            [[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]],
        ], dtype=np.int32)

        bubbles.append({
            "bbox": bbox,
            "confidence": 0.9,
            "center": (cx, cy),
            "area": float(bbox[2] * bbox[3]),
            "contour": contour_rect,
            "from_mask": True,
        })

    bubbles.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    logger.info("mask_to_bubbles: %d box(es) from segmentation", len(bubbles))
    return bubbles


def segment_page_safe(image: np.ndarray, **kwargs) -> Optional[np.ndarray]:
    """Segment page, return None on error."""
    try:
        return segment_page(image, **kwargs)
    except Exception as e:
        logger.warning("Text segmentation failed: %s", e)
        return None
