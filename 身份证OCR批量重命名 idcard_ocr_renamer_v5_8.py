# -*- coding: utf-8 -*-
"""
身份证批量识别重命名 V5.7 多线程加速版
- 多候选区域搜索 + 正面/背面评分 + 证据分级
- 批量处理使用线程池，每线程独立 OCR 引擎
- 显著提升批量处理速度
"""

import os
import sys
import re
import json
import math
import shutil
import threading
import logging
import queue
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk


# ============================================================
# 基础配置
# ============================================================

APP_TITLE = "身份证 OCR 批量重命名 V5.7（多线程版）"

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}

CONFIG_FILE = Path(__file__).resolve().parent / "idcard_config.json"

CARD_RATIO = 85.60 / 54.00
CARD_W = 1284
CARD_H = 810
MIN_CARD_AREA_RATIO = 0.035

NAME_MIN_LEN = 2
NAME_MAX_LEN = 4
DEBUG_SAVE = False

# 最大并行线程数（可手动调整，建议等于 CPU 核心数）
MAX_WORKERS = min(os.cpu_count() or 2, 4)


# ============================================================
# 默认干扰字（已移除"国"等常见姓名用字）
# ============================================================

DEFAULT_DIRTY_PREFIX = [
    "名", "姓", "多", "莛", "芾", "等", "荑",
    "矬", "奢", "娌", "琏", "妊", "各", "饧"
]

DEFAULT_DIRTY_SUFFIX = [
    "骥", "牲", "陛", "樊", "杰", "斌", "鑫", "亮",
    "怯", "牲别", "性別", "别", "性", "男", "女",
    "劓", "艮", "汊", "仅"
]


def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return (
                data.get("dirty_prefix", DEFAULT_DIRTY_PREFIX.copy()),
                data.get("dirty_suffix", DEFAULT_DIRTY_SUFFIX.copy()),
            )
        except Exception:
            pass
    return DEFAULT_DIRTY_PREFIX.copy(), DEFAULT_DIRTY_SUFFIX.copy()


def save_config(prefix_list, suffix_list):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dirty_prefix": prefix_list,
                    "dirty_suffix": suffix_list,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return True
    except Exception as e:
        logging.error("保存配置失败：%s", e)
        return False


DIRTY_PREFIX, DIRTY_SUFFIX = load_config()


# ============================================================
# tkinterdnd2
# ============================================================

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except Exception:
    HAS_DND = False
    DND_FILES = None
    TkinterDnD = None


# ============================================================
# RapidOCR（全局引擎 + 线程局部缓存）
# ============================================================

try:
    from rapidocr import RapidOCR
    RAPIDOCR_IMPORT_ERROR = None
except Exception as e:
    RapidOCR = None
    RAPIDOCR_IMPORT_ERROR = e

_thread_local = threading.local()


def get_ocr_engine():
    """为每个线程独立创建 OCR 引擎，避免竞争"""
    if not hasattr(_thread_local, "ocr_engine"):
        if RapidOCR is None:
            raise RuntimeError(
                "RapidOCR 导入失败：%s\n\n请执行：\npip install rapidocr onnxruntime" % RAPIDOCR_IMPORT_ERROR
            )
        _thread_local.ocr_engine = RapidOCR()
    return _thread_local.ocr_engine


def init_ocr():
    """主线程初始化（仅用于预加载）"""
    try:
        get_ocr_engine()
        return True
    except Exception as e:
        logging.error("OCR 初始化失败：%s", e)
        return False


# ============================================================
# 工具函数
# ============================================================

def imread_chinese(path):
    try:
        with open(str(path), "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:
        logging.error("读取图片失败：%s -> %s", path, e)
        return None


def imwrite_chinese(path, image):
    try:
        ext = Path(path).suffix or ".jpg"
        ok, buf = cv2.imencode(ext, image)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception as e:
        logging.error("保存图片失败：%s -> %s", path, e)
        return False


def order_quad_points(points):
    pts = np.asarray(points, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def polygon_area(points):
    return abs(cv2.contourArea(np.asarray(points, dtype=np.float32)))


def quad_size(points):
    pts = order_quad_points(points)
    tl, tr, br, bl = pts
    w1 = np.linalg.norm(tr - tl)
    w2 = np.linalg.norm(br - bl)
    h1 = np.linalg.norm(bl - tl)
    h2 = np.linalg.norm(br - tr)
    return max(w1, w2), max(h1, h2)


def quad_ratio(points):
    w, h = quad_size(points)
    if h <= 1:
        return 0
    return w / h


def rotate_if_portrait(img):
    h, w = img.shape[:2]
    if h > w * 1.15:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def _resize_card_for_probe(card):
    return cv2.resize(card, (900, 568), interpolation=cv2.INTER_AREA)


# ============================================================
# 多候选区域检测（核心改动）
# ============================================================

def detect_rectangles_from_edges(gray, min_area_ratio=0.012, max_area_ratio=0.90):
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < h * w * min_area_ratio or area > h * w * max_area_ratio:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        ratio = cw / max(ch, 1)
        if 1.20 <= ratio <= 2.05:
            rects.append((x, y, cw, ch))
    return rects


def detect_rectangles_from_threshold(gray, threshold=200, min_area_ratio=0.012, max_area_ratio=0.90):
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < h * w * min_area_ratio or area > h * w * max_area_ratio:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        ratio = cw / max(ch, 1)
        if 1.20 <= ratio <= 2.05:
            rects.append((x, y, cw, ch))
    return rects


def detect_rectangles_from_nonwhite(gray, max_white=248, min_area_ratio=0.012):
    mask = cv2.inRange(gray, 0, max_white)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < h * w * min_area_ratio:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        ratio = cw / max(ch, 1)
        if 1.20 <= ratio <= 2.05:
            rects.append((x, y, cw, ch))
    return rects


def detect_rectangles_partition(gray, axis='horizontal', min_area_ratio=0.012):
    h, w = gray.shape
    rects = []
    if axis == 'horizontal':
        parts = 2
        step = h // parts
        for i in range(parts):
            y1 = i * step
            y2 = min(h, (i+1)*step)
            sub = gray[y1:y2, :]
            sub_rects = detect_rectangles_from_threshold(
                sub, threshold=200, min_area_ratio=min_area_ratio, max_area_ratio=0.95
            )
            for x, y, cw, ch in sub_rects:
                rects.append((x, y1+y, cw, ch))
    else:  # vertical
        parts = 2
        step = w // parts
        for i in range(parts):
            x1 = i * step
            x2 = min(w, (i+1)*step)
            sub = gray[:, x1:x2]
            sub_rects = detect_rectangles_from_threshold(
                sub, threshold=200, min_area_ratio=min_area_ratio, max_area_ratio=0.95
            )
            for x, y, cw, ch in sub_rects:
                rects.append((x1+x, y, cw, ch))
    return rects


def gather_all_candidates(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    candidates = []

    # 1. 四边形检测（原有）
    quads = find_card_quads(img)
    for _, quad in quads:
        x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
        candidates.append((x, y, w, h))

    # 2. 边缘矩形
    rects = detect_rectangles_from_edges(gray_blur)
    candidates.extend(rects)

    # 3. 亮区域矩形
    rects = detect_rectangles_from_threshold(gray_blur, threshold=200)
    candidates.extend(rects)

    # 4. 非白区域
    rects = detect_rectangles_from_nonwhite(gray_blur)
    candidates.extend(rects)

    # 5. 水平分区
    rects = detect_rectangles_partition(gray_blur, axis='horizontal')
    candidates.extend(rects)

    # 6. 垂直分区
    rects = detect_rectangles_partition(gray_blur, axis='vertical')
    candidates.extend(rects)

    # 去重：合并重叠过大的框
    unique = []
    for x, y, w, h in candidates:
        if w <= 0 or h <= 0:
            continue
        keep = True
        for ux, uy, uw, uh in unique:
            xa = max(x, ux)
            ya = max(y, uy)
            xb = min(x+w, ux+uw)
            yb = min(y+h, uy+uh)
            inter = max(0, xb-xa) * max(0, yb-ya)
            union = w*h + uw*uh - inter
            if union > 0 and inter / union > 0.65:
                keep = False
                break
        if keep:
            unique.append((x, y, w, h))

    return unique


# ============================================================
# 正面/背面评分系统
# ============================================================

def run_ocr(image):
    engine = get_ocr_engine()
    if engine is None:
        return []
    try:
        result = engine(image, use_det=True, use_cls=True, use_rec=True)
        return parse_ocr_result(result)
    except TypeError:
        try:
            result = engine(image)
            return parse_ocr_result(result)
        except Exception as e:
            logging.error("OCR失败：%s", e)
            return []
    except Exception as e:
        logging.error("OCR失败：%s", e)
        return []


def parse_ocr_result(result):
    output = []
    if result is None:
        return output
    try:
        txts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        boxes = getattr(result, "boxes", None)
        if txts is not None:
            for i, text in enumerate(txts):
                score = 0.0
                if scores is not None and i < len(scores):
                    try:
                        score = float(scores[i])
                    except Exception:
                        score = 0.0
                box = boxes[i] if boxes is not None and i < len(boxes) else None
                output.append({"text": str(text), "score": score, "box": box})
            return output
    except Exception:
        pass
    try:
        data = result
        if isinstance(data, tuple):
            data = data[0]
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
            data = data[0]
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                box = item[0]
                text = ""
                score = 0.0
                if len(item) >= 2 and isinstance(item[1], (list, tuple)) and len(item[1]) >= 1:
                    text = item[1][0]
                    if len(item[1]) >= 2:
                        try:
                            score = float(item[1][1])
                        except Exception:
                            score = 0.0
                else:
                    text = item[1]
                    if len(item) >= 3:
                        try:
                            score = float(item[2])
                        except Exception:
                            score = 0.0
                output.append({"text": str(text), "score": score, "box": box})
    except Exception as e:
        logging.error("解析 RapidOCR 返回值失败：%s", e)
    return output


def score_card(card, probe_results=None):
    if card is None:
        return -1e9, -1e9, False
    if probe_results is None:
        probe_results = run_ocr(_resize_card_for_probe(card))
    texts = [normalize_text(x.get('text', '')) for x in probe_results if normalize_text(x.get('text', ''))]
    joined = ''.join(texts)

    front_score = 0.0
    back_score = 0.0

    front_terms = {
        '姓名': 55, '性别': 18, '民族': 18, '出生': 18,
        '住址': 18, '公民身份号码': 30, '公民': 10, '身份': 8,
    }
    for term, pts in front_terms.items():
        if term in joined:
            front_score += pts
    if '姓' in joined and '名' in joined:
        front_score += 22
    if re.search(r'\d{17}[0-9Xx]', joined):
        front_score += 25

    back_terms = {'签发机关': 45, '有效期限': 35}
    for term, pts in back_terms.items():
        if term in joined:
            back_score += pts

    if front_score > 0 and back_score > 0:
        front_score += 30
        back_score -= 20

    if '居民身份证' in joined and '姓名' not in joined:
        back_score += 30

    has_name_label = '姓名' in joined or ('姓' in joined and '名' in joined)

    return front_score, back_score, has_name_label


# ============================================================
# 正方形检测（四边形路线，保留）
# ============================================================

def quad_angle_score(points):
    pts = order_quad_points(points)
    scores = []
    for i in range(4):
        p0 = pts[i - 1]
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        a = p0 - p1
        b = p2 - p1
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 2 or nb < 2:
            continue
        c = np.clip(float(np.dot(a, b) / (na * nb)), -1, 1)
        angle = math.degrees(math.acos(c))
        scores.append(max(0.0, 1.0 - abs(angle - 90.0) / 55.0))
    return float(np.mean(scores)) if scores else 0.0


def candidate_score(points, image_shape):
    h, w = image_shape[:2]
    area = polygon_area(points)
    if area <= 0:
        return -1e9
    ratio = quad_ratio(points)
    if ratio <= 0:
        return -1e9
    area_ratio = area / float(w * h)
    if area_ratio < 0.012:
        return -1e9
    ratio_error = abs(math.log(max(ratio, 0.01) / CARD_RATIO))
    ratio_score = max(0.0, 1.0 - ratio_error * 1.75)
    angle_score = quad_angle_score(points)
    area_score = min(area_ratio / 0.20, 1.0)
    return ratio_score * 0.55 + angle_score * 0.30 + area_score * 0.15


def _add_quad_candidate(candidates, pts, score, image_shape):
    if pts is None or len(pts) != 4:
        return
    pts = order_quad_points(np.asarray(pts, dtype=np.float32))
    area = polygon_area(pts)
    h, w = image_shape[:2]
    if area < h * w * 0.012:
        return
    ratio = quad_ratio(pts)
    if ratio < 1.20 or ratio > 2.05:
        return
    candidates.append((float(score), pts))


def find_card_quads(img, max_candidates=12):
    if img is None:
        return []
    oh, ow = img.shape[:2]
    scale = min(1.0, 1800.0 / max(oh, ow))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    ih, iw = gray.shape[:2]
    img_area = ih * iw
    candidates = []

    for low, high, close_size, close_iter in ((35, 120, 5, 2), (50, 150, 7, 2), (80, 180, 5, 1)):
        edges = cv2.Canny(gray, low, high)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8), iterations=close_iter)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < img_area * 0.012:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in (0.018, 0.025, 0.035):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    pts = approx.reshape(4, 2)
                    _add_quad_candidate(candidates, pts, candidate_score(pts, gray.shape), gray.shape)

    for threshold in (150, 180, 205, 225, 240):
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < img_area * 0.012 or area > img_area * 0.90:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                _add_quad_candidate(candidates, pts, candidate_score(pts, gray.shape) + 0.015, gray.shape)

    unique = []
    for score, pts in sorted(candidates, key=lambda x: x[0], reverse=True):
        x1, y1, x2, y2 = cv2.boundingRect(pts.astype(np.int32))
        keep = True
        for _, old in unique:
            ox1, oy1, ox2, oy2 = cv2.boundingRect(old.astype(np.int32))
            xa, ya = max(x1, ox1), max(y1, oy1)
            xb, yb = min(x1 + x2, ox1 + ox2), min(y1 + y2, oy1 + oy2)
            inter = max(0, xb - xa) * max(0, yb - ya)
            union = x2 * y2 + ox2 * oy2 - inter
            if union > 0 and inter / union > 0.65:
                keep = False
                break
        if keep:
            unique.append((score, pts))
        if len(unique) >= max_candidates:
            break

    result = []
    for score, pts in unique:
        if scale != 1:
            pts = pts / scale
        result.append((score, order_quad_points(pts)))
    return result


def perspective_warp(img, quad):
    if quad is None:
        return None
    src = order_quad_points(quad)
    w, h = quad_size(src)
    if w < h:
        src = np.array([src[1], src[2], src[3], src[0]], dtype=np.float32)
        w, h = h, w
    dst = np.array([[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (CARD_W, CARD_H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def crop_rect(img, rect):
    x, y, w, h = rect
    return img[y:y+h, x:x+w]


def normalize_card(img, log_callback=None):
    rects = gather_all_candidates(img)
    if rects:
        scored_cards = []
        for idx, (x, y, w, h) in enumerate(rects):
            crop = crop_rect(img, (x, y, w, h))
            if crop is None or crop.size == 0:
                continue
            card = cv2.resize(crop, (CARD_W, CARD_H), interpolation=cv2.INTER_CUBIC)
            front, back, has_name = score_card(card)
            if front > back + 30:
                scored_cards.append((front, card))
        if scored_cards:
            scored_cards.sort(key=lambda x: x[0], reverse=True)
            best_card = scored_cards[0][1]
            if log_callback:
                log_callback(f"从 {len(scored_cards)} 个正面候选中选择最高分 {scored_cards[0][0]:.1f}")
            return best_card, "多候选区域+正面评分筛选"

    quads = find_card_quads(img)
    if quads:
        for _, quad in quads:
            card = perspective_warp(img, quad)
            if card is None:
                continue
            front, back, has_name = score_card(card)
            if front > back + 30:
                return card, "四边形检测+正面评分"
        if quads:
            card = perspective_warp(img, quads[0][1])
            if card is not None:
                return card, "四边形检测（无正面评分，采用第一个）"

    crop = crop_nonwhite_region(img)
    if crop is not None:
        crop = rotate_if_portrait(crop)
        return crop, "非白区域裁剪（fallback）"
    return rotate_if_portrait(img), "原图fallback"


def crop_nonwhite_region(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 0, 245)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    h, w = gray.shape[:2]
    possible = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < h * w * 0.008:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        ratio = cw / max(ch, 1)
        if 1.20 <= ratio <= 2.10:
            possible.append((area, x, y, cw, ch))
    if not possible:
        return None
    _, x, y, cw, ch = max(possible, key=lambda z: z[0])
    px, py = max(5, int(cw * .02)), max(5, int(ch * .02))
    return img[max(0, y-py):min(h, y+ch+py), max(0, x-px):min(w, x+cw+px)]


# ============================================================
# 图像预处理
# ============================================================

def upscale(img, scale=2.0):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def build_ocr_variants(img):
    variants = []
    base = upscale(img, 2.5)
    variants.append(("原始放大", base))
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    variants.append(("灰度CLAHE", enhanced))
    denoise = cv2.GaussianBlur(enhanced, (3, 3), 0)
    variants.append(("去噪", denoise))
    blur = cv2.GaussianBlur(enhanced, (0, 0), 2)
    sharpen = cv2.addWeighted(enhanced, 1.7, blur, -0.7, 0)
    variants.append(("锐化", sharpen))
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("OTSU", otsu))
    adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    variants.append(("自适应阈值", adaptive))
    return variants


# ============================================================
# 文本处理 / 证据分级
# ============================================================

FIXED_IDCARD_WORDS = {
    '姓名', '性别', '民族', '出生', '住址', '公民', '身份', '号码',
    '身份证', '居民身份证', '中华人民共和国', '签发机关', '有效期限',
    '长期', '中华人民共和国居民身份证', '公民身份号码',
}

FIXED_IDCARD_CHUNKS = {
    '中华', '华人', '人民', '民共', '共和国', '和国', '居民', '身份',
    '证件', '签发', '机关', '有效', '期限', '公民', '号码',
}


def normalize_text(text):
    if text is None:
        return ''
    return str(text).replace(' ', '').replace('\n', '').replace('\r', '').replace('\t', '')


def chinese_only(text):
    return re.sub(r'[^\u4e00-\u9fa5]', '', text or '')


def is_invalid_name(name):
    if not name:
        return True
    invalid_patterns = [
        '中华人民', '共和国', '居民身份', '签发机关', '有效期限',
        '中华', '人民', '居民', '身份', '证件', '机关', '期限',
        '公民身份', '号码'
    ]
    for pat in invalid_patterns:
        if pat in name:
            return True
    if name in FIXED_IDCARD_WORDS or name in FIXED_IDCARD_CHUNKS:
        return True
    return False


def clean_name(name):
    if not name:
        return None, False
    name = normalize_text(name)
    for label in ('姓名', '妊名', '性别', '牲别', '性別'):
        name = name.replace(label, '')
    name = chinese_only(name)
    if not (NAME_MIN_LEN <= len(name) <= NAME_MAX_LEN):
        return None, False
    if is_invalid_name(name):
        return None, False
    # 干扰字清洗（首部/尾部）
    changed = True
    while changed and len(name) > NAME_MIN_LEN:
        changed = False
        for dirty in sorted(DIRTY_PREFIX, key=len, reverse=True):
            if len(name) > NAME_MIN_LEN and dirty and name.startswith(dirty):
                test = name[len(dirty):]
                if len(test) >= NAME_MIN_LEN:
                    name = test
                    changed = True
                    break
    changed = True
    while changed and len(name) > NAME_MIN_LEN:
        changed = False
        for dirty in sorted(DIRTY_SUFFIX, key=len, reverse=True):
            if len(name) > NAME_MIN_LEN and dirty and name.endswith(dirty):
                test = name[:-len(dirty)]
                if len(test) >= NAME_MIN_LEN:
                    name = test
                    changed = True
                    break
    if not (NAME_MIN_LEN <= len(name) <= NAME_MAX_LEN):
        return None, False
    if is_invalid_name(name):
        return None, False
    return name, True


def _is_name_label(text):
    text = normalize_text(text)
    if text == '姓名' or text == '妊名':
        return True
    chars = list(text)
    if len(chars) >= 2 and '姓' in chars and '名' in chars:
        idxs = [i for i, c in enumerate(chars) if c in ('姓','名')]
        if len(idxs) >= 2 and chars.index('姓') < chars.index('名'):
            return True
    return False


def _name_from_text(text):
    text = normalize_text(text)
    if not text:
        return []
    out = []
    patterns = [
        r'(?:姓名|妊名)[:：]?([\u4e00-\u9fa5]{2,6})',
    ]
    for p in patterns:
        for m in re.finditer(p, text):
            raw = m.group(1)
            for stop in ('性别', '牲别', '性別', '民族', '出生', '住址', '公民', '身份', '号码'):
                if stop in raw:
                    raw = raw.split(stop, 1)[0]
            name, ok = clean_name(raw)
            if ok:
                out.append(name)
    return out


def _box_info(item):
    box = item.get('box')
    if box is None:
        return None
    try:
        arr = np.asarray(box, dtype=np.float32).reshape(-1, 2)
        if arr.shape[0] < 4:
            return None
        x1, y1 = np.min(arr[:,0]), np.min(arr[:,1])
        x2, y2 = np.max(arr[:,0]), np.max(arr[:,1])
        return float(x1), float(y1), float(x2), float(y2)
    except Exception:
        return None


def locate_name_from_boxes(results):
    items = []
    for item in results:
        text = normalize_text(item.get('text', ''))
        box = _box_info(item)
        if not text or box is None:
            continue
        x1, y1, x2, y2 = box
        items.append((text, float(item.get('score', 0)), x1, y1, x2, y2))
    items.sort(key=lambda z: (z[3], z[2]))
    found = []
    for text, score, x1, y1, x2, y2 in items:
        if not _is_name_label(text):
            continue
        cy = (y1+y2)/2
        label_h = max(1.0, y2-y1)
        near = []
        for t, s, ax1, ay1, ax2, ay2 in items:
            if ax1 < x2 - 2:
                continue
            acy = (ay1+ay2)/2
            if abs(acy-cy) <= max(label_h, ay2-ay1) * 0.75 + 10:
                if ax1 - x2 <= 430:
                    near.append((ax1, t, s, ay1, ay2))
        near.sort(key=lambda z: z[0])
        if near:
            text_right = ''.join(x[1] for x in near[:2])
            m = re.match(r'[\u4e00-\u9fa5]{2,6}', text_right)
            if m:
                name, ok = clean_name(m.group(0))
                if ok:
                    found.append((name, 120 + min(score,1.0)*30, 'A级-姓名标签定位'))
    return found


def extract_candidates_from_text(text, allow_weak=False):
    candidates = []
    for name in _name_from_text(text):
        candidates.append((name, 140, 'B级-文本标签'))
    if allow_weak:
        t = normalize_text(text)
        if re.fullmatch(r'[\u4e00-\u9fa5]{2,4}', t or ''):
            name, ok = clean_name(t)
            if ok:
                candidates.append((name, 40, 'C级-弱候选'))
    return candidates


def extract_name_from_ocr(results, roi_mode=False):
    if not results:
        return None, 0.0, []
    candidates = []
    # A级优先
    for name, base, source in locate_name_from_boxes(results):
        candidates.append((name, base, source))
    # B级
    for item in results:
        text = normalize_text(item.get('text',''))
        score = float(item.get('score',0))
        for name, base, source in extract_candidates_from_text(text, allow_weak=False):
            candidates.append((name, base + score*40, source))
    # C级（仅 roi_mode=True 且没有 A/B 级时允许）
    if roi_mode and not candidates:
        for item in results:
            text = normalize_text(item.get('text',''))
            score = float(item.get('score',0))
            for name, base, source in extract_candidates_from_text(text, allow_weak=True):
                candidates.append((name, base + score*20, source))

    if not candidates:
        return None, 0.0, []

    totals, counts, sources = {}, {}, {}
    for name, score, source in candidates:
        totals[name] = totals.get(name, 0.0) + score
        counts[name] = counts.get(name, 0) + 1
        sources.setdefault(name, set()).add(source)
    ranked=[]
    for name in totals:
        source_bonus = 50 if any('A级' in s for s in sources[name]) else 0
        ranked.append((name, totals[name] + min(counts[name]-1,4)*45 + source_bonus, counts[name], len(sources[name])))
    ranked.sort(key=lambda x:x[1], reverse=True)
    best=ranked[0]
    return best[0], best[1], ranked[:10]


# ============================================================
# 姓名区域
# ============================================================

def crop_relative(img, x1, y1, x2, y2):
    h,w=img.shape[:2]
    xx1=max(0,min(w-1,int(w*x1))); yy1=max(0,min(h-1,int(h*y1)))
    xx2=max(xx1+1,min(w,int(w*x2))); yy2=max(yy1+1,min(h,int(h*y2)))
    return img[yy1:yy2,xx1:xx2]


def build_name_rois(card):
    return [
        ('姓名标准区', crop_relative(card, .06,.145,.56,.34)),
        ('姓名窄区', crop_relative(card, .11,.165,.50,.30)),
        ('姓名宽区', crop_relative(card, .025,.115,.68,.38)),
        ('姓名扩展区', crop_relative(card, .04,.13,.72,.34)),
        ('姓名整行区', crop_relative(card, .02,.10,.80,.42)),
    ]


def dynamic_name_roi(card):
    probe = run_ocr(_resize_card_for_probe(card))
    names=[]
    for item in probe:
        text=normalize_text(item.get('text',''))
        box=_box_info(item)
        if box and _is_name_label(text):
            names.append((box,item))
    if not names:
        return None, probe
    box,_=min(names,key=lambda z:z[0][1])
    x1,y1,x2,y2=box
    sx=card.shape[1]/900.0; sy=card.shape[0]/568.0
    x1=int(x1*sx); x2=int(min(card.shape[1], (x2+390)*sx))
    y1=int(max(0,(y1-22)*sy)); y2=int(min(card.shape[0], (y2+35)*sy))
    if x2<=x1 or y2<=y1:
        return None, probe
    return card[y1:y2,x1:x2], probe


def recognize_idcard_name(original_img, log_callback=None, debug_dir=None, debug_name=None):
    if original_img is None:
        return None
    card, method = normalize_card(original_img, log_callback=log_callback)
    if log_callback:
        log_callback(f'身份证区域：{method}')
    if card is None:
        return None

    dynamic_roi, probe = dynamic_name_roi(card)
    vote = {}
    evidence = {}

    if dynamic_roi is not None and dynamic_roi.size:
        if log_callback: log_callback('识别姓名区域：OCR动态定位区')
        for variant_label, image in build_ocr_variants(dynamic_roi):
            results=run_ocr(image)
            name,score,ranked=extract_name_from_ocr(results,roi_mode=True)
            if log_callback:
                text=' | '.join(normalize_text(x.get('text','')) for x in results if normalize_text(x.get('text','')))
                if text: log_callback(f'OCR动态定位区/{variant_label}：{text[:120]}')
            if name:
                vote[name]=vote.get(name,0)+1+min(score/120.0,1.5)
                evidence.setdefault(name,[]).append(f'动态/{variant_label}')
        if vote:
            ranked_vote=sorted(vote.items(),key=lambda x:x[1],reverse=True)
            best,bv=ranked_vote[0]; second=ranked_vote[1][1] if len(ranked_vote)>1 else 0
            if log_callback: log_callback('动态姓名候选：'+'、'.join(f'{n}({s:.1f})' for n,s in ranked_vote[:5]))
            if len(evidence.get(best,[]))>=2 or (bv>=3.0 and bv>second*1.6):
                if log_callback: log_callback(f'姓名定位成功：{best}')
                return best

    vote={}; evidence={}
    for roi_label,roi in build_name_rois(card):
        if roi is None or roi.size==0: continue
        if log_callback: log_callback(f'识别姓名区域：{roi_label}')
        for variant_label,image in build_ocr_variants(roi):
            results=run_ocr(image)
            name,score,ranked=extract_name_from_ocr(results,roi_mode=True)
            if log_callback:
                text=' | '.join(normalize_text(x.get('text','')) for x in results if normalize_text(x.get('text','')))
                if text: log_callback(f'{roi_label}/{variant_label}：{text[:120]}')
            if name:
                vote[name]=vote.get(name,0)+1+min(score/120.0,1.5)
                evidence.setdefault(name,[]).append(f'{roi_label}/{variant_label}')
    if vote:
        ranked_vote=sorted(vote.items(),key=lambda x:x[1],reverse=True)
        best,bv=ranked_vote[0]; second=ranked_vote[1][1] if len(ranked_vote)>1 else 0
        if log_callback: log_callback('姓名候选：'+'、'.join(f'{n}({s:.1f})' for n,s in ranked_vote[:5]))
        if len(evidence.get(best,[]))>=2 or (bv>=3.8 and bv>second*1.8):
            if log_callback: log_callback(f'姓名区域可靠结果：{best}')
            return best

    if log_callback: log_callback('姓名专区结果不足，进入整张身份证 OCR 兜底（严格模式）。')
    all_results=[]
    for label,image in build_ocr_variants(card)[:4]:
        results=run_ocr(image)
        all_results.extend(results)
        if log_callback:
            text=' | '.join(normalize_text(x.get('text','')) for x in results if normalize_text(x.get('text','')))
            if text: log_callback(f'整卡/{label}：{text[:180]}')
    name,score,ranked=extract_name_from_ocr(all_results,roi_mode=False)
    if log_callback and ranked:
        log_callback('整卡 OCR 候选：'+'、'.join(f'{n}({s:.1f})' for n,s,_ in ranked[:5]))
    if name:
        if any(n==name for n,_,_ in [(x[0],x[1],x[2]) for x in ranked]) and score>=150:
            if log_callback: log_callback(f'整卡 OCR 最终姓名：{name}')
            return name
    if log_callback: log_callback('无法可靠确定姓名，保持原文件名。')
    return None


# ============================================================
# 重名处理
# ============================================================

def unique_path(path):
    path = Path(path)
    if not path.exists():
        return path
    base = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.parent / f"{base}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


# ============================================================
# 单张处理（线程安全）
# ============================================================

def process_single_image(img_path, output_dir=None, log_callback=None):
    img_path = Path(img_path)
    if not img_path.exists():
        return img_path, None
    img = imread_chinese(img_path)
    if img is None:
        if log_callback:
            log_callback(f"无法读取：{img_path}")
        return img_path, None

    if log_callback:
        log_callback("========================================")
        log_callback(f"处理：{img_path.name}")

    name = recognize_idcard_name(
        img,
        log_callback=log_callback,
        debug_dir=img_path.parent / "_ocr_debug",
        debug_name=img_path.stem
    )

    if not name:
        if log_callback:
            log_callback(f"失败：没有可靠识别姓名 -> {img_path.name}")
        return img_path, None

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        new_path = out_dir / f"{name}{img_path.suffix}"
    else:
        new_path = img_path.parent / f"{name}{img_path.suffix}"
    new_path = unique_path(new_path)

    try:
        if output_dir:
            shutil.copy2(str(img_path), str(new_path))
            action = "复制"
        else:
            img_path.rename(new_path)
            action = "重命名"
        if log_callback:
            log_callback(f"成功：{action} {img_path.name} -> {new_path.name}")
        return img_path, new_path
    except Exception as e:
        if log_callback:
            log_callback(f"文件操作失败：{e}")
        return img_path, None


# ============================================================
# 批量处理（多线程并行）
# ============================================================

def batch_process(folder_path, output_dir=None, log_callback=None, progress_callback=None):
    folder = Path(folder_path)
    if not folder.is_dir():
        if log_callback:
            log_callback(f"不是文件夹：{folder}")
        return

    image_files = sorted(
        [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS and "_ocr_debug" not in p.parts],
        key=lambda p: str(p).lower()
    )
    total = len(image_files)
    if total == 0:
        if log_callback:
            log_callback("没有找到支持的图片。")
        return

    if log_callback:
        log_callback(f"共找到 {total} 张图片，使用 {MAX_WORKERS} 线程并行处理...")

    success = 0
    failed = 0
    modified_dirs = set()

    # 使用线程池执行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for idx, img_file in enumerate(image_files):
            # 每个任务传入一个独立的日志回调（使用闭包捕获 idx）
            def make_callback(idx):
                def cb(msg):
                    if log_callback:
                        log_callback(f"[{idx+1}/{total}] {msg}")
                return cb
            future = executor.submit(
                process_single_image,
                img_file,
                output_dir,
                make_callback(idx)
            )
            futures[future] = idx

        # 收集结果并更新进度
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
            old, new = future.result()
            if new:
                success += 1
                modified_dirs.add(new.parent)
            else:
                failed += 1

    if log_callback:
        log_callback("========================================")
        log_callback(f"批量处理完成：成功 {success} 张，失败 {failed} 张。")
    if modified_dirs:
        refresh_multiple_folders(modified_dirs)


# ============================================================
# Windows 文件夹刷新
# ============================================================

def refresh_folder(folder_path):
    if sys.platform != "win32":
        return
    try:
        import ctypes
        SHCNE_UPDATEDIR = 0x00001000
        SHCNF_PATHW = 0x0005
        ctypes.windll.shell32.SHChangeNotify(SHCNE_UPDATEDIR, SHCNF_PATHW, ctypes.c_wchar_p(str(folder_path)), None)
    except Exception:
        pass


def refresh_multiple_folders(folder_paths):
    for folder in set(folder_paths):
        refresh_folder(folder)


# ============================================================
# 干扰字管理（界面固定大小）
# ============================================================

class DirtyCharManager:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("干扰字管理")
        self.window.geometry("620x700")
        self.window.resizable(False, False)
        self.window.transient(parent)

        self.temp_prefix = DIRTY_PREFIX.copy()
        self.temp_suffix = DIRTY_SUFFIX.copy()

        frame = ttk.LabelFrame(self.window, text="当前干扰字")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.display = tk.Text(frame, height=12, font=("微软雅黑", 10))
        self.display.pack(fill="both", expand=True, padx=5, pady=5)
        self.update_display()

        f1 = ttk.Frame(self.window)
        f1.pack(fill="x", padx=10, pady=5)
        ttk.Label(f1, text="首部干扰字（逗号分隔）：").pack(anchor="w")
        r1 = ttk.Frame(f1)
        r1.pack(fill="x")
        self.prefix_entry = ttk.Entry(r1)
        self.prefix_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(r1, text="添加", command=self.add_prefix).pack(side="left", padx=5)

        f2 = ttk.Frame(self.window)
        f2.pack(fill="x", padx=10, pady=5)
        ttk.Label(f2, text="尾部干扰字（逗号分隔）：").pack(anchor="w")
        r2 = ttk.Frame(f2)
        r2.pack(fill="x")
        self.suffix_entry = ttk.Entry(r2)
        self.suffix_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(r2, text="添加", command=self.add_suffix).pack(side="left", padx=5)

        buttons = ttk.Frame(self.window)
        buttons.pack(pady=10)
        ttk.Button(buttons, text="保存", command=self.save_and_close).pack(side="left", padx=5)
        ttk.Button(buttons, text="取消", command=self.window.destroy).pack(side="left", padx=5)

    def parse_input(self, text):
        text = text.replace("，", ",").replace("、", ",").replace("；", ",").replace(";", ",")
        return [x.strip() for x in text.split(",") if x.strip()]

    def add_prefix(self):
        items = self.parse_input(self.prefix_entry.get())
        for item in items:
            if item not in self.temp_prefix:
                self.temp_prefix.append(item)
        self.prefix_entry.delete(0, tk.END)
        self.update_display()

    def add_suffix(self):
        items = self.parse_input(self.suffix_entry.get())
        for item in items:
            if item not in self.temp_suffix:
                self.temp_suffix.append(item)
        self.suffix_entry.delete(0, tk.END)
        self.update_display()

    def update_display(self):
        self.display.delete("1.0", tk.END)
        self.display.insert(tk.END, "【首部】\n" + "、".join(self.temp_prefix) + "\n\n")
        self.display.insert(tk.END, "【尾部】\n" + "、".join(self.temp_suffix))

    def save_and_close(self):
        global DIRTY_PREFIX, DIRTY_SUFFIX
        DIRTY_PREFIX = self.temp_prefix.copy()
        DIRTY_SUFFIX = self.temp_suffix.copy()
        if save_config(DIRTY_PREFIX, DIRTY_SUFFIX):
            messagebox.showinfo("成功", "干扰字配置已保存并立即生效。")
            self.window.destroy()
        else:
            messagebox.showerror("错误", "保存配置失败。")


# ============================================================
# 主 GUI
# ============================================================

BaseTk = TkinterDnD.Tk if HAS_DND else tk.Tk


class App(BaseTk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("820x720")
        self.minsize(720, 600)
        self.path_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.ocr_ready = False
        self.processing = False

        self.create_widgets()
        self.setup_drag_drop()
        self.after(200, self.start_ocr_init)

    def create_widgets(self):
        input_frame = ttk.LabelFrame(self, text="输入设置", padding=8)
        input_frame.pack(fill="x", padx=10, pady=8)

        row1 = ttk.Frame(input_frame)
        row1.pack(fill="x", pady=4)
        ttk.Label(row1, text="图片 / 文件夹：").pack(side="left")
        ttk.Entry(row1, textvariable=self.path_var).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(row1, text="选择图片", command=self.browse_image).pack(side="left")
        ttk.Button(row1, text="选择文件夹", command=self.browse_folder).pack(side="left", padx=(5, 0))

        row2 = ttk.Frame(input_frame)
        row2.pack(fill="x", pady=4)
        ttk.Label(row2, text="输出目录：").pack(side="left")
        ttk.Entry(row2, textvariable=self.output_var).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(row2, text="选择", command=self.select_output).pack(side="left")

        option_frame = ttk.Frame(input_frame)
        option_frame.pack(fill="x", pady=6)
        ttk.Button(option_frame, text="🧹 干扰字管理", command=self.open_dirty_manager).pack(side="left")
        ttk.Label(option_frame, text="支持：实拍身份证 / 白底截图 / 拼图").pack(side="left", padx=15)
        self.status_label = ttk.Label(option_frame, text="OCR：正在初始化...")
        self.status_label.pack(side="right")

        button_frame = ttk.Frame(self)
        button_frame.pack(fill="x", padx=10, pady=5)
        self.start_button = ttk.Button(button_frame, text="开始处理", command=self.start_processing)
        self.start_button.pack(side="left", padx=5)
        ttk.Button(button_frame, text="清空日志", command=self.clear_log).pack(side="left", padx=5)

        self.progress = ttk.Progressbar(self, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", padx=10, pady=5)

        log_frame = ttk.LabelFrame(self, text="运行日志", padding=5)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, state="disabled", font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)

    def setup_drag_drop(self):
        if not HAS_DND:
            return
        self.drop_target_register(DND_FILES)
        self.dnd_bind("<<Drop>>", self.on_drop)

    def on_drop(self, event):
        data = event.data
        if not data:
            return
        try:
            import shlex
            paths = shlex.split(data)
        except Exception:
            paths = data.split()
        if not paths:
            return
        path = paths[0].strip('{}"')
        if Path(path).exists():
            self.path_var.set(path)

    def start_ocr_init(self):
        self.log("正在初始化 RapidOCR + ONNX Runtime...")
        self.log("不会使用 PaddleOCR，也不会使用 EasyOCR。")

        def worker():
            try:
                success = init_ocr()
                self.after(0, self.ocr_init_done, success, None)
            except Exception as e:
                self.after(0, self.ocr_init_done, False, str(e))

        threading.Thread(target=worker, daemon=True).start()

    def ocr_init_done(self, success, error):
        if success:
            self.ocr_ready = True
            self.status_label.config(text="OCR：RapidOCR/ONNX 就绪")
            self.log("OCR 初始化成功。")
        else:
            self.ocr_ready = False
            self.status_label.config(text="OCR：初始化失败")
            self.log(f"OCR 初始化失败：{error}")
            messagebox.showerror("OCR 初始化失败", f"请先安装：\n\npip install rapidocr onnxruntime\n\n{error}")

    def browse_image(self):
        path = filedialog.askopenfilename(
            title="选择身份证图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"), ("所有文件", "*.*")]
        )
        if path:
            self.path_var.set(path)

    def browse_folder(self):
        path = filedialog.askdirectory(title="选择身份证图片文件夹")
        if path:
            self.path_var.set(path)

    def select_output(self):
        path = filedialog.askdirectory(title="选择输出目录（不选择则直接重命名原图）")
        if path:
            self.output_var.set(path)

    def log(self, text):
        def update():
            self.log_text.config(state="normal")
            self.log_text.insert(tk.END, str(text) + "\n")
            self.log_text.see(tk.END)
            self.log_text.config(state="disabled")
        try:
            self.after(0, update)
        except Exception:
            pass

    def clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state="disabled")

    def open_dirty_manager(self):
        DirtyCharManager(self)

    def update_progress(self, current, total):
        def update():
            self.progress["maximum"] = total
            self.progress["value"] = current
        self.after(0, update)

    def start_processing(self):
        if self.processing:
            messagebox.showwarning("提示", "正在处理中，请等待完成。")
            return
        if not self.ocr_ready:
            messagebox.showerror("错误", "OCR 尚未初始化完成。")
            return
        raw_path = self.path_var.get().strip()
        if not raw_path:
            messagebox.showwarning("提示", "请选择图片或文件夹。")
            return
        path = Path(raw_path)
        if not path.exists():
            messagebox.showerror("错误", f"路径不存在：\n{raw_path}")
            return
        output = self.output_var.get().strip() or None
        self.processing = True
        self.start_button.config(state="disabled")
        self.progress["value"] = 0
        self.clear_log()
        threading.Thread(target=self.worker, args=(path, output), daemon=True).start()

    def worker(self, path, output):
        try:
            if path.is_file():
                if path.suffix.lower() not in IMAGE_EXTS:
                    self.log(f"不支持的图片格式：{path.suffix}")
                    return
                self.log(f"开始处理单张图片：{path}")
                old, new = process_single_image(path, output_dir=output, log_callback=self.log)
                self.update_progress(1, 1)
                if new:
                    self.log("处理完成。")
                else:
                    self.log("处理失败：未获得可靠姓名。")
            elif path.is_dir():
                self.log(f"开始批量处理：{path}")
                batch_process(path, output_dir=output, log_callback=self.log, progress_callback=self.update_progress)
            else:
                self.log("未知路径类型。")
        except Exception as e:
            import traceback
            self.log(f"发生异常：{e}")
            self.log(traceback.format_exc())
        finally:
            def finish():
                self.processing = False
                self.start_button.config(state="normal")
            self.after(0, finish)


# ============================================================
# main
# ============================================================

def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
