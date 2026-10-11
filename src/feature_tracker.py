"""Сопровождение признаков для MSCKF: KLT, проверка вперед-назад, RANSAC.

Выход — словарь {id: (u, v)} на каждый кадр (пиксели кадра без
дисторсии). Признак теряет id при любой ошибке сопровождения; новые
признаки добираются по сетке, чтобы покрытие кадра было равномерным.
"""
from __future__ import annotations

import cv2
import numpy as np


class KltTracker:
    def __init__(self, *, max_features: int = 150, min_distance_px: int = 20,
                 fb_threshold_px: float = 0.7, ransac_px: float = 1.0, quality: float = 0.01) -> None:
        self.max_features = max_features
        self.min_distance = min_distance_px
        self.fb = fb_threshold_px
        self.ransac = ransac_px
        self.quality = quality
        self._prev: np.ndarray | None = None
        self._pts = np.zeros((0, 2), np.float32)
        self._ids = np.zeros(0, np.int64)
        self._next_id = 0
        self.counts = {"tracked": 0, "lost_klt": 0, "lost_fb": 0, "lost_ransac": 0, "new": 0}

    def track(self, gray: np.ndarray) -> dict[int, tuple[float, float]]:
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError("Нужен кадр uint8 в оттенках серого")
        if self._prev is not None and len(self._pts):
            p0 = self._pts.reshape(-1, 1, 2)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self._prev, gray, p0, None, winSize=(21, 21), maxLevel=3)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self._prev, p1, None, winSize=(21, 21), maxLevel=3)
            ok = (st.ravel() == 1) & (st2.ravel() == 1)
            self.counts["lost_klt"] += int(np.sum(~ok))
            fb = np.linalg.norm((back - p0).reshape(-1, 2), axis=1)
            good = ok & (fb < self.fb)
            self.counts["lost_fb"] += int(np.sum(ok & ~good))
            idx = np.where(good)[0]
            if len(idx) >= 8:
                _, mask = cv2.findFundamentalMat(p0[idx], p1[idx], cv2.FM_RANSAC, self.ransac, 0.999)
                if mask is not None:
                    inl = mask.ravel() == 1
                    self.counts["lost_ransac"] += int(np.sum(~inl))
                    idx = idx[inl]
            self._pts = p1.reshape(-1, 2)[idx]
            self._ids = self._ids[idx]
            h, w = gray.shape
            inside = (self._pts[:, 0] >= 0) & (self._pts[:, 0] < w) & (self._pts[:, 1] >= 0) & (self._pts[:, 1] < h)
            self._pts, self._ids = self._pts[inside], self._ids[inside]
            self.counts["tracked"] += len(self._ids)
        need = self.max_features - len(self._pts)
        if need > 0:
            mask = np.full(gray.shape, 255, np.uint8)
            for x, y in self._pts:
                cv2.circle(mask, (int(x), int(y)), self.min_distance, 0, -1)
            new = cv2.goodFeaturesToTrack(gray, need, self.quality, self.min_distance, mask=mask, blockSize=7)
            if new is not None:
                new = new.reshape(-1, 2).astype(np.float32)
                ids = np.arange(self._next_id, self._next_id + len(new))
                self._next_id += len(new)
                self._pts = np.vstack([self._pts, new])
                self._ids = np.concatenate([self._ids, ids])
                self.counts["new"] += len(new)
        self._prev = gray
        return {int(i): (float(p[0]), float(p[1])) for i, p in zip(self._ids, self._pts)}
