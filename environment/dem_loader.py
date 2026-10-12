"""Minimal GeoTIFF loader for SRTM-style 16-bit DEM files (numpy only)."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np

@dataclass(frozen=True)
class DEMInfo:
    path: str
    width: int
    height: int
    pixel_scale: Tuple[float, float, float]   # degrees per pixel (x, y, z)
    tiepoint: Tuple[float, float, float, float, float, float]
    # tiepoint: (raster_x, raster_y, raster_z, lon, lat, h)

def _read_tags(path: str) -> dict:
    with open(path, "rb") as f:
        head = f.read(8)
        if head[:2] != b"II":
            raise NotImplementedError("only little-endian TIFF is supported")
        magic = struct.unpack("<H", head[2:4])[0]
        if magic != 42:
            raise NotImplementedError("only classic TIFF (magic 42) is supported")
        ifd_offset = struct.unpack("<I", head[4:8])[0]
        f.seek(ifd_offset)
        n = struct.unpack("<H", f.read(2))[0]
        tags = {}
        for _ in range(n):
            data = f.read(12)
            tag, typ, cnt = struct.unpack("<HHI", data[:8])
            tags[tag] = (typ, cnt, data[8:])
        return tags

def _read_values(path: str, tags: dict, tag: int):
    typ, cnt, val = tags[tag]
    size_map = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 12: 8}
    with open(path, "rb") as f:
        if cnt * size_map[typ] <= 4:
            raw = val[:4]
        else:
            off = struct.unpack("<I", val[:4])[0]
            f.seek(off)
            raw = f.read(cnt * size_map[typ])
        if typ == 3:
            return struct.unpack("<" + "H" * cnt, raw[: cnt * 2])
        if typ == 4:
            return struct.unpack("<" + "I" * cnt, raw[: cnt * 4])
        if typ == 5:
            return [
                struct.unpack("<II", raw[i * 8 : i * 8 + 8]) for i in range(cnt)
            ]
        if typ == 12:
            return struct.unpack("<" + "d" * cnt, raw[: cnt * 8])
        return raw

def load_dem_array(path: str) -> Tuple[np.ndarray, DEMInfo]:
    """Load an uncompressed 16-bit grayscale GeoTIFF DEM into memory."""
    tags = _read_tags(path)
    width = _read_values(path, tags, 256)[0]
    height = _read_values(path, tags, 257)[0]
    bits = _read_values(path, tags, 258)[0]
    compression = _read_values(path, tags, 259)[0]
    sample_format = _read_values(path, tags, 339)[0] if 339 in tags else 1
    if bits != 16 or sample_format not in (1, 2):
        raise NotImplementedError(
            "only 16-bit unsigned/signed integer DEM samples are supported"
        )
    if compression != 1:
        raise NotImplementedError("only uncompressed TIFF DEM is supported")

    strip_offsets = np.asarray(_read_values(path, tags, 273), dtype=np.int64)
    strip_counts = np.asarray(_read_values(path, tags, 279), dtype=np.int64)
    rows_per_strip = _read_values(path, tags, 278)[0] if 278 in tags else 1

    array = np.zeros((height, width), dtype=np.uint16)
    with open(path, "rb") as f:
        for strip_idx, offset in enumerate(strip_offsets):
            f.seek(int(offset))
            row0 = strip_idx * rows_per_strip
            row1 = min(row0 + rows_per_strip, height)
            nbytes = int(strip_counts[strip_idx])
            data = np.frombuffer(f.read(nbytes), dtype="<u2")
            expected = (row1 - row0) * width
            if data.size != expected:
                raise ValueError(
                    f"strip {strip_idx} has {data.size} samples, expected {expected}"
                )
            array[row0:row1, :] = data.reshape(row1 - row0, width)

    ps = tuple(float(v) for v in _read_values(path, tags, 33550))
    tp = tuple(float(v) for v in _read_values(path, tags, 33922))
    info = DEMInfo(path=path, width=width, height=height,
                   pixel_scale=ps, tiepoint=tp)
    return array, info

def crop_geometry(
    info: DEMInfo,
    center_lat_lon: Tuple[float, float],
    crop_size_m: float,
) -> Tuple[int, int, int, int, int, int]:
    """Return (row0, row1, col0, col1, nrows, ncols) for a crop."""
    lat_c, lon_c = center_lat_lon
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat_c))
    px_h_m = info.pixel_scale[1] * m_per_deg_lat
    px_w_m = info.pixel_scale[0] * m_per_deg_lon
    nrows = int(round(crop_size_m / px_h_m))
    ncols = int(round(crop_size_m / px_w_m))
    col_c = (lon_c - info.tiepoint[3]) / info.pixel_scale[0]
    row_c = (info.tiepoint[4] - lat_c) / info.pixel_scale[1]
    col0 = int(round(col_c - ncols / 2.0))
    row0 = int(round(row_c - nrows / 2.0))
    col1, row1 = col0 + ncols, row0 + nrows
    if col0 < 0 or row0 < 0 or col1 > info.width or row1 > info.height:
        raise ValueError(
            f"DEM crop center {center_lat_lon} is outside the raster bounds "
            f"({info.width}x{info.height}); choose another center."
        )
    return row0, row1, col0, col1, nrows, ncols

def fill_nodata_nearest(crop: np.ndarray, nodata_values) -> np.ndarray:
    """Fill nodata pixels with the nearest valid 4-connected neighbor."""
    crop = np.asarray(crop, dtype=float).copy()
    mask = np.isin(crop, nodata_values)
    if not mask.any():
        return crop
    h, w = crop.shape
    while mask.any():
        ys, xs = np.where(mask)
        changed = False
        for y, x in zip(ys, xs):
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx]:
                    crop[y, x] = crop[ny, nx]
                    changed = True
                    break
        mask = np.isin(crop, nodata_values)
        if not changed:
            break
    return crop

def normalize_elevation(
    crop: np.ndarray,
    mode: str,
    max_relief: float,
) -> np.ndarray:
    """Apply the configured real-DEM elevation mapping."""
    if mode == "absolute":
        return crop
    h_min, h_max = float(crop.min()), float(crop.max())
    if mode == "detrend":
        return crop - h_min
    if h_max - h_min <= 1e-9:
        return np.zeros_like(crop)
    return (crop - h_min) / (h_max - h_min) * max_relief

def bilinear_sample(arr: np.ndarray, fx: float, fy: float) -> float:
    """Bilinear sample at fractional (col, row) coordinates."""
    h, w = arr.shape
    fx = float(np.clip(fx, 0.0, w - 1.0))
    fy = float(np.clip(fy, 0.0, h - 1.0))
    x0 = int(math.floor(fx))
    y0 = int(math.floor(fy))
    x0 = min(x0, w - 2)
    y0 = min(y0, h - 2)
    tx = fx - x0
    ty = fy - y0
    v00 = arr[y0, x0]
    v10 = arr[y0, x0 + 1]
    v01 = arr[y0 + 1, x0]
    v11 = arr[y0 + 1, x0 + 1]
    return float(
        v00 * (1.0 - tx) * (1.0 - ty)
        + v10 * tx * (1.0 - ty)
        + v01 * (1.0 - tx) * ty
        + v11 * tx * ty
    )
