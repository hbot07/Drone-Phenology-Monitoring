from __future__ import annotations

from typing import Any, Dict

import numpy as np


def compute_crown_attributes(geometry) -> Dict[str, Any]:
    centroid = geometry.centroid
    area = geometry.area
    perimeter = geometry.length
    bounds = geometry.bounds

    compactness = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0
    try:
        min_rect = geometry.minimum_rotated_rectangle
        coords = list(min_rect.exterior.coords)
        side1 = np.linalg.norm(np.array(coords[0]) - np.array(coords[1]))
        side2 = np.linalg.norm(np.array(coords[1]) - np.array(coords[2]))
        major_axis = max(side1, side2)
        minor_axis = min(side1, side2)
        eccentricity = minor_axis / major_axis if major_axis > 0 else 1.0
    except Exception:
        eccentricity = 1.0

    aspect_ratio = (bounds[3] - bounds[1]) / (bounds[2] - bounds[0]) if bounds[2] != bounds[0] else 1.0

    return {
        "geometry": geometry,
        "centroid": centroid,
        "area": float(area),
        "perimeter": float(perimeter),
        "compactness": float(compactness),
        "eccentricity": float(eccentricity),
        "aspect_ratio": float(aspect_ratio),
        "bounds": bounds,
    }


def compute_iou(g1, g2) -> float:
    try:
        intersection = g1.intersection(g2).area
        union = g1.union(g2).area
    except Exception:
        intersection = 0.0
        union = g1.area + g2.area
    return float(intersection / union) if union > 0 else 0.0
