from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .geometry import compute_iou
from .models import MatchCaseConfig, NodeId


def weighted_similarity(
    a1: Dict[str, Any],
    a2: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    max_dist: float = 100.0,
) -> Tuple[float, Dict[str, float]]:
    if weights is None:
        weights = {"spatial": 0.4, "area": 0.2, "shape": 0.2, "iou": 0.2}

    centroid_dist = a1["centroid"].distance(a2["centroid"])
    spatial_sim = max(0.0, 1.0 - (centroid_dist / max_dist))

    area_sim = min(a1["area"], a2["area"]) / max(a1["area"], a2["area"]) if max(a1["area"], a2["area"]) > 0 else 0.0

    compactness_sim = 1.0 - abs(a1["compactness"] - a2["compactness"])
    eccentricity_sim = 1.0 - abs(a1["eccentricity"] - a2["eccentricity"])
    shape_sim = (compactness_sim + eccentricity_sim) / 2.0

    iou_sim = compute_iou(a1["geometry"], a2["geometry"])

    total = (
        weights.get("spatial", 0.0) * spatial_sim
        + weights.get("area", 0.0) * area_sim
        + weights.get("shape", 0.0) * shape_sim
        + weights.get("iou", 0.0) * iou_sim
    )

    return float(total), {
        "spatial": float(spatial_sim),
        "area": float(area_sim),
        "shape": float(shape_sim),
        "iou": float(iou_sim),
        "total": float(total),
    }


def compute_pair_metrics(prev_attrs: Dict[str, Any], curr_attrs: Dict[str, Any], max_dist: float) -> Dict[str, float]:
    prev_geom = prev_attrs["geometry"]
    curr_geom = curr_attrs["geometry"]

    try:
        intersection_area = prev_geom.intersection(curr_geom).area
    except Exception:
        intersection_area = 0.0

    try:
        union_area = prev_geom.union(curr_geom).area
    except Exception:
        union_area = prev_attrs["area"] + curr_attrs["area"] - intersection_area

    prev_area = prev_attrs["area"] if prev_attrs["area"] > 0 else 1e-6
    curr_area = curr_attrs["area"] if curr_attrs["area"] > 0 else 1e-6

    overlap_prev = intersection_area / prev_area
    overlap_curr = intersection_area / curr_area
    iou = intersection_area / union_area if union_area > 0 else 0.0

    centroid_dist = prev_attrs["centroid"].distance(curr_attrs["centroid"])
    base_similarity, parts = weighted_similarity(prev_attrs, curr_attrs, max_dist=max_dist)

    prev_radius = np.sqrt(prev_area / np.pi)
    curr_radius = np.sqrt(curr_area / np.pi)
    mean_radius = max((prev_radius + curr_radius) / 2.0, 1e-3)

    area_ratio = curr_area / prev_area if prev_area > 0 else np.inf
    if not np.isfinite(area_ratio) or area_ratio <= 0:
        balanced_area_ratio = 0.0
    else:
        balanced_area_ratio = area_ratio if area_ratio <= 1 else 1 / area_ratio

    try:
        prev_contains_curr = prev_geom.buffer(0).contains(curr_geom)
    except Exception:
        prev_contains_curr = False

    try:
        curr_contains_prev = curr_geom.buffer(0).contains(prev_geom)
    except Exception:
        curr_contains_prev = False

    return {
        "intersection_area": float(intersection_area),
        "union_area": float(union_area),
        "overlap_prev": float(overlap_prev),
        "overlap_curr": float(overlap_curr),
        "iou": float(iou),
        "centroid_dist": float(centroid_dist),
        "base_similarity": float(base_similarity),
        "spatial_similarity": float(parts["spatial"]),
        "area_similarity": float(parts["area"]),
        "shape_similarity": float(parts["shape"]),
        "mean_radius": float(mean_radius),
        "area_ratio": float(area_ratio if np.isfinite(area_ratio) else 0.0),
        "balanced_area_ratio": float(balanced_area_ratio),
        "prev_contains_curr": bool(prev_contains_curr),
        "curr_contains_prev": bool(curr_contains_prev),
    }


def classify_match_case(
    prev_node: NodeId,
    curr_node: NodeId,
    features: Dict[str, float],
    prev_overlap_counts: Dict[NodeId, int],
    curr_overlap_counts: Dict[NodeId, int],
    overlap_gate: float,
    mode: str = "balanced",
) -> str:
    if features["prev_contains_curr"] or features["curr_contains_prev"]:
        return "containment"

    overlap_prev = features["overlap_prev"]
    overlap_curr = features["overlap_curr"]
    iou = features["iou"]
    centroid_dist = features["centroid_dist"]
    prev_count = prev_overlap_counts.get(prev_node, 0)
    curr_count = curr_overlap_counts.get(curr_node, 0)

    if mode == "lite":
        min_overlap_one_to_one = max(overlap_gate, 0.10)
        min_iou_one_to_one = max(overlap_gate * 0.5, 0.04)
        if (
            overlap_prev >= min_overlap_one_to_one
            and overlap_curr >= min_overlap_one_to_one
            and iou >= min_iou_one_to_one
            and centroid_dist < 50.0
        ):
            return "one_to_one"
        near_gate = max(overlap_gate * 0.5, 0.01)
        if centroid_dist < 50.0 and (overlap_prev >= near_gate or overlap_curr >= near_gate):
            return "nearby"

    elif mode == "strict":
        min_overlap_for_one_to_one = max(overlap_gate, 0.30)
        min_iou_for_one_to_one = max(overlap_gate / 2, 0.10)
        if (
            prev_count == 1
            and curr_count == 1
            and overlap_prev >= min_overlap_for_one_to_one
            and overlap_curr >= min_overlap_for_one_to_one
            and iou >= min_iou_for_one_to_one
        ):
            return "one_to_one"
        if centroid_dist < 30.0 and (overlap_prev >= overlap_gate or overlap_curr >= overlap_gate):
            return "nearby"

    else:  # balanced
        min_overlap_one_to_one = max(overlap_gate, 0.15)
        min_iou_one_to_one = max(overlap_gate * 0.5, 0.05)
        if (
            overlap_prev >= min_overlap_one_to_one
            and overlap_curr >= min_overlap_one_to_one
            and iou >= min_iou_one_to_one
            and centroid_dist < 40.0
        ):
            return "one_to_one"
        near_gate = max(overlap_gate * 0.5, 0.02)
        if centroid_dist < 35.0 and (overlap_prev >= near_gate or overlap_curr >= near_gate):
            return "nearby"

    return "none"


def score_candidate(base_similarity: float, similarity_parts: Dict[str, float], features: Dict[str, float], config: MatchCaseConfig) -> float:
    centroid_factor = 1.0 - min(1.0, features["centroid_dist"] / (features["mean_radius"] * 3.0))

    components = {
        "base": base_similarity,
        "spatial": similarity_parts.get("spatial", 0.0),
        "area": similarity_parts.get("area", 0.0),
        "shape": similarity_parts.get("shape", 0.0),
        "iou": features["iou"],
        "overlap_prev": features["overlap_prev"],
        "overlap_curr": features["overlap_curr"],
        "centroid": max(0.0, centroid_factor),
        "area_balance": features.get("balanced_area_ratio", 0.0),
    }

    score = 0.0
    for key, weight in config.scoring_weights.items():
        score += weight * components.get(key, 0.0)
    return float(score)


def select_candidates_by_case(
    candidates: List[Dict[str, Any]],
    configs: Dict[str, MatchCaseConfig],
    case_order: List[str],
    max_dist: float,
    min_base_similarity: float = 0.0,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    used_prev: Dict[NodeId, int] = defaultdict(int)
    used_curr: Dict[NodeId, int] = defaultdict(int)

    configs = {name: replace(cfg) for name, cfg in configs.items()}

    for case_name in case_order:
        config = configs.get(case_name)
        if not config:
            continue

        case_candidates = [cand for cand in candidates if cand["case"] == case_name]
        if not case_candidates:
            continue

        enriched: List[Dict[str, Any]] = []
        for cand in case_candidates:
            prev_attrs = cand["prev_attrs"]
            curr_attrs = cand["curr_attrs"]
            features = cand["features"]

            if config.max_centroid_dist is not None and features["centroid_dist"] > config.max_centroid_dist:
                continue
            if features["iou"] < config.min_iou:
                continue
            if features["overlap_prev"] < config.min_overlap_prev:
                continue
            if features["overlap_curr"] < config.min_overlap_curr:
                continue

            base_similarity, parts = weighted_similarity(
                prev_attrs,
                curr_attrs,
                weights=config.base_similarity_weights,
                max_dist=max_dist,
            )
            if min_base_similarity and base_similarity < min_base_similarity:
                continue

            score = score_candidate(base_similarity, parts, features, config)
            if score < config.similarity_threshold:
                continue

            cand = dict(cand)
            cand["base_similarity"] = float(base_similarity)
            cand["similarity_parts"] = {k: float(v) for k, v in parts.items()}
            cand["score"] = float(score)
            enriched.append(cand)

        if not enriched:
            continue

        if config.mutual_best:
            best_prev: Dict[NodeId, Dict[str, Any]] = {}
            best_curr: Dict[NodeId, Dict[str, Any]] = {}

            for cand in enriched:
                prev_node = cand["prev_node"]
                curr_node = cand["curr_node"]

                if not config.allow_multiple and (used_prev.get(prev_node, 0) or used_curr.get(curr_node, 0)):
                    continue

                if prev_node not in best_prev or cand["score"] > best_prev[prev_node]["score"]:
                    best_prev[prev_node] = cand
                if curr_node not in best_curr or cand["score"] > best_curr[curr_node]["score"]:
                    best_curr[curr_node] = cand

            for cand in enriched:
                prev_node = cand["prev_node"]
                curr_node = cand["curr_node"]

                if best_prev.get(prev_node) is not cand or best_curr.get(curr_node) is not cand:
                    continue

                if not config.allow_multiple and (used_prev.get(prev_node, 0) or used_curr.get(curr_node, 0)):
                    continue

                if config.max_edges_per_prev is not None and used_prev[prev_node] >= config.max_edges_per_prev:
                    continue
                if config.max_edges_per_curr is not None and used_curr[curr_node] >= config.max_edges_per_curr:
                    continue

                selected.append(cand)
                used_prev[prev_node] += 1
                used_curr[curr_node] += 1

        else:
            enriched.sort(key=lambda c: c["score"], reverse=True)
            for cand in enriched:
                prev_node = cand["prev_node"]
                curr_node = cand["curr_node"]

                if not config.allow_multiple and (used_prev.get(prev_node, 0) or used_curr.get(curr_node, 0)):
                    continue

                if config.max_edges_per_prev is not None and used_prev[prev_node] >= config.max_edges_per_prev:
                    continue
                if config.max_edges_per_curr is not None and used_curr[curr_node] >= config.max_edges_per_curr:
                    continue

                selected.append(cand)
                used_prev[prev_node] += 1
                used_curr[curr_node] += 1

    return selected


def ultra_relaxed_case_configs() -> Dict[str, MatchCaseConfig]:
    return {
        "one_to_one": MatchCaseConfig(
            name="one_to_one",
            base_similarity_weights={"spatial": 0.50, "area": 0.15, "shape": 0.05, "iou": 0.30},
            scoring_weights={"base": 0.60, "iou": 0.10, "overlap_prev": 0.05, "overlap_curr": 0.05, "centroid": 0.20},
            similarity_threshold=0.25,
            min_iou=0.01,
            min_overlap_prev=0.10,
            min_overlap_curr=0.10,
            max_centroid_dist=50.0,
            allow_multiple=True,
            max_edges_per_prev=3,
            max_edges_per_curr=3,
        ),
        "containment": MatchCaseConfig(
            name="containment",
            base_similarity_weights={"spatial": 0.50, "area": 0.20, "shape": 0.10, "iou": 0.20},
            scoring_weights={"base": 0.40, "overlap_prev": 0.30, "overlap_curr": 0.30},
            similarity_threshold=0.25,
            min_overlap_prev=0.25,
            min_overlap_curr=0.25,
            max_centroid_dist=150.0,
            allow_multiple=True,
            max_edges_per_prev=2,
            max_edges_per_curr=2,
        ),
        "nearby": MatchCaseConfig(
            name="nearby",
            base_similarity_weights={"spatial": 0.85, "area": 0.10, "shape": 0.03, "iou": 0.02},
            scoring_weights={"base": 0.70, "centroid": 0.30},
            similarity_threshold=0.20,
            min_overlap_prev=0.05,
            min_overlap_curr=0.05,
            max_centroid_dist=200.0,
            allow_multiple=True,
            max_edges_per_prev=2,
            max_edges_per_curr=2,
        ),
    }


def strict_aligned_configs() -> Dict[str, MatchCaseConfig]:
    return {
        "one_to_one": MatchCaseConfig(
            name="one_to_one",
            base_similarity_weights={"iou": 0.45, "spatial": 0.35, "area": 0.15, "shape": 0.05},
            scoring_weights={"base": 0.40, "iou": 0.30, "overlap_prev": 0.10, "overlap_curr": 0.10, "centroid": 0.10},
            similarity_threshold=0.40,
            min_iou=0.30,
            min_overlap_prev=0.20,
            min_overlap_curr=0.20,
            max_centroid_dist=10.0,
            allow_multiple=True,
            max_edges_per_prev=2,
            max_edges_per_curr=2,
        ),
        "containment": MatchCaseConfig(
            name="containment",
            base_similarity_weights={"iou": 0.40, "spatial": 0.35, "area": 0.20, "shape": 0.05},
            scoring_weights={"base": 0.30, "overlap_prev": 0.35, "overlap_curr": 0.35},
            similarity_threshold=0.35,
            min_iou=0.30,
            min_overlap_prev=0.30,
            min_overlap_curr=0.30,
            max_centroid_dist=15.0,
            allow_multiple=True,
            max_edges_per_prev=2,
            max_edges_per_curr=2,
        ),
        "nearby": MatchCaseConfig(
            name="nearby",
            base_similarity_weights={"iou": 0.35, "spatial": 0.45, "area": 0.15, "shape": 0.05},
            scoring_weights={"base": 0.60, "centroid": 0.40},
            similarity_threshold=0.30,
            min_iou=0.10,
            min_overlap_prev=0.10,
            min_overlap_curr=0.10,
            max_centroid_dist=20.0,
            allow_multiple=True,
            max_edges_per_prev=2,
            max_edges_per_curr=2,
        ),
    }
