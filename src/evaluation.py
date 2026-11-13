"""
Competition Metric (Panoptic Quality) and Submission Verification Utilities.
"""
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util


def rle_encode(mask: np.ndarray) -> str:
    """Encodes a 2D binary numpy array (H, W) into an ASCII COCO RLE string."""
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def rle_decode(rle_str: str, shape: Tuple[int, int] = (2048, 2048)) -> np.ndarray:
    """Decodes an ASCII COCO RLE string into a 2D binary uint8 numpy array."""
    rle_obj = {"size": list(shape), "counts": rle_str.encode("ascii")}
    return mask_util.decode(rle_obj).astype(np.uint8)


def compute_panoptic_quality(
    pred_rles: List[Dict[str, Any]],
    gt_rles: List[Dict[str, Any]],
    iou_threshold: float = 0.50,
) -> Dict[str, float]:
    """
    Computes exact single-class COCO Panoptic Quality (PQ = SQ * RQ):
      - True Positive (TP): IoU(pred, gt) > 0.50 (unique by theorem since masks are disjoint)
      - Segmentation Quality (SQ): sum_{(p,g) in TP} IoU(p, g) / |TP|
      - Recognition Quality (RQ): |TP| / (|TP| + 0.5*|FP| + 0.5*|FN|)
    """
    if not gt_rles and not pred_rles:
        return {"pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 0, "fp": 0, "fn": 0, "sum_iou": 0.0}
    if not gt_rles:
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": len(pred_rles), "fn": 0, "sum_iou": 0.0}
    if not pred_rles:
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": 0, "fn": len(gt_rles), "sum_iou": 0.0}

    iou_mat = mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles))
    hit = iou_mat > iou_threshold
    sum_iou = float(iou_mat[hit].sum())
    tp = int(hit.sum())
    fp = int((hit.sum(axis=1) == 0).sum())
    fn = int((hit.sum(axis=0) == 0).sum())
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = sum_iou / denom if denom > 0 else 0.0
    sq = sum_iou / tp if tp > 0 else 0.0
    rq = tp / denom if denom > 0 else 0.0
    return {"pq": pq, "sq": sq, "rq": rq, "tp": tp, "fp": fp, "fn": fn, "sum_iou": sum_iou}


def verify_submission_contract(
    df: pd.DataFrame,
    expected_stems: int = 180,
    check_overlaps: bool = True,
) -> Dict[str, Any]:
    """
    Verifies all structural, geometric, and topological requirements for a Kaggle
    Solar Filament Segmentation Challenge 2026 submission DataFrame.
    """
    required_cols = ["filament_id", "segmentation_rle"]
    assert list(df.columns[:2]) == required_cols, f"Invalid columns: {list(df.columns)}"
    assert df.isna().sum().sum() == 0, "Submission contains NaN/null values"
    assert df["filament_id"].is_unique, "Duplicate filament_id entries detected"
    assert (df["segmentation_rle"].str.len() > 0).all(), "Empty RLE strings detected"

    temp = df.copy()
    temp["stem"] = temp["filament_id"].str.rsplit("_", n=1).str[0]
    temp["station"] = temp["stem"].str[-2]
    n_stems = int(temp["stem"].nunique())
    assert n_stems == expected_stems, f"Expected {expected_stems} stems, found {n_stems}"

    overlap_violations = 0
    total_area = 0
    areas = []
    if check_overlaps:
        for stem, g in temp.groupby("stem"):
            canvas = np.zeros((2048, 2048), dtype=bool)
            for rle_str in g["segmentation_rle"]:
                m = rle_decode(rle_str).astype(bool)
                a = int(m.sum())
                areas.append(a)
                total_area += a
                if (canvas & m).any():
                    overlap_violations += 1
                canvas |= m

    assert overlap_violations == 0, f"Found {overlap_violations} overlapping instance masks!"

    station_counts = temp.groupby("station")["filament_id"].count().to_dict()
    return {
        "num_rows": len(df),
        "num_stems": n_stems,
        "overlap_violations": overlap_violations,
        "mean_area_px": float(np.mean(areas)) if areas else 0.0,
        "median_area_px": float(np.median(areas)) if areas else 0.0,
        "total_foreground_px": total_area,
        "station_counts": station_counts,
    }
