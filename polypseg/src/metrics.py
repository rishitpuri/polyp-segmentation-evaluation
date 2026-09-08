"""Per-image segmentation metrics.

Region metrics (Dice, IoU, precision, recall) plus boundary metrics
(Boundary F1 at a 2-pixel tolerance, and 95th-percentile Hausdorff
distance), which matter clinically for resection-margin accuracy and are
where small/flat polyps typically fail.

All functions take single-image binary numpy arrays so results can be
aggregated per stratum afterwards.
"""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

EPS = 1e-7


def _binary(x, thr=0.5):
    return (x > thr).astype(np.uint8)


def region_metrics(pred, gt):
    pred, gt = _binary(pred), _binary(gt)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, 1 - gt).sum())
    fn = float(np.logical_and(1 - pred, gt).sum())
    dice = (2 * tp) / (2 * tp + fp + fn + EPS)
    iou = tp / (tp + fp + fn + EPS)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    return dict(dice=dice, iou=iou, precision=precision, recall=recall)


def _boundary(mask):
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    return (mask - eroded).astype(bool)


def _tolerance_px(shape, tolerance_frac: float) -> int:
    """Scale the boundary tolerance to image size.

    A fixed pixel tolerance is not comparable across a dataset whose images
    range from 332x487 to 1920x1072: predictions are made at a fixed network
    resolution and upsampled, so achievable boundary precision grows with
    image size. We therefore express tolerance as a fraction of the image
    diagonal (default 0.5%), following the boundary-IoU convention.
    """
    h, w = shape[:2]
    diag = float(np.hypot(h, w))
    return max(1, int(round(tolerance_frac * diag)))


def boundary_f1(pred, gt, tolerance: int = 2):
    """F1 between boundary pixels, counting a match if within `tolerance` px."""
    pred, gt = _binary(pred), _binary(gt)
    pb, gb = _boundary(pred), _boundary(gt)
    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0
    dist_to_gt = distance_transform_edt(~gb)
    dist_to_pred = distance_transform_edt(~pb)
    precision = (dist_to_gt[pb] <= tolerance).mean()
    recall = (dist_to_pred[gb] <= tolerance).mean()
    return float(2 * precision * recall / (precision + recall + EPS))


def hd95(pred, gt):
    """95th-percentile symmetric Hausdorff distance in pixels.

    Returns NaN when either mask is empty (undefined), so it can be
    excluded from averages rather than silently biasing them.
    """
    pred, gt = _binary(pred), _binary(gt)
    pb, gb = _boundary(pred), _boundary(gt)
    if pb.sum() == 0 or gb.sum() == 0:
        return float("nan")
    d_pred_to_gt = distance_transform_edt(~gb)[pb]
    d_gt_to_pred = distance_transform_edt(~pb)[gb]
    return float(max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95)))


def lesion_metrics(pred, gt, iou_thresholds=(0.25, 0.5), min_area_frac: float = 2e-4):
    """Per-lesion detection statistics.

    Region-overlap metrics score a whole image as one region, so an image
    containing several lesions is penalised only in proportion to the missed
    lesions' combined area. For screening, the clinically meaningful quantity
    is what fraction of individual lesions is found.

    Each ground-truth connected component is matched against predicted
    components; a lesion counts as detected when some predicted component
    reaches the IoU threshold. Predicted components matching no lesion are
    counted as false positives. Components below `min_area_frac` of the image
    are discarded as noise.
    """
    pred, gt = _binary(pred), _binary(gt)
    h, w = gt.shape
    min_area = max(1.0, min_area_frac * h * w)

    def components(mask):
        n, lab = cv2.connectedComponents(mask, connectivity=8)
        out = []
        for i in range(1, n):
            comp = (lab == i)
            if comp.sum() >= min_area:
                out.append(comp)
        return out

    gt_comps = components(gt)
    pred_comps = components(pred)

    res = {"n_lesions": len(gt_comps), "n_pred_components": len(pred_comps)}
    matched_pred = set()
    for thr in iou_thresholds:
        detected = 0
        for g in gt_comps:
            best, best_j = 0.0, None
            for j, p in enumerate(pred_comps):
                inter = float(np.logical_and(g, p).sum())
                if inter == 0:
                    continue
                iou = inter / (float(np.logical_or(g, p).sum()) + EPS)
                if iou > best:
                    best, best_j = iou, j
            if best >= thr:
                detected += 1
                if best_j is not None:
                    matched_pred.add(best_j)
        key = f"{thr:g}".replace(".", "")
        res[f"detected_iou{key}"] = detected
        res[f"det_rate_iou{key}"] = detected / len(gt_comps) if gt_comps else float("nan")
    res["n_false_positives"] = max(0, len(pred_comps) - len(matched_pred))
    return res


def all_metrics(pred, gt, tolerance_frac: float = 0.005):
    """All metrics for one image.

    Boundary tolerance and HD95 are additionally reported scale-normalised
    (as a percentage of the image diagonal) so they are comparable across
    the heterogeneous image sizes in Kvasir-SEG and across datasets.
    """
    m = region_metrics(pred, gt)
    tol = _tolerance_px(gt.shape, tolerance_frac)
    m["boundary_f1"] = boundary_f1(pred, gt, tol)
    m["boundary_f1_2px"] = boundary_f1(pred, gt, 2)
    m["tolerance_px"] = tol
    h = hd95(pred, gt)
    m["hd95"] = h
    diag = float(np.hypot(*gt.shape[:2]))
    m["hd95_pct_diag"] = float("nan") if np.isnan(h) else 100.0 * h / diag
    m.update(lesion_metrics(pred, gt))
    return m
