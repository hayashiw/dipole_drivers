import numpy as np

from scipy.spatial import ConvexHull, Delaunay, Voronoi

__all__ = [
    "fit_minimum_enclosing_circle",
    "fit_maximum_enclosed_circle",
]

def _circ2(a, b):
    return 0.5 * (a + b), 0.5 * np.linalg.norm(a - b)

def _circ3(a, b, c):
    ax, ay = a; bx, by = b; cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-14:
        return None
    a2, b2, c2 = ax*ax + ay*ay, bx*bx + by*by, cx*cx + cy*cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    ctr = np.array([ux, uy])
    return ctr, np.linalg.norm(ctr - a)

def fit_minimum_enclosing_circle(
    points: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    seed: int = 0
):
    r"""Smallest circle enclosing a 2D cloud (Welzl, move-to-front).

    Exact, O(n) expected. Returns (center[2], radius).
    Prefilters with the convex hull for speed

    Args:
        points: (n, 2) array of points.
        seed: Random seed for reproducibility.
    """
    assert points.ndim == 2 and points.shape[1] == 2, \
        f"Input points must be a 2D array of shape (n, 2). Got shape {points.shape}."
    pts = np.unique(np.asarray(points, dtype=float), axis=0)
    pts = pts[ConvexHull(pts).vertices]

    pts = pts[np.random.default_rng(seed).permutation(len(pts))]
    c, r = pts[0].copy(), 0.0
    for i in range(len(pts)):
        if np.linalg.norm(pts[i] - c) <= r + 1e-12:
            continue
        c, r = pts[i].copy(), 0.0
        for j in range(i):
            if np.linalg.norm(pts[j] - c) <= r + 1e-12:
                continue
            c, r = _circ2(pts[i], pts[j])
            for k in range(j):
                if np.linalg.norm(pts[k] - c) <= r + 1e-12:
                    continue
                res = _circ3(pts[i], pts[j], pts[k])
                if res is not None:
                    c, r = res
    return c, r

def fit_maximum_enclosed_circle(
    points: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    enclose: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None,
):
    r"""Largest empty circle enclosed by a 2D cloud.

    Returns (center[2], radius).

    Args:
        points: (n, 2) array of points.
    """
    assert points.ndim == 2 and points.shape[1] == 2, \
        f"Input points must be a 2D array of shape (n, 2). Got shape {points.shape}."
    pts = np.unique(np.asarray(points, dtype=float), axis=0)
    N = pts.shape[0]
    assert N >= 2, f"At least 2 unique points are required. Got {N} unique points."

    if N == 2:
        center = 0.5 * (pts[0] + pts[1])
        radius = 0.5 * np.linalg.norm(pts[0] - pts[1])
        return center, radius

    hull_pts = pts[ConvexHull(pts).vertices]
    starts = hull_pts
    ends = np.roll(hull_pts, -1, axis=0)
    t = np.linspace(0.0, 1.0, 101)[None, :, None]
    edge_samples = (starts[:, None, :] * (1.0 - t) + ends[:, None, :] * t).reshape(-1, 2)

    candidates = [Voronoi(pts).vertices, edge_samples, np.mean(pts, axis=0)[None, :]]
    if enclose is not None:
        enc = np.atleast_2d(np.asarray(enclose, dtype=float))
        candidates.append(Voronoi(np.vstack([pts, enc])).vertices)
    candidates = np.vstack(candidates)
    candidates = candidates[Delaunay(pts).find_simplex(candidates) >= 0]

    radii = np.linalg.norm(candidates[:, None, :] - pts[None, :, :], axis=2).min(axis=1)
    if enclose is not None:
        needed = np.linalg.norm(candidates[:, None, :] - enc[None, :, :], axis=2).max(axis=1)
        feasible = radii >= needed
        if not feasible.any():
            raise ValueError(
                "No empty circle centred within the convex hull encloses the "
                "given points.")
        candidates, radii = candidates[feasible], radii[feasible]

    k = int(np.argmax(radii))
    return candidates[k], radii[k]
