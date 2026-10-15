"""Features for HMM map matching (emission + transition), following the CS229 paper.

Everything is precomputed into flat numpy arrays (class Prep) so Viterbi stays fast.
Candidate rows come from candidates.pkl: one row per (GPS point, nearby road).
"""
import numpy as np
import pandas as pd
import shapely

MPH_40 = 17.88          # 40 mph in m/s (speed-limit feature threshold in the paper)


class Prep:
    """Flat arrays used by the matcher. 'rows' = row numbers of the (sorted) candidate table."""
    pass


def estimate_sigma(true_dist):
    """GPS noise sigma from the distance of points to their true road (Newson & Krumm's
    median-based estimate). Pass only TRAINING points to avoid leaking the test set."""
    return 1.4826 * float(np.median(true_dist))


def gauss_distance(d, sigma):
    """Emission feature 1 (paper): Gaussian of the distance to the road.
    Scaled to peak at 1 (the paper's 1/(sigma*sqrt(2*pi)) factor is dropped so that all
    features live on a 0..1 scale)."""
    return np.exp(-(d ** 2) / (2.0 * sigma ** 2))


def sech_distance(d, sigma):
    """Change B: hyperbolic-secant distance feature (heavier tails than Gaussian).
    Same scale sigma (same variance), peak 1 at d = 0."""
    return 1.0 / np.cosh(np.pi * d / (2.0 * sigma))


def heading_and_speed(gps, k=3, min_move=6.0):
    """The Seattle file has no speed/heading, so derive them from the last k seconds.
    Uses only past points (real-time safe). Returns speed (m/s), unit heading (hx, hy) and
    a 'moving' flag (heading is meaningless when the car barely moved)."""
    x, y, t = gps['x'].values, gps['y'].values, gps['t'].values
    n = len(gps)
    j = np.maximum(np.arange(n) - k, 0)
    dx, dy, dt = x - x[j], y - y[j], t - t[j]
    dist = np.hypot(dx, dy)
    speed = np.where(dt > 0, dist / np.where(dt > 0, dt, 1.0), 0.0)
    moving = dist >= min_move
    safe = np.where(dist > 0, dist, 1.0)
    return speed, dx / safe, dy / safe, moving


def prepare(gps, roads, cands, sigma, k=3, min_move=6.0):
    """Build all per-point and per-candidate arrays."""
    P = Prep()
    n = len(gps)
    P.n = n
    P.gx, P.gy, P.t = gps['x'].values, gps['y'].values, gps['t'].values
    P.speed, hx, hy, moving = heading_and_speed(gps, k, min_move)
    P.moving = moving
    P.sigma = sigma

    cands = cands.sort_values(['pt_idx', 'edge_id']).reset_index(drop=True)
    P.c_pt = cands['pt_idx'].values.astype(np.int64)
    P.c_edge = cands['edge_id'].values.astype(np.int64)
    P.c_dist = cands['dist'].values.astype(float)
    P.c_along = cands['along'].values.astype(float)
    P.c_len = cands['road_len'].values.astype(float)
    pts = np.arange(n)
    P.st = np.searchsorted(P.c_pt, pts, side='left')
    P.en = np.searchsorted(P.c_pt, pts, side='right')
    if not (P.en > P.st).all():
        bad = int((P.en <= P.st).sum())
        raise ValueError(f'{bad} GPS points have no candidate roads - increase the candidate radius')

    # map candidate rows to road table rows
    assert roads['edge_id'].is_unique, 'duplicate edge ids in roads table'
    ridx = pd.Series(np.arange(len(roads)), index=roads['edge_id'].values)
    c_ri = ridx.loc[P.c_edge].values
    geoms = roads['geometry'].values[c_ri]
    two_way = roads['two_way'].values[c_ri] == 1
    limit = roads['speed'].values[c_ri]

    # point on the road closest to the GPS point, and the road's direction there
    pp = shapely.line_interpolate_point(geoms, P.c_along)
    P.c_px, P.c_py = shapely.get_x(pp), shapely.get_y(pp)
    p0 = shapely.line_interpolate_point(geoms, np.clip(P.c_along - 1.0, 0, P.c_len))
    p1 = shapely.line_interpolate_point(geoms, np.clip(P.c_along + 1.0, 0, P.c_len))
    rdx, rdy = shapely.get_x(p1) - shapely.get_x(p0), shapely.get_y(p1) - shapely.get_y(p0)
    rn = np.hypot(rdx, rdy)
    ok = rn > 0
    rdx, rdy = rdx / np.where(ok, rn, 1.0), rdy / np.where(ok, rn, 1.0)

    # midpoints of each candidate's road (used by transition feature 2)
    mid = shapely.line_interpolate_point(geoms, 0.5, normalized=True)
    P.c_mx, P.c_my = shapely.get_x(mid), shapely.get_y(mid)

    # ---- emission features, one row per candidate ----
    f1g = gauss_distance(P.c_dist, sigma)
    f1s = sech_distance(P.c_dist, sigma)

    # feature 2: direction. |cos| for two-way roads; one-way roads only count if we go with them
    dot = hx[P.c_pt] * rdx + hy[P.c_pt] * rdy
    f2 = np.where(two_way, np.abs(dot), np.maximum(dot, 0.0))
    f2 = np.where(moving[P.c_pt] & ok, f2, 0.5)          # neutral when heading unknown

    # feature 3: speed limit. 0 if we drive >40 mph and >=20% above this road's limit
    v = P.speed[P.c_pt]
    f3 = np.where((v > MPH_40) & (v >= 1.2 * limit), 0.0, 1.0)

    P.f1_gauss, P.f1_sech, P.f2, P.f3 = f1g, f1s, f2, f3
    P.F_gauss = np.column_stack([f1g, f2, f3])      # baseline emission features
    P.F_sech = np.column_stack([f1s, f2, f3])       # change B emission features
    return P


def trans_feats(P, ra, rb, i, H, mode='midpoint'):
    """Transition features for every pair (a in ra at point i-1) -> (b in rb at point i).

    omega1 (backtrack): 0 if b was already on a's recent path (and b != a), else 1.
    omega2 (distance discrepancy): 1 / (|d(GPS_i, GPS_i-1) - d(road a, road b)| + 1).
        mode='midpoint'  : d(road a, road b) = distance between the roads' midpoints (paper)
        mode='projection': distance between the projected points (|along diff| on same road)
    H : (len(ra), L) array with the most recent road ids on each predecessor's best path
    Returns two (len(ra), len(rb)) matrices."""
    d_gps = np.hypot(P.gx[i] - P.gx[i - 1], P.gy[i] - P.gy[i - 1])
    ea, eb = P.c_edge[ra], P.c_edge[rb]
    if mode == 'midpoint':
        d_road = np.hypot(P.c_mx[ra][:, None] - P.c_mx[rb][None, :],
                          P.c_my[ra][:, None] - P.c_my[rb][None, :])
    else:
        d_proj = np.hypot(P.c_px[ra][:, None] - P.c_px[rb][None, :],
                          P.c_py[ra][:, None] - P.c_py[rb][None, :])
        same = ea[:, None] == eb[None, :]
        d_road = np.where(same, np.abs(P.c_along[rb][None, :] - P.c_along[ra][:, None]), d_proj)
    om2 = 1.0 / (np.abs(d_gps - d_road) + 1.0)
    seen = (H[:, None, :] == eb[None, :, None]).any(axis=2)
    om1 = 1.0 - (seen & (ea[:, None] != eb[None, :])).astype(float)
    return om1, om2
# emission and transition feature
