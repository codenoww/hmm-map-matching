# accuracy and F1 evaluation
import numpy as np
import pandas as pd
from shapely.geometry import Point


def assign_true_roads(gps, roads, truth, window=15):
    """Give each GPS point its true road.
    Walks forward along the ground-truth route; for each point it picks the
    closest road among the next `window` roads on the route, and never goes back."""
    geom = dict(zip(roads['edge_id'], roads['geometry']))
    route = [e for e in truth['edge_id'].tolist() if e in geom]
    print('route length used:', len(route), 'of', len(truth))

    k = 0
    out_edge, out_dist = [], []
    for x, y in zip(gps['x'].values, gps['y'].values):
        p = Point(x, y)
        hi = min(k + window, len(route))
        ds = [geom[route[j]].distance(p) for j in range(k, hi)]
        best = k + int(np.argmin(ds))
        k = best
        out_edge.append(route[best])
        out_dist.append(min(ds))
    return pd.DataFrame({'pt_idx': np.arange(len(gps)),
                         'true_edge': out_edge,
                         'dist_to_true': out_dist})
