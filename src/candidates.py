# finds candidate roads near each GPS point
import numpy as np
import pandas as pd
from shapely import STRtree
from shapely.geometry import Point


def build_candidates(gps, roads, radius=50.0):
    """One row per (GPS point, nearby road)."""
    geoms = roads['geometry'].values
    edge_ids = roads['edge_id'].values
    tree = STRtree(geoms)                       # fast spatial index
    rows = []
    for i, (x, y) in enumerate(zip(gps['x'].values, gps['y'].values)):
        p = Point(x, y)
        for j in tree.query(p.buffer(radius)):
            g = geoms[j]
            d = g.distance(p)
            if d <= radius:
                rows.append((i, edge_ids[j], d, g.project(p), g.length))
    return pd.DataFrame(rows, columns=['pt_idx', 'edge_id', 'dist', 'along', 'road_len'])
