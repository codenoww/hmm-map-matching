# loads GPS points, road network and ground truth
import re
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, box

# Seattle is in UTM zone 10N (EPSG:32610): this converts degrees to metres
TO_M = Transformer.from_crs(4326, 32610, always_xy=True)
TO_LL = Transformer.from_crs(32610, 4326, always_xy=True)

def load_gps(path):
    df = pd.read_csv(path, sep='\t', header=0)

    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        'Date (UTC)': 'date',
        'Time (UTC)': 'time',
        'Latitude': 'lat',
        'Longitude': 'lon'
    })

    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')

    df = df.dropna(subset=['lat', 'lon']).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid GPS coordinates found in gps_data.txt")

    df['timestamp'] = pd.to_datetime(
        df['date'].astype(str) + ' ' + df['time'].astype(str),
        format='%d-%b-%Y %H:%M:%S',
        errors='coerce'
    )

    df = df.dropna(subset=['timestamp']).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid GPS timestamps found.")

    df['t'] = (
        df['timestamp'] - df['timestamp'].iloc[0]
    ).dt.total_seconds()

    df['x'], df['y'] = TO_M.transform(
        df['lon'].to_numpy(),
        df['lat'].to_numpy()
    )

    return df


def load_roads(path):
    rows = []
    skipped = 0

    with open(path, 'r', errors='replace') as f:
        next(f, None)  # Skip header

        for line in f:
            try:
                m = re.search(r'LINESTRING\s*\((.*?)\)', line)

                if not m:
                    skipped += 1
                    continue

                head = line[:m.start()].strip().split('\t')

                if len(head) < 6:
                    skipped += 1
                    continue

                pts = []
                for p in m.group(1).split(','):
                    values = p.strip().split()

                    if len(values) >= 2:
                        pts.append(tuple(map(float, values)))

                if len(pts) < 2:
                    skipped += 1
                    continue

                # Detect longitude-first or latitude-first coordinates
                if abs(pts[0][0]) > 90:
                    lons, lats = zip(*[(p[0], p[1]) for p in pts])
                else:
                    lats, lons = zip(*[(p[0], p[1]) for p in pts])

                xs, ys = TO_M.transform(lons, lats)

                rows.append({
                    'edge_id': int(head[0]),
                    'from_node': head[1],
                    'to_node': head[2],
                    'two_way': int(head[3]),
                    'speed': float(head[4]),
                    'vertex_count': int(head[5]),
                    'geometry': LineString(zip(xs, ys))
                })

            except (ValueError, TypeError, IndexError):
                skipped += 1

    roads = pd.DataFrame(rows)

    if roads.empty:
        raise ValueError(
            "No valid roads loaded. Check the road_network.txt format."
        )

    print("Roads loaded:", len(roads))
    print("Skipped lines:", skipped)

    return roads


def load_truth(path):
    import pandas as pd

    df = pd.read_csv(path, sep='\t', header=0)

    print("Ground-truth columns:", df.columns.tolist())
    print("Raw rows:", len(df))

    # The first column contains the Edge ID
    df = df.iloc[:, :2].copy()
    df.columns = ['edge_id', 'direction']

    df['edge_id'] = pd.to_numeric(
        df['edge_id'], errors='coerce'
    )

    df = df.dropna(subset=['edge_id']).reset_index(drop=True)
    df['edge_id'] = df['edge_id'].astype('int64')

    print("Valid ground-truth rows:", len(df))

    return df


def crop_roads(roads, gps, margin=500):
    """Keep only roads inside the GPS trace's bounding box (+ margin metres)."""
    area = box(gps.x.min() - margin, gps.y.min() - margin,
               gps.x.max() + margin, gps.y.max() + margin)
    keep = roads.geometry.apply(lambda g: g.intersects(area))
    return roads[keep].reset_index(drop=True)
