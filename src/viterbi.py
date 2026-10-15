"""Viterbi map matching: whole-trace, sliding window with buffer, and a nearest-road baseline."""
import numpy as np
from features import trans_feats

EPS = 1e-9


def _hist_from_edges(edges, L):
    """Right-aligned history array of length L (padded with -1)."""
    h = np.full(L, -1, dtype=np.int64)
    edges = np.asarray(edges, dtype=np.int64)[-L:]
    if len(edges):
        h[L - len(edges):] = edges
    return h


def decode_segment(P, F, i0, i1, w_em, w_tr, prev_row=None, prev_hist=None,
                   mode='midpoint', L=15):
    """Best road sequence for GPS points i0..i1 (inclusive).

    F        : (n_candidates, 3) emission features (P.F_gauss or P.F_sech)
    w_em     : 3 emission weights, w_tr : 2 transition weights (each group sums to 1)
    prev_row : candidate row already decided for point i0-1 (None = start of the trace)
    prev_hist: recent road ids on that decided path (length L)
    Returns the list of candidate rows (one per point)."""
    rows = np.arange(P.st[i0], P.en[i0])
    lem = np.log(F[rows] @ w_em + EPS)
    if prev_row is None:
        score = lem
        H = np.full((len(rows), L), -1, dtype=np.int64)
        H[:, -1] = P.c_edge[rows]
    else:
        ra = np.array([prev_row])
        Ha = prev_hist[None, :]
        om1, om2 = trans_feats(P, ra, rows, i0, Ha, mode)
        score = np.log(w_tr[0] * om1[0] + w_tr[1] * om2[0] + EPS) + lem
        H = np.concatenate([np.repeat(Ha[:, 1:], len(rows), axis=0), P.c_edge[rows][:, None]], axis=1)

    all_rows, back = [rows], []
    for i in range(i0 + 1, i1 + 1):
        rb = np.arange(P.st[i], P.en[i])
        om1, om2 = trans_feats(P, rows, rb, i, H, mode)
        lt = np.log(w_tr[0] * om1 + w_tr[1] * om2 + EPS)
        tot = score[:, None] + lt
        best_a = tot.argmax(axis=0)
        score = tot[best_a, np.arange(len(rb))] + np.log(F[rb] @ w_em + EPS)
        H = np.concatenate([H[best_a][:, 1:], P.c_edge[rb][:, None]], axis=1)
        back.append(best_a)
        all_rows.append(rb)
        rows = rb

    j = int(score.argmax())
    path = [all_rows[-1][j]]
    for k in range(len(back) - 1, -1, -1):
        j = int(back[k][j])
        path.append(all_rows[k][j])
    return path[::-1]


def sliding_decode(P, F, w_em, w_tr, T, N, i_start, i_end, mode='midpoint', L=15):
    """Real-time matching. Window = last T points; the decision for point i is made at time
    i+N (a delay of N points), so N 'future' points can correct it. Decisions never change.
    Needs N <= T-1. Returns {point index: candidate row}."""
    assert 0 <= N <= T - 1, 'buffer N must satisfy 0 <= N <= T-1'
    final = {}
    path, s = None, i_start
    for t in range(i_start, i_end + 1):
        s = max(i_start, t - T + 1)
        if s > i_start:
            prev_row = final[s - 1]
            past = [P.c_edge[final[j]] for j in range(max(i_start, s - L), s)]
            prev_hist = _hist_from_edges(past, L)
        else:
            prev_row, prev_hist = None, None
        path = decode_segment(P, F, s, t, w_em, w_tr, prev_row, prev_hist, mode, L)
        d = t - N
        if d >= i_start:
            final[d] = path[d - s]
    for d in range(max(i_start, i_end - N + 1), i_end + 1):      # flush the last N points
        final[d] = path[d - s]
    return final


def nearest_road(P, i_start, i_end):
    """Baseline with no HMM: pick the closest candidate road for every point."""
    return {i: P.st[i] + int(np.argmin(P.c_dist[P.st[i]:P.en[i]])) for i in range(i_start, i_end + 1)}


def accuracy(P, chosen, true_edge, i_start, i_end):
    """Fraction of points whose chosen road equals the true road (1 - the paper's error E)."""
    if isinstance(chosen, dict):
        pred = np.array([P.c_edge[chosen[i]] for i in range(i_start, i_end + 1)])
    else:
        pred = np.array([P.c_edge[r] for r in chosen])
    return float((pred == np.asarray(true_edge)[i_start:i_end + 1]).mean())
# sliding-window Viterb
