"""Learn the feature weights from labelled transitions (paper Section 4.3).

Every pair (road a at point i-1) -> (road b at point i) is one training example with 5 features
[emission f1, f2, f3 of b, transition omega1, omega2]. Label +1 = the true transition, -1 = all others
(only about 5-10% of examples are +1, so class weights are used)."""
import warnings
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import f1_score
from features import trans_feats

warnings.filterwarnings('ignore', message='.*penalty.*')   # newer scikit-learn nags about 'penalty'

C_GRID = [0.01, 0.1, 1]        # the paper found C = 1 best


def build_pairs(P, F, true_edge, i_lo, i_hi, mode='midpoint', L=15):
    """Training examples for points i_lo..i_hi. Steps where the true road is not among the
    candidates are skipped. Returns X (n,5), y (n,) with y in {+1,-1}."""
    true_edge = np.asarray(true_edge, dtype=np.int64)
    Xs, ys = [], []
    for i in range(max(i_lo, 1), i_hi + 1):
        ra = np.arange(P.st[i - 1], P.en[i - 1])
        rb = np.arange(P.st[i], P.en[i])
        ea, eb = P.c_edge[ra], P.c_edge[rb]
        ta = np.where(ea == true_edge[i - 1])[0]
        tb = np.where(eb == true_edge[i])[0]
        if len(ta) == 0 or len(tb) == 0:
            continue
        # recent history = the TRUE path before point i-1, plus the predecessor itself
        past = true_edge[max(0, i - L):i - 1]
        base = np.full(L - 1, -1, dtype=np.int64)
        if len(past):
            base[L - 1 - len(past):] = past
        H = np.hstack([np.tile(base, (len(ra), 1)), ea[:, None]])
        om1, om2 = trans_feats(P, ra, rb, i, H, mode)
        na, nb = len(ra), len(rb)
        X = np.empty((na * nb, 5))
        X[:, :3] = np.tile(F[rb], (na, 1))
        X[:, 3] = om1.ravel()
        X[:, 4] = om2.ravel()
        y = -np.ones((na, nb))
        y[ta[0], tb[0]] = 1
        Xs.append(X)
        ys.append(y.ravel())
    return np.vstack(Xs), np.concatenate(ys)


def make_model(kind):
    """'svm' = the paper's L1 SVM (baseline); 'logreg' = change A (L1 logistic regression)."""
    if kind == 'svm':
        return LinearSVC(penalty='l1', loss='squared_hinge', dual=False,
                         class_weight='balanced', tol=1e-3, max_iter=20000)
    if kind == 'logreg':
        return LogisticRegression(penalty='l1', solver='liblinear',
                                  class_weight='balanced', tol=1e-3, max_iter=2000)
    raise ValueError(kind)


def fit_model(X, y, kind, c_grid=C_GRID, folds=5):
    """Tune C by cross-validation on contiguous blocks (no shuffling: neighbouring GPS points
    are almost identical, so shuffled folds would leak). Returns (fitted model, best C)."""
    gs = GridSearchCV(make_model(kind), {'C': list(c_grid)}, scoring='f1',
                      cv=KFold(n_splits=folds, shuffle=False))
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_['C']


def to_weights(coef):
    """Turn model coefficients into HMM weights: drop negatives, then rescale each group
    (3 emission, 2 transition) to sum to 1, as in the paper (||w||_1 = 1)."""
    w = np.clip(np.asarray(coef, dtype=float).ravel(), 0, None)

    def norm(v):
        s = v.sum()
        return v / s if s > 0 else np.ones(len(v)) / len(v)

    return norm(w[:3]), norm(w[3:])


def classifier_f1(model, X, y):
    return float(f1_score(y, model.predict(X), pos_label=1))
# SVM baseline and logistic regression weight learnin
