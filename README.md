# Real-Time HMM Map Matching (UE24CS352A ML mini-project)

Real-time map matching with a Hidden Markov Model and Viterbi, based on the Stanford CS229 paper "Real-Time Dense Map Matching with Naive Hidden Markov Models: Delay versus Accuracy". Run on the Seattle (Newson & Krumm) dataset, plus a small change of our own.

**Team:** Anushka (PES1UG24CS074) and [Person A name, SRN] — PES University

## What it does
GPS points are noisy. Map matching snaps each point to the road the vehicle was really on.
- Hidden state = road segment, observation = GPS point.
- Emission clues: distance to road, heading vs road direction, speed vs speed limit.
- Transition clues: no backtracking, distance between GPS points vs distance between roads.
- Viterbi finds the best whole chain of roads; a sliding window (T) with buffer (N) makes it real-time.
- Clue weights are learned from labelled transitions with an L1 linear classifier (paper Sec. 4.3).

## Our change
| | Paper (baseline) | Ours |
|---|---|---|
| A. Weight learner | L1 SVM | L1 logistic regression |
| B. Distance feature | Gaussian exp(-d²/2σ²) | Hyperbolic secant 1/cosh(πd/2σ) |

Run as a 2×2 experiment (`notebooks/02_model_and_experiments.ipynb`, section 7).
The switches are `make_model(kind)` in `src/train_weights.py` and `P.F_gauss` / `P.F_sech` in `src/features.py`.

## Results (test = last 30% of the drive, ~2,260 points)
| Method | Distance | Accuracy | Error |
|---|---|---|---|
| Nearest road (no HMM) | - | 91.99% | 8.01% |
| HMM, hand-picked weights | Gaussian | 95.04% | 4.96% |
| **L1 SVM (baseline)** | Gaussian | 95.00% | 5.00% |
| L1 SVM | Sech (B) | 95.04% | 4.96% |
| L1 logistic regression (A) | Gaussian | 94.95% | 5.05% |
| L1 logistic regression (A) | Sech (B) | 95.09% | 4.91% |

- The HMM cuts error from 8.0% to about 5.0% versus nearest-road.
- Our changes are within noise (spread 0.14 points ≈ 3 test points): the method is insensitive to the classifier and the exact distance score.
- Delay vs accuracy (T=15): error 7.3% at N=0, 5.6% at N=3, 5.0% at N=14. See `results/error_vs_buffer.png`.

Absolute numbers are not comparable with the paper (different city/data); we compare only against our own baseline.

## How to run
1. Get the data (see `data/README.md`) into a Drive folder.
2. Run `notebooks/01_data_and_candidates.ipynb` (builds `gps.pkl`, `roads.pkl`, `candidates.pkl`, `truth_per_point.pkl`).
3. Run `notebooks/02_model_and_experiments.ipynb` (writes `src/`, trains, evaluates, saves `results/`).

## Layout
- `src/` – data loading, candidates, features, Viterbi (full + sliding window), weight learning, evaluation
- `notebooks/` – data preparation and experiments (Colab)
- `results/` – results table, delay-vs-accuracy data and plot, demo map
- `report/` – 2-page write-up
