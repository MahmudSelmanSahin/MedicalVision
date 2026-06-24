"""
===========================================================================
HIPERPARAMETRE OPTIMIZASYONU - Grid | Random | Bayesian(Optuna)
===========================================================================
SYZ2026 / MedicalVision

Tek arayuz: run_hpo(...) -> (best_hp, best_score, all_trials)
Her denemede pipeline YENIDEN kurulur (hp enjekte edilir) ve cv_evaluate ile
CV-MCC ortalamasi hesaplanir. Bu, sarmalanmis estimator'larda parametre-yolu
sorununu ve sizintiyi onler. HPO sirasinda CV ucuz tutulur (n_repeats=1).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.registry import build_pipeline          # noqa: E402
from validation.cv import cv_evaluate               # noqa: E402

# Model bazli arama uzaylari (clf parametre isimleri) -----------------------
SPACES: dict[str, dict] = {
    "xgboost": {"max_depth": [3, 4, 6], "learning_rate": [0.03, 0.1, 0.2],
                "n_estimators": [100, 200], "subsample": [0.8, 1.0]},
    "lightgbm": {"max_depth": [-1, 4, 8], "learning_rate": [0.03, 0.1, 0.2],
                 "n_estimators": [100, 200], "num_leaves": [15, 31, 63]},
    "catboost": {"depth": [3, 4, 6], "learning_rate": [0.03, 0.1, 0.2]},
    "random_forest": {"max_depth": [None, 6, 12], "min_samples_leaf": [1, 2, 4]},
    "extra_trees": {"max_depth": [None, 6, 12], "min_samples_leaf": [1, 2, 4]},
    "adaboost": {"n_estimators": [50, 100, 200], "learning_rate": [0.5, 1.0]},
    "svm": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]},
    "knn": {"n_neighbors": [3, 5, 11, 15], "weights": ["uniform", "distance"]},
    "logreg": {"C": [0.01, 0.1, 1.0, 10.0]},
}


def _score(model, scenario, ablation, X, y, hp, seed, augment_fn=None):
    pipe = build_pipeline(model, scenario, ablation, y, seed=seed, hp=hp)
    _, _, info = cv_evaluate(pipe, X, y, n_splits=5, n_repeats=1, seed=seed,
                             augment_fn=augment_fn)
    return info.get("cv_mcc_mean", -1.0)


def _grid(space):
    keys = list(space)
    for combo in itertools.product(*[space[k] for k in keys]):
        yield dict(zip(keys, combo))


def run_hpo(model, scenario, ablation, X, y, *, method="random",
            n_iter=15, seed=42, augment_fn=None):
    space = SPACES.get(model, {})
    if not space:
        return {}, _score(model, scenario, ablation, X, y, {}, seed, augment_fn), []

    trials = []

    def evaluate(hp):
        s = _score(model, scenario, ablation, X, y, hp, seed, augment_fn)
        trials.append({"hp": hp, "score": s})
        return s

    if method == "grid":
        for hp in _grid(space):
            evaluate(hp)

    elif method == "bayesian":
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            hp = {}
            for k, vals in space.items():
                hp[k] = trial.suggest_categorical(k, vals)
            return evaluate(hp)

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_iter, show_progress_bar=False)

    else:  # random
        rng = np.random.RandomState(seed)
        seen = set()
        for _ in range(n_iter):
            hp = {k: vals[rng.randint(len(vals))] for k, vals in space.items()}
            key = tuple(sorted((k, str(v)) for k, v in hp.items()))
            if key in seen:
                continue
            seen.add(key)
            evaluate(hp)

    best = max(trials, key=lambda t: t["score"])
    return best["hp"], best["score"], trials
