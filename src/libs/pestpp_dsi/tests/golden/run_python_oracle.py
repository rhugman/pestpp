#!/usr/bin/env python3
"""Generate pyemu DSI golden artefacts for pestpp_dsi C++ parity tests.

Plan §10.2: pre-train pyemu's NormalScoreTransformer + DSI on a fixed
seed, dump the internals, and let the C++ side load them and assert
pmat / predict() agreement to 1e-8 max relative.

We sidestep cross-compiler `std::normal_distribution` non-portability
(plan §5.2) by serialising the per-column z_scores produced by pyemu's
sample-then-sort generator. The C++ side pre-seeds those into its
NormalScoreTransform via `set_state(...)` and skips its own RNG.

Usage:
    python run_python_oracle.py <output_dir>

Outputs (whitespace-delimited text, %.17g precision):
    meta.txt           : single line "n_train n_obs n_components n_test"
    train.txt          : (n_train x n_obs) training data, raw scale
    latent.txt         : (n_test x n_components) latent pvals
    predicted.txt      : (n_test x n_obs) pyemu's DSI.predict(latent)
    ovals.txt          : (n_obs,) DSI.ovals
    s_kept.txt         : (n_components,) DSI.s
    pmat.txt           : (n_obs x n_components) DSI.pmat
    ns_<obs>.txt       : (n_train x 2) z_scores, originals — one per column
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure vendored pyemu is preferred over any system install.
# Script lives at pestpp/src/libs/pestpp_dsi/tests/golden/<this>.py.
# Walk parents until we find the dependencies/pyemu directory.
_HERE = Path(__file__).resolve()
_PYEMU = None
for _parent in _HERE.parents:
    candidate = _parent / "dependencies" / "pyemu" / "pyemu" / "__init__.py"
    if candidate.exists():
        _PYEMU = _parent / "dependencies" / "pyemu"
        break
if _PYEMU is None:
    raise SystemExit("could not locate dependencies/pyemu relative to this script")
sys.path.insert(0, str(_PYEMU))

import numpy as np
import pandas as pd
from pyemu.emulators import DSI


SEED = 30
N_TRAIN = 50
N_OBS = 25            # smaller than the plan's 100 to keep ns_*.txt count
                      # tractable while still exercising the full path
N_TEST = 10


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    obs_names = [f"o{j:03d}" for j in range(N_OBS)]
    train = rng.standard_normal((N_TRAIN, N_OBS))
    train_df = pd.DataFrame(train, columns=obs_names)

    # Force numpy's legacy global RNG to a fixed seed so pyemu's
    # sample-then-sort z-score generator is reproducible.
    np.random.seed(SEED)

    dsi = DSI(
        data=train_df,
        transforms=[{"type": "normal_score", "columns": list(obs_names)}],
        rowwise_groups=None,
        energy_threshold=1.0,
    )
    dsi.fit()

    # Hand the already-trained pipeline a small held-out batch in
    # latent space and capture predict() output.
    n_components = dsi.pmat.shape[1]
    latent_rng = np.random.default_rng(SEED + 1)
    latent = latent_rng.standard_normal((N_TEST, n_components))
    latent_df = pd.DataFrame(
        latent,
        index=[f"r{i}" for i in range(N_TEST)],
        columns=[f"p_{i}" for i in range(n_components)],
    )
    predicted = dsi.predict(latent_df)
    if isinstance(predicted, pd.Series):
        predicted = predicted.to_frame().T

    # --- Write artefacts in DSI obs order (what pmat rows correspond to) ---
    dsi_obs = list(dsi.ovals.index) if hasattr(dsi.ovals, "index") else list(obs_names)
    assert dsi_obs == obs_names, "DSI rearranged columns; oracle assumes identity order"

    ovals = (
        dsi.ovals.values if hasattr(dsi.ovals, "values") else np.asarray(dsi.ovals)
    )
    pmat = np.asarray(dsi.pmat)
    s_kept = np.asarray(dsi.s)

    # pyemu's TransformerPipeline.inverse_transform casts to float32
    # internally (transformers.py:743 + 753) — that downcast costs ~1e-7
    # relative precision and is plausibly an upstream bug. To validate
    # the native C++ DSI's math without inheriting that defect, we
    # also dump a "high-precision" predict that runs each transformer
    # stage inverse manually in float64. Plan §2.3 — "we don't improve
    # DSI's underlying math" — applies; this just gives us a tighter
    # reference for the golden test.
    sim_t_arr = ovals[:, None] + pmat @ latent.T   # (n_obs, n_test) float64
    sim_t_df = pd.DataFrame(sim_t_arr.T, columns=obs_names,
                            index=latent_df.index)
    pipe_obj = dsi.transformer_pipeline
    predicted_hp = sim_t_df.copy()
    for stage_obj, cols in reversed(pipe_obj.pipeline.transformers):
        cols_to = cols if cols is not None else list(predicted_hp.columns)
        valid_cols = [c for c in cols_to if c in predicted_hp.columns]
        sub = predicted_hp[valid_cols].copy()        # stays float64
        inv = stage_obj.inverse_transform(sub)       # float64 ndarray
        predicted_hp.loc[:, valid_cols] = np.asarray(inv, dtype=np.float64)
    np.savetxt(out / "predicted_hp.txt",
               predicted_hp.loc[:, obs_names].values, fmt="%.17g")

    # Intermediate sim_t in NS-z space — what pyemu computes via
    #   ovals[:, None] + pmat @ latent.T  (n_obs, n_test)
    # before applying NS inverse. Lets the C++ test isolate any
    # SVD-vs-NS-inverse contribution to disagreement.
    sim_t_py = ovals[:, None] + pmat @ latent.T   # (n_obs, n_test)
    sim_t_py = sim_t_py.T                          # (n_test, n_obs)

    np.savetxt(out / "train.txt", train, fmt="%.17g")
    np.savetxt(out / "latent.txt", latent, fmt="%.17g")
    np.savetxt(out / "predicted.txt",
               predicted.loc[:, obs_names].values, fmt="%.17g")
    np.savetxt(out / "sim_t.txt", sim_t_py, fmt="%.17g")
    np.savetxt(out / "ovals.txt", ovals.reshape(-1), fmt="%.17g")
    np.savetxt(out / "s_kept.txt", s_kept.reshape(-1), fmt="%.17g")
    np.savetxt(out / "pmat.txt", pmat, fmt="%.17g")

    # --- NS per-column state ---
    # transformer_pipeline is an AutobotsAssemble; its inner pipeline
    # stores (transformer_obj, columns) tuples.
    ns_stage = None
    pipe = dsi.transformer_pipeline
    if pipe is not None and getattr(pipe, "pipeline", None) is not None:
        for stage_obj, _cols in pipe.pipeline.transformers:
            if hasattr(stage_obj, "column_parameters") and stage_obj.column_parameters:
                ns_stage = stage_obj
                break
    assert ns_stage is not None, "Could not locate NormalScoreTransformer in pipeline"
    for j, col in enumerate(obs_names):
        params = ns_stage.column_parameters[col]
        z = np.asarray(params["z_scores"]).reshape(-1)
        o = np.asarray(params["originals"]).reshape(-1)
        assert z.shape == o.shape, f"{col}: z/o shape mismatch"
        stacked = np.column_stack([z, o])
        np.savetxt(out / f"ns_{col}.txt", stacked, fmt="%.17g")

    # --- meta.txt ---
    with open(out / "meta.txt", "w", encoding="utf-8") as f:
        f.write(f"{N_TRAIN} {N_OBS} {n_components} {N_TEST}\n")
        for nm in obs_names:
            f.write(f"{nm}\n")

    print(f"OK  wrote oracle artefacts to {out}")
    print(f"    n_train={N_TRAIN} n_obs={N_OBS} n_components={n_components} n_test={N_TEST}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
