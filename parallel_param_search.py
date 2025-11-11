"""Two-stage coarse-to-fine grid search and ablation for the event-clustering
pipeline."""

from __future__ import annotations

import os
import random
import pickle
from itertools import product, combinations
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from event_clustering_pipeline import run_meta_pipeline, reseed

GROUPS = ["object", "infrastructure", "toponym", "actor", "organization", "event"]
TYPE_PAIRS_ALL = [p for p in combinations(GROUPS, 2)]

# geo
DIST_COARSE = [100, 2_000, 8_000, 16_000]
GEO_ALPHA_COARSE = [0.5, 1.5, 2.5]
DIST_FINE_STEP, GEO_ALPHA_FINE_STEP = 500, 0.5
DIST_BOUNDS, GEO_ALPHA_BOUNDS = (100, 20_000), (0.5, 3.0)

# semantic
SEM_DTHR_COARSE, SEM_ALPHA_COARSE = [0.2, 0.6, 1.0], [0.5, 1.5, 2.5]
SEM_DTHR_FINE_STEP, SEM_ALPHA_FINE_STEP = 0.2, 0.5
SEM_DTHR_BOUNDS, SEM_ALPHA_BOUNDS = (0.2, 1.0), (0.5, 3.0)

TOP_K, SEED = 5, 42
SCORE_KEY: str | None = None

TIME_INTERVAL, WINDOW, STEP = "week", "1w", "1w"
COMM_PARAMS = {"resolution": 0.8, "random_state": SEED}
SIM_THR, TEXT_ALPHA = 0.5, 0.4

DATA_KWARGS = dict(
    nodes_csv="entities.csv",
    relations_csv="events2012_data_20k.csv",
    texts_csv="data_20k.csv",
    community_method="leiden",
)

reseed(SEED)


def neighbourhood(val: float, step: float, lo: float, hi: float, *, is_int: bool = False) -> List[float]:
    vals = {val - step, val, val + step}
    if is_int:
        return sorted({max(lo, min(hi, int(v))) for v in vals})
    return sorted({round(max(lo, min(hi, v)), 3) for v in vals})


def make_intra(def_thr: float, alpha: float, include: List[str] = GROUPS) -> Dict[str, Any]:
    return {
        "include_types": include,
        "default_thr": def_thr,
        "alpha": alpha,
        "model_name": "sentence-transformers/paraphrase-MiniLM-L12-v2",
        "device": None,
        "fp16": True,
        "top_k": 256,
    }


def make_cross(alpha: float, pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
    return {
        "type_pairs": pairs,
        "sim_threshold": 0.5,
        "alpha": alpha,
        "model_name": "sentence-transformers/paraphrase-MiniLM-L12-v2",
        "device": None,
        "fp16": True,
        "top_k": 128,
    }

def run_pipeline(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    reseed(SEED)
    _, scores, params = run_meta_pipeline(**kwargs)
    return {"scores": scores, "params": params, **kwargs}

def build_stage1() -> List[Dict[str, Any]]:
    jobs = []
    for d_thr, g_a, s_thr, s_a in product(
        DIST_COARSE, GEO_ALPHA_COARSE, SEM_DTHR_COARSE, SEM_ALPHA_COARSE
    ):
        jobs.append(
            dict(
                DATA_KWARGS,
                interval=TIME_INTERVAL,
                window=WINDOW,
                step=STEP,
                stage="stage1",
                community_params=COMM_PARAMS,
                geo_edges={"dist_thr": d_thr, "alpha": g_a, "cache_file": "toponyms_cache.csv"},
                intra_type_edges=make_intra(s_thr, s_a),
                cross_type_edges=make_cross(s_a, TYPE_PAIRS_ALL),
                sim_thr=SIM_THR,
                text_alpha=TEXT_ALPHA,
            )
        )
    return jobs


def build_stage2(top_cfgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    jobs = []
    seen: set[Tuple] = set()
    for cfg in top_cfgs:
        geo = cfg["geo_edges"]
        sem = cfg["intra_type_edges"]
        d_vals = neighbourhood(geo["dist_thr"], DIST_FINE_STEP, *DIST_BOUNDS, is_int=True)
        g_vals = neighbourhood(geo["alpha"], GEO_ALPHA_FINE_STEP, *GEO_ALPHA_BOUNDS)
        t_vals = neighbourhood(sem["default_thr"], SEM_DTHR_FINE_STEP, *SEM_DTHR_BOUNDS)
        s_vals = neighbourhood(sem["alpha"], SEM_ALPHA_FINE_STEP, *SEM_ALPHA_BOUNDS)
        for d, g_a, thr, s_a in product(d_vals, g_vals, t_vals, s_vals):
            key = (d, g_a, thr, s_a)
            if key in seen:
                continue
            seen.add(key)
            jobs.append(
                dict(
                    DATA_KWARGS,
                    interval=TIME_INTERVAL,
                    window=WINDOW,
                    step=STEP,
                    stage="stage2",
                    community_params=COMM_PARAMS,
                    geo_edges={"dist_thr": d, "alpha": g_a, "cache_file": "toponyms_cache.csv"},
                    intra_type_edges=make_intra(thr, s_a),
                    cross_type_edges=make_cross(s_a, TYPE_PAIRS_ALL),
                    sim_thr=SIM_THR,
                    text_alpha=TEXT_ALPHA,
                )
            )
    return jobs


def run_jobs(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    with Pool(processes=3, initializer=reseed, initargs=(SEED,)) as pool:
        for out in tqdm(pool.imap_unordered(run_pipeline, jobs), total=len(jobs)):
            results.append(out)
    return results

def l1o_variant(base: Dict[str, Any], drop: str) -> Dict[str, Any]:
    inc = [g for g in GROUPS if g != drop]
    pairs = [p for p in TYPE_PAIRS_ALL if drop not in p]
    thr = base["intra_type_edges"]["default_thr"]
    a = base["intra_type_edges"]["alpha"]
    var = base.copy()
    var["stage"] = "ablation"
    var["intra_type_edges"] = make_intra(thr, a, inc)
    var["cross_type_edges"] = make_cross(a, pairs)
    return var


def a1i_variant(base: Dict[str, Any], only: str) -> Dict[str, Any]:
    inc = [only]
    pairs = [(only, g) for g in GROUPS if g != only]
    thr = base["intra_type_edges"]["default_thr"]
    a = base["intra_type_edges"]["alpha"]
    var = base.copy()
    var["stage"] = "ablation"
    var["intra_type_edges"] = make_intra(thr, a, inc)
    var["cross_type_edges"] = make_cross(a, pairs)
    return var


def disable_methods(cfg: Dict[str, Any], to_disable: Tuple[str, ...]) -> Dict[str, Any]:
    new = {k: v for k, v in cfg.items() if k not in {"geo_edges", "intra_type_edges", "cross_type_edges"}}
    for m in ["geo_edges", "intra_type_edges", "cross_type_edges"]:
        if m not in to_disable:
            new[m] = cfg[m]
        else:
            new[m] = None
    new["stage"] = "ablation"
    return new


if __name__ == "__main__":
    out_dir = Path("grid_search")
    out_dir.mkdir(exist_ok=True)

    stage1_res = run_jobs(build_stage1())
    SCORE_KEY = SCORE_KEY or next(iter(stage1_res[0]["scores"]))
    stage1_res.sort(key=lambda r: r["scores"][SCORE_KEY], reverse=True)
    top_cfgs = stage1_res[:TOP_K]

    stage2_res = run_jobs(build_stage2(top_cfgs))
    grid_res = stage1_res + stage2_res

    with (out_dir / "grid_search_results.pkl").open("wb") as f:
        pickle.dump(grid_res, f)

    flat_rows = []
    for row in grid_res:
        base = {
            "stage": row["stage"],
            "resolution": row["community_params"]["resolution"],
            "sim_thr": row["sim_thr"],
            "text_alpha": row["text_alpha"],
            "geo_dist": row["geo_edges"]["dist_thr"],
            "geo_alpha": row["geo_edges"]["alpha"],
            "sem_thr": row["intra_type_edges"]["default_thr"],
            "sem_alpha": row["intra_type_edges"]["alpha"],
        }
        base.update({f"score_{k}": v for k, v in row["scores"].items()})
        base.update({f"param_{k}": v for k, v in row["params"].items()})
        flat_rows.append(base)

    pd.DataFrame(flat_rows).to_csv(out_dir / "grid_search_results.csv", index=False)

    best_cfg = max(grid_res, key=lambda r: r["scores"][SCORE_KEY])
    ablation_jobs = []
    for g in GROUPS:
        ablation_jobs.append(l1o_variant(best_cfg, g))
        ablation_jobs.append(a1i_variant(best_cfg, g))

    METHODS = ["geo_edges", "intra_type_edges", "cross_type_edges"]
    for r in range(1, len(METHODS) + 1):
        for subset in combinations(METHODS, r):
            job = disable_methods(best_cfg, subset)
            job["abl_mode"] = "OFF_" + "+".join(s.split("_")[0].upper() for s in subset)
            ablation_jobs.append(job)

    ablation_res = run_jobs(ablation_jobs)

    base_scores = best_cfg["scores"]
    rows = []
    for r in ablation_res:
        mode = r.pop("abl_mode", r.get("stage", "abl"))
        delta = {f"delta_{k}": base_scores[k] - r["scores"][k] for k in base_scores}
        rows.append({"mode": mode, **r["scores"], **delta})
    pd.DataFrame(rows).to_csv(out_dir / "ablation_results.csv", index=False)

    print("Average grid metrics")
    for m in base_scores:
        avg = sum(rec["scores"][m] for rec in grid_res) / len(grid_res)
        print(f"{m}: {avg}")
    print(
        f"Finished: grid={len(grid_res)} configs (Stage1 {len(stage1_res)} + Stage2 {len(stage2_res)}), ablation={len(rows)} runs. Results -> {out_dir.resolve()}"
    )

