#!/usr/bin/env python3
"""
Full lat–lon grid pairwise hypothesis tests for CMA and CID (WeatherBench2 zarr → ``acor_test``).

- Loads two forecasts + ERA5 obs, runs ``acor``’s ``acor_test`` at every valid grid cell.
- ``--method cma`` (default) or ``--method cid``; outputs are tagged so you can run both.

Use ``--check_zarr`` to only load and print array shapes (no inference).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import xarray as xr

from acor import acor_test
from config import (  # type: ignore
    ACOR_VARIANCE_DEFAULT,
    DATA_DIR,
    FORECAST_NAMES,
    LEAD_TIME_HOURS,
    OBS_PATH,
    OUTPUTS_DIR,
    PRECIP_VARIABLE,
    forecast_zarr_path,
)

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **kwargs):
        return it


def standardize_dims(dataset: xr.Dataset) -> xr.Dataset:
    dim_mapping = {}
    if "time" not in dataset.dims:
        time_candidates = [d for d in dataset.dims if "time" in d.lower()]
        if time_candidates:
            dim_mapping[time_candidates[0]] = "time"
    if "latitude" not in dataset.dims and "lat" in dataset.dims:
        dim_mapping["lat"] = "latitude"
    if "longitude" not in dataset.dims and "lon" in dataset.dims:
        dim_mapping["lon"] = "longitude"
    return dataset.rename(dim_mapping) if dim_mapping else dataset


def make_latitude_increasing(dataset: xr.Dataset) -> xr.Dataset:
    lat_name = next((d for d in ("latitude", "lat") if d in dataset.dims), None)
    if lat_name is None:
        raise ValueError("No latitude dimension found")
    lat = dataset[lat_name].values
    if (np.diff(lat) < 0).all():
        dataset = dataset.sel({lat_name: lat[::-1]})
    return dataset


def open_zarr_fct_all(forecast_path: str, obs_path: str, forecast_names: List[str]):
    obs = make_latitude_increasing(standardize_dims(xr.open_zarr(obs_path)))

    def load_fct(name: str):
        zpath = forecast_zarr_path(forecast_path, name)
        if not os.path.isdir(zpath):
            raise FileNotFoundError(
                f"Forecast zarr not found (expected WeatherBench2 layout): {zpath}"
            )
        return make_latitude_increasing(standardize_dims(xr.open_zarr(zpath)))

    return load_fct(forecast_names[0]), load_fct(forecast_names[1]), obs


def load_aligned_precip_arrays(forecast_path, obs_path, forecast_names):
    f1, f2, obs = open_zarr_fct_all(forecast_path, obs_path, forecast_names)
    forecast_times = f1.time.values
    obs_times = obs.time.values
    valid_f = [
        t for t in forecast_times if t + np.timedelta64(LEAD_TIME_HOURS, "h") in obs_times
    ]
    valid_o = [t + np.timedelta64(LEAD_TIME_HOURS, "h") for t in valid_f]
    filtered_obs = obs.sel(time=valid_o)
    latitudes = obs.latitude.values
    longitudes = obs.longitude.values
    var = PRECIP_VARIABLE
    var_obs = filtered_obs[var].values

    def slice_fct(fc):
        s = fc.sel(time=valid_f).assign_coords(time=valid_o)
        return s[var].values

    fct1, fct2 = slice_fct(f1), slice_fct(f2)
    n_lat, n_lon = len(latitudes), len(longitudes)
    if var_obs.shape[1] != n_lat or var_obs.shape[2] != n_lon:
        print("Transposing (time, lon, lat) -> (time, lat, lon)")
        var_obs = np.transpose(var_obs, (0, 2, 1))
        fct1 = np.transpose(fct1, (0, 2, 1))
        fct2 = np.transpose(fct2, (0, 2, 1))
    return {
        "fct1": fct1,
        "fct2": fct2,
        "var_obs": var_obs,
        "latitudes": latitudes,
        "longitudes": longitudes,
        "n_lat": n_lat,
        "n_lon": n_lon,
        "forecast_names": list(forecast_names),
    }


def acor_pairwise_from_python(
    y: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    method: str = "cma",
    iid: bool = False,
    conf_level: float = 0.95,
    alternative: str = "two.sided",
    variance: str = ACOR_VARIANCE_DEFAULT,
    fisher: bool = False,
):
    y = np.asarray(y, dtype=np.float64).ravel()
    x1 = np.asarray(x1, dtype=np.float64).ravel()
    x2 = np.asarray(x2, dtype=np.float64).ravel()
    n = y.size
    if x1.size != n or x2.size != n:
        raise ValueError("y, x1, x2 must have the same length")
    if n < 3:
        raise ValueError("acor_test requires at least 3 observations")

    X = np.column_stack([x1, x2])
    res = acor_test(
        X,
        y,
        method=method,
        alternative=alternative,
        conf_level=conf_level,
        iid=iid,
        fisher=fisher,
        variance=variance,
    )

    est = np.asarray(res.estimate, dtype=np.float64).ravel()
    if est.size != 2:
        raise RuntimeError(f"expected 2 estimates, got shape {est.shape}")

    p_ind = np.asarray([res.results[0]["pvalue"], res.results[1]["pvalue"]], dtype=np.float64)
    pw = res.pairwise_results[0]
    ci_lo = float(pw["ci_lower"])
    ci_hi = float(pw["ci_upper"])
    global_p = float(res.pvalue)

    V = np.asarray(res.variance, dtype=np.float64)
    if V.shape != (2, 2):
        raise RuntimeError(f"expected 2x2 variance matrix, got {V.shape}")
    S = V / n
    var_diff = float(S[0, 0] - S[0, 1] - S[1, 0] + S[1, 1])

    return est, p_ind, np.array([ci_lo, ci_hi]), global_p, var_diff


def run_acor_grid(
    fct1: np.ndarray,
    fct2: np.ndarray,
    var_obs: np.ndarray,
    forecast_names: List[str],
    output_dir: str = OUTPUTS_DIR,
    *,
    method: str = "cma",
    iid: bool = False,
    conf_level: float = 0.95,
    alternative: str = "two.sided",
    variance: str = ACOR_VARIANCE_DEFAULT,
    fisher: bool = False,
    lat_slice: Optional[slice] = None,
    lon_slice: Optional[slice] = None,
    save_txt: bool = True,
) -> Dict[str, np.ndarray]:
    os.makedirs(output_dir, exist_ok=True)

    n_lat, n_lon = int(var_obs.shape[1]), int(var_obs.shape[2])
    if fct1.shape != var_obs.shape or fct2.shape != var_obs.shape:
        raise ValueError(
            f"shape mismatch: var_obs {var_obs.shape}, fct1 {fct1.shape}, fct2 {fct2.shape}"
        )

    lat_range = range(n_lat) if lat_slice is None else range(n_lat)[lat_slice]
    lon_range = range(n_lon) if lon_slice is None else range(n_lon)[lon_slice]

    score_fct1 = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    score_fct2 = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    p_fct1 = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    p_fct2 = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    ci_lower = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    ci_upper = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    global_grid = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    variance_grid = np.full((n_lat, n_lon), np.nan, dtype=np.float64)

    n_points = len(lat_range) * len(lon_range)
    for lat_idx in tqdm(lat_range, desc="latitude", unit="row"):
        for lon_idx in lon_range:
            fct1_series = fct1[:, lat_idx, lon_idx]
            fct2_series = fct2[:, lat_idx, lon_idx]
            obs_series = var_obs[:, lat_idx, lon_idx]

            if not (
                np.all(np.isfinite(obs_series))
                and np.all(np.isfinite(fct1_series))
                and np.all(np.isfinite(fct2_series))
            ):
                continue

            try:
                scores, pvals, ci, g_p, vdiff = acor_pairwise_from_python(
                    obs_series,
                    fct1_series,
                    fct2_series,
                    method=method,
                    iid=iid,
                    conf_level=conf_level,
                    alternative=alternative,
                    variance=variance,
                    fisher=fisher,
                )
                score_fct1[lat_idx, lon_idx] = scores[0]
                score_fct2[lat_idx, lon_idx] = scores[1]
                p_fct1[lat_idx, lon_idx] = pvals[0]
                p_fct2[lat_idx, lon_idx] = pvals[1]
                ci_lower[lat_idx, lon_idx] = ci[0]
                ci_upper[lat_idx, lon_idx] = ci[1]
                global_grid[lat_idx, lon_idx] = g_p
                variance_grid[lat_idx, lon_idx] = vdiff
            except Exception as e:
                print(f"Error at lat {lat_idx}, lon {lon_idx}: {e}", file=sys.stderr)

    name_fct1, name_fct2 = forecast_names[0], forecast_names[1]

    np.savez(
        os.path.join(output_dir, f"full_grid_{method}_python.npz"),
        score_fct1=score_fct1,
        score_fct2=score_fct2,
        p_fct1=p_fct1,
        p_fct2=p_fct2,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        global_grid=global_grid,
        variance=variance_grid,
        forecast_names=np.array(forecast_names, dtype=object),
    )

    if save_txt:
        np.savetxt(os.path.join(output_dir, f"{method}_{name_fct1}.txt"), score_fct1)
        np.savetxt(os.path.join(output_dir, f"{method}_{name_fct2}.txt"), score_fct2)
        np.savetxt(os.path.join(output_dir, f"p_{method}_{name_fct1}.txt"), p_fct1)
        np.savetxt(os.path.join(output_dir, f"p_{method}_{name_fct2}.txt"), p_fct2)
        np.savetxt(
            os.path.join(output_dir, f"ci_lower_{method}_{name_fct1}_{name_fct2}.txt"),
            ci_lower,
        )
        np.savetxt(
            os.path.join(output_dir, f"ci_upper_{method}_{name_fct1}_{name_fct2}.txt"),
            ci_upper,
        )
        np.savetxt(
            os.path.join(output_dir, f"p_global_{method}_{name_fct1}_{name_fct2}.txt"),
            global_grid,
        )
        np.savetxt(
            os.path.join(output_dir, f"variance_{method}_{name_fct1}_{name_fct2}.txt"),
            variance_grid,
        )

    if lat_slice is None and lon_slice is None:
        print(f"Processed full grid ({n_points} points). Saved under {output_dir}")
    else:
        print(f"Processed {n_points} grid points (subset). Saved under {output_dir}")
    return {
        "score_fct1": score_fct1,
        "score_fct2": score_fct2,
        "p_fct1": p_fct1,
        "p_fct2": p_fct2,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "global_grid": global_grid,
        "variance": variance_grid,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run acor_test on full lat-lon grid (pure Python) for CMA or CID."
    )
    parser.add_argument("--forecast_dir", type=str, default=DATA_DIR)
    parser.add_argument("--obs_path", type=str, default=OBS_PATH)
    parser.add_argument("--output_dir", type=str, default=OUTPUTS_DIR)
    parser.add_argument(
        "--method",
        type=str,
        choices=("cma", "cid"),
        default="cma",
        help="Monotone method passed to acor_test: cma (default) or cid (C-index / pairwise).",
    )
    parser.add_argument("--iid", action="store_true", help="Pass iid=True to acor_test")
    parser.add_argument("--conf_level", type=float, default=0.95)
    parser.add_argument("--alternative", type=str, default="two.sided")
    parser.add_argument(
        "--variance",
        type=str,
        choices=("ij", "plugin"),
        default=ACOR_VARIANCE_DEFAULT,
        help="acor_test variance (default: plugin)",
    )
    parser.add_argument("--fisher", action="store_true")
    parser.add_argument("--no_save_txt", action="store_true")
    parser.add_argument("--check_zarr", action="store_true")
    args = parser.parse_args()

    print(f"Loading zarr: forecast_dir={args.forecast_dir!r}, obs={args.obs_path!r}")
    data = load_aligned_precip_arrays(args.forecast_dir, args.obs_path, FORECAST_NAMES)
    print(
        f"Grid size: {data['n_lat']} lat x {data['n_lon']} lon "
        f"= {data['n_lat'] * data['n_lon']} points; fct1 shape {data['fct1'].shape}"
    )
    if args.check_zarr:
        return

    print(
        f"Running acor_test grid: method={args.method!r}, variance={args.variance!r}, "
        f"alternative={args.alternative!r}, iid={args.iid}"
    )
    run_acor_grid(
        data["fct1"],
        data["fct2"],
        data["var_obs"],
        list(data["forecast_names"]),
        output_dir=args.output_dir,
        method=args.method,
        iid=args.iid,
        conf_level=args.conf_level,
        alternative=args.alternative,
        variance=args.variance,
        fisher=args.fisher,
        save_txt=not args.no_save_txt,
    )


if __name__ == "__main__":
    main()
