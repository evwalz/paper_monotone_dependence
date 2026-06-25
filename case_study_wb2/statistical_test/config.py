# config.py - Central configuration for the statistical_test pipeline (Python acor_test)
import os
from pathlib import Path

# Anchor to this file so defaults work no matter the shell cwd (unlike ``./`` or ``../``).
_STATISTICAL_TEST = Path(__file__).resolve().parent
_CASE_STUDY_WB2 = _STATISTICAL_TEST.parent

# Forecast members (prefix names in WeatherBench2-style zarr filenames)
FORECAST_NAMES = ["graphcast_ifs", "ifs_hres"]

# Variable / lead (must match ``case_study_wb2/download_data.py`` output under DATA_DIR)
PRECIP_VARIABLE = "total_precipitation_24hr"
LEAD_TIME_HOURS = 24

# Grid resolution is not stored in these filenames; it is whatever your zarr stores contain.
# Default ``download_data.py`` uses ``--resolution 240x121`` (equatorial 240×121).
WB2_RESOLUTION = "240x121"

# Same default tree as ``download_data.py`` (``fct_data/`` under ``case_study_wb2/``).
DATA_DIR = str(_CASE_STUDY_WB2 / "fct_data")
# ``outputs/`` = full-grid grids (``.npz`` / ``.txt``). ``plots/`` = PDFs (same name as ``../plots/`` in WB2).
OUTPUTS_DIR = str(_STATISTICAL_TEST / "outputs")
PLOTS_DIR = str(_STATISTICAL_TEST / "plots")

# Passed explicitly to ``acor_test(..., method=..., variance=...)`` (see ``full_grid_acor_test.py``).
ACOR_VARIANCE_DEFAULT = "plugin"

OBS_PATH = str(
    Path(DATA_DIR) / f"era5_obs_{PRECIP_VARIABLE}_2020.zarr"
)


def forecast_zarr_path(forecast_dir: str, forecast_name: str) -> str:
    """Local zarr path for a forecast member (matches default ``download_data.py`` naming)."""
    stem = f"{forecast_name}_{PRECIP_VARIABLE}_{LEAD_TIME_HOURS}h_2020"
    return os.path.join(forecast_dir, f"{stem}.zarr")
