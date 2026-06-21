# Data Flow Before pvlib

## Why Estimated Parameters Exist

Station observation files may contain only:

```text
dtime, observe_power, observe_ghi
```

The clear-sky physics branch still needs solar geometry, pressure, PWV, and AOD-like atmospheric priors. These values are obtained as follows:

| Parameter | Source before pvlib | Training treatment |
|---|---|---|
| `mu0_target`, `apparent_zenith`, `dni_extra` | deterministic time + site geometry | valid derived physics |
| `pressure_pa` | standard atmosphere from altitude | estimated, mask `pressure_pa_is_observed=0` |
| `pwv_cm` | `PWAT * 0.1` if available, else default `1.5` | observed/joined if PWAT exists, default otherwise |
| `aod700` | constant `0.08` | default, mask `aod700_is_observed=0` |
| weather fields | native weather file or timestamp join | marked by `weather_join_status` |

The training model receives both values and masks. This keeps a unified tensor shape without pretending that estimated/default values are true station observations.

## Clear-Sky Backend Priority

When `pvlib` is installed, `clear_sky_backend=auto` uses `pvlib.clearsky.simplified_solis` and records:

```text
clear_sky_backend = pvlib_simplified_solis
```

When `pvlib` is unavailable, `auto` uses the local REST2-style backend and records:

```text
clear_sky_backend = rest2_numpy
```

The model-ready schema remains unchanged. Only clear-sky physics values are updated, especially:

```text
ghi_clear_target
dni_clear_target
dhi_clear_target
t_clr_dd
smart_persistence_ghi
```

Use `scripts/compare_clear_sky_backends.py` to inspect the difference between `pvlib`, `rest2_numpy`, and the legacy `fallback_rest2_like` implementation before training on a new site.

## Mock Data Policy

Mock data is generated in two stages:

1. Strict raw CSV files match the user-provided sample schemas.
2. Enriched CSV files are generated from raw CSV files via timestamp joining and physics feature generation.
3. Model-ready CSV files drop source-specific raw columns and keep a fixed column order for training.

This is the preferred workflow because it tests both the ingestion layer and the final model-ready table.

Recommended directories:

```text
data/mock_raw/          strict source-like CSV files
data/mock_enriched/     debug CSV files with source-specific raw columns
data/mock_model_ready/  fixed-schema training CSV files
```
