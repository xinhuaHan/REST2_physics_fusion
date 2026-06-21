# Physics Feature Contract

All generated physics CSV files must include the following columns:

```text
dtime
source_type
station_name
input_ghi
ghi_clear_target
dni_clear_target
dhi_clear_target
mu0_target
apparent_zenith
dni_extra
pressure_pa
pwv_cm
aod700
t_clr_dd
clear_sky_index_last
smart_persistence_ghi
feature_quality_flags
```

Source and missingness columns should also be preserved:

```text
input_ghi_source
temp_c_source
wind_speed_source
wind_dir_source
precip_source
sdwe_source
pwat_source
pwv_cm_source
pressure_pa_source
aod700_source
weather_join_status
temp_c_is_observed
wind_speed_is_observed
wind_dir_is_observed
precip_is_observed
sdwe_is_observed
pwat_is_observed
pwv_cm_is_observed
pressure_pa_is_observed
aod700_is_observed
weather_is_joined
```

These fields are part of the data contract. They allow the model and evaluator to distinguish true station/weather observations from joined forecasts, altitude-estimated pressure, default AOD, and default PWV.

Units:

```text
irradiance: W/m^2
temperature: Celsius
pressure_pa: Pa
pwv_cm: cm
wind_speed: m/s
wind_dir: degree
```

The model should treat `DNI/DHI` clear-sky values as physics priors, not labels.

Additional clear-sky/weather fusion priors are included in the model-ready physics vector:

```text
clear_sky_index_current
weather_attenuation_prior
weather_adjusted_clear_sky_ghi
clear_sky_residual_last
clear_sky_gap
ghi_clear_horizon_5min
ghi_clear_horizon_4h
ghi_clear_horizon_1d
mu0_horizon_5min
mu0_horizon_4h
mu0_horizon_1d
weather_adjusted_clear_sky_ghi_horizon_5min
weather_adjusted_clear_sky_ghi_horizon_4h
weather_adjusted_clear_sky_ghi_horizon_1d
```

These columns are not final forecasts. They are physical information features used by the training model to learn weather-conditioned attenuation and residual correction.

For supervised targets such as `target_ghi_4h`, the model must use the matching horizon-aligned clear-sky feature (`ghi_clear_horizon_4h`) as its physical baseline. This prevents the model from using the current nighttime clear-sky value to predict a future daytime target.

## Clear-Sky Backend Contract

Supported clear-sky backend values are:

```text
auto
pvlib_simplified_solis
rest2_numpy
fallback_rest2_like
```

`auto` uses `pvlib_simplified_solis` when `pvlib` is installed. If `pvlib` is unavailable, it falls back to `rest2_numpy`.

`rest2_numpy` is the project REST2-style offline backend. It uses solar geometry plus `pressure_pa`, `pwv_cm`, `aod700`, `mu0_target`, and `dni_extra` to generate `ghi_clear_target`, `dni_clear_target`, `dhi_clear_target`, and transmittance-derived features.

`fallback_rest2_like` is kept only as a lightweight compatibility backend. It should not be the default training baseline when `pvlib` or `rest2_numpy` is available.

The Torch implementation mirrors the continuous REST2 core only. Gradients are supported through `pressure_pa`, `pwv_cm`, `aod700`, `mu0_target`, and `dni_extra`; timestamp parsing and solar geometry remain offline preprocessing.

## Model-Ready CSV Contract

`data/mock_model_ready/*.csv` must use exactly the same columns and column order across all sources. It must not retain source-specific columns such as `GHI`, `observe_ghi`, `interval`, `report_date`, or `report_time`.

Source-specific fields are allowed in enriched/debug CSV files only.

## Estimated Parameters Before pvlib

`pressure_pa` is estimated from station altitude using a standard atmosphere equation unless a real pressure column is present in a future dataset.

`pwv_cm` is computed from `PWAT` when available. If `PWAT` is missing, the pipeline uses `1.5 cm` and marks `pwv_cm_source=default_pwv_1p5cm`.

`aod700` is fixed to `0.08` and marked as `aod700_source=default`. This is a placeholder until pvlib/REST2 calibration or site-level AOD priors are introduced.

## Legacy Output Directory

`data/mock_training/` is a legacy debug directory from before the fixed-schema split. Do not use it as the training entrypoint. Use `data/mock_model_ready/` for model training and `data/mock_enriched/` for field-source debugging.
