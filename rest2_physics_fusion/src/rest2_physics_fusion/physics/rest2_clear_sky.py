from __future__ import annotations

import numpy as np
import pandas as pd


REST2_BACKEND = "rest2_numpy"
PVLIB_BACKEND = "pvlib_simplified_solis"
FALLBACK_BACKEND = "fallback_rest2_like"


def tau_rayleigh(pressure_pa: np.ndarray) -> np.ndarray:
    return 0.00877 * (pressure_pa / 101325.0)


def tau_water(pwv_cm: np.ndarray) -> np.ndarray:
    return 0.2385 * np.power(np.clip(pwv_cm, 0.01, None), 0.3035)


def _relative_airmass_from_mu0(mu0: np.ndarray) -> np.ndarray:
    mu0_safe = np.clip(mu0, 1e-6, 1.0)
    zenith_deg = np.rad2deg(np.arccos(mu0_safe))
    return 1.0 / (mu0_safe + 0.50572 * np.power(np.clip(96.07995 - zenith_deg, 1e-3, None), -1.6364))


def _rest2_components_numpy(
    *,
    mu0: np.ndarray,
    pressure_pa: np.ndarray,
    pwv_cm: np.ndarray,
    aod700: np.ndarray,
    dni_extra: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute a compact REST2-style clear-sky approximation.

    The full REST2 model splits the spectrum into two bands and uses a richer
    atmospheric state. This implementation keeps the same engineering role for
    this project: pressure/PWV/AOD-sensitive clear-sky priors that are stable
    offline and easy to mirror in Torch for differentiable calibration.
    """

    mu0_raw = np.asarray(mu0, dtype=float)
    mu0_safe = np.clip(mu0_raw, 1e-4, 1.0)
    pressure_safe = np.clip(np.asarray(pressure_pa, dtype=float), 5.0e4, 1.1e5)
    pwv_safe = np.clip(np.asarray(pwv_cm, dtype=float), 0.05, 10.0)
    aod_safe = np.clip(np.asarray(aod700, dtype=float), 0.0, 2.0)
    dni_extra_safe = np.clip(np.asarray(dni_extra, dtype=float), 0.0, None)

    air_mass = _relative_airmass_from_mu0(mu0_safe)
    pressure_ratio = pressure_safe / 101325.0
    rayleigh_mass = air_mass * pressure_ratio

    tau_r = tau_rayleigh(pressure_safe)
    tau_w = tau_water(pwv_safe)
    tau_a = aod_safe

    # REST2-inspired broadband transmittance terms. The coefficients are chosen
    # for numerical stability and to preserve physical monotonicity with respect
    # to pressure, precipitable water, and aerosol optical depth.
    t_rayleigh = np.exp(-0.0903 * np.power(rayleigh_mass, 0.84) * (1.0 + rayleigh_mass - np.power(rayleigh_mass, 1.01)))
    t_gases = np.exp(-0.0127 * np.power(rayleigh_mass, 0.26))
    water_path = pwv_safe * air_mass
    t_water = np.exp(-0.2385 * water_path / np.power(1.0 + 20.07 * water_path, 0.45))
    t_aerosol = np.exp(-aod_safe * np.power(air_mass, 0.92))

    direct_transmittance = np.clip(t_rayleigh * t_gases * t_water * t_aerosol, 0.0, 1.0)
    dni_clear = dni_extra_safe * direct_transmittance

    rayleigh_diffuse = 0.5 * (1.0 - t_rayleigh) * t_gases * t_water * t_aerosol
    aerosol_diffuse = 0.75 * (1.0 - t_aerosol) * t_rayleigh * t_gases * t_water
    diffuse_transmittance = np.clip(rayleigh_diffuse + aerosol_diffuse, 0.0, 0.45)
    dhi_clear = dni_extra_safe * mu0_raw * diffuse_transmittance
    ghi_clear = dni_clear * mu0_raw + dhi_clear

    night = mu0_raw <= 0.0
    dni_clear = np.where(night, 0.0, dni_clear)
    dhi_clear = np.where(night, 0.0, dhi_clear)
    ghi_clear = np.where(night, 0.0, ghi_clear)
    direct_transmittance = np.where(night, 0.0, direct_transmittance)

    return {
        "tau_rayleigh": tau_r,
        "tau_water": tau_w,
        "tau_aer": tau_a,
        "t_rayleigh": t_rayleigh,
        "t_gases": t_gases,
        "t_water": t_water,
        "t_aerosol": t_aerosol,
        "t_direct": direct_transmittance,
        "ghi_clear_target": np.maximum(ghi_clear, 0.0),
        "dni_clear_target": np.maximum(dni_clear, 0.0),
        "dhi_clear_target": np.maximum(dhi_clear, 0.0),
    }


def _assign_clear_sky_outputs(out: pd.DataFrame, components: dict[str, np.ndarray], backend: str) -> pd.DataFrame:
    out["tau_rayleigh"] = components["tau_rayleigh"]
    out["tau_water"] = components["tau_water"]
    out["tau_aer"] = components["tau_aer"]
    out["ghi_clear_target"] = components["ghi_clear_target"]
    out["dni_clear_target"] = components["dni_clear_target"]
    out["dhi_clear_target"] = components["dhi_clear_target"]
    out["clear_sky_backend"] = backend
    out["t_clr_dd"] = (out["dni_clear_target"] / (out["dni_extra"] + 1e-6)).clip(lower=0.0, upper=1.0)
    return out


def add_rest2_numpy_clear_sky(df: pd.DataFrame) -> pd.DataFrame:
    """Add the project REST2 clear-sky baseline with a stable NumPy backend."""

    out = df.copy()
    components = _rest2_components_numpy(
        mu0=out["mu0_target"].to_numpy(dtype=float),
        pressure_pa=out["pressure_pa"].to_numpy(dtype=float),
        pwv_cm=out["pwv_cm"].to_numpy(dtype=float),
        aod700=out["aod700"].to_numpy(dtype=float),
        dni_extra=out["dni_extra"].to_numpy(dtype=float),
    )
    return _assign_clear_sky_outputs(out, components, REST2_BACKEND)


def add_rest2_like_clear_sky(df: pd.DataFrame) -> pd.DataFrame:
    """Add a stable REST2-like clear-sky approximation.

    This is intentionally lightweight for offline feature generation. The
    feature contract is stable, so a stricter REST2 or pvlib backend can replace
    this function later without changing downstream data loaders.
    """

    out = df.copy()
    mu0_raw = out["mu0_target"].to_numpy(dtype=float)
    mu0 = np.clip(mu0_raw, 1e-4, 1.0)
    pressure_pa = out["pressure_pa"].to_numpy(dtype=float)
    pwv_cm = out["pwv_cm"].to_numpy(dtype=float)
    aod700 = np.clip(out["aod700"].to_numpy(dtype=float), 0.0, None)
    dni_extra = out["dni_extra"].to_numpy(dtype=float)

    tau_r = tau_rayleigh(pressure_pa)
    tau_w = tau_water(pwv_cm)
    tau_a = aod700
    optical_depth = tau_r + tau_w + 0.9 * tau_a
    t_beam = np.exp(-optical_depth / mu0)

    dni_clear = dni_extra * t_beam
    ghi_haurwitz = 1098.0 * mu0_raw * np.exp(-0.059 / mu0)
    ghi_clear = np.maximum(ghi_haurwitz * np.exp(-0.15 * tau_a), dni_clear * mu0_raw)
    dhi_clear = np.maximum(ghi_clear - dni_clear * mu0_raw, 0.0)

    night = mu0_raw <= 0.0
    dni_clear[night] = 0.0
    ghi_clear[night] = 0.0
    dhi_clear[night] = 0.0

    out["tau_rayleigh"] = tau_r
    out["tau_water"] = tau_w
    out["tau_aer"] = tau_a
    out["ghi_clear_target"] = ghi_clear
    out["dni_clear_target"] = dni_clear
    out["dhi_clear_target"] = dhi_clear
    out["clear_sky_backend"] = FALLBACK_BACKEND
    out["t_clr_dd"] = (out["dni_clear_target"] / (out["dni_extra"] + 1e-6)).clip(lower=0.0, upper=1.0)
    return out


def add_pvlib_simplified_solis_clear_sky(df: pd.DataFrame) -> pd.DataFrame:
    """Add pvlib simplified Solis clear-sky irradiance.

    This mirrors the clear-sky backend used by the FARMS prototype while
    preserving the project feature contract.
    """

    try:
        import pvlib
    except Exception as exc:
        raise ImportError("pvlib is not installed") from exc

    out = df.copy()
    clearsky = pvlib.clearsky.simplified_solis(
        apparent_elevation=out["apparent_elevation"].to_numpy(dtype=float),
        aod700=out["aod700"].to_numpy(dtype=float),
        precipitable_water=out["pwv_cm"].to_numpy(dtype=float),
        pressure=out["pressure_pa"].to_numpy(dtype=float),
        dni_extra=out["dni_extra"].to_numpy(dtype=float),
    )
    out["tau_rayleigh"] = tau_rayleigh(out["pressure_pa"].to_numpy(dtype=float))
    out["tau_water"] = tau_water(out["pwv_cm"].to_numpy(dtype=float))
    out["tau_aer"] = np.clip(out["aod700"].to_numpy(dtype=float), 0.0, None)
    out["ghi_clear_target"] = np.asarray(clearsky["ghi"], dtype=float)
    out["dni_clear_target"] = np.asarray(clearsky["dni"], dtype=float)
    out["dhi_clear_target"] = np.asarray(clearsky["dhi"], dtype=float)
    out["clear_sky_backend"] = PVLIB_BACKEND
    out["t_clr_dd"] = (out["dni_clear_target"] / (out["dni_extra"] + 1e-6)).clip(lower=0.0, upper=1.0)
    return out


def add_clear_sky(df: pd.DataFrame, backend: str = "auto") -> pd.DataFrame:
    backend_key = str(backend or "auto").lower()
    if backend_key in {"pvlib", "pvlib_simplified_solis"}:
        return add_pvlib_simplified_solis_clear_sky(df)
    if backend_key in {"rest2", "rest2_numpy", "rest2_like"}:
        return add_rest2_numpy_clear_sky(df)
    if backend_key in {"fallback", "fallback_rest2_like"}:
        return add_rest2_like_clear_sky(df)
    if backend_key != "auto":
        raise ValueError(f"Unsupported clear-sky backend: {backend}")

    try:
        return add_pvlib_simplified_solis_clear_sky(df)
    except Exception:
        return add_rest2_numpy_clear_sky(df)
