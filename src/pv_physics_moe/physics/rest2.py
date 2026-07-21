from __future__ import annotations
from typing import Mapping
import torch
from torch import nn


PHYSICS_FEATURES = (
    "ghi_clear_target", "dni_clear_target", "dhi_clear_target", "mu0_target",
    "apparent_zenith", "dni_extra", "pressure_pa", "pwv_cm", "aod700",
    "t_clr_dd", "clear_sky_index_last", "smart_persistence_ghi",
    "clear_sky_index_current", "weather_attenuation_prior",
    "weather_adjusted_clear_sky_ghi", "clear_sky_residual_last", "clear_sky_gap",
    "ghi_clear_horizon_5min", "ghi_clear_horizon_4h", "ghi_clear_horizon_1d",
    "mu0_horizon_5min", "mu0_horizon_4h", "mu0_horizon_1d",
    "weather_adjusted_clear_sky_ghi_horizon_5min",
    "weather_adjusted_clear_sky_ghi_horizon_4h",
    "weather_adjusted_clear_sky_ghi_horizon_1d",
)


def _airmass(mu0: torch.Tensor) -> torch.Tensor:
    mu = mu0.clamp(1e-6, 1.0)
    zenith = torch.rad2deg(torch.arccos(mu))
    return 1.0 / (mu + 0.50572 * (96.07995 - zenith).clamp_min(1e-3).pow(-1.6364))


def rest2_torch_clear_sky(
    *, mu0: torch.Tensor, pressure_pa: torch.Tensor, pwv_cm: torch.Tensor,
    aod700: torch.Tensor, dni_extra: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable REST2-style clear-sky core copied from the source model contract."""
    dtype, device = mu0.dtype, mu0.device
    pressure = pressure_pa.to(device=device, dtype=dtype).clamp(5e4, 1.1e5)
    pwv = pwv_cm.to(device=device, dtype=dtype).clamp(0.05, 10.0)
    aod = aod700.to(device=device, dtype=dtype).clamp(0.0, 2.0)
    dni0 = dni_extra.to(device=device, dtype=dtype).clamp_min(0.0)
    mu_safe = mu0.clamp(1e-4, 1.0)
    air_mass = _airmass(mu_safe)
    rayleigh_mass = air_mass * pressure / 101325.0
    t_rayleigh = torch.exp(-0.0903 * rayleigh_mass.pow(0.84) * (1.0 + rayleigh_mass - rayleigh_mass.pow(1.01)))
    t_gases = torch.exp(-0.0127 * rayleigh_mass.pow(0.26))
    water_path = pwv * air_mass
    t_water = torch.exp(-0.2385 * water_path / (1.0 + 20.07 * water_path).pow(0.45))
    t_aerosol = torch.exp(-aod * air_mass.pow(0.92))
    direct_t = (t_rayleigh * t_gases * t_water * t_aerosol).clamp(0.0, 1.0)
    dni = dni0 * direct_t
    rayleigh_diffuse = 0.5 * (1.0 - t_rayleigh) * t_gases * t_water * t_aerosol
    aerosol_diffuse = 0.75 * (1.0 - t_aerosol) * t_rayleigh * t_gases * t_water
    diffuse_t = (rayleigh_diffuse + aerosol_diffuse).clamp(0.0, 0.45)
    dhi = dni0 * mu0 * diffuse_t
    ghi = dni * mu0 + dhi
    daylight = mu0 > 0.0
    zero = torch.zeros_like(ghi)
    return {
        "ghi_clear_target": torch.where(daylight, ghi.clamp_min(0.0), zero),
        "dni_clear_target": torch.where(daylight, dni.clamp_min(0.0), zero),
        "dhi_clear_target": torch.where(daylight, dhi.clamp_min(0.0), zero),
        "t_direct": torch.where(daylight, direct_t, zero),
    }


def _as_bh(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.ndim == 0:
        return value.expand_as(reference)
    if value.ndim == 1:
        return value[:, None].expand_as(reference)
    if value.shape != reference.shape:
        return value.expand_as(reference)
    return value


class Rest2FeatureBuilder(nn.Module):
    """Build the 26-value REST2 physical modality for every forecast step.

    Required inputs: ``mu0``, ``input_ghi``. Atmospheric values have safe defaults.
    Every tensor may be [B] or [B,H]; the output is [B,H,26] in physical units.
    """
    def __init__(self, sky_index_max: float = 2.0) -> None:
        super().__init__()
        self.sky_index_max = float(sky_index_max)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if "mu0" not in inputs or "input_ghi" not in inputs:
            raise KeyError("REST2 inputs require 'mu0' and 'input_ghi'")
        mu0 = inputs["mu0"].float()
        if mu0.ndim == 1:
            mu0 = mu0[:, None]
        input_ghi = _as_bh(inputs["input_ghi"], mu0)
        pressure = _as_bh(inputs.get("pressure_pa", torch.tensor(101325.0)), mu0)
        pwv = _as_bh(inputs.get("pwv_cm", torch.tensor(1.5)), mu0)
        aod = _as_bh(inputs.get("aod700", torch.tensor(0.08)), mu0)
        dni_extra = _as_bh(inputs.get("dni_extra", torch.tensor(1361.0)), mu0)
        clear = rest2_torch_clear_sky(
            mu0=mu0, pressure_pa=pressure, pwv_cm=pwv, aod700=aod, dni_extra=dni_extra,
        )
        ghi, dni, dhi = clear["ghi_clear_target"], clear["dni_clear_target"], clear["dhi_clear_target"]
        previous_ghi = _as_bh(inputs.get("previous_ghi", input_ghi), mu0)
        previous_clear = _as_bh(inputs.get("previous_clear_ghi", ghi), mu0)
        valid_previous = previous_clear > 20.0
        sky_last = torch.where(valid_previous, previous_ghi / previous_clear.clamp_min(1.0), torch.zeros_like(ghi)).clamp(0.0, self.sky_index_max)
        valid_current = ghi > 20.0
        sky_current = torch.where(valid_current, input_ghi / ghi.clamp_min(1.0), torch.zeros_like(ghi)).clamp(0.0, self.sky_index_max)
        precip = _as_bh(inputs.get("precip", torch.tensor(0.0)), mu0).clamp_min(0.0)
        wind = _as_bh(inputs.get("wind_speed", torch.tensor(0.0)), mu0).clamp_min(0.0)
        weather_available = _as_bh(inputs.get("weather_available", torch.tensor(1.0)), mu0).clamp(0.0, 1.0)
        attenuation = 1.0 - 0.025 * (pwv - 1.5).clamp_min(0.0) - 0.18 * (1.0 - torch.exp(-precip))
        attenuation = attenuation - 0.35 * (1.0 - sky_last).clamp(0.0, 1.0) + 0.03 * torch.tanh(wind / 6.0)
        attenuation = (attenuation * (0.7 + 0.3 * weather_available)).clamp(0.05, 1.2)
        adjusted = ghi * attenuation
        zenith = torch.rad2deg(torch.arccos(mu0.clamp(0.0, 1.0)))
        residual_last = previous_ghi - previous_clear
        gap = (ghi - input_ghi).clamp_min(0.0)
        values = (
            ghi, dni, dhi, mu0, zenith, dni_extra, pressure, pwv, aod, clear["t_direct"],
            sky_last, sky_last * ghi, sky_current, attenuation, adjusted, residual_last, gap,
            ghi, ghi, ghi, mu0, mu0, mu0, adjusted, adjusted, adjusted,
        )
        return torch.stack(values, dim=-1)

