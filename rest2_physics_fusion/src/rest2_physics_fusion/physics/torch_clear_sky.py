from __future__ import annotations

import torch


def tau_rayleigh_torch(pressure_pa: torch.Tensor) -> torch.Tensor:
    return 0.00877 * (pressure_pa / 101325.0)


def tau_water_torch(pwv_cm: torch.Tensor) -> torch.Tensor:
    return 0.2385 * torch.pow(torch.clamp(pwv_cm, min=0.01), 0.3035)


def _relative_airmass_from_mu0(mu0: torch.Tensor) -> torch.Tensor:
    mu0_safe = torch.clamp(mu0, min=1e-6, max=1.0)
    zenith_deg = torch.rad2deg(torch.arccos(mu0_safe))
    zenith_term = torch.clamp(96.07995 - zenith_deg, min=1e-3)
    return 1.0 / (mu0_safe + 0.50572 * torch.pow(zenith_term, -1.6364))


def rest2_torch_clear_sky(
    *,
    mu0: torch.Tensor,
    pressure_pa: torch.Tensor,
    pwv_cm: torch.Tensor,
    aod700: torch.Tensor,
    dni_extra: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable REST2-style clear-sky core.

    Gradients are intended for continuous atmospheric inputs such as pressure,
    precipitable water, and aerosol optical depth. Time parsing and solar
    geometry stay outside autograd in this first implementation.
    """

    dtype = mu0.dtype
    device = mu0.device
    pressure_pa = pressure_pa.to(device=device, dtype=dtype)
    pwv_cm = pwv_cm.to(device=device, dtype=dtype)
    aod700 = aod700.to(device=device, dtype=dtype)
    dni_extra = dni_extra.to(device=device, dtype=dtype)

    mu0_raw = mu0
    mu0_safe = torch.clamp(mu0_raw, min=1e-4, max=1.0)
    pressure_safe = torch.clamp(pressure_pa, min=5.0e4, max=1.1e5)
    pwv_safe = torch.clamp(pwv_cm, min=0.05, max=10.0)
    aod_safe = torch.clamp(aod700, min=0.0, max=2.0)
    dni_extra_safe = torch.clamp(dni_extra, min=0.0)

    air_mass = _relative_airmass_from_mu0(mu0_safe)
    pressure_ratio = pressure_safe / 101325.0
    rayleigh_mass = air_mass * pressure_ratio

    tau_r = tau_rayleigh_torch(pressure_safe)
    tau_w = tau_water_torch(pwv_safe)
    tau_a = aod_safe

    t_rayleigh = torch.exp(
        -0.0903
        * torch.pow(rayleigh_mass, 0.84)
        * (1.0 + rayleigh_mass - torch.pow(rayleigh_mass, 1.01))
    )
    t_gases = torch.exp(-0.0127 * torch.pow(rayleigh_mass, 0.26))
    water_path = pwv_safe * air_mass
    t_water = torch.exp(-0.2385 * water_path / torch.pow(1.0 + 20.07 * water_path, 0.45))
    t_aerosol = torch.exp(-aod_safe * torch.pow(air_mass, 0.92))

    direct_transmittance = torch.clamp(t_rayleigh * t_gases * t_water * t_aerosol, min=0.0, max=1.0)
    dni_clear = dni_extra_safe * direct_transmittance

    rayleigh_diffuse = 0.5 * (1.0 - t_rayleigh) * t_gases * t_water * t_aerosol
    aerosol_diffuse = 0.75 * (1.0 - t_aerosol) * t_rayleigh * t_gases * t_water
    diffuse_transmittance = torch.clamp(rayleigh_diffuse + aerosol_diffuse, min=0.0, max=0.45)
    dhi_clear = dni_extra_safe * mu0_raw * diffuse_transmittance
    ghi_clear = dni_clear * mu0_raw + dhi_clear

    daylight = mu0_raw > 0.0
    zeros = torch.zeros_like(ghi_clear)
    dni_clear = torch.where(daylight, dni_clear, zeros)
    dhi_clear = torch.where(daylight, dhi_clear, zeros)
    ghi_clear = torch.where(daylight, ghi_clear, zeros)
    direct_transmittance = torch.where(daylight, direct_transmittance, zeros)

    return {
        "tau_rayleigh": tau_r,
        "tau_water": tau_w,
        "tau_aer": tau_a,
        "t_rayleigh": t_rayleigh,
        "t_gases": t_gases,
        "t_water": t_water,
        "t_aerosol": t_aerosol,
        "t_direct": direct_transmittance,
        "ghi_clear_target": torch.clamp(ghi_clear, min=0.0),
        "dni_clear_target": torch.clamp(dni_clear, min=0.0),
        "dhi_clear_target": torch.clamp(dhi_clear, min=0.0),
    }
