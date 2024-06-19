from typing import Dict

import torch
from torch import Tensor


def blood_volume_prime(
    sum_plasma_infusions: Tensor,
    sum_rbc_infusions: Tensor,
    sum_platelet_infusions: Tensor,
    sum_fluid_infusions: Tensor,
    k_bleed: Tensor,
    blood_volume: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of blood volume.

    This function represents the differential equation governing blood volume dynamics.
    It considers infusions of plasma, RBCs, platelets, and fluids, as well as blood loss
    due to bleeding and fluid regulation through urination.

    Args:
        sum_plasma_infusions: Total plasma infusion rate (mL/min).
        sum_rbc_infusions: Total RBC infusion rate (mL/min).
        sum_platelet_infusions: Total platelet infusion rate (mL/min).
        sum_fluid_infusions: Total fluid infusion rate (mL/min).
        k_bleed: Bleeding rate (1/min).
        blood_volume: Current blood volume (mL).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of blood volume (mL/min).
    """
    d_blood_volume_dt = (
        sum_plasma_infusions
        + sum_rbc_infusions
        + sum_platelet_infusions
        + sum_fluid_infusions
        - k_bleed * blood_volume
        + (config["HEMODYNAMICS"]["S_BLOOD_VOLUME"] - blood_volume)
        * config["HEMODYNAMICS"]["K_URINATE"]
    )
    return d_blood_volume_dt


def d_bv_dt(
    sum_plasma_infusions: Tensor,
    sum_rbc_infusions: Tensor,
    sum_platelet_infusions: Tensor,
    sum_fluid_infusions: Tensor,
    k_bleed: Tensor,
    blood_volume: Tensor,
    config: Dict,
) -> Tensor:
    """
    Alias for blood_volume_prime. Calculates the rate of change of blood volume.

    This function is an alias for `blood_volume_prime` and is included for consistency
    with the original model, where it might be used in other modules.

    Args:
        sum_plasma_infusions: Total plasma infusion rate (mL/min).
        sum_rbc_infusions: Total RBC infusion rate (mL/min).
        sum_platelet_infusions: Total platelet infusion rate (mL/min).
        sum_fluid_infusions: Total fluid infusion rate (mL/min).
        k_bleed: Bleeding rate (1/min).
        blood_volume: Current blood volume (mL).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of blood volume (mL/min).
    """
    return blood_volume_prime(
        sum_plasma_infusions,
        sum_rbc_infusions,
        sum_platelet_infusions,
        sum_fluid_infusions,
        k_bleed,
        blood_volume,
        config,
    )


def blood_pressure_unadjusted_prime(
    _blood_pressure: Tensor, no: Tensor, config: Dict
) -> Tensor:
    """
    Calculates the rate of change of unadjusted blood pressure.

    This function models the dynamics of blood pressure without the direct influence
    of blood volume. It considers the baseline blood pressure regulation and the
    effect of nitric oxide (NO) on blood pressure.

    Args:
        _blood_pressure: Current blood pressure (mmHg).
        no: Current nitric oxide level (umol/L).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of unadjusted blood pressure (mmHg/min).
    """
    d_blood_pressure_unadjusted_dt = (
        config["HEMODYNAMICS"]["S_BLOOD_PRESSURE"] - _blood_pressure
    ) * config["HEMODYNAMICS"]["K_BASELINE_BLOOD_PRESSURE"] + (
        config["INFLAMMATION"]["S_NO"] - no
    ) * _blood_pressure * config[
        "HEMODYNAMICS"
    ][
        "K_BLOOD_PRESSURE_NO"
    ]
    return d_blood_pressure_unadjusted_dt


def blood_pressure(
    blood_pressure_unadjusted: Tensor, blood_volume: Tensor, config: Dict
) -> Tensor:
    """
    Calculates the blood pressure, adjusted for blood volume.

    This function adjusts the unadjusted blood pressure based on the current blood volume
    using a sigmoid-like function. This reflects the direct relationship between blood
    volume and blood pressure.

    Args:
        blood_pressure_unadjusted: Unadjusted blood pressure (mmHg).
        blood_volume: Current blood volume (mL).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The adjusted blood pressure (mmHg).
    """
    adjusted_blood_pressure = (
        blood_pressure_unadjusted
        * 2
        * blood_volume**2
        / (config["HEMODYNAMICS"]["S_BLOOD_VOLUME"] ** 2 + blood_volume**2)
    )
    return adjusted_blood_pressure
