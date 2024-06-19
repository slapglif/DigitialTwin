from typing import Dict

import torch
from torch import Tensor


def k_bleed(
    trauma: Tensor, clot: Tensor, blood_pressure: Tensor, config: Dict
) -> Tensor:
    """
    Calculates the bleeding rate.

    This function models the rate of blood loss based on the current _trauma level,
    the extent of clot formation, and the blood pressure. It assumes that higher _trauma
    increases bleeding, while clot formation reduces it. Blood pressure also influences
    bleeding, with higher pressure leading to increased blood loss.

    Args:
        trauma: Current _trauma level (unitless).
        clot: Current clot level (unitless).
        blood_pressure: Current blood pressure (mmHg).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The bleeding rate (1/min).
    """
    # Calculate the bleeding tendency based on _trauma and clot formation.
    bleeding_tendency = config["COAGULATION"]["K_BLEED_TRAUMA"] * trauma - clot

    # Ensure bleeding tendency is non-negative (no bleeding if tendency is negative).
    bleeding_tendency = torch.maximum(bleeding_tendency, torch.tensor(0.0))

    # Scale bleeding rate by blood pressure relative to steady-state blood pressure.
    bleeding_rate = (
        config["COAGULATION"]["K_BLEED_CONTROL"]
        * bleeding_tendency
        * (blood_pressure / config["HEMODYNAMICS"]["S_BLOOD_PRESSURE"])
    )
    return bleeding_rate


def rbc_prime(
    rbc: Tensor,
    blood_volume: Tensor,
    sum_rbc_infusions: Tensor,
    _k_bleed: Tensor,
    d_bv_dt: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of red blood cells (RBCs).

    This function models the dynamics of RBC concentration in the blood. It accounts for
    RBC production, infusions, blood loss due to bleeding, and changes in blood volume.

    Args:
        rbc: Current RBC concentration (RBCs/mL).
        blood_volume: Current blood volume (mL).
        sum_rbc_infusions: Total RBC infusion rate (mL/min).
        _k_bleed: Bleeding rate (1/min).
        d_bv_dt: Rate of change of blood volume (mL/min).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of RBC concentration (RBCs/(mL*min)).
    """
    # Calculate the rate of RBC return to baseline.
    rbc_return_to_baseline = (config["COAGULATION"]["S_RBC"] - rbc) * config[
        "COAGULATION"
    ]["K_BASELINE_RBC"]

    # Calculate the effect of RBC infusions, adjusted for blood volume.
    rbc_infusion_effect = (
        (1 / blood_volume)
        * (
            config["COAGULATION"]["RBC_INFUSION_BONUS_CONC"]
            * config["COAGULATION"]["S_RBC"]
            - rbc
        )
        * sum_rbc_infusions
    )

    # Calculate the RBC loss due to bleeding.
    rbc_loss_from_bleeding = _k_bleed * rbc

    # Calculate the RBC change due to blood volume changes.
    rbc_change_from_bv = rbc * d_bv_dt / blood_volume

    # Combine all the effects to get the overall rate of change of RBCs.
    d_rbc_dt = (
        rbc_return_to_baseline
        + rbc_infusion_effect
        - rbc_loss_from_bleeding
        - rbc_change_from_bv
    )
    return d_rbc_dt


def inactive_coag_factor_prime(
    inactive_coag_factor: Tensor,
    blood_volume: Tensor,
    sum_plasma_infusions: Tensor,
    sum_platelet_infusions: Tensor,
    _k_bleed: Tensor,
    trauma: Tensor,
    il6: Tensor,
    d_bv_dt: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of inactive coagulation factors.

    This function models the dynamics of inactive coagulation factor concentration in the blood.
    It considers the baseline production, infusions of plasma and platelets (which contain
    coagulation factors), consumption due to activation by _trauma and IL-6, blood loss
    from bleeding, and changes in blood volume.

    Args:
        inactive_coag_factor: Current inactive coagulation factor concentration (pg/mL).
        blood_volume: Current blood volume (mL).
        sum_plasma_infusions: Total plasma infusion rate (mL/min).
        sum_platelet_infusions: Total platelet infusion rate (mL/min).
        _k_bleed: Bleeding rate (1/min).
        trauma: Current _trauma level (unitless).
        il6: Current IL-6 level (pg/mL).
        d_bv_dt: Rate of change of blood volume (mL/min).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of inactive coagulation factor concentration (pg/(mL*min)).
    """
    # Calculate the rate of return to baseline for inactive coagulation factors.
    inactive_coag_return_to_baseline = config["COAGULATION"][
        "K_BASELINE_INACTIVE_COAG_FACTOR"
    ] * (config["COAGULATION"]["S_INACTIVE_COAG_FACTOR"] - inactive_coag_factor)

    # Calculate the effect of plasma and platelet infusions, adjusted for blood volume.
    coag_infusion_effect = (
        (1 / blood_volume)
        * (
            config["COAGULATION"]["PLASMA_INFUSION_COAG_BONUS"]
            * config["COAGULATION"]["S_INACTIVE_COAG_FACTOR"]
            - inactive_coag_factor
        )
        * (sum_plasma_infusions + sum_platelet_infusions)
    )

    # Calculate the consumption of inactive coagulation factors due to activation by _trauma and IL-6.
    coag_activation = (
        inactive_coag_factor
        * blood_volume
        * (
            config["COAGULATION"]["K_COAG_FACTOR_ACTIVATION_TRAUMA"] * trauma
            + config["COAGULATION"]["K_COAG_FACTOR_ACTIVATION_IL6"] * il6
        )
    )

    # Calculate the loss of inactive coagulation factors due to bleeding.
    inactive_coag_loss_from_bleeding = _k_bleed * inactive_coag_factor

    # Calculate the change in inactive coagulation factors due to blood volume changes.
    inactive_coag_change_from_bv = inactive_coag_factor * d_bv_dt / blood_volume

    # Combine all the effects to get the overall rate of change of inactive coagulation factors.
    d_inactive_coag_factor_dt = (
        inactive_coag_return_to_baseline
        + coag_infusion_effect
        - coag_activation
        - inactive_coag_loss_from_bleeding
        - inactive_coag_change_from_bv
    )
    return d_inactive_coag_factor_dt


def active_coag_factor_prime(
    inactive_coag_factor: Tensor,
    blood_volume: Tensor,
    trauma: Tensor,
    il6: Tensor,
    _k_bleed: Tensor,
    active_anti_coag: Tensor,
    inactive_anti_coag: Tensor,
    platelets: Tensor,
    active_coag_factor: Tensor,  # Added missing argument
    d_bv_dt: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of active coagulation factors.

    This function models the dynamics of active coagulation factor concentration in the blood.
    It considers the activation from inactive coagulation factors, natural decay, inhibition
    by active anticoagulants, consumption during clot formation, blood loss from bleeding,
    and changes in blood volume.

    Args:
        inactive_coag_factor: Current inactive coagulation factor concentration (pg/mL).
        blood_volume: Current blood volume (mL).
        trauma: Current _trauma level (unitless).
        il6: Current IL-6 level (pg/mL).
        _k_bleed: Bleeding rate (1/min).
        active_anti_coag: Current active anticoagulant concentration (pg/mL).
        inactive_anti_coag: Current inactive anticoagulant concentration (pg/mL).
        platelets: Current platelet concentration (platelets/mL).
        active_coag_factor: Current active coagulation factor concentration (pg/mL). # This was missing!
        d_bv_dt: Rate of change of blood volume (mL/min).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of active coagulation factor concentration (pg/(mL*min)).
    """
    # Calculate the activation of coagulation factors from the inactive form.
    coag_activation = (
        inactive_coag_factor
        * blood_volume
        * (
            config["COAGULATION"]["K_COAG_FACTOR_ACTIVATION_TRAUMA"] * trauma
            + config["COAGULATION"]["K_COAG_FACTOR_ACTIVATION_IL6"] * il6
        )
    )

    # Calculate the natural decay of active coagulation factors.
    active_coag_decay = (
        config["COAGULATION"]["D_ACTIVE_COAG_FACTOR"] * active_coag_factor
    )

    # Calculate the enhancement of active coagulation factor decay due to active anticoagulants.
    enhanced_decay = (
        active_coag_decay
        * config["COAGULATION"]["K_ACTIVE_COAG_FACTOR_ACTIVE_ANTI_COAG_ENHANCE"]
        * config["COAGULATION"]["K_ACTIVE_COAG_FACTOR_ACTIVE_ANTI_COAG_COLLIDE"]
        * active_anti_coag
        * blood_volume
    )

    # Calculate the inhibition of active coagulation factors by inactive anticoagulants.
    inhibition_by_inactive_anticoag = (
        config["COAGULATION"]["K_INACTIVE_ANTI_COAG_ACTIVE_COAG_FACTOR"]
        * blood_volume
        * inactive_anti_coag
        * active_coag_factor
    )

    # Calculate the consumption of active coagulation factors during clot formation.
    clot_formation_consumption = (
        config["COAGULATION"]["ACTIVECOAGFACTOR_PORTION_OF_CLOT_CONVERTER"]
        * config["COAGULATION"]["K_CLOT"]
        * blood_volume
        * platelets
        * active_coag_factor
    )

    # Calculate the loss of active coagulation factors due to bleeding.
    active_coag_loss_from_bleeding = _k_bleed * active_coag_factor

    # Calculate the change in active coagulation factors due to blood volume changes.
    active_coag_change_from_bv = active_coag_factor * d_bv_dt / blood_volume

    # Combine all the effects to get the overall rate of change of active coagulation factors.
    d_active_coag_factor_dt = (
        coag_activation
        - active_coag_decay
        - enhanced_decay
        - inhibition_by_inactive_anticoag
        - clot_formation_consumption
        - active_coag_loss_from_bleeding
        - active_coag_change_from_bv
    )
    return d_active_coag_factor_dt


def pro_coag_total(inactive_coag_factor: Tensor, active_coag_factor: Tensor) -> Tensor:
    """
    Calculates the total procoagulant level.

    This function simply sums the concentrations of inactive and active coagulation factors
    to represent the total procoagulant potential in the blood.

    Args:
        inactive_coag_factor: Current inactive coagulation factor concentration (pg/mL).
        active_coag_factor: Current active coagulation factor concentration (pg/mL).

    Returns:
        torch.Tensor: The total procoagulant level (pg/mL).
    """
    return inactive_coag_factor + active_coag_factor


def inactive_anti_coag_prime(
    inactive_anti_coag: Tensor,
    blood_volume: Tensor,
    sum_plasma_infusions: Tensor,
    sum_platelet_infusions: Tensor,
    _k_bleed: Tensor,
    active_coag_factor: Tensor,
    d_bv_dt: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of inactive anticoagulants.

    This function models the dynamics of inactive anticoagulant concentration in the blood.
    It considers the baseline production, infusions of plasma and platelets (which may contain
    anticoagulants), consumption due to interaction with active coagulation factors, blood loss
    from bleeding, and changes in blood volume.

    Args:
        inactive_anti_coag: Current inactive anticoagulant concentration (pg/mL).
        blood_volume: Current blood volume (mL).
        sum_plasma_infusions: Total plasma infusion rate (mL/min).
        sum_platelet_infusions: Total platelet infusion rate (mL/min).
        _k_bleed: Bleeding rate (1/min).
        active_coag_factor: Current active coagulation factor concentration (pg/mL).
        d_bv_dt: Rate of change of blood volume (mL/min).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of inactive anticoagulant concentration (pg/(mL*min)).
    """
    # Calculate the rate of return to baseline for inactive anticoagulants.
    inactive_anticoag_return_to_baseline = config["COAGULATION"][
        "K_BASELINE_INACTIVE_ANTI_COAG"
    ] * (config["COAGULATION"]["S_INACTIVE_ANTI_COAG"] - inactive_anti_coag)

    # Calculate the effect of plasma and platelet infusions, adjusted for blood volume.
    anticoag_infusion_effect = (
        (1 / blood_volume)
        * (
            config["COAGULATION"]["PLASMA_INFUSION_COAG_BONUS"]
            * config["COAGULATION"]["S_INACTIVE_ANTI_COAG"]
            - inactive_anti_coag
        )
        * (sum_plasma_infusions + sum_platelet_infusions)
    )

    # Calculate the consumption of inactive anticoagulants due to interaction with active coagulation factors.
    interaction_with_active_coag = (
        config["COAGULATION"]["K_INACTIVE_ANTI_COAG_ACTIVE_COAG_FACTOR"]
        * blood_volume
        * inactive_anti_coag
        * active_coag_factor
    )

    # Calculate the loss of inactive anticoagulants due to bleeding.
    inactive_anticoag_loss_from_bleeding = _k_bleed * inactive_anti_coag

    # Calculate the change in inactive anticoagulants due to blood volume changes.
    inactive_anticoag_change_from_bv = inactive_anti_coag * d_bv_dt / blood_volume

    # Combine all the effects to get the overall rate of change of inactive anticoagulants.
    d_inactive_anti_coag_dt = (
        inactive_anticoag_return_to_baseline
        + anticoag_infusion_effect
        - interaction_with_active_coag
        - inactive_anticoag_loss_from_bleeding
        - inactive_anticoag_change_from_bv
    )
    return d_inactive_anti_coag_dt


def active_anti_coag_prime(
    active_anti_coag: Tensor,
    inactive_anti_coag: Tensor,
    blood_volume: Tensor,
    active_coag_factor: Tensor,
    sum_plasma_infusions: Tensor,
    sum_platelet_infusions: Tensor,
    _k_bleed: Tensor,
    d_bv_dt: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of active anticoagulants.

    This function models the dynamics of active anticoagulant concentration in the blood.
    It considers the baseline production, activation from the inactive form, infusions
    of plasma and platelets (which may contain anticoagulants), consumption due to interaction
    with active coagulation factors, blood loss from bleeding, and changes in blood volume.

    Args:
        active_anti_coag: Current active anticoagulant concentration (pg/mL).
        inactive_anti_coag: Current inactive anticoagulant concentration (pg/mL).
        blood_volume: Current blood volume (mL).
        active_coag_factor: Current active coagulation factor concentration (pg/mL).
        sum_plasma_infusions: Total plasma infusion rate (mL/min).
        sum_platelet_infusions: Total platelet infusion rate (mL/min).
        _k_bleed: Bleeding rate (1/min).
        d_bv_dt: Rate of change of blood volume (mL/min).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of active anticoagulant concentration (pg/(mL*min)).
    """
    # Calculate the rate of return to baseline for active anticoagulants.
    active_anticoag_return_to_baseline = config["COAGULATION"][
        "K_BASELINE_ACTIVE_ANTI_COAG"
    ] * (config["COAGULATION"]["S_ACTIVE_ANTI_COAG"] - active_anti_coag)

    # Calculate the activation of anticoagulants from the inactive form.
    anticoag_activation = (
        config["COAGULATION"]["K_INACTIVE_ANTI_COAG_ACTIVE_COAG_FACTOR"]
        * blood_volume
        * inactive_anti_coag
        * active_coag_factor
    )

    # Calculate the effect of plasma and platelet infusions, adjusted for blood volume.
    anticoag_infusion_effect = (
        (1 / blood_volume)
        * (
            config["COAGULATION"]["PLASMA_INFUSION_COAG_BONUS"]
            * config["COAGULATION"]["S_ACTIVE_ANTI_COAG"]
            - active_anti_coag
        )
        * (sum_plasma_infusions + sum_platelet_infusions)
    )

    # Calculate the consumption of active anticoagulants due to interaction with active coagulation factors.
    interaction_with_active_coag = (
        config["COAGULATION"]["K_ACTIVE_COAG_FACTOR_ACTIVE_ANTI_COAG_COLLIDE"]
        * active_coag_factor
        * active_anti_coag
        * blood_volume
    )

    # Calculate the loss of active anticoagulants due to bleeding.
    active_anticoag_loss_from_bleeding = _k_bleed * active_anti_coag

    # Calculate the change in active anticoagulants due to blood volume changes.
    active_anticoag_change_from_bv = active_anti_coag * d_bv_dt / blood_volume

    # Combine all the effects to get the overall rate of change of active anticoagulants.
    d_active_anti_coag_dt = (
        active_anticoag_return_to_baseline
        + anticoag_activation
        + anticoag_infusion_effect
        - interaction_with_active_coag
        - active_anticoag_loss_from_bleeding
        - active_anticoag_change_from_bv
    )
    return d_active_anti_coag_dt


def anti_coag_total(inactive_anti_coag: Tensor, active_anti_coag: Tensor) -> Tensor:
    """
    Calculates the total anticoagulant level.

    This function sums the concentrations of inactive and active anticoagulants to represent
    the total anticoagulant potential in the blood.

    Args:
        inactive_anti_coag: Current inactive anticoagulant concentration (pg/mL).
        active_anti_coag: Current active anticoagulant concentration (pg/mL).

    Returns:
        torch.Tensor: The total anticoagulant level (pg/mL).
    """
    return inactive_anti_coag + active_anti_coag


def platelets_prime(
    platelets: Tensor,
    blood_volume: Tensor,
    sum_platelet_infusions: Tensor,
    _platelet_source_enhance: Tensor,
    active_coag_factor: Tensor,
    _k_bleed: Tensor,
    d_bv_dt: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of platelets.

    This function models the dynamics of platelet concentration in the blood. It considers
    baseline production, enhancement of production due to _trauma, infusions, consumption
    during clot formation, blood loss from bleeding, and changes in blood volume.

    Args:
        platelets: Current platelet concentration (platelets/mL).
        blood_volume: Current blood volume (mL).
        sum_platelet_infusions: Total platelet infusion rate (mL/min).
        _platelet_source_enhance: Enhancement factor for platelet production due to _trauma.
        active_coag_factor: Current active coagulation factor concentration (pg/mL).
        _k_bleed: Bleeding rate (1/min).
        d_bv_dt: Rate of change of blood volume (mL/min).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of platelet concentration (platelets/(mL*min)).
    """
    # Calculate the baseline platelet production rate.
    baseline_platelet_production = config["COAGULATION"]["K_BASELINE_PLATELETS"] * (
        config["COAGULATION"]["S_PLATELETS"] * (1 + _platelet_source_enhance)
        - platelets
    )

    # Calculate the effect of platelet infusions, adjusted for blood volume.
    platelet_infusion_effect = (
        (1 / blood_volume)
        * (
            config["COAGULATION"]["PLATELET_INFUSION_BONUS_CONC"]
            * config["COAGULATION"]["S_PLATELETS"]
            - platelets
        )
        * sum_platelet_infusions
    )

    # Calculate the consumption of platelets during clot formation.
    clot_formation_consumption = (
        config["COAGULATION"]["PLATELETS_PORTION_OF_CLOT_CONVERTER"]
        * config["COAGULATION"]["K_CLOT"]
        * blood_volume
        * platelets
        * active_coag_factor
    )

    # Calculate the loss of platelets due to bleeding.
    platelet_loss_from_bleeding = _k_bleed * platelets

    # Calculate the change in platelets due to blood volume changes.
    platelet_change_from_bv = platelets * d_bv_dt / blood_volume

    # Combine all the effects to get the overall rate of change of platelets.
    d_platelets_dt = (
        baseline_platelet_production
        + platelet_infusion_effect
        - clot_formation_consumption
        - platelet_loss_from_bleeding
        - platelet_change_from_bv
    )
    return d_platelets_dt


def platelet_source_enhance(trauma: Tensor, config: Dict) -> Tensor:
    """
    Calculates the enhancement factor for platelet production due to _trauma.

    This function models the increase in platelet production in response to _trauma,
    representing a phenomenon known as reactive thrombocytosis.

    Args:
        trauma: Current _trauma level (unitless).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The enhancement factor for platelet production.
    """
    return (
        config["COAGULATION"]["K_PLATELET_SOURCE_ENHANCE"]
        * trauma**2
        / (trauma**2 + config["INFLAMMATION"]["MAX_ISS"] ** 2)
    )


def clot_prime(
    platelets: Tensor,
    active_coag_factor: Tensor,
    rbc: Tensor,
    trauma: Tensor,
    blood_volume: Tensor,
    clot: Tensor,  # Added the missing argument
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of clot formation.

    This function models the dynamics of clot formation based on the concentrations of
    platelets, active coagulation factors, and RBCs, as well as the _trauma level.
    It also considers clot breakdown (fibrinolysis) influenced by _trauma.

    Args:
        platelets: Current platelet concentration (platelets/mL).
        active_coag_factor: Current active coagulation factor concentration (pg/mL).
        rbc: Current RBC concentration (RBCs/mL).
        trauma: Current _trauma level (unitless).
        blood_volume: Current blood volume (mL).
        clot: Current clot level (unitless). # This was missing!
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of clot formation (unitless/min).
    """
    # Calculate the rate of clot formation.
    clot_formation_rate = (
        config["COAGULATION"]["K_CLOT"]
        * blood_volume
        * platelets
        * active_coag_factor
        * (1 + config["COAGULATION"]["K_CLOT_RBC"] * rbc)
    )

    # Calculate the rate of clot breakdown (fibrinolysis).
    clot_breakdown_rate = (
        config["COAGULATION"]["D_CLOT"]
        * clot
        * (1 + config["COAGULATION"]["K_FIBRINOLYSIS"] * trauma)
    )

    # Combine formation and breakdown rates to get the overall rate of change.
    d_clot_dt = clot_formation_rate - clot_breakdown_rate
    return d_clot_dt
