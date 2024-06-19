from typing import Tuple, Dict

import torch
from torch import Tensor


def fs(v: Tensor, x: Tensor, hill: int) -> Tensor:
    """
    Calculates the sigmoid function.

    Args:
        v: Value.
        x: Threshold.
        hill: Hill coefficient.

    Returns:
        Sigmoid function value.
    """
    return x**hill / (v**hill + x**hill)


def fm(v: Tensor, x: Tensor, hill: int) -> Tensor:
    """
    Calculates the inverse sigmoid function.

    Args:
        v: Value.
        x: Threshold.
        hill: Hill coefficient.

    Returns:
        Inverse sigmoid function value.
    """
    return v**hill / (v**hill + x**hill)


def square(time: Tensor, ton: Tensor, toff: Tensor) -> Tensor:
    """
    Calculates a square wave function.

    Args:
        time: Current time.
        ton: Time when the square wave turns on.
        toff: Time when the square wave turns off.

    Returns:
        Square wave function value.
    """
    return torch.heaviside(time - ton, torch.tensor(1.0)) - torch.heaviside(
        time - toff, torch.tensor(1.0)
    )


def calculate_infusions(
    t: Tensor, config: dict
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Calculates the infusion rates for plasma, fluids, RBCs, and platelets at a given time.

    Args:
        t: Current time.
        config: Dictionary containing model parameters.

    Returns:
        Tuple containing the infusion rates for plasma, fluids, RBCs, and platelets.
    """

    def infusion(
        _t: Tensor, ton: float, toff: float, dose: float, duration: float
    ) -> Tensor:
        """
        Calculates the infusion rate at a given time.

        Args:
            _t: Current time.
            ton: Time when the infusion starts.
            toff: Time when the infusion ends.
            dose: Total dose of the infusion.
            duration: Duration of the infusion.

        Returns:
            Infusion rate at time t.
        """
        return torch.where(
            (_t >= ton) & (_t < toff), dose / duration, torch.tensor(0.0)
        )

    sum_plasma_infusions = torch.tensor(0.0)
    sum_fluid_infusions = torch.tensor(0.0)
    sum_rbc_infusions = torch.tensor(0.0)
    sum_platelet_infusions = torch.tensor(0.0)

    for i in range(1, 11):
        sum_plasma_infusions += infusion(
            t,
            config["INFUSION_SETTINGS"][f"TON_PLASMA_INFUSION{i}"],
            config["INFUSION_SETTINGS"][f"TON_PLASMA_INFUSION{i}"]
            + config["INFUSION_SETTINGS"][f"T_PLASMA_INFUSION_DURATION{i}"],
            config["INFUSION_SETTINGS"][f"PLASMA_DOSE{i}"],
            config["INFUSION_SETTINGS"][f"T_PLASMA_INFUSION_DURATION{i}"],
        )
        sum_fluid_infusions += infusion(
            t,
            config["INFUSION_SETTINGS"][f"TON_FLUID_INFUSION{i}"],
            config["INFUSION_SETTINGS"][f"TON_FLUID_INFUSION{i}"]
            + config["INFUSION_SETTINGS"][f"T_FLUID_INFUSION_DURATION{i}"],
            config["INFUSION_SETTINGS"][f"FLUID_DOSE{i}"],
            config["INFUSION_SETTINGS"][f"T_FLUID_INFUSION_DURATION{i}"],
        )
        sum_rbc_infusions += infusion(
            t,
            config["INFUSION_SETTINGS"][f"TON_RBC_INFUSION{i}"],
            config["INFUSION_SETTINGS"][f"TON_RBC_INFUSION{i}"]
            + config["INFUSION_SETTINGS"][f"T_RBC_INFUSION_DURATION{i}"],
            config["INFUSION_SETTINGS"][f"RBC_DOSE{i}"],
            config["INFUSION_SETTINGS"][f"T_RBC_INFUSION_DURATION{i}"],
        )
        sum_platelet_infusions += infusion(
            t,
            config["INFUSION_SETTINGS"][f"TON_PLATELET_INFUSION{i}"],
            config["INFUSION_SETTINGS"][f"TON_PLATELET_INFUSION{i}"]
            + config["INFUSION_SETTINGS"][f"T_PLATELET_INFUSION_DURATION{i}"],
            config["INFUSION_SETTINGS"][f"PLATELET_DOSE{i}"],
            config["INFUSION_SETTINGS"][f"T_PLATELET_INFUSION_DURATION{i}"],
        )

    return (
        sum_plasma_infusions,
        sum_fluid_infusions,
        sum_rbc_infusions,
        sum_platelet_infusions,
    )


def o2sat(o2sat0: Tensor, epal: Tensor, vent_on: Tensor, config: Dict) -> Tensor:
    """
    Calculates the oxygen saturation.

    Args:
        o2sat0: Initial oxygen saturation.
        epal: Activated epithelial cells in the lung.
        vent_on: Indicator variable for whether the ventilator is on.
        config: Dictionary containing model parameters.

    Returns:
        Oxygen saturation.
    """
    return (
        o2sat0
        * config["INFLAMMATION"]["VENT_PAR1"]
        / (
            config["INFLAMMATION"]["VENT_PAR1"]
            + config["INFLAMMATION"]["VENT_PAR2"]
            * epal
            / (
                config["INFLAMMATION"]["VENT_PAR2"]
                + config["INFLAMMATION"]["VENT_PAR3"] * vent_on
            )
        )
    )


def trauma(
    t: Tensor,
    iss: Tensor,
    second_trauma_iss_mut: Tensor,
    third_trauma_iss_mut: Tensor,
    ton_second_trauma: Tensor,
    ton_third_trauma: Tensor,
    t_initial: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the _trauma level at a given time.

    Args:
        t: Current time.
        iss: Initial Injury Severity Score.
        second_trauma_iss_mut: Injury Severity Score of the second _trauma.
        third_trauma_iss_mut: Injury Severity Score of the third _trauma.
        ton_second_trauma: Time of the second _trauma.
        ton_third_trauma: Time of the third _trauma.
        t_initial: Initial time.
        config: Dictionary containing model parameters.

    Returns:
        Trauma level at time t.
    """
    return (
        (iss + second_trauma_iss_mut + third_trauma_iss_mut)
        * config["INFLAMMATION"]["RECOVERY_SLOWNESS"]
        / (
            config["INFLAMMATION"]["RECOVERY_SLOWNESS"]
            + (
                t
                - torch.where(
                    t < ton_second_trauma,
                    t_initial,
                    torch.where(
                        t < ton_third_trauma, ton_second_trauma, ton_third_trauma
                    ),
                )
            )
            ** 2
        )
    )


def damage(
    blood_pressure: Tensor,
    il6: Tensor,
    il6l: Tensor,
    il6s: Tensor,
    _o2sat: Tensor,
    _trauma: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the damage score based on various physiological parameters.

    Args:
        blood_pressure: Current blood pressure (mmHg).
        il6: Concentration of IL-6 in the blood (pg/mL).
        il6l: Concentration of IL-6 in the lung (pg/mL).
        il6s: Concentration of IL-6 in the splanchnic (pg/mL).
        _o2sat: Oxygen saturation (%).
        _trauma: Current _trauma level (unitless).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The calculated damage score.
    """
    o2sat_damage = (
        config["INFLAMMATION"]["K_DAMAGE_O2SAT"]
        * torch.maximum(
            config["INFLAMMATION"]["O2SAT_DAMAGE_THRESHOLD"] - _o2sat,
            torch.tensor(0.0),
        )
        / config["INFLAMMATION"]["MAX_O2SAT"]
    )
    il6_damage = config["INFLAMMATION"]["K_DAMAGE_IL6"] * fm(
        torch.maximum(
            il6 - config["INFLAMMATION"]["IL6_DAMAGE_THRESHOLD"], torch.tensor(0.0)
        )
        + (1 - config["INFLAMMATION"]["USE_BLOOD_IL6_ONLY"])
        * torch.maximum(
            il6l - config["INFLAMMATION"]["IL6_DAMAGE_THRESHOLD"], torch.tensor(0.0)
        )
        + (1 - config["INFLAMMATION"]["USE_BLOOD_IL6_ONLY"])
        * torch.maximum(
            il6s - config["INFLAMMATION"]["IL6_DAMAGE_THRESHOLD"], torch.tensor(0.0)
        ),
        config["INFLAMMATION"]["X_DAMAGE_IL6"],
        2,
    )
    bp_damage = config["INFLAMMATION"]["K_DAMAGE_BP"] * fm(
        torch.maximum(
            config["INFLAMMATION"]["BP_DAMAGE_THRESHOLD"] - blood_pressure,
            torch.tensor(0.0),
        ),
        config["INFLAMMATION"]["X_DAMAGE_BP"],
        2,
    )
    trauma_damage = (
        config["INFLAMMATION"]["K_DAMAGE_TRAUMA"]
        * _trauma
        / config["INFLAMMATION"]["MAX_ISS"]
    )

    total_damage = bp_damage + il6_damage + o2sat_damage + trauma_damage
    return total_damage
