from typing import Dict

from torch import Tensor

from core.utils import fm, fs


def vma(
    mr: Tensor,
    blood_volume: Tensor,
    trauma: Tensor,
    tnf: Tensor,
    il1: Tensor,
    active_coag_factor: Tensor,
    il10: Tensor,
    il6: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of monocyte activation in the blood.

    This function models the activation of resting monocytes (Mr) into activated monocytes (Ma)
    in the blood compartment. The activation is driven by various factors, including TNF, IL-1,
    _trauma, and active coagulation factors. IL-10 and IL-6 have inhibitory effects on this process.

    Args:
        mr: Concentration of resting monocytes in the blood (cells/mL).
        blood_volume: Current blood volume (mL).
        trauma: Current _trauma level (unitless).
        tnf: Concentration of TNF in the blood (pg/mL).
        il1: Concentration of IL-1 in the blood (pg/mL).
        active_coag_factor: Concentration of active coagulation factors in the blood (pg/mL).
        il10: Concentration of IL-10 in the blood (pg/mL).
        il6: Concentration of IL-6 in the blood (pg/mL).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of monocyte activation in the blood (cells/(mL*min)).
    """
    activation_rate = (
        mr
        * blood_volume
        * (
            config["INFLAMMATION"]["KMTNF"]
            * fm(tnf, config["INFLAMMATION"]["XMTNF"], 2)
            + config["INFLAMMATION"]["KM1"] * fm(il1, config["INFLAMMATION"]["XM1"], 2)
            + config["INFLAMMATION"]["KMTR"] * trauma
            + config["INFLAMMATION"]["KM_COAG"]
            * fm(active_coag_factor, config["INFLAMMATION"]["XM_COAG"], 2)
        )
        * (
            config["INFLAMMATION"]["XM10"]
            / (
                config["INFLAMMATION"]["XM10"]
                + (config["INFLAMMATION"]["KM10"] * il10) ** 2
            )
        )
        * (
            config["INFLAMMATION"]["XM6"]
            / (
                config["INFLAMMATION"]["XM6"]
                + (config["INFLAMMATION"]["KM6"] * il6) ** 2
            )
        )
    )
    return activation_rate


def mr_prime(
    mr: Tensor,
    _vma: Tensor,
    k_bleed: Tensor,
    d_bv_dt: Tensor,
    blood_volume: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of resting monocytes (Mr) in the blood.

    This function models the dynamics of resting monocytes in the blood, considering their
    baseline production, activation into activated monocytes, loss due to bleeding, and
    changes in blood volume.

    Args:
        mr: Concentration of resting monocytes in the blood (cells/mL).
        _vma: Rate of monocyte activation in the blood (cells/(mL*min)).
        k_bleed: Bleeding rate (1/min).
        d_bv_dt: Rate of change of blood volume (mL/min).
        blood_volume: Current blood volume (mL).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of resting monocytes in the blood (cells/(mL*min)).
    """
    mo_source_enhance = config["INFLAMMATION"]["K_MO_SOURCE_ENHANCE"] * _vma
    d_mr_dt = (
        (config["INFLAMMATION"]["S_MR"] * (1 + mo_source_enhance) - mr)
        * config["INFLAMMATION"]["D_MO_BLOOD"]
        - _vma
        - k_bleed * mr
        - mr * d_bv_dt / blood_volume
    )
    return d_mr_dt


def ma_prime(
    _vma: Tensor,
    ma: Tensor,
    k_b_t_ma: Tensor,
    k_bleed: Tensor,
    d_bv_dt: Tensor,
    blood_volume: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of activated monocytes (Ma) in the blood.

    This function models the dynamics of activated monocytes in the blood, considering their
    production from resting monocytes, migration to tissues, loss due to bleeding, and
    changes in blood volume.

    Args:
        _vma: Rate of monocyte activation in the blood (cells/(mL*min)).
        ma: Concentration of activated monocytes in the blood (cells/mL).
        k_b_t_ma: Rate of monocyte migration from blood to tissue (1/min).
        k_bleed: Bleeding rate (1/min).
        d_bv_dt: Rate of change of blood volume (mL/min).
        blood_volume: Current blood volume (mL).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of activated monocytes in the blood (cells/(mL*min)).
    """
    d_ma_dt = (
        _vma
        - ma * (config["INFLAMMATION"]["D_MO_BLOOD"] + k_b_t_ma + k_bleed)
        - ma * d_bv_dt / blood_volume
    )
    return d_ma_dt


def mo_blood_total(mr: Tensor, ma: Tensor) -> Tensor:
    """
    Calculates the total monocyte count in the blood.

    This function simply sums the concentrations of resting and activated monocytes to
    represent the total monocyte count in the blood.

    Args:
        mr: Concentration of resting monocytes in the blood (cells/mL).
        ma: Concentration of activated monocytes in the blood (cells/mL).

    Returns:
        torch.Tensor: The total monocyte count in the blood (cells/mL).
    """
    return mr + ma


def vna(
    nr: Tensor,
    blood_volume: Tensor,
    trauma: Tensor,
    tnf: Tensor,
    il1: Tensor,
    active_coag_factor: Tensor,
    il10: Tensor,
    il6: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of neutrophil activation in the blood.

    This function models the activation of resting neutrophils (Nr) into activated neutrophils (Na)
    in the blood compartment. The activation is driven by various factors, including TNF, IL-1,
    _trauma, and active coagulation factors. IL-10 and IL-6 have inhibitory effects on this process.

    Args:
        nr: Concentration of resting neutrophils in the blood (cells/mL).
        blood_volume: Current blood volume (mL).
        trauma: Current _trauma level (unitless).
        tnf: Concentration of TNF in the blood (pg/mL).
        il1: Concentration of IL-1 in the blood (pg/mL).
        active_coag_factor: Concentration of active coagulation factors in the blood (pg/mL).
        il10: Concentration of IL-10 in the blood (pg/mL).
        il6: Concentration of IL-6 in the blood (pg/mL).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of neutrophil activation in the blood (cells/(mL*min)).
    """
    activation_rate = (
        nr
        * blood_volume
        * (
            config["INFLAMMATION"]["KNTNF"]
            * fm(tnf, config["INFLAMMATION"]["XNTNF"], 2)
            + config["INFLAMMATION"]["KN1"] * fm(il1, config["INFLAMMATION"]["XN1"], 2)
            + config["INFLAMMATION"]["KNTR"] * trauma
            + config["INFLAMMATION"]["KN_COAG"]
            * fm(active_coag_factor, config["INFLAMMATION"]["XN_COAG"], 2)
        )
        * (
            config["INFLAMMATION"]["XN10"]
            / (
                config["INFLAMMATION"]["XN10"]
                + (config["INFLAMMATION"]["KN10"] * il10) ** 2
            )
        )
        * (
            config["INFLAMMATION"]["XN6"]
            / (
                config["INFLAMMATION"]["XN6"]
                + (config["INFLAMMATION"]["KN6"] * il6) ** 2
            )
        )
    )
    return activation_rate


def nr_prime(
    nr: Tensor,
    _vna: Tensor,
    k_bleed: Tensor,
    d_bv_dt: Tensor,
    blood_volume: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of resting neutrophils (Nr) in the blood.

    This function models the dynamics of resting neutrophils in the blood, considering their
    baseline production, activation into activated neutrophils, loss due to bleeding, and
    changes in blood volume.

    Args:
        nr: Concentration of resting neutrophils in the blood (cells/mL).
        _vna: Rate of neutrophil activation in the blood (cells/(mL*min)).
        k_bleed: Bleeding rate (1/min).
        d_bv_dt: Rate of change of blood volume (mL/min).
        blood_volume: Current blood volume (mL).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of resting neutrophils in the blood (cells/(mL*min)).
    """
    nu_source_enhance = config["INFLAMMATION"]["K_NU_SOURCE_ENHANCE"] * _vna
    d_nr_dt = (
        (config["INFLAMMATION"]["S_NR"] * (1 + nu_source_enhance) - nr)
        * config["INFLAMMATION"]["D_NU_BLOOD"]
        - _vna
        - k_bleed * nr
        - nr * d_bv_dt / blood_volume
    )
    return d_nr_dt


def na_prime(
    _vna: Tensor,
    na: Tensor,
    k_b_t_na: Tensor,
    k_bleed: Tensor,
    d_bv_dt: Tensor,
    blood_volume: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of activated neutrophils (Na) in the blood.

    This function models the dynamics of activated neutrophils in the blood, considering their
    production from resting neutrophils, migration to tissues, loss due to bleeding, and
    changes in blood volume.

    Args:
        _vna: Rate of neutrophil activation in the blood (cells/(mL*min)).
        na: Concentration of activated neutrophils in the blood (cells/mL).
        k_b_t_na: Rate of neutrophil migration from blood to tissue (1/min).
        k_bleed: Bleeding rate (1/min).
        d_bv_dt: Rate of change of blood volume (mL/min).
        blood_volume: Current blood volume (mL).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of activated neutrophils in the blood (cells/(mL*min)).
    """
    d_na_dt = (
        _vna
        - na * (config["INFLAMMATION"]["D_NU_BLOOD"] + k_b_t_na + k_bleed)
        - na * d_bv_dt / blood_volume
    )
    return d_na_dt


def nu_blood_total(nr: Tensor, na: Tensor) -> Tensor:
    """
    Calculates the total neutrophil count in the blood.

    This function simply sums the concentrations of resting and activated neutrophils to
    represent the total neutrophil count in the blood.

    Args:
        nr: Concentration of resting neutrophils in the blood (cells/mL).
        na: Concentration of activated neutrophils in the blood (cells/mL).

    Returns:
        torch.Tensor: The total neutrophil count in the blood (cells/mL).
    """
    return nr + na


def il1_prime(
    na: Tensor,
    ma: Tensor,
    epal: Tensor,
    epas: Tensor,
    il1: Tensor,
    _nu_blood_total: Tensor,
    _mo_blood_total: Tensor,
    _epl_total: Tensor,
    _eps_total: Tensor,
    blood_volume: Tensor,
    k_bleed: Tensor,
    d_bv_dt: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of IL-1 in the blood.

    This function models the dynamics of IL-1 concentration in the blood, considering its
    production by neutrophils, monocytes, and epithelial cells, its natural decay, consumption
    by neutrophils, monocytes, and epithelial cells, loss due to bleeding, and changes in blood volume.

    Args:
        na: Concentration of activated neutrophils in the blood (cells/mL).
        ma: Concentration of activated monocytes in the blood (cells/mL).
        epal: Concentration of activated epithelial cells in the lung (cells).
        epas: Concentration of activated epithelial cells in the splanchnic (cells).
        il1: Concentration of IL-1 in the blood (pg/mL).
        _nu_blood_total: Total neutrophil count in the blood (cells/mL).
        _mo_blood_total: Total monocyte count in the blood (cells/mL).
        _epl_total: Total epithelial cell count in the lung (cells).
        _eps_total: Total epithelial cell count in the splanchnic (cells).
        blood_volume: Current blood volume (mL).
        k_bleed: Bleeding rate (1/min).
        d_bv_dt: Rate of change of blood volume (mL/min).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of IL-1 concentration in the blood (pg/(mL*min)).
    """
    d_il1_dt = (
        config["INFLAMMATION"]["ALPHA_1_PRODUCTION"]
        * (
            config["INFLAMMATION"]["K1N"] * na
            + config["INFLAMMATION"]["K1M"] * ma
            + config["INFLAMMATION"]["K1EP"] * (epal + epas) / blood_volume
        )
        - il1 * (config["INFLAMMATION"]["D_1"] + k_bleed)
        - (
            il1
            * blood_volume
            * (
                config["INFLAMMATION"]["KN1"]
                * fm(_nu_blood_total, config["INFLAMMATION"]["X1N"], 2)
                + config["INFLAMMATION"]["KM1"]
                * fm(_mo_blood_total, config["INFLAMMATION"]["X1M"], 2)
                + (config["INFLAMMATION"]["K_EP1"] / blood_volume)
                * (
                    fm(_epl_total, config["INFLAMMATION"]["X1EP"], 2)
                    + fm(_eps_total, config["INFLAMMATION"]["X1EP"], 2)
                )
            )
        )
        - il1 * d_bv_dt / blood_volume
    )
    return d_il1_dt


def tnf_prime(
    na: Tensor,
    ma: Tensor,
    tnf: Tensor,
    _nu_blood_total: Tensor,
    _mo_blood_total: Tensor,
    _epl_total: Tensor,
    _eps_total: Tensor,
    blood_volume: Tensor,
    k_bleed: Tensor,
    d_bv_dt: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of TNF in the blood.

    This function models the dynamics of TNF concentration in the blood, considering its
    production by neutrophils and monocytes, its natural decay, consumption by neutrophils,
    monocytes, and epithelial cells, loss due to bleeding, and changes in blood volume.

    Args:
        na: Concentration of activated neutrophils in the blood (cells/mL).
        ma: Concentration of activated monocytes in the blood (cells/mL).
        tnf: Concentration of TNF in the blood (pg/mL).
        _nu_blood_total: Total neutrophil count in the blood (cells/mL).
        _mo_blood_total: Total monocyte count in the blood (cells/mL).
        _epl_total: Total epithelial cell count in the lung (cells).
        _eps_total: Total epithelial cell count in the splanchnic (cells).
        blood_volume: Current blood volume (mL).
        k_bleed: Bleeding rate (1/min).
        d_bv_dt: Rate of change of blood volume (mL/min).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of TNF concentration in the blood (pg/(mL*min)).
    """
    d_tnf_dt = (
        config["INFLAMMATION"]["ALPHA_TNF_PRODUCTION"]
        * (config["INFLAMMATION"]["KTNFN"] * na + config["INFLAMMATION"]["KTNFM"] * ma)
        - tnf * (config["INFLAMMATION"]["D_TNF"] + k_bleed)
        - (
            tnf
            * blood_volume
            * (
                config["INFLAMMATION"]["KNTNF"]
                * fm(_nu_blood_total, config["INFLAMMATION"]["XTNFN"], 2)
                + config["INFLAMMATION"]["KMTNF"]
                * fm(_mo_blood_total, config["INFLAMMATION"]["XTNFM"], 2)
                + (config["INFLAMMATION"]["K_EP_TNF"] / blood_volume)
                * (
                    fm(_epl_total, config["INFLAMMATION"]["XTNF_EP"], 2)
                    + fm(_eps_total, config["INFLAMMATION"]["XTNF_EP"], 2)
                )
            )
        )
        - tnf * d_bv_dt / blood_volume
    )
    return d_tnf_dt


def il6_prime(
    na: Tensor,
    ma: Tensor,
    epal: Tensor,
    epas: Tensor,
    il6: Tensor,
    _nu_blood_total: Tensor,
    _mo_blood_total: Tensor,
    _epl_total: Tensor,
    _eps_total: Tensor,
    blood_volume: Tensor,
    k_bleed: Tensor,
    d_bv_dt: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of IL-6 in the blood.

    This function models the dynamics of IL-6 concentration in the blood, considering its
    production by neutrophils, monocytes, and epithelial cells, its natural decay, consumption
    by neutrophils, monocytes, and epithelial cells, loss due to bleeding, and changes in blood volume.

    Args:
        na: Concentration of activated neutrophils in the blood (cells/mL).
        ma: Concentration of activated monocytes in the blood (cells/mL).
        epal: Concentration of activated epithelial cells in the lung (cells).
        epas: Concentration of activated epithelial cells in the splanchnic (cells).
        il6: Concentration of IL-6 in the blood (pg/mL).
        _nu_blood_total: Total neutrophil count in the blood (cells/mL).
        _mo_blood_total: Total monocyte count in the blood (cells/mL).
        _epl_total: Total epithelial cell count in the lung (cells).
        _eps_total: Total epithelial cell count in the splanchnic (cells).
        blood_volume: Current blood volume (mL).
        k_bleed: Bleeding rate (1/min).
        d_bv_dt: Rate of change of blood volume (mL/min).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of IL-6 concentration in the blood (pg/(mL*min)).
    """
    d_il6_dt = (
        config["INFLAMMATION"]["ALPHA_6_PRODUCTION"]
        * (
            config["INFLAMMATION"]["K6N"] * na
            + config["INFLAMMATION"]["K6M"] * ma
            + config["INFLAMMATION"]["K6EP"] * (epal + epas) / blood_volume
        )
        - il6 * (config["INFLAMMATION"]["D_6"] + k_bleed)
        - il6
        * blood_volume
        * (
            config["INFLAMMATION"]["PG_PER_MIN_PER_ML_PER_CELLS"]
            * (
                config["INFLAMMATION"]["KN6"] * _nu_blood_total
                + config["INFLAMMATION"]["KM6"] * _mo_blood_total
            )
            + (config["INFLAMMATION"]["K_EP6"] / blood_volume)
            * (
                fm(_epl_total, config["INFLAMMATION"]["X6EP"], 2)
                + fm(_eps_total, config["INFLAMMATION"]["X6EP"], 2)
            )
        )
        - il6 * d_bv_dt / blood_volume
    )
    return d_il6_dt


def il10_prime(
    na: Tensor,
    ma: Tensor,
    epal: Tensor,
    epas: Tensor,
    il10: Tensor,
    _nu_blood_total: Tensor,
    _mo_blood_total: Tensor,
    _epl_total: Tensor,
    _eps_total: Tensor,
    blood_volume: Tensor,
    k_bleed: Tensor,
    d_bv_dt: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of IL-10 in the blood.

    This function models the dynamics of IL-10 concentration in the blood, considering its
    production by neutrophils, monocytes, and epithelial cells, its baseline production,
    consumption by neutrophils, monocytes, and epithelial cells, loss due to bleeding, and
    changes in blood volume.

    Args:
        na: Concentration of activated neutrophils in the blood (cells/mL).
        ma: Concentration of activated monocytes in the blood (cells/mL).
        epal: Concentration of activated epithelial cells in the lung (cells).
        epas: Concentration of activated epithelial cells in the splanchnic (cells).
        il10: Concentration of IL-10 in the blood (pg/mL).
        _nu_blood_total: Total neutrophil count in the blood (cells/mL).
        _mo_blood_total: Total monocyte count in the blood (cells/mL).
        _epl_total: Total epithelial cell count in the lung (cells).
        _eps_total: Total epithelial cell count in the splanchnic (cells).
        blood_volume: Current blood volume (mL).
        k_bleed: Bleeding rate (1/min).
        d_bv_dt: Rate of change of blood volume (mL/min).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of IL-10 concentration in the blood (pg/(mL*min)).
    """
    d_il10_dt = (
        config["INFLAMMATION"]["ALPHA_10_PRODUCTION"]
        * (
            config["INFLAMMATION"]["K10N"] * na
            + config["INFLAMMATION"]["K10M"] * ma
            + config["INFLAMMATION"]["K10EP"] * (epal + epas) / blood_volume
        )
        + (config["INFLAMMATION"]["S_10"] - il10)
        * config["INFLAMMATION"]["K_BASELINE_IL10"]
        - k_bleed * il10
        - il10
        * blood_volume
        * config["INFLAMMATION"]["PG_PER_CELLS_PER_MIN"]
        * (
            config["INFLAMMATION"]["ONE_PER_ML"]
            * (
                config["INFLAMMATION"]["KN10"] * _nu_blood_total
                + config["INFLAMMATION"]["KM10"] * _mo_blood_total
            )
            + (config["INFLAMMATION"]["K_EP10"] / blood_volume)
            * (_epl_total + _eps_total)
        )
        - il10 * d_bv_dt / blood_volume
    )
    return d_il10_dt


def inos_prime(
    inos: Tensor,  # Added missing argument
    ma: Tensor,
    na: Tensor,
    epal: Tensor,
    epas: Tensor,
    il10: Tensor,
    il10l: Tensor,
    il10s: Tensor,
    no: Tensor,
    blood_volume: Tensor,
    k_bleed: Tensor,
    d_bv_dt: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of inducible nitric oxide synthase (iNOS) in the blood.

    This function models the dynamics of iNOS concentration in the blood, considering its
    production by monocytes, neutrophils, and epithelial cells, its natural decay, and
    loss due to bleeding and changes in blood volume. The production of iNOS is also
    influenced by IL-10 and NO levels.

    Args:
        inos: Concentration of iNOS in the blood (umol/L).
        ma: Concentration of activated monocytes in the blood (cells/mL).
        na: Concentration of activated neutrophils in the blood (cells/mL).
        epal: Concentration of activated epithelial cells in the lung (cells).
        epas: Concentration of activated epithelial cells in the splanchnic (cells).
        il10: Concentration of IL-10 in the blood (pg/mL).
        il10l: Concentration of IL-10 in the lung (pg/mL).
        il10s: Concentration of IL-10 in the splanchnic (pg/mL).
        no: Concentration of nitric oxide in the blood (umol/L).
        blood_volume: Current blood volume (mL).
        k_bleed: Bleeding rate (1/min).
        d_bv_dt: Rate of change of blood volume (mL/min).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of iNOS concentration in the blood (umol/(L*min)).
    """
    d_inos_dt = (
        (
            (
                config["INFLAMMATION"]["KINOSM"] * ma
                + config["INFLAMMATION"]["KINOSN"] * na
            )
            * fs(il10, config["INFLAMMATION"]["XINOS10"], 2)
            * fs(no, config["INFLAMMATION"]["XINOSNO"], 2)
        )
        + (config["INFLAMMATION"]["KINOSEP"] / blood_volume)
        * (
            epal
            * fs(il10 + il10l, config["INFLAMMATION"]["XINOS10"], 2)
            * fs(no, config["INFLAMMATION"]["XINOSNO"], 2)
            + epas
            * fs(il10 + il10s, config["INFLAMMATION"]["XINOS10"], 2)
            * fs(no, config["INFLAMMATION"]["XINOSNO"], 2)
        )
        - inos * (config["INFLAMMATION"]["D_INOS"] + k_bleed)
        - inos * d_bv_dt / blood_volume
    )
    return d_inos_dt


def enos_prime(
    enos: Tensor,
    pe: Tensor,
    blood_volume: Tensor,
    k_bleed: Tensor,
    d_bv_dt: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of endothelial nitric oxide synthase (eNOS) in the blood.

    This function models the dynamics of eNOS concentration in the blood, considering its
    baseline production, natural decay, and loss due to bleeding and changes in blood volume.
    The production of eNOS is also influenced by endotoxin (pe) levels, although this effect
    is currently inactive in the model.

    Args:
        enos: Concentration of eNOS in the blood (umol/L).
        pe: Concentration of endotoxin in the blood (not used currently).
        blood_volume: Current blood volume (mL).
        k_bleed: Bleeding rate (1/min).
        d_bv_dt: Rate of change of blood volume (mL/min).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of eNOS concentration in the blood (umol/(L*min)).
    """
    d_enos_dt = (
        config["INFLAMMATION"]["SOURCE_ENOS"]
        * fs(pe, config["INFLAMMATION"]["XENOSPE"], 2)
        / blood_volume
        - enos * (config["INFLAMMATION"]["D_ENOS"] + k_bleed)
        - enos * d_bv_dt / blood_volume
    )
    return d_enos_dt


def no_prime(
    inos: Tensor,
    enos: Tensor,
    ma: Tensor,
    na: Tensor,
    epal: Tensor,
    epas: Tensor,
    no: Tensor,
    blood_volume: Tensor,
    k_bleed: Tensor,
    d_bv_dt: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of nitric oxide (NO) in the blood.

    This function models the dynamics of NO concentration in the blood, considering its
    production by iNOS, eNOS, monocytes, neutrophils, and epithelial cells, its natural decay,
    and loss due to bleeding and changes in blood volume.

    Args:
        inos: Concentration of iNOS in the blood (umol/L).
        enos: Concentration of eNOS in the blood (umol/L).
        ma: Concentration of activated monocytes in the blood (cells/mL).
        na: Concentration of activated neutrophils in the blood (cells/mL).
        epal: Concentration of activated epithelial cells in the lung (cells).
        epas: Concentration of activated epithelial cells in the splanchnic (cells).
        no: Concentration of nitric oxide in the blood (umol/L).
        blood_volume: Current blood volume (mL).
        k_bleed: Bleeding rate (1/min).
        d_bv_dt: Rate of change of blood volume (mL/min).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of NO concentration in the blood (umol/(L*min)).
    """
    d_no_dt = (
        config["INFLAMMATION"]["K_NO_INOS"] * inos
        + config["INFLAMMATION"]["K_NO_ENOS"] * enos
        + config["INFLAMMATION"]["K_NO_MA"] * ma
        + config["INFLAMMATION"]["K_NO_NA"] * na
        + config["INFLAMMATION"]["K_NO_EP"] * (epal + epas) / blood_volume
        - no * (config["INFLAMMATION"]["D_NO"] + k_bleed + d_bv_dt / blood_volume)
    )
    return d_no_dt


def vmal(
    mrl: Tensor,
    tnfl: Tensor,
    il1l: Tensor,
    trauma: Tensor,
    il10l: Tensor,  # Corrected typo: il101 -> il10l
    il6l: Tensor,  # Corrected typo: il6l -> il6l
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of monocyte activation in the lung.

    This function models the activation of resting monocytes (MrL) into activated monocytes (MaL)
    in the lung compartment. The activation is driven by TNF (TNFL), IL-1 (IL1L), and _trauma.
    IL-10 and IL-6 have inhibitory effects on this process.

    Args:
        mrl: Concentration of resting monocytes in the lung (cells/mL).
        tnfl: Concentration of TNF in the lung (pg/mL).
        il1l: Concentration of IL-1 in the lung (pg/mL).
        trauma: Current _trauma level (unitless).
        il10l: Concentration of IL-10 in the lung (pg/mL). # Corrected typo
        il6l: Concentration of IL-6 in the lung (pg/mL).  # Corrected typo
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of monocyte activation in the lung (cells/(mL*min)).
    """
    activation_rate = (
        mrl
        * config["INFLAMMATION"]["V_LECF"]
        * (
            config["INFLAMMATION"]["KMTNF"]
            * fm(tnfl, config["INFLAMMATION"]["XMTNF"], 2)
            + config["INFLAMMATION"]["KM1"] * fm(il1l, config["INFLAMMATION"]["XM1"], 2)
            + config["INFLAMMATION"]["KMTR"] * trauma
        )
        * (
            config["INFLAMMATION"]["XM10"]
            / (
                config["INFLAMMATION"]["XM10"]
                + (config["INFLAMMATION"]["KM10"] * il10l) ** 2
            )
        )
        * (
            config["INFLAMMATION"]["XM6"]
            / (
                config["INFLAMMATION"]["XM6"]
                + (config["INFLAMMATION"]["KM6"] * il6l) ** 2
            )
        )
    )
    return activation_rate


def mrl_prime(
    mrl: Tensor,
    _vmal: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of resting monocytes (MrL) in the lung.

    This function models the dynamics of resting monocytes in the lung, considering their
    baseline production, activation into activated monocytes, and natural death.

    Args:
        mrl: Concentration of resting monocytes in the lung (cells/mL).
        _vmal: Rate of monocyte activation in the lung (cells/(mL*min)).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of resting monocytes in the lung (cells/(mL*min)).
    """
    mo_source_enhance_l = config["INFLAMMATION"]["K_MO_SOURCE_ENHANCE"] * _vmal
    d_mrl_dt = (
        config["INFLAMMATION"]["S_MR"] * (1 + mo_source_enhance_l) - mrl
    ) * config["INFLAMMATION"]["D_MR_TIS_LUNG"] - _vmal
    return d_mrl_dt


def mal_prime(
    _vmal: Tensor,
    mal: Tensor,
    ma: Tensor,
    blood_volume: Tensor,
    k_b_t_ma: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of activated monocytes (MaL) in the lung.

    This function models the dynamics of activated monocytes in the lung, considering their
    production from resting monocytes, migration from the blood, and natural death.

    Args:
        _vmal: Rate of monocyte activation in the lung (cells/(mL*min)).
        mal: Concentration of activated monocytes in the lung (cells/mL).
        ma: Concentration of activated monocytes in the blood (cells/mL).
        blood_volume: Current blood volume (mL).
        k_b_t_ma: Rate of monocyte migration from blood to tissue (1/min).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of activated monocytes in the lung (cells/(mL*min)).
    """
    d_mal_dt = (
        _vmal
        - mal * config["INFLAMMATION"]["D_MA_TIS_LUNG"]
        + k_b_t_ma
        * (config["INFLAMMATION"]["K_B_T_MA_L"])
        * (ma * blood_volume / config["INFLAMMATION"]["V_LECF"])
    )
    return d_mal_dt


def mo_lung_total(mrl: Tensor, mal: Tensor) -> Tensor:
    """
    Calculates the total monocyte count in the lung.

    This function simply sums the concentrations of resting and activated monocytes to
    represent the total monocyte count in the lung.

    Args:
        mrl: Concentration of resting monocytes in the lung (cells/mL).
        mal: Concentration of activated monocytes in the lung (cells/mL).

    Returns:
        torch.Tensor: The total monocyte count in the lung (cells/mL).
    """
    return mrl + mal


def vnal(
    nrl: Tensor,
    tnfl: Tensor,
    il1l: Tensor,
    trauma: Tensor,
    il10l: Tensor,  # Added missing argument
    il6l: Tensor,  # Added missing argument
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of neutrophil activation in the lung.

    This function models the activation of resting neutrophils (NrL) into activated neutrophils (NaL)
    in the lung compartment. The activation is driven by TNF (TNFL), IL-1 (IL1L), and _trauma.
    IL-10 and IL-6 have inhibitory effects on this process.

    Args:
        nrl: Concentration of resting neutrophils in the lung (cells/mL).
        tnfl: Concentration of TNF in the lung (pg/mL).
        il1l: Concentration of IL-1 in the lung (pg/mL).
        trauma: Current _trauma level (unitless).
        il10l: Concentration of IL-10 in the lung (pg/mL).
        il6l: Concentration of IL-6 in the lung (pg/mL).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of neutrophil activation in the lung (cells/(mL*min)).
    """
    activation_rate = (
        nrl
        * config["INFLAMMATION"]["V_LECF"]
        * (
            config["INFLAMMATION"]["KNTNF"]
            * fm(tnfl, config["INFLAMMATION"]["XNTNF"], 2)
            + config["INFLAMMATION"]["KN1"] * fm(il1l, config["INFLAMMATION"]["XN1"], 2)
            + config["INFLAMMATION"]["KNTR"] * trauma
        )
        * (
            config["INFLAMMATION"]["XN10"]
            / (
                config["INFLAMMATION"]["XN10"]
                + (config["INFLAMMATION"]["KN10"] * il10l) ** 2
            )
        )
        * (
            config["INFLAMMATION"]["XN6"]
            / (
                config["INFLAMMATION"]["XN6"]
                + (config["INFLAMMATION"]["KN6"] * il6l) ** 2
            )
        )
    )
    return activation_rate


def nrl_prime(
    nrl: Tensor,
    _vnal: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of resting neutrophils (NrL) in the lung.

    This function models the dynamics of resting neutrophils in the lung, considering their
    baseline production, activation into activated neutrophils, and natural death.

    Args:
        nrl: Concentration of resting neutrophils in the lung (cells/mL).
        _vnal: Rate of neutrophil activation in the lung (cells/(mL*min)).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of resting neutrophils in the lung (cells/(mL*min)).
    """
    nu_source_enhance_l = config["INFLAMMATION"]["K_NU_SOURCE_ENHANCE"] * _vnal
    d_nrl_dt = (
        config["INFLAMMATION"]["S_NR"] * (1 + nu_source_enhance_l) - nrl
    ) * config["INFLAMMATION"]["D_NU_TIS_LUNG"] - _vnal
    return d_nrl_dt


def nal_prime(
    _vnal: Tensor,
    nal: Tensor,
    na: Tensor,
    blood_volume: Tensor,
    k_b_t_na: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of activated neutrophils (NaL) in the lung.

    This function models the dynamics of activated neutrophils in the lung, considering their
    production from resting neutrophils, migration from the blood, and natural death.

    Args:
        _vnal: Rate of neutrophil activation in the lung (cells/(mL*min)).
        nal: Concentration of activated neutrophils in the lung (cells/mL).
        na: Concentration of activated neutrophils in the blood (cells/mL).
        blood_volume: Current blood volume (mL).
        k_b_t_na: Rate of neutrophil migration from blood to tissue (1/min).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of activated neutrophils in the lung (cells/(mL*min)).
    """
    d_nal_dt = (
        _vnal
        - nal * config["INFLAMMATION"]["D_NU_TIS_LUNG"]
        + k_b_t_na
        * (config["INFLAMMATION"]["K_B_T_NA_L"])
        * (na * blood_volume / config["INFLAMMATION"]["V_LECF"])
    )
    return d_nal_dt


def nu_lung_total(nrl: Tensor, nal: Tensor) -> Tensor:
    """
    Calculates the total neutrophil count in the lung.

    This function simply sums the concentrations of resting and activated neutrophils to
    represent the total neutrophil count in the lung.

    Args:
        nrl: Concentration of resting neutrophils in the lung (cells/mL).
        nal: Concentration of activated neutrophils in the lung (cells/mL).

    Returns:
        torch.Tensor: The total neutrophil count in the lung (cells/mL).
    """
    return nrl + nal


def vepal(
    eprl: Tensor,
    tnf: Tensor,
    tnfl: Tensor,
    il6: Tensor,
    il6l: Tensor,
    il1: Tensor,
    il1l: Tensor,
    trauma: Tensor,
    active_coag_factor: Tensor,
    il10: Tensor,
    il10l: Tensor,
    blood_volume: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of epithelial cell activation in the lung.

    This function models the activation of resting epithelial cells (EPrL) into activated
    epithelial cells (EPaL) in the lung compartment. The activation is driven by TNF, IL-6,
    IL-1, _trauma, and active coagulation factors. IL-10 has an inhibitory effect on this process.

    Args:
        eprl: Number of resting epithelial cells in the lung (cells).
        tnf: Concentration of TNF in the blood (pg/mL).
        tnfl: Concentration of TNF in the lung (pg/mL).
        il6: Concentration of IL-6 in the blood (pg/mL).
        il6l: Concentration of IL-6 in the lung (pg/mL).
        il1: Concentration of IL-1 in the blood (pg/mL).
        il1l: Concentration of IL-1 in the lung (pg/mL).
        trauma: Current _trauma level (unitless).
        active_coag_factor: Concentration of active coagulation factors in the blood (pg/mL).
        il10: Concentration of IL-10 in the blood (pg/mL).
        il10l: Concentration of IL-10 in the lung (pg/mL).
        blood_volume: Current blood volume (mL).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of epithelial cell activation in the lung (cells/min).
    """
    activation_rate = (
        eprl
        * (
            config["INFLAMMATION"]["K_EP_TNF"]
            * fm(
                tnf * blood_volume + tnfl * config["INFLAMMATION"]["V_LECF"],
                config["INFLAMMATION"]["X_EP_TNF"],
                2,
            )
            + config["INFLAMMATION"]["K_EP6"]
            * fm(
                il6 * blood_volume + il6l * config["INFLAMMATION"]["V_LECF"],
                config["INFLAMMATION"]["X_EP6"],
                2,
            )
            + config["INFLAMMATION"]["K_EP1"]
            * fm(
                il1 * blood_volume + il1l * config["INFLAMMATION"]["V_LECF"],
                config["INFLAMMATION"]["X_EP1"],
                2,
            )
            + config["INFLAMMATION"]["KEPTR"] * trauma
            + config["INFLAMMATION"]["K_EP_COAG"]
            * fm(
                active_coag_factor * blood_volume,
                config["INFLAMMATION"]["X_EP_COAG"],
                2,
            )
        )
        * (
            config["INFLAMMATION"]["XEP10"]
            / (
                config["INFLAMMATION"]["XEP10"]
                + (
                    config["INFLAMMATION"]["K_EP10"]
                    * (il10 * blood_volume + il10l * config["INFLAMMATION"]["V_LECF"])
                )
                ** 2
            )
        )
    )
    return activation_rate


def eprl_prime(
    _vepal: Tensor,
    eprl: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of resting epithelial cells (EPrL) in the lung.

    This function models the dynamics of resting epithelial cells in the lung, considering their
    activation into activated epithelial cells and natural death.

    Args:
        _vepal: Rate of epithelial cell activation in the lung (cells/min).
        eprl: Number of resting epithelial cells in the lung (cells).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of resting epithelial cells in the lung (cells/min).
    """
    d_eprl_dt = -_vepal - config["INFLAMMATION"]["D_EP"] * (
        eprl - config["INFLAMMATION"]["S_EPRL"]
    )
    return d_eprl_dt


def epal_prime(
    _vepal: Tensor,
    epal: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of activated epithelial cells (EPaL) in the lung.

    This function models the dynamics of activated epithelial cells in the lung, considering their
    production from resting epithelial cells and natural death.

    Args:
        _vepal: Rate of epithelial cell activation in the lung (cells/min).
        epal: Number of activated epithelial cells in the lung (cells).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of activated epithelial cells in the lung (cells/min).
    """
    d_epal_dt = _vepal - config["INFLAMMATION"]["D_EP"] * epal
    return d_epal_dt


def epl_total(eprl: Tensor, epal: Tensor) -> Tensor:
    """
    Calculates the total epithelial cell count in the lung.

    This function simply sums the number of resting and activated epithelial cells to
    represent the total epithelial cell count in the lung.

    Args:
        eprl: Number of resting epithelial cells in the lung (cells).
        epal: Number of activated epithelial cells in the lung (cells).

    Returns:
        torch.Tensor: The total epithelial cell count in the lung (cells).
    """
    return eprl + epal


def il1l_prime(
    nal: Tensor,
    mal: Tensor,
    epal: Tensor,
    il1l: Tensor,
    _nu_lung_total: Tensor,
    _mo_lung_total: Tensor,
    _epl_total: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of IL-1 (IL1L) in the lung.

    This function models the dynamics of IL-1 concentration in the lung, considering its
    production by neutrophils, monocytes, and epithelial cells, its natural decay, and
    consumption by neutrophils, monocytes, and epithelial cells.

    Args:
        nal: Concentration of activated neutrophils in the lung (cells/mL).
        mal: Concentration of activated monocytes in the lung (cells/mL).
        epal: Number of activated epithelial cells in the lung (cells).
        il1l: Concentration of IL-1 in the lung (pg/mL).
        _nu_lung_total: Total neutrophil count in the lung (cells/mL).
        _mo_lung_total: Total monocyte count in the lung (cells/mL).
        _epl_total: Total epithelial cell count in the lung (cells).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of IL-1 concentration in the lung (pg/(mL*min)).
    """
    d_il1l_dt = (
        config["INFLAMMATION"]["ALPHA_1_PRODUCTION"]
        * (
            config["INFLAMMATION"]["K1N"] * nal
            + config["INFLAMMATION"]["K1M"] * mal
            + config["INFLAMMATION"]["K1EP"] * epal / config["INFLAMMATION"]["V_LECF"]
        )
        - config["INFLAMMATION"]["D_1"] * il1l
        - (
            il1l
            * config["INFLAMMATION"]["V_LECF"]
            * (
                config["INFLAMMATION"]["KN1"]
                * fm(_nu_lung_total, config["INFLAMMATION"]["X1N"], 2)
                + config["INFLAMMATION"]["KM1"]
                * fm(_mo_lung_total, config["INFLAMMATION"]["X1M"], 2)
                + (config["INFLAMMATION"]["K_EP1"] / config["INFLAMMATION"]["V_LECF"])
                * fm(_epl_total, config["INFLAMMATION"]["X1EP"], 2)
            )
        )
    )
    return d_il1l_dt


def tnfl_prime(
    nal: Tensor,
    mal: Tensor,
    tnfl: Tensor,
    _nu_lung_total: Tensor,
    _mo_lung_total: Tensor,
    _epl_total: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of TNF (TNFL) in the lung.

    This function models the dynamics of TNF concentration in the lung, considering its
    production by neutrophils and monocytes, its natural decay, and consumption by neutrophils,
    monocytes, and epithelial cells.

    Args:
        nal: Concentration of activated neutrophils in the lung (cells/mL).
        mal: Concentration of activated monocytes in the lung (cells/mL).
        tnfl: Concentration of TNF in the lung (pg/mL).
        _nu_lung_total: Total neutrophil count in the lung (cells/mL).
        _mo_lung_total: Total monocyte count in the lung (cells/mL).
        _epl_total: Total epithelial cell count in the lung (cells).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of TNF concentration in the lung (pg/(mL*min)).
    """
    d_tnfl_dt = (
        config["INFLAMMATION"]["ALPHA_TNF_PRODUCTION"]
        * (
            config["INFLAMMATION"]["KTNFN"] * nal
            + config["INFLAMMATION"]["KTNFM"] * mal
        )
        - config["INFLAMMATION"]["D_TNF"] * tnfl
        - (
            tnfl
            * config["INFLAMMATION"]["V_LECF"]
            * (
                config["INFLAMMATION"]["KNTNF"]
                * fm(_nu_lung_total, config["INFLAMMATION"]["XTNFN"], 2)
                + config["INFLAMMATION"]["KMTNF"]
                * fm(_mo_lung_total, config["INFLAMMATION"]["XTNFM"], 2)
                + (
                    config["INFLAMMATION"]["K_EP_TNF"]
                    / config["INFLAMMATION"]["V_LECF"]
                )
                * fm(_epl_total, config["INFLAMMATION"]["XTNF_EP"], 2)
            )
        )
    )
    return d_tnfl_dt


def il6l_prime(
    nal: Tensor,
    mal: Tensor,
    epal: Tensor,
    il6l: Tensor,
    _nu_lung_total: Tensor,
    _mo_lung_total: Tensor,
    _epl_total: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of IL-6 (IL6L) in the lung.

    This function models the dynamics of IL-6 concentration in the lung, considering its
    production by neutrophils, monocytes, and epithelial cells, its natural decay, and
    consumption by neutrophils, monocytes, and epithelial cells.

    Args:
        nal: Concentration of activated neutrophils in the lung (cells/mL).
        mal: Concentration of activated monocytes in the lung (cells/mL).
        epal: Number of activated epithelial cells in the lung (cells).
        il6l: Concentration of IL-6 in the lung (pg/mL).
        _nu_lung_total: Total neutrophil count in the lung (cells/mL).
        _mo_lung_total: Total monocyte count in the lung (cells/mL).
        _epl_total: Total epithelial cell count in the lung (cells).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of IL-6 concentration in the lung (pg/(mL*min)).
    """
    d_il6l_dt = (
        config["INFLAMMATION"]["ALPHA_6_PRODUCTION"]
        * (
            config["INFLAMMATION"]["K6N"] * nal
            + config["INFLAMMATION"]["K6M"] * mal
            + config["INFLAMMATION"]["K6EP"] * epal / config["INFLAMMATION"]["V_LECF"]
        )
        - config["INFLAMMATION"]["D_6"] * il6l
        - il6l
        * config["INFLAMMATION"]["V_LECF"]
        * (
            config["INFLAMMATION"]["PG_PER_MIN_PER_ML_PER_CELLS"]
            * (
                config["INFLAMMATION"]["KN6"] * _nu_lung_total
                + config["INFLAMMATION"]["KM6"] * _mo_lung_total
            )
            + (config["INFLAMMATION"]["K_EP6"] / config["INFLAMMATION"]["V_LECF"])
            * fm(_epl_total, config["INFLAMMATION"]["X6EP"], 2)
        )
    )
    return d_il6l_dt


def il10l_prime(
    nal: Tensor,
    mal: Tensor,
    epal: Tensor,
    il10l: Tensor,
    _nu_lung_total: Tensor,
    _mo_lung_total: Tensor,
    _epl_total: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of IL-10 (IL10L) in the lung.

    This function models the dynamics of IL-10 concentration in the lung, considering its
    production by neutrophils, monocytes, and epithelial cells, its baseline production,
    consumption by neutrophils, monocytes, and epithelial cells.

    Args:
        nal: Concentration of activated neutrophils in the lung (cells/mL).
        mal: Concentration of activated monocytes in the lung (cells/mL).
        epal: Number of activated epithelial cells in the lung (cells).
        il10l: Concentration of IL-10 in the lung (pg/mL).
        _nu_lung_total: Total neutrophil count in the lung (cells/mL).
        _mo_lung_total: Total monocyte count in the lung (cells/mL).
        _epl_total: Total epithelial cell count in the lung (cells).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of IL-10 concentration in the lung (pg/(mL*min)).
    """
    d_il10l_dt = (
        config["INFLAMMATION"]["ALPHA_10_PRODUCTION"]
        * (
            config["INFLAMMATION"]["K10N"] * nal
            + config["INFLAMMATION"]["K10M"] * mal
            + config["INFLAMMATION"]["K10EP"] * epal / config["INFLAMMATION"]["V_LECF"]
        )
        + config["INFLAMMATION"]["K_BASELINE_IL10"]
        * (config["INFLAMMATION"]["S_10"] - il10l)
        - il10l
        * config["INFLAMMATION"]["V_LECF"]
        * config["INFLAMMATION"]["PG_PER_CELLS_PER_MIN"]
        * (
            config["INFLAMMATION"]["ONE_PER_ML"]
            * (
                config["INFLAMMATION"]["KN10"] * _nu_lung_total
                + config["INFLAMMATION"]["KM10"] * _mo_lung_total
            )
            + (config["INFLAMMATION"]["K_EP10"] / config["INFLAMMATION"]["V_LECF"])
            * _epl_total
        )
    )
    return d_il10l_dt


# --------------------------------------------------------------------------------------------------
# Splanchnic Compartment Functions
# --------------------------------------------------------------------------------------------------


def damp_prime(
    damp: Tensor,
    _eps_total: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of damage-associated molecular patterns (DAMPs) in the splanchnic.

    This function models the dynamics of DAMPs in the splanchnic compartment, considering their
    consumption by activated epithelial cells (EPS_total).

    Args:
        damp: Concentration of DAMPs in the splanchnic (pg/mL).
        _eps_total: Total epithelial cell count in the splanchnic (cells).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of DAMP concentration in the splanchnic (pg/(mL*min)).
    """
    d_damp_dt = (
        -damp
        * config["INFLAMMATION"]["K_EP_DAMP"]
        * fm(_eps_total, config["INFLAMMATION"]["XDAMP_EP"], 2)
    )
    return d_damp_dt


def vmas(
    mrs: Tensor,
    tnfs: Tensor,
    il1s: Tensor,
    trauma: Tensor,
    il10s: Tensor,  # Corrected typo: il10s -> il10s
    il6s: Tensor,  # Corrected typo: il6s -> il6s
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of monocyte activation in the splanchnic.

    This function models the activation of resting monocytes (MrS) into activated monocytes (MaS)
    in the splanchnic compartment. The activation is driven by TNF (TNFS), IL-1 (IL1S), and _trauma.
    IL-10 and IL-6 have inhibitory effects on this process.

    Args:
        mrs: Concentration of resting monocytes in the splanchnic (cells/mL).
        tnfs: Concentration of TNF in the splanchnic (pg/mL).
        il1s: Concentration of IL-1 in the splanchnic (pg/mL).
        trauma: Current _trauma level (unitless).
        il10s: Concentration of IL-10 in the splanchnic (pg/mL). # Corrected typo
        il6s: Concentration of IL-6 in the splanchnic (pg/mL).  # Corrected typo
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of monocyte activation in the splanchnic (cells/(mL*min)).
    """
    activation_rate = (
        mrs
        * config["INFLAMMATION"]["V_SECF"]
        * (
            config["INFLAMMATION"]["KMTNF"]
            * fm(tnfs, config["INFLAMMATION"]["XMTNF"], 2)
            + config["INFLAMMATION"]["KM1"] * fm(il1s, config["INFLAMMATION"]["XM1"], 2)
            + config["INFLAMMATION"]["KMTR"] * trauma
        )
        * (
            config["INFLAMMATION"]["XM10"]
            / (
                config["INFLAMMATION"]["XM10"]
                + (config["INFLAMMATION"]["KM10"] * il10s) ** 2
            )
        )
        * (
            config["INFLAMMATION"]["XM6"]
            / (
                config["INFLAMMATION"]["XM6"]
                + (config["INFLAMMATION"]["KM6"] * il6s) ** 2
            )
        )
    )
    return activation_rate


def mrs_prime(
    mrs: Tensor,
    _vmas: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of resting monocytes (MrS) in the splanchnic.

    This function models the dynamics of resting monocytes in the splanchnic, considering their
    baseline production, activation into activated monocytes, and natural death.

    Args:
        mrs: Concentration of resting monocytes in the splanchnic (cells/mL).
        _vmas: Rate of monocyte activation in the splanchnic (cells/(mL*min)).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of resting monocytes in the splanchnic (cells/(mL*min)).
    """
    mo_source_enhance_s = config["INFLAMMATION"]["K_MO_SOURCE_ENHANCE"] * _vmas
    d_mrs_dt = (
        config["INFLAMMATION"]["S_MR"] * (1 + mo_source_enhance_s) - mrs
    ) * config["INFLAMMATION"]["D_MR_TIS_LUNG"] - _vmas
    return d_mrs_dt


def mas_prime(
    _vmas: Tensor,
    mas: Tensor,
    ma: Tensor,
    blood_volume: Tensor,
    k_b_t_ma: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of activated monocytes (MaS) in the splanchnic.

    This function models the dynamics of activated monocytes in the splanchnic, considering their
    production from resting monocytes, migration from the blood, and natural death.

    Args:
        _vmas: Rate of monocyte activation in the splanchnic (cells/(mL*min)).
        mas: Concentration of activated monocytes in the splanchnic (cells/mL).
        ma: Concentration of activated monocytes in the blood (cells/mL).
        blood_volume: Current blood volume (mL).
        k_b_t_ma: Rate of monocyte migration from blood to tissue (1/min).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of activated monocytes in the splanchnic (cells/(mL*min)).
    """
    d_mas_dt = (
        _vmas
        - mas * config["INFLAMMATION"]["D_MA_TIS_LUNG"]
        + k_b_t_ma
        * (1 - config["INFLAMMATION"]["K_B_T_MA_L"])
        * (ma * blood_volume / config["INFLAMMATION"]["V_SECF"])
    )
    return d_mas_dt


def mo_splanchnic_total(mrs: Tensor, mas: Tensor) -> Tensor:
    """
    Calculates the total monocyte count in the splanchnic.

    This function simply sums the concentrations of resting and activated monocytes to
    represent the total monocyte count in the splanchnic.

    Args:
        mrs: Concentration of resting monocytes in the splanchnic (cells/mL).
        mas: Concentration of activated monocytes in the splanchnic (cells/mL).

    Returns:
        torch.Tensor: The total monocyte count in the splanchnic (cells/mL).
    """
    return mrs + mas


def vnas(
    nrs: Tensor,
    tnfs: Tensor,
    il1s: Tensor,
    trauma: Tensor,
    il10s: Tensor,  # Added missing argument
    il6s: Tensor,  # Added missing argument
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of neutrophil activation in the splanchnic.

    This function models the activation of resting neutrophils (NrS) into activated neutrophils (NaS)
    in the splanchnic compartment. The activation is driven by TNF (TNFS), IL-1 (IL1S), and _trauma.
    IL-10 and IL-6 have inhibitory effects on this process.

    Args:
        nrs: Concentration of resting neutrophils in the splanchnic (cells/mL).
        tnfs: Concentration of TNF in the splanchnic (pg/mL).
        il1s: Concentration of IL-1 in the splanchnic (pg/mL).
        trauma: Current _trauma level (unitless).
        il10s: Concentration of IL-10 in the splanchnic (pg/mL).
        il6s: Concentration of IL-6 in the splanchnic (pg/mL).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of neutrophil activation in the splanchnic (cells/(mL*min)).
    """
    activation_rate = (
        nrs
        * config["INFLAMMATION"]["V_SECF"]
        * (
            config["INFLAMMATION"]["KNTNF"]
            * fm(tnfs, config["INFLAMMATION"]["XNTNF"], 2)
            + config["INFLAMMATION"]["KN1"] * fm(il1s, config["INFLAMMATION"]["XN1"], 2)
            + config["INFLAMMATION"]["KNTR"] * trauma
        )
        * (
            config["INFLAMMATION"]["XN10"]
            / (
                config["INFLAMMATION"]["XN10"]
                + (config["INFLAMMATION"]["KN10"] * il10s) ** 2
            )
        )
        * (
            config["INFLAMMATION"]["XN6"]
            / (
                config["INFLAMMATION"]["XN6"]
                + (config["INFLAMMATION"]["KN6"] * il6s) ** 2
            )
        )
    )
    return activation_rate


def nrs_prime(
    nrs: Tensor,
    _vnas: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of resting neutrophils (NrS) in the splanchnic.

    This function models the dynamics of resting neutrophils in the splanchnic, considering their
    baseline production, activation into activated neutrophils, and natural death.

    Args:
        nrs: Concentration of resting neutrophils in the splanchnic (cells/mL).
        _vnas: Rate of neutrophil activation in the splanchnic (cells/(mL*min)).
        config: Dictionary containing model parameters.
           Returns:
        torch.Tensor: The rate of change of resting neutrophils in the splanchnic (cells/(mL*min)).
    """
    nu_source_enhance_s = config["INFLAMMATION"]["K_NU_SOURCE_ENHANCE"] * _vnas
    d_nrs_dt = (
        config["INFLAMMATION"]["S_NR"] * (1 + nu_source_enhance_s) - nrs
    ) * config["INFLAMMATION"]["D_NU_TIS_LUNG"] - _vnas
    return d_nrs_dt


def nas_prime(
    _vnas: Tensor,
    nas: Tensor,
    na: Tensor,
    blood_volume: Tensor,
    k_b_t_na: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of activated neutrophils (NaS) in the splanchnic.

    This function models the dynamics of activated neutrophils in the splanchnic, considering their
    production from resting neutrophils, migration from the blood, and natural death.

    Args:
        _vnas: Rate of neutrophil activation in the splanchnic (cells/(mL*min)).
        nas: Concentration of activated neutrophils in the splanchnic (cells/mL).
        na: Concentration of activated neutrophils in the blood (cells/mL).
        blood_volume: Current blood volume (mL).
        k_b_t_na: Rate of neutrophil migration from blood to tissue (1/min).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of activated neutrophils in the splanchnic (cells/(mL*min)).
    """
    d_nas_dt = (
        _vnas
        - nas * config["INFLAMMATION"]["D_NU_TIS_LUNG"]
        + k_b_t_na
        * (1 - config["INFLAMMATION"]["K_B_T_NA_L"])
        * (na * blood_volume / config["INFLAMMATION"]["V_SECF"])
    )
    return d_nas_dt


def nu_splanchnic_total(nrs: Tensor, nas: Tensor) -> Tensor:
    """
    Calculates the total neutrophil count in the splanchnic.

    This function simply sums the concentrations of resting and activated neutrophils to
    represent the total neutrophil count in the splanchnic.

    Args:
        nrs: Concentration of resting neutrophils in the splanchnic (cells/mL).
        nas: Concentration of activated neutrophils in the splanchnic (cells/mL).

    Returns:
        torch.Tensor: The total neutrophil count in the splanchnic (cells/mL).
    """
    return nrs + nas


def vepas(
    eprs: Tensor,
    damp: Tensor,
    tnf: Tensor,
    tnfs: Tensor,
    il6: Tensor,
    il6s: Tensor,
    il1: Tensor,
    il1s: Tensor,
    trauma: Tensor,
    active_coag_factor: Tensor,
    il10: Tensor,
    il10s: Tensor,
    blood_volume: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of epithelial cell activation in the splanchnic.

    This function models the activation of resting epithelial cells (EPrS) into activated
    epithelial cells (EPaS) in the splanchnic compartment. The activation is driven by DAMPs,
    TNF, IL-6, IL-1, _trauma, and active coagulation factors. IL-10 has an inhibitory effect
    on this process.

    Args:
        eprs: Number of resting epithelial cells in the splanchnic (cells).
        damp: Concentration of DAMPs in the splanchnic (pg/mL).
        tnf: Concentration of TNF in the blood (pg/mL).
        tnfs: Concentration of TNF in the splanchnic (pg/mL).
        il6: Concentration of IL-6 in the blood (pg/mL).
        il6s: Concentration of IL-6 in the splanchnic (pg/mL).
        il1: Concentration of IL-1 in the blood (pg/mL).
        il1s: Concentration of IL-1 in the splanchnic (pg/mL).
        trauma: Current _trauma level (unitless).
        active_coag_factor: Concentration of active coagulation factors in the blood (pg/mL).
        il10: Concentration of IL-10 in the blood (pg/mL).
        il10s: Concentration of IL-10 in the splanchnic (pg/mL).
        blood_volume: Current blood volume (mL).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of epithelial cell activation in the splanchnic (cells/min).
    """
    activation_rate = (
        eprs
        * (
            config["INFLAMMATION"]["ALPHA_DAMP_SENS"]
            * config["INFLAMMATION"]["K_EP_DAMP"]
            * fm(
                damp * config["INFLAMMATION"]["V_SECF"],
                config["INFLAMMATION"]["X_EP_DAMP"],
                2,
            )
            + config["INFLAMMATION"]["K_EP_TNF"]
            * fm(
                tnf * blood_volume + tnfs * config["INFLAMMATION"]["V_SECF"],
                config["INFLAMMATION"]["X_EP_TNF"],
                2,
            )
            + config["INFLAMMATION"]["K_EP6"]
            * fm(
                il6 * blood_volume + il6s * config["INFLAMMATION"]["V_SECF"],
                config["INFLAMMATION"]["X_EP6"],
                2,
            )
            + config["INFLAMMATION"]["K_EP1"]
            * fm(
                il1 * blood_volume + il1s * config["INFLAMMATION"]["V_SECF"],
                config["INFLAMMATION"]["X_EP1"],
                2,
            )
            + config["INFLAMMATION"]["KEPTR"] * trauma
            + config["INFLAMMATION"]["K_EP_COAG"]
            * fm(
                active_coag_factor * blood_volume,
                config["INFLAMMATION"]["X_EP_COAG"],
                2,
            )
        )
        * (
            config["INFLAMMATION"]["XEP10"]
            / (
                config["INFLAMMATION"]["XEP10"]
                + (
                    config["INFLAMMATION"]["K_EP10"]
                    * (il10 * blood_volume + il10s * config["INFLAMMATION"]["V_SECF"])
                )
                ** 2
            )
        )
    )
    return activation_rate


def eprs_prime(
    _vepas: Tensor,
    eprs: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of resting epithelial cells (EPrS) in the splanchnic.

    This function models the dynamics of resting epithelial cells in the splanchnic, considering
    their activation into activated epithelial cells and natural death.

    Args:
        _vepas: Rate of epithelial cell activation in the splanchnic (cells/min).
        eprs: Number of resting epithelial cells in the splanchnic (cells).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of resting epithelial cells in the splanchnic (cells/min).
    """
    d_eprs_dt = -_vepas - config["INFLAMMATION"]["D_EP"] * (
        eprs - config["INFLAMMATION"]["S_EPRS"]
    )
    return d_eprs_dt


def epas_prime(
    _vepas: Tensor,
    epas: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of activated epithelial cells (EPaS) in the splanchnic.

    This function models the dynamics of activated epithelial cells in the splanchnic, considering
    their production from resting epithelial cells and natural death.

    Args:
        _vepas: Rate of epithelial cell activation in the splanchnic (cells/min).
        epas: Number of activated epithelial cells in the splanchnic (cells).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of activated epithelial cells in the splanchnic (cells/min).
    """
    d_epas_dt = _vepas - config["INFLAMMATION"]["D_EP"] * epas
    return d_epas_dt


def eps_total(eprs: Tensor, epas: Tensor) -> Tensor:
    """
    Calculates the total epithelial cell count in the splanchnic.

    This function simply sums the number of resting and activated epithelial cells to
    represent the total epithelial cell count in the splanchnic.

    Args:
        eprs: Number of resting epithelial cells in the splanchnic (cells).
        epas: Number of activated epithelial cells in the splanchnic (cells).

    Returns:
        torch.Tensor: The total epithelial cell count in the splanchnic (cells).
    """
    return eprs + epas


def il1s_prime(
    nas: Tensor,
    mas: Tensor,
    epas: Tensor,
    il1s: Tensor,
    _nu_splanchnic_total: Tensor,
    _mo_splanchnic_total: Tensor,
    _eps_total: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of IL-1 (IL1S) in the splanchnic.

    This function models the dynamics of IL-1 concentration in the splanchnic, considering its
    production by neutrophils, monocytes, and epithelial cells, its natural decay, and
    consumption by neutrophils, monocytes, and epithelial cells.

    Args:
        nas: Concentration of activated neutrophils in the splanchnic (cells/mL).
        mas: Concentration of activated monocytes in the splanchnic (cells/mL).
        epas: Number of activated epithelial cells in the splanchnic (cells).
        il1s: Concentration of IL-1 in the splanchnic (pg/mL).
        _nu_splanchnic_total: Total neutrophil count in the splanchnic (cells/mL).
        _mo_splanchnic_total: Total monocyte count in the splanchnic (cells/mL).
        _eps_total: Total epithelial cell count in the splanchnic (cells).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of IL-1 concentration in the splanchnic (pg/(mL*min)).
    """
    d_il1s_dt = (
        config["INFLAMMATION"]["ALPHA_1_PRODUCTION"]
        * (
            config["INFLAMMATION"]["K1N"] * nas
            + config["INFLAMMATION"]["K1M"] * mas
            + config["INFLAMMATION"]["K1EP"] * epas / config["INFLAMMATION"]["V_SECF"]
        )
        - config["INFLAMMATION"]["D_1"] * il1s
        - (
            il1s
            * config["INFLAMMATION"]["V_SECF"]
            * (
                config["INFLAMMATION"]["KN1"]
                * fm(_nu_splanchnic_total, config["INFLAMMATION"]["X1N"], 2)
                + config["INFLAMMATION"]["KM1"]
                * fm(_mo_splanchnic_total, config["INFLAMMATION"]["X1M"], 2)
                + (config["INFLAMMATION"]["K_EP1"] / config["INFLAMMATION"]["V_SECF"])
                * fm(_eps_total, config["INFLAMMATION"]["X1EP"], 2)
            )
        )
    )
    return d_il1s_dt


def tnfs_prime(
    nas: Tensor,
    mas: Tensor,
    tnfs: Tensor,
    _nu_splanchnic_total: Tensor,
    _mo_splanchnic_total: Tensor,
    _eps_total: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of TNF (TNFS) in the splanchnic.

    This function models the dynamics of TNF concentration in the splanchnic, considering its
    production by neutrophils and monocytes, its natural decay, and consumption by neutrophils,
    monocytes, and epithelial cells.

    Args:
        nas: Concentration of activated neutrophils in the splanchnic (cells/mL).
        mas: Concentration of activated monocytes in the splanchnic (cells/mL).
        tnfs: Concentration of TNF in the splanchnic (pg/mL).
        _nu_splanchnic_total: Total neutrophil count in the splanchnic (cells/mL).
        _mo_splanchnic_total: Total monocyte count in the splanchnic (cells/mL).
        _eps_total: Total epithelial cell count in the splanchnic (cells).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of TNF concentration in the splanchnic (pg/(mL*min)).
    """
    d_tnfs_dt = (
        config["INFLAMMATION"]["ALPHA_TNF_PRODUCTION"]
        * (
            config["INFLAMMATION"]["KTNFN"] * nas
            + config["INFLAMMATION"]["KTNFM"] * mas
        )
        - config["INFLAMMATION"]["D_TNF"] * tnfs
        - (
            tnfs
            * config["INFLAMMATION"]["V_SECF"]
            * (
                config["INFLAMMATION"]["KNTNF"]
                * fm(_nu_splanchnic_total, config["INFLAMMATION"]["XTNFN"], 2)
                + config["INFLAMMATION"]["KMTNF"]
                * fm(_mo_splanchnic_total, config["INFLAMMATION"]["XTNFM"], 2)
                + (
                    config["INFLAMMATION"]["K_EP_TNF"]
                    / config["INFLAMMATION"]["V_SECF"]
                )
                * fm(_eps_total, config["INFLAMMATION"]["XTNF_EP"], 2)
            )
        )
    )
    return d_tnfs_dt


def il6s_prime(
    nas: Tensor,
    mas: Tensor,
    epas: Tensor,
    il6s: Tensor,
    _nu_splanchnic_total: Tensor,
    _mo_splanchnic_total: Tensor,
    _eps_total: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of IL-6 (IL6S) in the splanchnic.

    This function models the dynamics of IL-6 concentration in the splanchnic, considering its
    production by neutrophils, monocytes, and epithelial cells, its natural decay, and
    consumption by neutrophils, monocytes, and epithelial cells.

    Args:
        nas: Concentration of activated neutrophils in the splanchnic (cells/mL).
        mas: Concentration of activated monocytes in the splanchnic (cells/mL).
        epas: Number of activated epithelial cells in the splanchnic (cells).
        il6s: Concentration of IL-6 in the splanchnic (pg/mL).
        _nu_splanchnic_total: Total neutrophil count in the splanchnic (cells/mL).
        _mo_splanchnic_total: Total monocyte count in the splanchnic (cells/mL).
        _eps_total: Total epithelial cell count in the splanchnic (cells).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of IL-6 concentration in the splanchnic (pg/(mL*min)).
    """
    d_il6s_dt = (
        config["INFLAMMATION"]["ALPHA_6_PRODUCTION"]
        * (
            config["INFLAMMATION"]["K6N"] * nas
            + config["INFLAMMATION"]["K6M"] * mas
            + config["INFLAMMATION"]["K6EP"] * epas / config["INFLAMMATION"]["V_SECF"]
        )
        - config["INFLAMMATION"]["D_6"] * il6s
        - il6s
        * config["INFLAMMATION"]["V_SECF"]
        * (
            config["INFLAMMATION"]["PG_PER_MIN_PER_ML_PER_CELLS"]
            * (
                config["INFLAMMATION"]["KN6"] * _nu_splanchnic_total
                + config["INFLAMMATION"]["KM6"] * _mo_splanchnic_total
            )
            + (config["INFLAMMATION"]["K_EP6"] / config["INFLAMMATION"]["V_SECF"])
            * fm(_eps_total, config["INFLAMMATION"]["X6EP"], 2)
        )
    )
    return d_il6s_dt


def il10s_prime(
    nas: Tensor,
    mas: Tensor,
    epas: Tensor,
    il10s: Tensor,
    _nu_splanchnic_total: Tensor,
    _mo_splanchnic_total: Tensor,
    _eps_total: Tensor,
    config: Dict,
) -> Tensor:
    """
    Calculates the rate of change of IL-10 (IL10S) in the splanchnic.

    This function models the dynamics of IL-10 concentration in the splanchnic, considering its
    production by neutrophils, monocytes, and epithelial cells, its baseline production, and
    consumption by neutrophils, monocytes, and epithelial cells.

    Args:
        nas: Concentration of activated neutrophils in the splanchnic (cells/mL).
        mas: Concentration of activated monocytes in the splanchnic (cells/mL).
        epas: Number of activated epithelial cells in the splanchnic (cells).
        il10s: Concentration of IL-10 in the splanchnic (pg/mL).
        _nu_splanchnic_total: Total neutrophil count in the splanchnic (cells/mL).
        _mo_splanchnic_total: Total monocyte count in the splanchnic (cells/mL).
        _eps_total: Total epithelial cell count in the splanchnic (cells).
        config: Dictionary containing model parameters.

    Returns:
        torch.Tensor: The rate of change of IL-10 concentration in the splanchnic (pg/(mL*min)).
    """
    d_il10s_dt = (
        config["INFLAMMATION"]["ALPHA_10_PRODUCTION"]
        * (
            config["INFLAMMATION"]["K10N"] * nas
            + config["INFLAMMATION"]["K10M"] * mas
            + config["INFLAMMATION"]["K10EP"] * epas / config["INFLAMMATION"]["V_SECF"]
        )
        + config["INFLAMMATION"]["K_BASELINE_IL10"]
        * (config["INFLAMMATION"]["S_10"] - il10s)
        - il10s
        * config["INFLAMMATION"]["V_SECF"]
        * config["INFLAMMATION"]["PG_PER_CELLS_PER_MIN"]
        * (
            config["INFLAMMATION"]["ONE_PER_ML"]
            * (
                config["INFLAMMATION"]["KN10"] * _nu_splanchnic_total
                + config["INFLAMMATION"]["KM10"] * _mo_splanchnic_total
            )
            + (config["INFLAMMATION"]["K_EP10"] / config["INFLAMMATION"]["V_SECF"])
            * _eps_total
        )
    )
    return d_il10s_dt
