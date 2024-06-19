import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
import yaml
from einops import rearrange
from loguru import logger
from matplotlib import pyplot as plt
from torchdiffeq import odeint
from tqdm import tqdm

from core.coagulation import (
    active_anti_coag_prime,
    active_coag_factor_prime,
    clot_prime,
    inactive_anti_coag_prime,
    inactive_coag_factor_prime,
    k_bleed,
    platelet_source_enhance,
    platelets_prime,
    rbc_prime,
)
from core.hemodynamics import (
    blood_pressure,
    blood_pressure_unadjusted_prime,
    blood_volume_prime,
    d_bv_dt,
)
from core.inflammation import (
    damp_prime,
    enos_prime,
    epal_prime,
    epl_total,
    eprl_prime,
    epas_prime,
    eps_total,
    eprs_prime,
    il10_prime,
    il10l_prime,
    il10s_prime,
    il1_prime,
    il1l_prime,
    il1s_prime,
    il6_prime,
    il6l_prime,
    il6s_prime,
    inos_prime,
    ma_prime,
    mal_prime,
    mas_prime,
    mo_blood_total,
    mo_lung_total,
    mo_splanchnic_total,
    mr_prime,
    mrl_prime,
    mrs_prime,
    na_prime,
    nal_prime,
    nas_prime,
    no_prime,
    nr_prime,
    nrl_prime,
    nrs_prime,
    nu_blood_total,
    nu_lung_total,
    nu_splanchnic_total,
    tnf_prime,
    tnfl_prime,
    tnfs_prime,
    vma,
    vmal,
    vmas,
    vna,
    vnal,
    vnas,
    vepal,
    vepas,
)
from core.utils import calculate_infusions, o2sat, trauma, damage

# Set up logging with Loguru
logger.add(
    os.path.join("logs", "trauma_model_{time}.log"),
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    level="INFO",
    rotation="10 MB",
    compression="zip",
)


def trauma_model(t: torch.Tensor, state: torch.Tensor, _config: Dict) -> torch.Tensor:
    """
    Defines the differential equations for the _trauma model.

    This function calculates the derivatives of all state variables at a given time point.
    It encapsulates the core logic of the _trauma model, describing the interactions between
    hemodynamics, coagulation, inflammation, and other physiological processes.

    Args:
        t: Current time point (min).
        state: Current state of the model, a tensor containing all state variables.
        _config: Dictionary containing model parameters and settings loaded from _config.yaml.

    Returns:
        torch.Tensor: A tensor containing the derivatives of all state variables at time t.
    """
    (
        blood_volume,
        blood_pressure_unadjusted,
        rbc,
        inactive_coag_factor,
        active_coag_factor,
        inactive_anti_coag,
        active_anti_coag,
        platelets,
        clot,
        mr,
        ma,
        mrl,
        mal,
        mrs,
        mas,
        nr,
        na,
        nrl,
        nal,
        nrs,
        nas,
        eprl,
        epal,
        eprs,
        epas,
        tnf,
        tnfl,
        tnfs,
        il1,
        il1l,
        il1s,
        il6,
        il6l,
        il6s,
        il10,
        il10l,
        il10s,
        inos,
        enos,
        no,
        damp,
        auc_damage,
        second_trauma_iss_mut,
        third_trauma_iss_mut,
        vent_on,
        time_of_death_mut,
    ) = torch.unbind(rearrange(state, "b (n a) -> n b a", a=1), dim=0)

    (
        sum_plasma_infusions,
        sum_fluid_infusions,
        sum_rbc_infusions,
        sum_platelet_infusions,
    ) = calculate_infusions(t, _config)

    k_b_t_ma = _config["INFLAMMATION"]["K_B_T_MA0"] * (epal + epas)
    k_b_t_na = _config["INFLAMMATION"]["K_B_T_NA0"] * (epal + epas)

    _k_bleed = k_bleed(
        trauma(
            t,
            iss,
            second_trauma_iss_mut,
            third_trauma_iss_mut,
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            _config
        ),
        clot,
        blood_pressure(blood_pressure_unadjusted, blood_volume, _config),
        _config,
    )

    _d_bv_dt = d_bv_dt(
        sum_plasma_infusions,
        sum_rbc_infusions,
        sum_platelet_infusions,
        sum_fluid_infusions,
        _k_bleed,
        blood_volume,
        _config,
    )

    d_blood_volume_dt = blood_volume_prime(
        sum_plasma_infusions,
        sum_rbc_infusions,
        sum_platelet_infusions,
        sum_fluid_infusions,
        _k_bleed,
        blood_volume,
        _config,
    )
    d_blood_pressure_unadjusted_dt = blood_pressure_unadjusted_prime(
        blood_pressure_unadjusted, no, _config
    )
    d_rbc_dt = rbc_prime(
        rbc, blood_volume, sum_rbc_infusions, _k_bleed, _d_bv_dt, _config
    )
    d_inactive_coag_factor_dt = inactive_coag_factor_prime(
        inactive_coag_factor,
        blood_volume,
        sum_plasma_infusions,
        sum_platelet_infusions,
        _k_bleed,
        trauma(
            t,
            iss,
            second_trauma_iss_mut,
            third_trauma_iss_mut,
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            _config
        ),
        il6,
        _d_bv_dt,
        _config,
    )
    d_active_coag_factor_dt = active_coag_factor_prime(
        inactive_coag_factor,
        blood_volume,
        trauma(
            t,
            iss,
            second_trauma_iss_mut,
            third_trauma_iss_mut,
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            _config
        ),
        il6,
        _k_bleed,
        active_anti_coag,
        inactive_anti_coag,
        platelets,
        active_coag_factor,
        _d_bv_dt,
        _config,
    )
    d_inactive_anti_coag_dt = inactive_anti_coag_prime(
        inactive_anti_coag,
        blood_volume,
        sum_plasma_infusions,
        sum_platelet_infusions,
        _k_bleed,
        active_coag_factor,
        _d_bv_dt,
        _config,
    )
    d_active_anti_coag_dt = active_anti_coag_prime(
        active_anti_coag,
        inactive_anti_coag,
        blood_volume,
        active_coag_factor,
        sum_plasma_infusions,
        sum_platelet_infusions,
        _k_bleed,
        _d_bv_dt,
        _config,
    )
    d_platelets_dt = platelets_prime(
        platelets,
        blood_volume,
        sum_platelet_infusions,
        platelet_source_enhance(
            trauma(
                t,
                iss,
                second_trauma_iss_mut,
                third_trauma_iss_mut,
                torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
                torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
                torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
                _config
            ),
            _config,
        ),
        active_coag_factor,
        _k_bleed,
        _d_bv_dt,
        _config,
    )
    d_clot_dt = clot_prime(
        platelets,
        active_coag_factor,
        rbc,
        trauma(
            t,
            iss,
            second_trauma_iss_mut,
            third_trauma_iss_mut,
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            _config
        ),
        blood_volume,
        clot,
        _config,
    )
    _vma = vma(
        mr,
        blood_volume,
        trauma(
            t,
            iss,
            second_trauma_iss_mut,
            third_trauma_iss_mut,
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            _config
        ),
        tnf,
        il1,
        active_coag_factor,
        il10,
        il6,
        _config,
    )
    d_mr_dt = mr_prime(mr, _vma, _k_bleed, _d_bv_dt, blood_volume, _config)
    d_ma_dt = ma_prime(_vma, ma, k_b_t_ma, _k_bleed, _d_bv_dt, blood_volume, _config)
    _vmal = vmal(
        mrl,
        tnfl,
        il1l,
        trauma(
            t,
            iss,
            second_trauma_iss_mut,
            third_trauma_iss_mut,
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            _config
        ),
        il10l,
        il6l,
        _config,
    )
    d_mrl_dt = mrl_prime(mrl, _vmal, _config)
    d_mal_dt = mal_prime(_vmal, mal, ma, blood_volume, k_b_t_ma, _config)
    _vmas = vmas(
        mrs,
        tnfs,
        il1s,
        trauma(
            t,
            iss,
            second_trauma_iss_mut,
            third_trauma_iss_mut,
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            _config
        ),
        il10s,
        il6s,
        _config,
    )
    d_mrs_dt = mrs_prime(mrs, _vmas, _config)
    d_mas_dt = mas_prime(_vmas, mas, ma, blood_volume, k_b_t_ma, _config)
    _vna = vna(
        nr,
        blood_volume,
        trauma(
            t,
            iss,
            second_trauma_iss_mut,
            third_trauma_iss_mut,
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            _config
        ),
        tnf,
        il1,
        active_coag_factor,
        il10,
        il6,
        _config,
    )
    d_nr_dt = nr_prime(nr, _vna, _k_bleed, _d_bv_dt, blood_volume, _config)
    d_na_dt = na_prime(_vna, na, k_b_t_na, _k_bleed, _d_bv_dt, blood_volume, _config)
    _vnal = vnal(
        nrl,
        tnfl,
        il1l,
        trauma(
            t,
            iss,
            second_trauma_iss_mut,
            third_trauma_iss_mut,
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            _config
        ),
        il10l,
        il6l,
        _config,
    )
    d_nrl_dt = nrl_prime(nrl, _vnal, _config)
    d_nal_dt = nal_prime(_vnal, nal, na, blood_volume, k_b_t_na, _config)
    _vnas = vnas(
        nrs,
        tnfs,
        il1s,
        trauma(
            t,
            iss,
            second_trauma_iss_mut,
            third_trauma_iss_mut,
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            _config
        ),
        il10s,
        il6s,
        _config,
    )
    d_nrs_dt = nrs_prime(nrs, _vnas, _config)
    d_nas_dt = nas_prime(_vnas, nas, na, blood_volume, k_b_t_na, _config)
    _vepal = vepal(
        eprl,
        tnf,
        tnfl,
        il6,
        il6l,
        il1,
        il1l,
        trauma(
            t,
            iss,
            second_trauma_iss_mut,
            third_trauma_iss_mut,
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            _config
        ),
        active_coag_factor,
        il10,
        il10l,
        blood_volume,
        _config,
    )
    d_eprl_dt = eprl_prime(_vepal, eprl, _config)
    d_epal_dt = epal_prime(_vepal, epal, _config)
    _vepas = vepas(
        eprs,
        damp,
        tnf,
        tnfs,
        il6,
        il6s,
        il1,
        il1s,
        trauma(
            t,
            iss,
            second_trauma_iss_mut,
            third_trauma_iss_mut,
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            _config
        ),
        active_coag_factor,
        il10,
        il10s,
        blood_volume,
        _config,
    )
    d_eprs_dt = eprs_prime(_vepas, eprs, _config)
    d_epas_dt = epas_prime(_vepas, epas, _config)
    d_tnf_dt = tnf_prime(
        na,
        ma,
        tnf,
        nu_blood_total(nr, na),
        mo_blood_total(mr, ma),
        epl_total(eprl, epal),
        eps_total(eprs, epas),
        blood_volume,
        _k_bleed,
        _d_bv_dt,
        _config,
    )
    d_tnfl_dt = tnfl_prime(
        nal,
        mal,
        tnfl,
        nu_lung_total(nrl, nal),
        mo_lung_total(mrl, mal),
        epl_total(eprl, epal),
        _config,
    )
    d_tnfs_dt = tnfs_prime(
        nas,
        mas,
        tnfs,
        nu_splanchnic_total(nrs, nas),
        mo_splanchnic_total(mrs, mas),
        eps_total(eprs, epas),
        _config,
    )
    d_il1_dt = il1_prime(
        na,
        ma,
        epal,
        epas,
        il1,
        nu_blood_total(nr, na),
        mo_blood_total(mr, ma),
        epl_total(eprl, epal),
        eps_total(eprs, epas),
        blood_volume,
        _k_bleed,
        _d_bv_dt,
        _config,
    )
    d_il1l_dt = il1l_prime(
        nal,
        mal,
        epal,
        il1l,
        nu_lung_total(nrl, nal),
        mo_lung_total(mrl, mal),
        epl_total(eprl, epal),
        _config,
    )
    d_il1s_dt = il1s_prime(
        nas,
        mas,
        epas,
        il1s,
        nu_splanchnic_total(nrs, nas),
        mo_splanchnic_total(mrs, mas),
        eps_total(eprs, epas),
        _config,
    )
    d_il6_dt = il6_prime(
        na,
        ma,
        epal,
        epas,
        il6,
        nu_blood_total(nr, na),
        mo_blood_total(mr, ma),
        epl_total(eprl, epal),
        eps_total(eprs, epas),
        blood_volume,
        _k_bleed,
        _d_bv_dt,
        _config,
    )
    d_il6l_dt = il6l_prime(
        nal,
        mal,
        epal,
        il6l,
        nu_lung_total(nrl, nal),
        mo_lung_total(mrl, mal),
        epl_total(eprl, epal),
        _config,
    )
    d_il6s_dt = il6s_prime(
        nas,
        mas,
        epas,
        il6s,
        nu_splanchnic_total(nrs, nas),
        mo_splanchnic_total(mrs, mas),
        eps_total(eprs, epas),
        _config,
    )
    d_il10_dt = il10_prime(
        na,
        ma,
        epal,
        epas,
        il10,
        nu_blood_total(nr, na),
        mo_blood_total(mr, ma),
        epl_total(eprl, epal),
        eps_total(eprs, epas),
        blood_volume,
        _k_bleed,
        _d_bv_dt,
        _config,
    )
    d_il10l_dt = il10l_prime(
        nal,
        mal,
        epal,
        il10l,
        nu_lung_total(nrl, nal),
        mo_lung_total(mrl, mal),
        epl_total(eprl, epal),
        _config,
    )
    d_il10s_dt = il10s_prime(
        nas,
        mas,
        epas,
        il10s,
        nu_splanchnic_total(nrs, nas),
        mo_splanchnic_total(mrs, mas),
        eps_total(eprs, epas),
        _config,
    )
    d_inos_dt = inos_prime(
        inos,
        ma,
        na,
        epal,
        epas,
        il10,
        il10l,
        il10s,
        no,
        blood_volume,
        _k_bleed,
        _d_bv_dt,
        _config,
    )
    # Access PE from _config
    pe = _config["INFLAMMATION"]["PE"]

    d_enos_dt = enos_prime(
        enos, pe, blood_volume, _k_bleed, _d_bv_dt, _config  # Pass pe to enos_prime
    )
    d_no_dt = no_prime(
        inos,
        enos,
        ma,
        na,
        epal,
        epas,
        no,
        blood_volume,
        _k_bleed,
        _d_bv_dt,
        _config,
    )
    d_damp_dt = damp_prime(damp, eps_total(eprs, epas), _config)
    d_auc_damage_dt = damage(
        blood_pressure(blood_pressure_unadjusted, blood_volume, _config),
        il6,
        il6l,
        il6s,
        o2sat(
            torch.tensor(_config["INFLAMMATION"]["O2SAT0"]),
            epal,
            vent_on,
            _config
        ),
        trauma(
            t,
            iss,
            second_trauma_iss_mut,
            third_trauma_iss_mut,
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            torch.tensor(_config["MODEL_PARAMETERS"]["T_INITIAL"]),
            _config
        ),
        _config,
    )
    d_second_trauma_iss_mut_dt = torch.tensor(0.0)
    d_third_trauma_iss_mut_dt = torch.tensor(0.0)
    d_vent_on_dt = torch.tensor(0.0)
    d_time_of_death_mut_dt = torch.tensor(0.0)

    # Stack the derivatives back into a single tensor
    d_state_dt = torch.stack(
        [
            d_blood_volume_dt,
            d_blood_pressure_unadjusted_dt,
            d_rbc_dt,
            d_inactive_coag_factor_dt,
            d_active_coag_factor_dt,
            d_inactive_anti_coag_dt,
            d_active_anti_coag_dt,
            d_platelets_dt,
            d_clot_dt,
            d_mr_dt,
            d_ma_dt,
            d_mrl_dt,
            d_mal_dt,
            d_mrs_dt,
            d_mas_dt,
            d_nr_dt,
            d_na_dt,
            d_nrl_dt,
            d_nal_dt,
            d_nrs_dt,
            d_nas_dt,
            d_eprl_dt,
            d_epal_dt,
            d_eprs_dt,
            d_epas_dt,
            d_tnf_dt,
            d_tnfl_dt,
            d_tnfs_dt,
            d_il1_dt,
            d_il1l_dt,
            d_il1s_dt,
            d_il6_dt,
            d_il6l_dt,
            d_il6s_dt,
            d_il10_dt,
            d_il10l_dt,
            d_il10s_dt,
            d_inos_dt,
            d_enos_dt,
            d_no_dt,
            d_damp_dt,
            d_auc_damage_dt,
            d_second_trauma_iss_mut_dt,
            d_third_trauma_iss_mut_dt,
            d_vent_on_dt,
            d_time_of_death_mut_dt,
        ],
        dim=0,
    )
    return rearrange(d_state_dt, "n b a -> b (n a)")


def run_simulation(_config: Dict) -> torch.Tensor:
    """
    Runs the _trauma model simulation.

    This function sets the initial conditions, defines the time span,
    solves the differential equations using `torchdiffeq.odeint`, and returns the simulation _results.

    Args:
        _config: Dictionary containing model parameters and settings loaded from _config.yaml.

    Returns:
        torch.Tensor: A tensor containing the simulation _results for all state variables over time.
    """
    logger.info("Setting initial conditions 🌱")
    y0 = torch.tensor(
        [
            _config["HEMODYNAMICS"]["S_BLOOD_VOLUME"],
            _config["HEMODYNAMICS"]["S_BLOOD_PRESSURE"],
            _config["COAGULATION"]["S_RBC"],
            _config["COAGULATION"]["S_INACTIVE_COAG_FACTOR"],
            0,
            _config["COAGULATION"]["S_INACTIVE_ANTI_COAG"],
            _config["COAGULATION"]["S_ACTIVE_ANTI_COAG"],
            _config["COAGULATION"]["S_PLATELETS"],
            0,
            _config["INFLAMMATION"]["S_MR"],
            0,
            _config["INFLAMMATION"]["S_MR"],
            0,
            _config["INFLAMMATION"]["S_MR"],
            0,
            _config["INFLAMMATION"]["S_NR"],
            0,
            _config["INFLAMMATION"]["S_NR"],
            0,
            _config["INFLAMMATION"]["S_NR"],
            0,
            _config["INFLAMMATION"]["S_EPRL"],
            0,
            _config["INFLAMMATION"]["S_EPRS"],
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            _config["INFLAMMATION"]["S_10"],
            _config["INFLAMMATION"]["S_10"],
            _config["INFLAMMATION"]["S_10"],
            0,
            _config["INFLAMMATION"]["S_ENOS"],
            _config["INFLAMMATION"]["S_NO"],
            _config["INFLAMMATION"]["ISS"]
            / _config["INFLAMMATION"]["MAX_ISS"]
            * _config["INFLAMMATION"]["DAMP_LOAD"]
            / _config["INFLAMMATION"]["V_SECF"],
            0,  # AUC_Damage
            0,  # Second_Trauma_ISS_mut
            0,  # Third_Trauma_ISS_mut
            0,  # vent_on
            -1000,  # time_of_death_mut
        ]
    )

    logger.info("Defining time span for simulation ⏱️")
    t_span = torch.linspace(
        _config["MODEL_PARAMETERS"]["T_INITIAL"],
        _config["MODEL_PARAMETERS"]["T_FINAL"],
        _config["MODEL_PARAMETERS"]["T_FINAL"]
        - _config["MODEL_PARAMETERS"]["T_INITIAL"]
        + 1,
    )

    logger.info("Solving differential equations using odeint 🧬")
    with tqdm(total=len(t_span), desc="Simulating") as pbar:
        solution = odeint(
            lambda t, y: trauma_model(t, y, _config),
            y0,
            t_span,
            method="rk4",
            options={"step_size": 1},
        )
        pbar.update(len(t_span))

    logger.info("Simulation complete! ✨")
    return solution


def save_results(_results: torch.Tensor, output_path: str, _config: Dict) -> None:
    """
    Saves the simulation results to a file and generates visualizations.

    Args:
        _results: Tensor containing the simulation results.
        output_path: Path to the output CSV file.
        _config: Dictionary containing model parameters and settings.
    """
    logger.info(f"Saving simulation _results to {output_path} 💾")

    # Convert _results to numpy array for easier handling
    _results = _results.detach().numpy()

    # Create a time array
    t_span = np.linspace(
        _config["MODEL_PARAMETERS"]["T_INITIAL"],
        _config["MODEL_PARAMETERS"]["T_FINAL"],
        _config["MODEL_PARAMETERS"]["T_FINAL"]
        - _config["MODEL_PARAMETERS"]["T_INITIAL"]
        + 1,
    )

    # Create a list of column names for the DataFrame
    column_names = [
        "blood_volume",
        "blood_pressure_unadjusted",
        "rbc",
        "inactive_coag_factor",
        "active_coag_factor",
        "inactive_anti_coag",
        "active_anti_coag",
        "platelets",
        "clot",
        "mr",
        "ma",
        "mrl",
        "mal",
        "mrs",
        "mas",
        "nr",
        "na",
        "nrl",
        "nal",
        "nrs",
        "nas",
        "eprl",
        "epal",
        "eprs",
        "epas",
        "tnf",
        "tnfl",
        "tnfs",
        "il1",
        "il1l",
        "il1s",
        "il6",
        "il6l",
        "il6s",
        "il10",
        "il10l",
        "il10s",
        "inos",
        "enos",
        "no",
        "damp",
        "auc_damage",
        "second_trauma_iss_mut",
        "third_trauma_iss_mut",
        "vent_on",
        "time_of_death_mut",
    ]

    # Create a Pandas DataFrame from the _results
    df = pd.DataFrame(data=_results, columns=column_names)
    df["time"] = t_span

    # Save the DataFrame to a CSV file
    df.to_csv(output_path, index=False)

    # --- Generate Visualizations ---
    logger.info("Generating visualizations 📊")

    # Create the 'output/figures' directory if it doesn't exist
    os.makedirs("output/figures", exist_ok=True)

    # Example 1: Blood Volume and Pressure
    plt.figure()
    plt.plot(df["time"], df["blood_volume"], label="Blood Volume (mL)")
    plt.plot(
        df["time"],
        blood_pressure(
            torch.tensor(df["blood_pressure_unadjusted"]),
            torch.tensor(df["blood_volume"]),
            _config,
        ),
        label="Blood Pressure (mmHg)",
    )
    plt.xlabel("Time (min)")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Blood Volume and Pressure Over Time")
    plt.savefig("output/figures/blood_volume_pressure.png")

    # Example 2: Cytokine Levels
    plt.figure()
    plt.plot(df["time"], df["tnf"], label="TNF (pg/mL)")
    plt.plot(df["time"], df["il1"], label="IL-1 (pg/mL)")
    plt.plot(df["time"], df["il6"], label="IL-6 (pg/mL)")
    plt.plot(df["time"], df["il10"], label="IL-10 (pg/mL)")
    plt.xlabel("Time (min)")
    plt.ylabel("Concentration (pg/mL)")
    plt.legend()
    plt.title("Cytokine Levels Over Time")
    plt.savefig("output/figures/cytokine_levels.png")

    # --- Extract Key Results ---
    logger.info("Extracting key _results 🔑")

    # Example 1: Maximum Damage
    max_damage = df["auc_damage"].max()
    logger.info(f"Maximum Damage: {max_damage:.2f}")

    # Example 2: Time of Death
    time_of_death = (
        df["time"][df["time_of_death_mut"] > 0].iloc[0]
        if any(df["time_of_death_mut"] > 0)
        else None
    )
    logger.info(f"Time of Death: {time_of_death}")

    # Example 3: AUC of IL-6
    auc_il6 = np.trapz(df["il6"], df["time"])
    logger.info(f"AUC of IL-6: {auc_il6:.2f}")


if __name__ == "__main__":
    # Load configuration from YAML file
    logger.info("Loading configuration from _config.yaml ⚙️")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Define global iss variable
    iss = torch.tensor(config["INFLAMMATION"]["ISS"])

    # Run the simulation
    results = run_simulation(config)

    # Save the _results
    output_file = os.path.join("output", "simulation_results.csv")
    save_results(results, output_file, config)

    logger.info("Finished! 🎉")
