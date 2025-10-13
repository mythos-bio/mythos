"""Utilities for the oxDNA simulator."""

import datetime
import functools
import operator
from pathlib import Path

import jax
import jax.numpy as jnp
import sympy
import pandas as pd

from jax_dna.input import oxdna_input
import jax_dna.utils.types as jd_types
from jax_dna.utils.types import oxDNAModelHType

ERR_CANNOT_PROCESS_SRC_H = "Cannot process src/model.h file. Failed parsing: {}"
ERR_INVALID_HEADER_TYPE = "Invalid header value variable {} with value {}"

SYMPY_EVAL_N: int = 32

DEFAULT_OXDNA_VARIABLE_MAPPER = {
    # fene
    "eps_backbone": "FENE_EPS",
    "delta_backbone": "FENE_DELTA",
    "r0_backbone": "FENE_R0_OXDNA",
    # excluded_volume
    "eps_exc": "EXCL_EPS",
    "sigma_backbone": "EXCL_S1",
    "sigma_base": "EXCL_S2",
    "sigma_back_base": "EXCL_S3",
    "sigma_base_back": "EXCL_S4",
    "dr_star_backbone": "EXCL_R1",
    "dr_star_base": "EXCL_R2",
    "dr_star_back_base": "EXCL_R3",
    "dr_star_base_back": "EXCL_R4",
    "b_backbone": "EXCL_B1",
    "b_base": "EXCL_B2",
    "b_back_base": "EXCL_B3",
    "b_base_back": "EXCL_B4",
    "dr_c_backbone": "EXCL_RC1",
    "dr_c_base": "EXCL_RC2",
    "dr_c_back_base": "EXCL_RC3",
    "dr_c_base_back": "EXCL_RC4",
    # stacking
    # func f1(dr_stack)
    "eps_stack_base": "STCK_BASE_EPS_OXDNA",
    "eps_stack_kt_coeff": "STCK_FACT_EPS_OXDNA",
    "a_stack": "STCK_A",
    "dr0_stack": "STCK_R0",
    "dr_c_stack": "STCK_RC",
    "dr_low_stack": "STCK_RLOW",
    "dr_high_stack": "STCK_RHIGH",
    "b_low_stack": "STCK_BLOW",
    "b_high_stack": "STCK_BHIGH",
    "dr_c_low_stack": "STCK_RCLOW",
    "dr_c_high_stack": "STCK_RCHIGH",
    # func f4(theta_4)
    "a_stack_4": "STCK_THETA4_A",
    "theta0_stack_4": "STCK_THETA4_T0",
    "delta_theta_star_stack_4": "STCK_THETA4_TS",
    "b_stack_4": "STCK_THETA4_B",
    "delta_theta_stack_4_c": "STCK_THETA4_TC",
    # func f4(theta_5p)
    "a_stack_5": "STCK_THETA5_A",
    "theta0_stack_5": "STCK_THETA5_T0",
    "delta_theta_star_stack_5": "STCK_THETA5_TS",
    "b_stack_5": "STCK_THETA5_B",
    "delta_theta_stack_5_c": "STCK_THETA5_TC",
    # func f4(theta_6p)
    "a_stack_6": "STCK_THETA6_A",
    "theta0_stack_6": "STCK_THETA6_T0",
    "delta_theta_star_stack_6": "STCK_THETA6_TS",
    "b_stack_6": "STCK_THETA6_B",
    "delta_theta_stack_6_c": "STCK_THETA6_TC",
    # func f5(-cos(phi1))
    "a_stack_1": "STCK_PHI1_A",
    "neg_cos_phi1_star_stack": "STCK_PHI1_XS",
    "b_neg_cos_phi1_stack": "STCK_PHI1_B",
    "neg_cos_phi1_c_stack": "STCK_PHI1_XC",
    # func f5(-cos(phi2))
    "a_stack_2": "STCK_PHI2_A",
    "neg_cos_phi2_star_stack": "STCK_PHI2_XS",
    "b_neg_cos_phi2_stack": "STCK_PHI2_B",
    "neg_cos_phi2_c_stack": "STCK_PHI2_XC",
    # hydrogen_bonding
    # func f1(dr_hb)
    "eps_hb": "HYDR_EPS_OXDNA",
    "a_hb": "HYDR_A",
    "dr0_hb": "HYDR_R0",
    "dr_c_hb": "HYDR_RC",
    "dr_low_hb": "HYDR_RLOW",
    "dr_high_hb": "HYDR_RHIGH",
    "b_low_hb": "HYDR_BLOW",
    "dr_c_low_hb": "HYDR_RCLOW",
    "b_high_hb": "HYDR_BHIGH",
    "dr_c_high_hb": "HYDR_RCHIGH",
    # func f4(theta_1)
    "a_hb_1": "HYDR_THETA1_A",
    "theta0_hb_1": "HYDR_THETA1_T0",
    "delta_theta_star_hb_1": "HYDR_THETA1_TS",
    "b_hb_1": "HYDR_THETA1_B",
    "delta_theta_hb_1_c": "HYDR_THETA1_TC",
    # func f4(theta_2)
    "a_hb_2": "HYDR_THETA2_A",
    "theta0_hb_2": "HYDR_THETA2_T0",
    "delta_theta_star_hb_2": "HYDR_THETA2_TS",
    "b_hb_2": "HYDR_THETA2_B",
    "delta_theta_hb_2_c": "HYDR_THETA2_TC",
    # func f4(theta_3)
    "a_hb_3": "HYDR_THETA3_A",
    "theta0_hb_3": "HYDR_THETA3_T0",
    "delta_theta_star_hb_3": "HYDR_THETA3_TS",
    "b_hb_3": "HYDR_THETA3_B",
    "delta_theta_hb_3_c": "HYDR_THETA3_TC",
    # func f4(theta_4)
    "a_hb_4": "HYDR_THETA4_A",
    "theta0_hb_4": "HYDR_THETA4_T0",
    "delta_theta_star_hb_4": "HYDR_THETA4_TS",
    "b_hb_4": "HYDR_THETA4_B",
    "delta_theta_hb_4_c": "HYDR_THETA4_TC",
    # func f4(theta_7)
    "a_hb_7": "HYDR_THETA7_A",
    "theta0_hb_7": "HYDR_THETA7_T0",
    "delta_theta_star_hb_7": "HYDR_THETA7_TS",
    "b_hb_7": "HYDR_THETA7_B",
    "delta_theta_hb_7_c": "HYDR_THETA7_TC",
    # func f4(theta_8)
    "a_hb_8": "HYDR_THETA8_A",
    "theta0_hb_8": "HYDR_THETA8_T0",
    "delta_theta_star_hb_8": "HYDR_THETA8_TS",
    "b_hb_8": "HYDR_THETA8_B",
    "delta_theta_hb_8_c": "HYDR_THETA8_TC",
    # cross_stacking
    # func f2(dr_cross)
    "k_cross": "CRST_K",
    "r0_cross": "CRST_R0",
    "dr_c_cross": "CRST_RC",
    "dr_low_cross": "CRST_RLOW",
    "dr_high_cross": "CRST_RHIGH",
    "b_low_cross": "CRST_BLOW",
    "dr_c_low_cross": "CRST_RCLOW",
    "b_high_cross": "CRST_BHIGH",
    "dr_c_high_cross": "CRST_RCHIGH",
    # func f4(theta_1)
    "a_cross_1": "CRST_THETA1_A",
    "theta0_cross_1": "CRST_THETA1_T0",
    "delta_theta_star_cross_1": "CRST_THETA1_TS",
    "b_cross_1": "CRST_THETA1_B",
    "delta_theta_cross_1_c": "CRST_THETA1_TC",
    # func f4(theta_2)
    "a_cross_2": "CRST_THETA2_A",
    "theta0_cross_2": "CRST_THETA2_T0",
    "delta_theta_star_cross_2": "CRST_THETA2_TS",
    "b_cross_2": "CRST_THETA2_B",
    "delta_theta_cross_2_c": "CRST_THETA2_TC",
    # func f4(theta_3)
    "a_cross_3": "CRST_THETA3_A",
    "theta0_cross_3": "CRST_THETA3_T0",
    "delta_theta_star_cross_3": "CRST_THETA3_TS",
    "b_cross_3": "CRST_THETA3_B",
    "delta_theta_cross_3_c": "CRST_THETA3_TC",
    # func f4(theta_4) + f4(pi - theta_4)
    "a_cross_4": "CRST_THETA4_A",
    "theta0_cross_4": "CRST_THETA4_T0",
    "delta_theta_star_cross_4": "CRST_THETA4_TS",
    "b_cross_4": "CRST_THETA4_B",
    "delta_theta_cross_4_c": "CRST_THETA4_TC",
    # func f4(theta_7) + f4(pi - theta_7)
    "a_cross_7": "CRST_THETA7_A",
    "theta0_cross_7": "CRST_THETA7_T0",
    "delta_theta_star_cross_7": "CRST_THETA7_TS",
    "b_cross_7": "CRST_THETA7_B",
    "delta_theta_cross_7_c": "CRST_THETA7_TC",
    # func f4(theta_8) + f4(pi - theta_8)
    "a_cross_8": "CRST_THETA8_A",
    "theta0_cross_8": "CRST_THETA8_T0",
    "delta_theta_star_cross_8": "CRST_THETA8_TS",
    "b_cross_8": "CRST_THETA8_B",
    "delta_theta_cross_8_c": "CRST_THETA8_TC",
    # coaxial_stacking
    # func f2(dr_coax)
    "k_coax": "CXST_K_OXDNA",
    "dr0_coax": "CXST_R0",
    "dr_c_coax": "CXST_RC",
    "dr_low_coax": "CXST_RLOW",
    "dr_high_coax": "CXST_RHIGH",
    "b_low_coax": "CXST_BLOW",
    "dr_c_low_coax": "CXST_RCLOW",
    "b_high_coax": "CXST_BHIGH",
    "dr_c_high_coax": "CXST_RCHIGH",
    # func f4(theta_1) + f4(2*pi - theta_1)
    "a_coax_1": "CXST_THETA1_A",
    "theta0_coax_1": "CXST_THETA1_T0_OXDNA",
    "delta_theta_star_coax_1": "CXST_THETA1_TS",
    "b_coax_1": "CXST_THETA1_B",
    "delta_theta_coax_1_c": "CXST_THETA1_TC",
    # func f4(theta_4)
    "a_coax_4": "CXST_THETA4_A",
    "theta0_coax_4": "CXST_THETA4_T0",
    "delta_theta_star_coax_4": "CXST_THETA4_TS",
    "b_coax_4": "CXST_THETA4_B",
    "delta_theta_coax_4_c": "CXST_THETA4_TC",
    # func f4(theta_5) + f4(pi - theta_5)
    "a_coax_5": "CXST_THETA5_A",
    "theta0_coax_5": "CXST_THETA5_T0",
    "delta_theta_star_coax_5": "CXST_THETA5_TS",
    "b_coax_5": "CXST_THETA5_B",
    "delta_theta_coax_5_c": "CXST_THETA5_TC",
    # func f4(theta_6) + f4(pi - theta_6)
    "a_coax_6": "CXST_THETA6_A",
    "theta0_coax_6": "CXST_THETA6_T0",
    "delta_theta_star_coax_6": "CXST_THETA6_TS",
    "b_coax_6": "CXST_THETA6_B",
    "delta_theta_coax_6_c": "CXST_THETA6_TC",
    # func f5(cos(phi3))
    "a_coax_3p": "CXST_PHI3_A",
    "cos_phi3_star_coax": "CXST_PHI3_XS",
    "b_cos_phi3_coax": "CXST_PHI3_B",
    "cos_phi3_c_coax": "CXST_PHI3_XC",
    # func f5(cos(phi4))
    "a_coax_4p": "CXST_PHI4_A",
    "cos_phi4_star_coax": "CXST_PHI4_XS",
    "b_cos_phi4_coax": "CXST_PHI4_B",
    "cos_phi4_c_coax": "CXST_PHI4_XC",
}

MIN_VALID_HEADER_TOKEN_COUNT = 3


def _parse_value_in(value: str) -> int | float | str:
    for t in (int, float):
        try:
            if t is float:
                tmp_value = value.replace("f", "").lower()
                parsed = float(sympy.parse_expr(tmp_value).evalf(n=SYMPY_EVAL_N))
            else:
                parsed = t(value)
        except (AttributeError, ValueError, SyntaxError, TypeError):
            continue
        else:
            return parsed

    return value


def _parse_value_out(value: int | float | str) -> str:  # noqa: PYI041 -- this is documentation specific
    if isinstance(value, int) or (isinstance(value, jax.Array) and (jnp.issubdtype(value.dtype, jnp.integer))):
        parsed_value = str(value)
    elif isinstance(value, float) or (isinstance(value, jax.Array) and (jnp.issubdtype(value.dtype, jnp.floating))):
        parsed_value = f"{value}f"
    elif isinstance(value, str):
        parsed_value = value
    else:
        raise TypeError(ERR_INVALID_HEADER_TYPE.format(type(value), value))
    return parsed_value


def read_src_h(src_h: Path) -> dict[str, int | float | str]:
    """Parse the src/model.h file to get the parameters."""
    params = {}
    with src_h.open("r") as f:
        for line in f:
            # this is a variable
            if line.startswith("#define") and "MODEL_H_" not in line:
                # We need to parse lines of the following varieties:
                # #define POS_BACK -0.4f
                # #define HYDR_F1 0
                # #define HYDR_THETA8_T0 (PI*0.5f)
                # #define HYDR_T3_MESH_POINTS HYDR_T2_MESH_POINTS
                # #define CXST_T5_MESH_POINTS 6   // perfetto

                parts = line.split()
                if (
                    len(parts) >= MIN_VALID_HEADER_TOKEN_COUNT
                    and (var_value := _parse_value_in(" ".join(parts[2:]).split("//")[0].strip())) is not None
                ):
                    params[parts[1]] = var_value
                else:
                    raise ValueError(ERR_CANNOT_PROCESS_SRC_H.format(line))

    return params


def write_src_h(src_h: Path, params: dict[str, tuple[oxDNAModelHType, int | float | str]]) -> None:
    """Write the src/model.h file with the new parameters."""
    with src_h.open("w") as f:
        f.write(
            "\n".join(
                [
                    "/**",
                    " * @file model.h",
                    f" * @date {datetime.datetime.now(tz=datetime.UTC).strftime('%b %d, %Y')}",
                    " * @author fromano -- modified by jax_dna",
                    " */",
                    "",
                    "#ifndef MODEL_H_",
                    "#define MODEL_H_\n",
                ]
            )
        )

        for key, value in params.items():
            try:
                parsed_value = _parse_value_out(value)
            except ValueError as e:
                raise ValueError(ERR_INVALID_HEADER_TYPE.format(key, value)) from e

            f.write(f"#define {key} {parsed_value}\n")
            if key == "FENE_DELTA":
                f.write(f"#define FENE_DELTA2 {value**2}f\n")

        f.write("#endif /* MODEL_H_ */\n")


def update_params(src_h: Path, new_params: list[jd_types.Params]) -> None:
    """Update the src/model.h file with the new parameters."""
    params = read_src_h(src_h)
    flattened_params = functools.reduce(operator.or_, new_params, {})
    for np in filter(lambda k: k in DEFAULT_OXDNA_VARIABLE_MAPPER, flattened_params):
        mapped_name = DEFAULT_OXDNA_VARIABLE_MAPPER[np]
        if mapped_name in params:
            params[mapped_name] = flattened_params[np]
        else:
            raise ValueError(f"Parameter {np} not found in src/model.h")

    write_src_h(src_h, params)


def read_energy(simulation_dir: Path) -> pd.DataFrame:
    """Read the energy.dat file from an oxDNA simulation.

    Args:
        simulation_dir: Path to the simulation directory containing the
            energy.dat and other simulation files. this directory must also
            contain the input file, which is read to determine the shape of the
            energy file.

    Returns:
        A pandas DataFrame containing the energy data. When umbrella sampling is
        enabled, the order parameter columns and weight column are also
        included, where the order parameter name comes from the
        "order_parameter" specification in the "op_file".
    """
    inputs = oxdna_input.read(simulation_dir / "input")
    energy_file = simulation_dir / inputs["energy_file"]
    energy_df_columns_base = [
        "time", "potential_energy", "acc_ratio_trans", "acc_ratio_rot", "acc_ratio_vol"
    ]

    # This is a space separated file, no header and the first row corresponds to
    # the 0th step, which does not match the trajectory file, so we skip it. The
    # resulting energy file should then be the same length as the trajectory and
    # each row corresponsds to the same step.
    energy_df = pd.read_table(energy_file, sep=r"\s+", header=None, skiprows=1)
    if not inputs.get("umbrella_sampling"):
        energy_df.columns = energy_df_columns_base
        return energy_df

    # get the order parameter types from the op_file
    with simulation_dir.joinpath(inputs["op_file"]).open("r") as f:
        order_param_types = [
            line.strip().split("=")[1].strip()
            for line in f
            if line.strip().startswith("order_parameter")
        ]

    energy_df_columns = energy_df_columns_base + order_param_types + ["weight"]
    energy_df.columns = energy_df_columns
    return energy_df
