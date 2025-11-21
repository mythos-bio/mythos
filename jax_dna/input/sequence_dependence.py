"""Functions for handling sequence-dependent inputs."""
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jax_dna.utils.constants import DNA_ALPHA
from jax_dna.utils.constants import NUCLEOTIDES_IDX as N_IDX


def read_ss_weights(file: str) -> dict[str, jnp.ndarray]:
    """Read and oxDNA sequence-dependent weights file to a dictionary.

    This produces a dictionary of the parameters that are influenced by this
    file, which include stacking parameters:
        - ss_stack_weights (from STCK_X_Y entries)
        - eps_stack_kt_coeff (from STCK_FACT_EPS)
    and hydrogen-bonding parameters:
        - ss_hb_weights (from HYDR_X_Y entries, requires on or both of HYDR_A_T
          or HYDR_T_A and HYDR_G_C or HYDR_C_G to be present)

    File should be of the form KEY=VALUE, one per line, whitespace is ignore and
    float values may use 'f' suffix, but it is not required.
    """
    param_map = {}
    with Path(file).open("r") as f:
        for line in f:
            if kv := line.strip().replace(" ",""):
                key, val = kv.split("=")
                param_map[key] = float(val.replace("f", ""))

    stack_weights = np.zeros((4,4), dtype=jnp.float64)
    for i, row in enumerate(DNA_ALPHA):
        for j, col in enumerate(DNA_ALPHA):
            stack_weights[i,j] = param_map[f"STCK_{row}_{col}"]

    # in oxDNA the bonding pair mirrors are always set the the same value, read
    # from only one of them
    hb_a_t = param_map.get("HYDR_A_T", param_map["HYDR_T_A"])
    hb_g_c = param_map.get("HYDR_G_C", param_map["HYDR_C_G"])
    hb_weights = np.zeros((4,4), dtype=jnp.float64)
    hb_weights[N_IDX["A"], N_IDX["T"]] = hb_weights[N_IDX["T"], N_IDX["A"]] = hb_a_t
    hb_weights[N_IDX["G"], N_IDX["C"]] = hb_weights[N_IDX["C"], N_IDX["G"]] = hb_g_c

    return {
        "eps_stack_kt_coeff": jnp.array(param_map["STCK_FACT_EPS"], dtype=jnp.float64),
        "ss_stack_weights": jnp.array(stack_weights, dtype=jnp.float64),
        "ss_hb_weights": jnp.array(hb_weights, dtype=jnp.float64),
    }
