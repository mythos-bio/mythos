"""Units for the oxDNA model."""

import mythos.utils.types as jd_types

ANGSTROMS_PER_OXDNA_LENGTH = 8.518
ANGSTROMS_PER_NM = 10
NM_PER_OXDNA_LENGTH = ANGSTROMS_PER_OXDNA_LENGTH / ANGSTROMS_PER_NM
PN_PER_OXDNA_FORCE = 48.63
JOULES_PER_OXDNA_ENERGY = 4.142e-20


def get_kt(t_kelvin: jd_types.ARR_OR_SCALAR) -> jd_types.ARR_OR_SCALAR:
    """Converts a temperature in Kelvin to kT in simulation units."""
    return 0.1 * t_kelvin / 300.0

def get_kt_from_c(t_celsius: jd_types.ARR_OR_SCALAR) -> jd_types.ARR_OR_SCALAR:
    """Converts a temperature in Celsius to kT in simulation units."""
    return get_kt(t_celsius + 273.15)

def get_kt_from_string(temp_str: str) -> float:
    """Converts a temperature string (e.g. '300K', '27C') to kT in simulation units."""
    if temp_str.endswith("K"):
        t_kelvin = float(temp_str.replace("K", ""))
        return get_kt(t_kelvin)
    if temp_str.endswith("C"):
        t_celsius = float(temp_str.replace("C", ""))
        return get_kt_from_c(t_celsius)
    raise ValueError(f"Invalid temperature string: {temp_str}")

def from_kt(kt: jd_types.ARR_OR_SCALAR) -> jd_types.ARR_OR_SCALAR:
    """Converts kT in simulation units to temperature in Kelvin."""
    return 300.0 * kt / 0.1

