"""Constant, homogeneous, isotropic material properties."""

from dataclasses import dataclass

from .validation import real_number


@dataclass(frozen=True)
class Material:
    conductivity_w_mk: float
    density_kg_m3: float
    specific_heat_j_kgk: float

    def __post_init__(self) -> None:
        real_number(self.conductivity_w_mk, "Conductivity (رسانندگی)", minimum=0, strict=True)
        real_number(self.density_kg_m3, "Density (چگالی)", minimum=0, strict=True)
        real_number(self.specific_heat_j_kgk, "Specific heat (گرمای ویژه)", minimum=0, strict=True)
