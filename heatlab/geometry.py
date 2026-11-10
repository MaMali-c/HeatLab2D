"""Uniform cell-centred mesh for a rectangular slab, in SI units."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .validation import InputError, integer, real_number

MAX_CELLS = 40_000


@dataclass(frozen=True)
class Grid:
    width_m: float
    height_m: float
    thickness_m: float
    nx: int
    ny: int

    def __post_init__(self) -> None:
        real_number(self.width_m, "Width (عرض)", minimum=0, strict=True)
        real_number(self.height_m, "Height (ارتفاع)", minimum=0, strict=True)
        real_number(self.thickness_m, "Thickness (ضخامت)", minimum=0, strict=True)
        integer(self.nx, "Cells along x (سلول در راستای x)", 1, MAX_CELLS)
        integer(self.ny, "Cells along y (سلول در راستای y)", 1, MAX_CELLS)
        if self.size > MAX_CELLS:
            raise InputError(f"Use at most {MAX_CELLS:,} cells. (تعداد سلول‌ها از حد مجاز بیشتر است.)")
        for value in (self.cell_volume, self.x_face_area, self.y_face_area):
            real_number(value, "Cell measure (اندازهٔ سلول)", minimum=0, strict=True)

    @property
    def size(self) -> int:
        return int(self.nx) * int(self.ny)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def dx(self) -> float:
        return self.width_m / self.nx

    @property
    def dy(self) -> float:
        return self.height_m / self.ny

    @property
    def x_face_area(self) -> float:
        """Area of a face normal to x."""
        return self.dy * self.thickness_m

    @property
    def y_face_area(self) -> float:
        return self.dx * self.thickness_m

    @property
    def cell_volume(self) -> float:
        return self.dx * self.x_face_area

    @property
    def x_m(self) -> NDArray[np.float64]:
        return (np.arange(self.nx) + 0.5) * self.dx

    @property
    def y_m(self) -> NDArray[np.float64]:
        return (np.arange(self.ny) + 0.5) * self.dy

    @property
    def x_faces_m(self) -> NDArray[np.float64]:
        return np.linspace(0.0, self.width_m, self.nx + 1)

    @property
    def y_faces_m(self) -> NDArray[np.float64]:
        return np.linspace(0.0, self.height_m, self.ny + 1)
