"""Uniform and time-independent conditions specified separately for each side."""

from dataclasses import dataclass
from enum import Enum

from .validation import InputError, real_number


class Kind(Enum):
    TEMPERATURE = "Fixed temperature (دمای ثابت)"
    FLUX = "Inward heat flux (شار گرمای ورودی)"
    INSULATED = "Insulated (عایق)"
    CONVECTION = "Convection (همرفت)"


class Side(Enum):
    LEFT = "Left (چپ)"
    RIGHT = "Right (راست)"
    BOTTOM = "Bottom (پایین)"
    TOP = "Top (بالا)"


@dataclass(frozen=True)
class Boundary:
    kind: Kind
    value: float = 0.0
    h_w_m2k: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, Kind):
            raise InputError("Select a valid boundary type. (نوع شرط مرزی معتبر را انتخاب کنید.)")
        real_number(self.h_w_m2k, "Convection coefficient (ضریب همرفت)", minimum=0)
        if self.kind is Kind.CONVECTION:
            real_number(self.h_w_m2k, "Convection coefficient (ضریب همرفت)", minimum=0, strict=True)
        elif self.h_w_m2k != 0:
            raise InputError("Specify h only for convection. (ضریب h فقط برای همرفت است.)")

        if self.kind in (Kind.TEMPERATURE, Kind.CONVECTION):
            real_number(self.value, "Boundary temperature [K] (دمای مرزی)", minimum=0)
        else:
            real_number(self.value, "Inward heat flux (شار گرمای ورودی)")
        if self.kind is Kind.INSULATED and self.value != 0:
            raise InputError("An insulated side requires zero flux. (شار ضلع عایق باید صفر باشد.)")

    @property
    def anchors_temperature(self) -> bool:
        return self.kind in (Kind.TEMPERATURE, Kind.CONVECTION)


@dataclass(frozen=True)
class Boundaries:
    left: Boundary
    right: Boundary
    bottom: Boundary
    top: Boundary

    def __post_init__(self) -> None:
        if any(not isinstance(self[side], Boundary) for side in Side):
            raise InputError("Provide a condition for each side. (برای هر ضلع یک شرط مرزی وارد کنید.)")

    def __getitem__(self, side: Side) -> Boundary:
        return getattr(self, side.name.lower())

    @property
    def has_temperature_anchor(self) -> bool:
        return any(self[side].anchors_temperature for side in Side)
