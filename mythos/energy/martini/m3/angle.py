"""Martini3 angle energy function."""

from typing import ClassVar

from mythos.energy.martini.m2.angle import Angle as Martini2Angle


class Angle(Martini2Angle):
    """Angle energy function for Martini3."""
    use_G96: ClassVar[bool] = False  # noqa: N815
