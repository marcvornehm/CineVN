#  Copyright (c) 2023. Marc Vornehm <marc.vornehm@fau.de>.

from enum import Enum, unique


@unique
class AsymmetricEchoMode(Enum):
    ASYMM_ECHO_WEAK = 0x1
    ASYMM_ECHO_STRONG = 0x2
    ASYMM_ECHO_HALF = 0x4


@unique
class PartialFourierFactor(Enum):
    PF_HALF = 0x01
    PF_5_8 = 0x02
    PF_6_8 = 0x04
    PF_7_8 = 0x08
    PF_OFF = 0x10
    PF_AUTO = 0x20


@unique
class Trajectory(Enum):
    TRAJECTORY_CARTESIAN = 0x01
    TRAJECTORY_RADIAL = 0x02
    TRAJECTORY_SPIRAL = 0x04
    TRAJECTORY_BLADE = 0x08
