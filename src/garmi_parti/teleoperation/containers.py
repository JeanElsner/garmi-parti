"""
Container dataclasses for robot and teleoperation entities.
"""

from __future__ import annotations

import abc
import dataclasses
import typing

import numpy as np
import panda_py
from panda_py import controllers, libfranka
from scipy.spatial import transform as tr


@dataclasses.dataclass
class JointPositions:
    """
    Joint positions container.
    """

    positions: np.ndarray


@dataclasses.dataclass
class JointVelocities:
    """
    Joint velocities container.
    """

    velocites: np.ndarray


@dataclasses.dataclass
class JointTorques:
    """
    Joint torques container.
    """

    torques: np.ndarray


@dataclasses.dataclass
class JointStates:
    """Joint state container."""

    q: JointPositions
    dq: JointVelocities
    tau_ext: JointTorques

    @classmethod
    def from_state(cls, state: libfranka.RobotState) -> JointStates:
        """Construct a joint states container from a libfranka robot state."""
        return cls(
            JointPositions(np.array(state.q)),
            JointVelocities(np.array(state.dq)),
            JointTorques(np.array(state.tau_ext_hat_filtered)),
        )


def _default_stiffness() -> np.ndarray:
    return np.array([600, 600, 600, 600, 250, 150, 50])


def _default_damping() -> np.ndarray:
    return np.array([50, 50, 50, 20, 20, 20, 10])


def _default_position() -> np.ndarray:
    return np.zeros(3)


@dataclasses.dataclass
class TeleopParams:
    """
    Parameters for a teleoperator.
    """

    hostname: str
    transform: tr.Rotation = dataclasses.field(default_factory=tr.Rotation.identity)
    q_idle: JointPositions | None = None
    q_teleop: JointPositions | None = None
    lock_translation: bool = False
    lock_rotation: bool = False
    # Joint parameters
    stiffness: np.ndarray = dataclasses.field(default_factory=_default_stiffness)
    damping: np.ndarray = dataclasses.field(default_factory=_default_damping)
    # Cartesian parameters
    linear_stiffness: float = 400
    angular_stiffness: float = 20
    nullspace_stiffness: float = 0.5
    damping_ratio: float = 1.0
    # Filter parameters
    filter_coeff: float = 1.0
    gain_force: float = 0.4
    gain_torque: float = 0.0
    gain_joint_torque: float = 0.8
    speed_factor: float = 0.2
    gain_drift: float = 8.0


@dataclasses.dataclass
class TeleopContainer:
    """
    Container utility class for multi-arm teleoperation.
    """

    arm: panda_py.Panda
    gripper: libfranka.Gripper
    controller: controllers.TorqueController = None
    params: TeleopParams = dataclasses.field(
        default_factory=lambda: TeleopParams("localhost")
    )
    position_init: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    orientation_init: tr.Rotation = dataclasses.field(
        default_factory=tr.Rotation.identity
    )
    orientation_init_inv: tr.Rotation = dataclasses.field(
        default_factory=tr.Rotation.identity
    )
    transform: tr.Rotation = dataclasses.field(default_factory=tr.Rotation.identity)
    transform_inv: tr.Rotation = dataclasses.field(default_factory=tr.Rotation.identity)

    def reinitialize(self) -> None:
        """
        Reinitialize the reference pose to the current robot pose.
        """
        self.position_init = self.arm.get_position()
        self.orientation_init = self.get_rotation()
        self.orientation_init_inv = self.orientation_init.inv()

    def get_rotation(self) -> tr.Rotation:
        """
        Get a `scipy.spatial.tr.Rotation` from the robot's current
        end effector orientation.

        Returns:
          scipy.spatial.tr.Rotation: Unit quaternion representing the
            the robot end effector's orientation.
        """
        return tr.Rotation.from_quat(self.arm.get_orientation(scalar_first=False))


@dataclasses.dataclass
class Displacement:
    """
    Container holding a relative pose, i.e. SE3 displacement.
    """

    linear: np.ndarray = dataclasses.field(default_factory=_default_position)
    angular: tr.Rotation = dataclasses.field(default_factory=tr.Rotation.identity)


@dataclasses.dataclass
class Wrench:
    """
    Wrench container.
    """

    force: np.ndarray
    torque: np.ndarray


@dataclasses.dataclass
class Pose:
    """
    Pose container.
    """

    position: np.ndarray = dataclasses.field(default_factory=_default_position)
    orientation: tr.Rotation = dataclasses.field(default_factory=tr.Rotation.identity)


T = typing.TypeVar("T")


@dataclasses.dataclass
class TwoArmContainer(abc.ABC, typing.Generic[T]):
    """
    Abstract base class for two-arm containers.
    """

    left: T
    right: T


@dataclasses.dataclass
class TwoArmTeleopContainer(TwoArmContainer[TeleopContainer]):
    """
    Two arm teleoperation container.
    """


@dataclasses.dataclass
class TwoArmDisplacement(TwoArmContainer[Displacement]):
    """
    Two-arm displacement container.
    """


@dataclasses.dataclass
class TwoArmWrench(TwoArmContainer[Wrench]):
    """
    Two-arm wrench container.
    """


@dataclasses.dataclass
class TwoArmJointPositions(TwoArmContainer[typing.Optional[JointPositions]]):
    """
    Two-arm joint positions container.
    """


@dataclasses.dataclass
class TwoArmJointVelocities(TwoArmContainer[typing.Optional[JointVelocities]]):
    """
    Two-arm joint positions container.
    """


@dataclasses.dataclass
class TwoArmJointStates(TwoArmContainer[typing.Optional[JointStates]]):
    """
    Two-arm joint states container.
    """


@dataclasses.dataclass
class TwoArmPose(TwoArmContainer[Pose]):
    """
    Two-arm pose container.
    """


@dataclasses.dataclass
class TwoArmJointTorques(TwoArmContainer[JointTorques]):
    """
    Two-arm joint torques container.
    """
