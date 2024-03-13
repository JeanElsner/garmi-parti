"""
Teleoperation interfaces for the Parti robot.
"""

from __future__ import annotations

import collections
import logging
import pickle
import time

import numpy as np
from scipy.spatial import transform

from .. import panda

# from pypartigp.zmq import Publisher
from ..teleoperation import containers, interfaces, utils

Q_IDLE_LEFT = containers.JointPositions(
    [0, -np.pi / 2, 0, -2 * np.pi / 4, 0, np.pi / 2, np.pi / 4]
)
Q_IDLE_RIGHT = Q_IDLE_LEFT
Q_TELEOP_LEFT = containers.JointPositions([0.02, -1.18, -0.06, -1.47, 0.04, 1.92, 0.75])
Q_TELEOP_RIGHT = containers.JointPositions(
    [-0.04, -1.16, 0.08, -1.57, 0.00, 2.05, 0.84]
)

LEFT_TRANSFORM = transform.Rotation.from_euler(
    "XYZ", [0, 90 / 180 * np.pi, -90 / 180 * np.pi]
)
RIGHT_TRANSFORM = transform.Rotation.from_euler(
    "XYZ", [0, 90 / 180 * np.pi, 90 / 180 * np.pi]
)

DAMPING = [80, 0, 0, 0, 0, 0, 0]


class Tickable:
    """
    Tickable class that can be mixed in to track runtime.
    """

    def __init__(
        self,
        window_size: int = 1000,
        interval: int = 2000,
        logger: logging.Logger | None = None,
    ) -> None:
        self._t_last = time.perf_counter()
        self._t_window: collections.deque[float] = collections.deque(maxlen=window_size)
        self.num_ticks = 0
        self.interval = interval
        self._logger = logger

    def tick(self) -> None:
        """
        Reset the internal time since the last tick was called.
        """
        t_curr = time.perf_counter()
        self._t_window.append(t_curr - self._t_last)
        self._t_last = t_curr
        self.num_ticks += 1

        if self._logger is not None and self.num_ticks % self.interval == 0:
            self._logger.debug(
                "Current frequency: %fHz", 1 / np.average(self._t_window)
            )


class CartesianLeader(panda.CartesianLeader, interfaces.TwoArmPandaInterface, Tickable):
    """
    Use PARTI or a similar system as a cartesian teleoperation leader device.
    """

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        left_hostname: str,
        right_hostname: str,
        window_size: int = 1000,
        interval: int = 2000,
        logger: logging.Logger | None = None,
    ) -> None:
        Tickable.__init__(self, window_size, interval, logger)  # type: ignore[call-arg]

        self.paused = False

        q_idle = containers.TwoArmJointPositions(left=Q_IDLE_LEFT, right=Q_IDLE_RIGHT)
        q_teleop = containers.TwoArmJointPositions(
            left=Q_TELEOP_LEFT, right=Q_TELEOP_RIGHT
        )

        left_transform = LEFT_TRANSFORM
        right_transform = RIGHT_TRANSFORM

        interfaces.TwoArmPandaInterface.__init__(
            self,
            containers.TeleopParams(
                left_hostname,
                left_transform,
                q_idle=q_idle.left,
                q_teleop=q_teleop.left,
                damping=DAMPING,
            ),
            containers.TeleopParams(
                right_hostname,
                right_transform,
                q_idle=q_idle.right,
                q_teleop=q_teleop.right,
                damping=DAMPING,
            ),
        )

    def get_command(self) -> bytes:
        if self.paused:
            return pickle.dumps(
                containers.TwoArmDisplacement(
                    containers.Displacement(), containers.Displacement()
                )
            )
        displacement = containers.TwoArmDisplacement(
            left=utils.compute_displacement(self.left),
            right=utils.compute_displacement(self.right),
        )
        return pickle.dumps(displacement)

    def set_command(self, command: bytes) -> None:
        wrench: containers.TwoArmWrench = pickle.loads(command)
        self._set_command(self.left, wrench.left)
        self._set_command(self.right, wrench.right)
        self.tick()

    def unpause(self) -> None:
        self.left.reinitialize()
        self.right.reinitialize()
        self.paused = False


class JointLeader(panda.JointLeader, interfaces.TwoArmPandaInterface, Tickable):
    """
    Use PARTI or a similar system as a joint-space teleoperation leader device.
    """

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        left_hostname: str,
        right_hostname: str,
        window_size: int = 1000,
        interval: int = 2000,
        logger: logging.Logger | None = None,
    ) -> None:
        Tickable.__init__(self, window_size, interval, logger)  # type: ignore[call-arg]

        q_idle = containers.TwoArmJointPositions(left=Q_IDLE_LEFT, right=Q_IDLE_RIGHT)
        q_teleop = containers.TwoArmJointPositions(
            left=Q_TELEOP_LEFT, right=Q_TELEOP_RIGHT
        )

        left_transform = LEFT_TRANSFORM
        right_transform = RIGHT_TRANSFORM

        interfaces.TwoArmPandaInterface.__init__(
            self,
            containers.TeleopParams(
                left_hostname,
                left_transform,
                q_idle=q_idle.left,
                q_teleop=q_teleop.left,
                damping=DAMPING,
            ),
            containers.TeleopParams(
                right_hostname,
                right_transform,
                q_idle=q_idle.right,
                q_teleop=q_teleop.right,
                damping=DAMPING,
            ),
        )

    def get_command(self) -> bytes:
        return pickle.dumps(
            containers.TwoArmJointStates(
                left=self._get_command(self.left), right=self._get_command(self.right)
            )
        )

    def set_command(self, command: bytes) -> None:
        joint_states: containers.TwoArmJointStates = pickle.loads(command)
        if joint_states.left is not None:
            self._set_command(self.left, joint_states.left)
        if joint_states.right is not None:
            self._set_command(self.right, joint_states.right)
        self.tick()

    def pause(self) -> None:
        self.left.arm.stop_controller()
        self.right.arm.stop_controller()

    def get_sync_command(self) -> bytes:
        return pickle.dumps(
            containers.TwoArmJointPositions(
                left=self._get_sync_command(self.left),
                right=self._get_sync_command(self.right),
            )
        )
