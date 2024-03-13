"""
Teleoperation interfaces for the Garmi robot.
"""

from __future__ import annotations

import pickle

from .. import panda
from ..teleoperation import containers, interfaces


class CartesianFollower(panda.CartesianFollower, interfaces.TwoArmPandaInterface):
    """
    Teleoperation interface that controls the two arms of Garmi
    in Cartesian space.
    """

    def get_command(self) -> bytes:
        return pickle.dumps(
            containers.TwoArmWrench(
                left=self._get_command(self.left), right=self._get_command(self.right)
            )
        )

    def set_command(self, command: bytes) -> None:
        displacement: containers.TwoArmDisplacement = pickle.loads(command)
        self._set_command(displacement.left, self.left)
        self._set_command(displacement.right, self.right)

    def pause(self) -> None:
        self.left.arm.stop_controller()
        self.right.arm.stop_controller()


class JointFollower(panda.JointFollower, interfaces.TwoArmPandaInterface):
    """
    Teleoperation interface that controls the two arms of Garmi
    in joint space.
    """

    def get_command(self) -> bytes:
        return pickle.dumps(
            containers.TwoArmJointStates(
                left=self._get_command(self.left), right=self._get_command(self.right)
            )
        )

    def set_command(self, command: bytes) -> None:
        joint_states: containers.TwoArmJointStates = pickle.loads(command)
        if joint_states.left is not None:
            self._set_command(joint_states.left, self.left)
        if joint_states.right is not None:
            self._set_command(joint_states.right, self.right)

    def pause(self) -> None:
        self.left.arm.stop_controller()
        self.right.arm.stop_controller()

    def set_sync_command(self, command: bytes) -> None:
        self.move_arms(pickle.loads(command))
