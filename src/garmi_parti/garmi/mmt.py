"""
Model-mediated teleoperation interfaces for the Garmi robot.
"""

from __future__ import annotations

from .. import garmi
from ..teleoperation.containers import TeleopParams


class Follower(garmi.JointFollower):
    """
    Use GARMI or a similar system as a model-mediated teleoperation follower device.
    """

    def __init__(
        self,
        left: TeleopParams,
        right: TeleopParams,
        has_left_gripper: bool = False,
        has_right_gripper: bool = False,
    ) -> None:
        super().__init__(left, right, has_left_gripper, has_right_gripper)
        self.close("left")
        self.close("right")

    def pause(self) -> None:
        pass

    def unpause(self) -> None:
        pass
