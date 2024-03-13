from __future__ import annotations

import pickle
import time
import unittest
from unittest import mock

import numpy as np
import pytest
from numpy import testing

from garmi_parti import parti
from garmi_parti.launchers import parti_teleop
from garmi_parti.teleoperation import containers


class TestParti(unittest.TestCase):
    def test_cartesian_leader(self) -> None:
        leader = parti.CartesianLeader("left", "right")
        leader.pre_teleop()
        leader.start_teleop()
        cmd = containers.TwoArmWrench(
            containers.Wrench(np.zeros(3), np.zeros(3)),
            containers.Wrench(np.zeros(3), np.zeros(3)),
        )
        leader.set_command(pickle.dumps(cmd))
        displacement: containers.TwoArmDisplacement = pickle.loads(leader.get_command())
        self.assert_displacement(displacement.left)
        self.assert_displacement(displacement.right)
        leader.pause()
        leader.unpause()
        leader.post_teleop()

    def test_joint_leader(self) -> None:
        leader = parti.JointLeader("left-host", "right-host")
        leader.pre_teleop()
        leader.start_teleop()
        leader.get_sync_command()
        cmd = containers.TwoArmJointStates(
            left=containers.JointStates(
                q=containers.JointPositions(np.zeros(7)),
                dq=containers.JointVelocities(np.zeros(7)),
                tau_ext=containers.JointTorques(np.zeros(7)),
            ),
            right=containers.JointStates(
                q=containers.JointPositions(np.zeros(7)),
                dq=containers.JointVelocities(np.zeros(7)),
                tau_ext=containers.JointTorques(np.zeros(7)),
            ),
        )
        leader.set_command(pickle.dumps(cmd))
        joint_states: containers.TwoArmJointStates = pickle.loads(leader.get_command())
        self.assert_joint_states(joint_states.left)
        self.assert_joint_states(joint_states.right)
        leader.pause()
        leader.unpause()
        leader.post_teleop()

    def assert_joint_states(self, joint_states: containers.JointStates | None) -> None:
        assert joint_states is not None
        if joint_states.dq is not None:
            testing.assert_allclose(joint_states.dq.velocites, np.zeros(7))
        if joint_states.q is not None:
            testing.assert_allclose(joint_states.q.positions, np.zeros(7))

    def assert_displacement(self, displacement: containers.Displacement) -> None:
        testing.assert_allclose(displacement.linear, np.zeros(3))
        testing.assert_allclose(displacement.angular.as_euler("XYZ"), np.zeros(3))

    @mock.patch("builtins.input", return_value="q")
    def test_entrypoints(self, *args) -> None:
        del args
        with pytest.raises(ConnectionRefusedError):  # noqa: PT012
            parti_teleop.main()
            time.sleep(5)

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_entrypoints_raises(self) -> None:
        with pytest.raises(RuntimeError) as ctx:
            parti_teleop.main()
        assert "PANDA_LEFT and PANDA_RIGHT" in str(ctx.value.args[0])
