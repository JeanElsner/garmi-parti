from __future__ import annotations

import pickle
import unittest
from unittest import mock

import numpy as np
import pytest
from numpy import testing

from garmi_parti import panda
from garmi_parti.launchers import panda_teleop
from garmi_parti.teleoperation import containers


class TestPanda(unittest.TestCase):
    def test_cartesian_leader(self):
        panda_params = containers.TeleopParams(
            "panda-host", q_idle=panda_teleop.Q_IDLE, q_teleop=panda_teleop.Q_TELEOP
        )
        leader = panda.CartesianLeader(panda_params)
        leader.pre_teleop()
        leader.start_teleop()
        cmd = containers.Wrench(np.zeros(3), np.zeros(3))
        leader.set_command(pickle.dumps(cmd))
        displacement: containers.Displacement = pickle.loads(leader.get_command())
        testing.assert_allclose(displacement.linear, np.zeros(3))
        testing.assert_allclose(displacement.angular.as_euler("XYZ"), np.zeros(3))
        leader.pause()
        leader.unpause()
        leader.open()
        leader.close()
        leader.post_teleop()

    def test_joint_leader(self):
        panda_params = containers.TeleopParams(
            "panda-host", q_idle=panda_teleop.Q_IDLE, q_teleop=panda_teleop.Q_TELEOP
        )
        leader = panda.JointLeader(panda_params)
        leader.pre_teleop()
        leader.start_teleop()
        sync_command: containers.JointPositions = pickle.loads(
            leader.get_sync_command()
        )
        testing.assert_allclose(sync_command.positions, np.zeros(7))
        cmd = containers.JointStates(
            q=containers.JointPositions(np.zeros(7)),
            dq=containers.JointVelocities(np.zeros(7)),
            tau_ext=containers.JointTorques(np.zeros(7)),
        )
        leader.set_command(pickle.dumps(cmd))
        joint_states: containers.JointStates = pickle.loads(leader.get_command())
        testing.assert_allclose(joint_states.tau_ext.torques, np.zeros(7))
        testing.assert_allclose(joint_states.dq.velocites, np.zeros(7))
        testing.assert_allclose(joint_states.q.positions, np.zeros(7))
        leader.pause()
        leader.unpause()
        leader.open()
        leader.close()
        leader.post_teleop()

    def test_cartesian_follower(self):
        panda_params = containers.TeleopParams(
            "panda-host", q_idle=panda_teleop.Q_IDLE, q_teleop=panda_teleop.Q_TELEOP
        )
        follower = panda.CartesianFollower(panda_params, True)
        follower.pre_teleop()
        follower.start_teleop()
        cmd = containers.Displacement()
        follower.set_command(pickle.dumps(cmd))
        wrench: containers.Wrench = pickle.loads(follower.get_command())
        testing.assert_allclose(wrench.force, np.zeros(3))
        testing.assert_allclose(wrench.torque, np.zeros(3))
        follower.open()
        follower.close()
        assert follower.panda.gripper.stop.call_count == 2
        assert follower.panda.gripper.grasp.call_count == 2
        follower.pause()
        follower.unpause()
        follower.post_teleop()

    def test_joint_follower(self):
        panda_params = containers.TeleopParams(
            "panda-host", q_idle=panda_teleop.Q_IDLE, q_teleop=panda_teleop.Q_TELEOP
        )
        follower = panda.JointFollower(panda_params, True)
        follower.pre_teleop()
        follower.start_teleop()
        follower.set_sync_command(pickle.dumps(containers.JointPositions(np.zeros(7))))
        cmd = containers.JointStates(
            q=containers.JointPositions(np.zeros(7)),
            dq=containers.JointVelocities(np.zeros(7)),
            tau_ext=containers.JointTorques(np.zeros(7)),
        )
        follower.set_command(pickle.dumps(cmd))
        joint_states: containers.JointStates = pickle.loads(follower.get_command())
        testing.assert_allclose(joint_states.tau_ext.torques, np.zeros(7))
        follower.open()
        follower.close()
        assert follower.panda.gripper.stop.call_count == 2
        assert follower.panda.gripper.grasp.call_count == 2
        follower.pause()
        follower.unpause()
        follower.post_teleop()

    @mock.patch("builtins.input", return_value="q")
    def test_entrypoints(self, *args):
        del args
        with pytest.raises(ConnectionRefusedError):
            panda_teleop.leader()

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_entrypoints_raises(self):
        with pytest.raises(RuntimeError) as ctx:
            panda_teleop.follower()
        assert "environment variable PANDA" in str(ctx.value.args[0])
        with pytest.raises(RuntimeError) as ctx:
            panda_teleop.leader()
        assert "environment variable PANDA" in str(ctx.value.args[0])
