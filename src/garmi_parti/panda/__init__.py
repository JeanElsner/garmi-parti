"""
Teleoperation interfaces for the Panda robot.
"""

from __future__ import annotations

import pickle

import numpy as np
from panda_py import controllers
from scipy.spatial import transform as tr

from ..teleoperation import containers, interfaces, utils


class CartesianLeader(interfaces.PandaInterface):
    """
    Teleoperation interface that uses a Panda robot as a haptic input
    device in Cartesian space.
    """

    def __init__(self, params: containers.TeleopParams) -> None:
        super().__init__(params)
        self.paused = False

    def _pre_teleop(self, container: containers.TeleopContainer) -> None:
        super()._pre_teleop(container)
        container.controller = controllers.AppliedForce(
            damping=container.params.damping, filter_coeff=container.params.filter_coeff
        )

    def get_command(self) -> bytes:
        if self.paused:
            return pickle.dumps(containers.Displacement())
        return pickle.dumps(utils.compute_displacement(self.panda))

    def set_command(self, command: bytes) -> None:
        wrench: containers.Wrench = pickle.loads(command)
        self._set_command(self.panda, wrench)

    def _set_command(
        self, container: containers.TeleopContainer, wrench: containers.Wrench
    ) -> None:
        self.fdir(container)
        force_d = -container.params.gain_force * (
            container.transform_inv.apply(wrench.force)
        )
        torque_d = -container.params.gain_torque * (
            container.transform_inv.apply(wrench.torque)
        )
        container.controller.set_control(np.r_[force_d, torque_d])

    def pause(self) -> None:
        self.paused = True

    def unpause(self) -> None:
        self.panda.reinitialize()
        self.paused = False


class JointLeader(interfaces.PandaInterface):
    """
    Teleoperation interface that uses a Panda robot as a haptic input
    device in joint space.
    """

    def __init__(self, params: containers.TeleopParams) -> None:
        super().__init__(params)

    def _pre_teleop(self, container: containers.TeleopContainer) -> None:
        super()._pre_teleop(container)
        container.controller = controllers.AppliedTorque(
            damping=container.params.damping, filter_coeff=container.params.filter_coeff
        )

    def get_command(self) -> bytes:
        return pickle.dumps(self._get_command(self.panda))

    def _get_command(
        self, container: containers.TeleopContainer
    ) -> containers.JointStates:
        return containers.JointStates.from_state(container.arm.get_state())

    def set_command(self, command: bytes) -> None:
        joint_states: containers.JointStates = pickle.loads(command)
        self._set_command(self.panda, joint_states)

    def _set_command(
        self,
        container: containers.TeleopContainer,
        joint_states: containers.JointStates,
    ) -> None:
        self.fdir(container)
        container.controller.set_control(
            -container.params.gain_joint_torque * np.array(joint_states.tau_ext.torques)
        )

    def get_sync_command(self) -> bytes:
        return pickle.dumps(self._get_sync_command(self.panda))

    def _get_sync_command(
        self, container: containers.TeleopContainer
    ) -> containers.JointPositions:
        return containers.JointPositions(container.arm.get_state().q)

    def pause(self) -> None:
        self.panda.arm.stop_controller()

    def unpause(self) -> None:
        self.start_teleop()


class CartesianFollower(interfaces.PandaInterface):
    """
    Teleoperation interface that controls a Panda robot in Cartesian space.
    """

    def _pre_teleop(self, container: containers.TeleopContainer) -> None:
        super()._pre_teleop(container)

        # Compliance parameters
        stiffness = np.zeros((6, 6))
        stiffness[:3, :3] = np.eye(3) * container.params.linear_stiffness
        stiffness[3:, 3:] = np.eye(3) * container.params.angular_stiffness
        container.controller = controllers.CartesianImpedance(
            impedance=stiffness,
            damping_ratio=container.params.damping_ratio,
            nullspace_stiffness=container.params.nullspace_stiffness,
            filter_coeff=container.params.filter_coeff,
        )

    def get_command(self) -> bytes:
        return pickle.dumps(self._get_command(self.panda))

    def _get_command(self, container: containers.TeleopContainer) -> containers.Wrench:
        wrench = container.arm.get_state().O_F_ext_hat_K
        return containers.Wrench(
            force=container.transform_inv.apply(wrench[:3]),
            torque=container.transform_inv.apply(wrench[3:]),
        )

    def set_command(self, command: bytes) -> None:
        displacement: containers.Displacement = pickle.loads(command)
        self._set_command(displacement, self.panda)

    def _set_command(
        self, command: containers.Displacement, container: containers.TeleopContainer
    ) -> None:
        self.fdir(container)
        if container.params.lock_translation:
            command.linear = np.zeros(3)
        if container.params.lock_rotation:
            command.angular = tr.Rotation.identity()
        position_d = container.position_init + container.transform.apply(command.linear)
        orientation_d = (
            container.transform
            * command.angular
            * container.transform_inv
            * container.orientation_init
        )
        container.controller.set_control(position_d, orientation_d.as_quat())

    def pause(self) -> None:
        self.panda.arm.stop_controller()

    def unpause(self) -> None:
        self.start_teleop()


class JointFollower(interfaces.PandaInterface):
    """
    Teleoperation interface that controls a Panda robot in joint space.
    """

    def _pre_teleop(self, container: containers.TeleopContainer) -> None:
        super()._pre_teleop(container)

        container.controller = controllers.IntegratedVelocity(
            stiffness=container.params.stiffness,
            damping=container.params.damping,
        )

    def get_command(self) -> bytes:
        return pickle.dumps(self._get_command(self.panda))

    def _get_command(
        self, container: containers.TeleopContainer
    ) -> containers.JointStates:
        return containers.JointStates.from_state(container.arm.get_state())

    def set_command(self, command: bytes) -> None:
        joint_states: containers.JointStates = pickle.loads(command)
        self._set_command(joint_states, self.panda)

    def _set_command(
        self,
        joint_states: containers.JointStates,
        container: containers.TeleopContainer,
    ) -> None:
        self.fdir(container)
        ctrl: controllers.IntegratedVelocity = container.controller
        error = ctrl.get_qd() - joint_states.q.positions
        dqd = joint_states.dq.velocites - container.params.gain_drift * error
        container.controller.set_control(dqd)

    def pause(self) -> None:
        self.panda.arm.stop_controller()

    def unpause(self) -> None:
        self.start_teleop()

    def set_sync_command(self, command: bytes) -> None:
        self.move_arm(pickle.loads(command))
