"""
Haptic simulation module for the Parti robot.
"""

from __future__ import annotations

import csv
import pathlib
import pickle
import threading

import dm_env
import numpy as np
import panda_py
import roslibpy
import spatialmath
import zmq
from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from dm_env import specs
from dm_robotics.agentflow.preprocessors import timestep_preprocessor
from dm_robotics.moma import effector
from dm_robotics.moma.sensors import robot_arm_sensor
from dm_robotics.moma.sensors.robot_arm_sensor import joint_observations
from dm_robotics.panda import arm, gripper
from dm_robotics.panda import parameters as params
from dm_robotics.transformations import transformations as tr

from ..teleoperation import containers

# right arm base frame in world frame
T_0_right0 = tr.pos_quat_to_hmat([0, -0.24, 0.5], [0.5, 0.5, 0.5, 0.5])

# plane frame in world frame
T_0_plane = tr.pos_quat_to_hmat([0.845, 0.012, 0.34], [1, 0, 0, 0])

# world frame in plane frame
T_plane_0 = tr.hmat_inv(T_0_plane)

DEADTIME = 1.0


class TeleopAgent:
    """Teleoperation agent for MMT.

    This agent uses interprocess communication (IPC) to communicate
    with a local teleoperation node. Run `parti-mmt` on the PARTI
    system and `garmi-mmt` on the GARMI system to run a full MMT scenario.
    """

    def __init__(
        self,
        arena: composer.Arena,
        spec: specs.BoundedArray,
        action: np.ndarray,
        use_ros: bool,
        ros_hostname: str = "localhost",
        rosbridge_port: int = 9090,
    ) -> None:
        self._spec = spec
        self._arena: composer.Arena = arena
        self._action = action
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("ipc:///tmp/parti-haptic-sim")
        self._obs: list[dict[str, np.ndarray]] = []
        self._object_qpos = np.array([0, 0, 0])
        self._object_qpos_offset = np.array([0.04, 0.023, 0.05])
        self._plane_declination = -0.1745
        self._ft = np.zeros(6)

        if use_ros:
            client = roslibpy.Ros(host=ros_hostname, port=rosbridge_port)
            client.run()
            listener = roslibpy.Topic(
                client, "/icg_tracker_1/pose", "geometry_msgs/PoseStamped"
            )
            listener.subscribe(self._object_callback)
            listener_2 = roslibpy.Topic(
                client, "/detect_plane/pose", "geometry_msgs/Pose"
            )
            listener_2.subscribe(self._plane_callback)
            listener_3 = roslibpy.Topic(client, "/ft_compensated", "geometry_msgs/WrenchStamped")
            listener_3.subscribe(self._ft_callback)

    def shutdown(self) -> None:
        """
        Shutdown, closing any open connections.
        """
        self.socket.close()
        self.context.term()
        save_obs(self._obs, "obs.csv")

    def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
        """
        Steps the agent.
        Receives a percept from the environment and returns an action.
        """
        self._comm(timestep)
        self._obs.append(timestep.observation)
        action = np.zeros(19)

        # # This controls the position of the object (x, y, theta)
        action[-3:] = self._object_qpos + self._object_qpos_offset
        # This controls the declination of the plane
        action[-4] = self._plane_declination
        return action

    def _ft_callback(self, message: dict) -> None:
        force = message["wrench"]["force"]
        torque = message["wrench"]["torque"]
        self._ft = np.array([force["x"], force["y"], force["z"], torque["x"], torque["y"], torque["z"]])

    def get_force_torque_obs(self, timestep: timestep_preprocessor.PreprocessorTimestep) -> np.ndarray:
        del timestep
        return self._ft

    def _plane_callback(self, message: dict) -> None:
        normal = [
            message["position"]["x"],
            message["position"]["y"],
            message["position"]["z"],
        ]
        global_normal = T_0_right0[:3, :3] @ normal
        orientation = tr.quat_between_vectors([0, 0, 1], global_normal[:3])

        self._plane_declination = tr.quat_to_euler(orientation)[0]

    def _object_callback(self, message: dict) -> None:
        # object in right arm base frame
        T_right0_object = tr.pos_quat_to_hmat(  # pylint: disable=invalid-name
            [
                message["pose"]["position"]["x"],
                message["pose"]["position"]["y"],
                message["pose"]["position"]["z"],
            ],
            [
                message["pose"]["orientation"]["w"],
                message["pose"]["orientation"]["x"],
                message["pose"]["orientation"]["y"],
                message["pose"]["orientation"]["z"],
            ],
        )
        # # we move the COM
        # COM = tr.pos_quat_to_hmat([0.06, -0.055, 0.015], [1, 0, 0, 0])
        # COM_inv = tr.hmat_inv(tr.pos_quat_to_hmat([0.06, -0.055, 0.015], [1, 0, 0, 0]))
        # object in plane frame
        T_plane_object = T_plane_0 @ T_0_right0 @ T_right0_object  # pylint: disable=invalid-name

        theta = spatialmath.SO3(T_plane_object[:3, :3]).rpy()
        theta = theta[2]

        # hack
        delta = spatialmath.SE2(theta) @ spatialmath.SE2(0.06, -0.055)

        self._object_qpos = np.array(
            [
                T_plane_object[0, 3] + delta.t[0],
                T_plane_object[1, 3] + delta.t[1],
                theta,
            ]
        )

    def _comm(self, timestep: dm_env.TimeStep) -> None:
        joint_states = containers.TwoArmJointStates(
            left=containers.JointStates(
                q=containers.JointPositions(timestep.observation["left_joint_pos"]),
                dq=containers.JointVelocities(timestep.observation["left_joint_vel"]),
                tau_ext=containers.JointTorques(
                    timestep.observation["left_joint_torques"]
                ),
            ),
            right=containers.JointStates(
                q=containers.JointPositions(timestep.observation["right_joint_pos"]),
                dq=containers.JointVelocities(timestep.observation["right_joint_vel"]),
                tau_ext=containers.JointTorques(
                    timestep.observation["right_joint_torques"]
                ),
            ),
        )
        self.socket.send(pickle.dumps(joint_states))


def save_obs(obs: list[dict[str, np.ndarray]], output_file: str) -> None:
    """Saves a list of observations to a CSV file."""
    # Extracting field names from the first row
    first_row = obs[0]
    fieldnames = []
    for key, value in first_row.items():
        if np.ndim(value) > 0:
            if len(value) > 1:
                fieldnames.extend([f"{key}_{i+1}" for i in range(len(value))])
            else:
                fieldnames.append(key)
        else:
            fieldnames.append(key)
    # Writing to CSV
    with pathlib.Path(output_file).open("w", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in obs:
            new_row = {}
            for key, value in row.items():
                if np.ndim(value) > 0:
                    if len(value) > 1:
                        for i, v in enumerate(value):
                            new_row[f"{key}_{i+1}"] = v
                    else:
                        new_row[key] = value[0]
                else:
                    new_row[key] = value.item()  # Convert 0-d array to scalar
            writer.writerow(new_row)


class SceneEffector(effector.Effector):
    """
    Effector used to update the state of tracked objects and the environment.
    """

    def __init__(self, plane: mjcf.Element, obj: list[mjcf.Element]) -> None:
        self._plane = plane
        self._object = obj
        self._object_obs = np.zeros(3, dtype=np.float32)
        self._virtual_object_obs = np.zeros(3, dtype=np.float32)
        self._updating_obs = np.zeros(1, dtype=np.float32)
        self._plane_obs = np.zeros(1, dtype=np.float32)
        self._virtual_plane_obs = np.zeros(1, dtype=np.float32)
        self._spec = None
        self._deadtime = 0

    def close(self) -> None:
        pass

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        pass

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        del physics
        if self._spec is None:
            self._spec = specs.BoundedArray(
                (4,),
                np.float32,
                np.full((4,), -10, dtype=np.float32),
                np.full((4,), 10, dtype=np.float32),
                "\t".join(
                    [
                        f"{self.prefix}_{n}"
                        for n in [
                            "plane",
                            "object_x",
                            "object_y",
                            "object_theta",
                        ]
                    ]
                ),
            )
        return self._spec

    @property
    def prefix(self) -> str:
        return "scene"

    def get_object_observations(
        self, timestep: timestep_preprocessor.PreprocessorTimestep
    ) -> np.ndarray:
        del timestep
        return self._object_obs

    def get_virtual_object_observations(
        self, timestep: timestep_preprocessor.PreprocessorTimestep
    ) -> np.ndarray:
        del timestep
        return self._virtual_object_obs

    def get_plane_observations(
        self, timestep: timestep_preprocessor.PreprocessorTimestep
    ) -> np.ndarray:
        del timestep
        return self._plane_obs

    def get_virtual_plane_observations(
        self, timestep: timestep_preprocessor.PreprocessorTimestep
    ) -> np.ndarray:
        del timestep
        return self._virtual_plane_obs

    def get_updating_observations(
        self, timestep: timestep_preprocessor.PreprocessorTimestep
    ) -> np.ndarray:
        del timestep
        return self._updating_obs

    def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
        self._object_obs = command[1:]
        self._virtual_object_obs = physics.bind(self._object).qpos[:].astype(np.float32)
        self._plane_obs = np.atleast_1d(command[0])
        self._virtual_plane_obs = physics.bind(self._plane).qpos[:].astype(np.float32)

        if physics.time() - self._deadtime < DEADTIME:
            return
        update = True
        self._updating_obs = np.ones(1, dtype=np.float32)
        allowed_collision = ["plane", "side_a", "side_b", "side_c", "side_d"]
        for contact in physics.data.contact:
            geom1_name = physics.model.id2name(contact.geom1, "geom")
            geom2_name = physics.model.id2name(contact.geom2, "geom")
            if (
                (
                    geom1_name in ["object_1", "object_2"]
                    or geom2_name in ["object_1", "object_2"]
                )
                and geom1_name not in allowed_collision
                and geom2_name not in allowed_collision
            ):
                update = False
                self._updating_obs = np.zeros(1, dtype=np.float32)
                self._deadtime = physics.time()
                break
        # only update scene if object is not in contact with element other than plane
        if update:
            # update object
            physics.bind(self._object).qpos[:] = command[1:]
            physics.bind(self._object).qvel[:] = np.zeros(3)
            # update plane
            physics.bind(self._plane).qpos[:] = command[0]
            physics.bind(self._plane).qvel[:] = 0


ENDEFFECTOR_XML_PATH = (
    pathlib.Path(__file__).parent / ".." / "assets" / "endeffector.xml"
)


class EndEffector(gripper.DummyHand):
    def _build(self, name: str = "endeffector") -> None:
        self._mjcf_root = mjcf.from_path(ENDEFFECTOR_XML_PATH)
        self._mjcf_root.model = name
        self._tool_center_point = self._mjcf_root.find("site", "TCP")


def make_endeffector(name: str) -> params.GripperParams:
    """Creates the endeffector."""
    name = f"{name}_gripper"
    gripper_model = EndEffector()
    gripper_effector = None
    return params.GripperParams(gripper_model, gripper_effector)


def make_gripper(name: str) -> params.GripperParams:
    """Creates a Panda gripper.

    We create the Panda gripper manually here because the connected
    hardware doesn't actually have grippers. If we were to use the
    grippers configured by dm-robotics-panda, the HIL mechanism would
    attempt to connect to nonexistent grippers.
    """
    name = f"{name}_gripper"
    gripper_model = gripper.PandaHand()
    gripper_sensor = gripper.PandaHandSensor(gripper_model, name)
    gripper_effector = gripper.PandaHandEffector(
        params.RobotParams(name=name), gripper_model, gripper_sensor
    )
    return params.GripperParams(gripper_model, gripper_effector)


def move_arms(
    left_hostname: str,
    right_hostname: str,
    q_left: np.ndarray,
    q_right: np.ndarray,
    speed_factor: float,
) -> None:
    """Move robot arms.

    Utility function that moves two robot arms simultaneously
    to the given joint positions.
    """
    panda_left = panda_py.Panda(left_hostname)
    panda_right = panda_py.Panda(right_hostname)
    t_left = threading.Thread(
        target=panda_left.move_to_joint_position, args=(q_left.positions, speed_factor)
    )
    t_right = threading.Thread(
        target=panda_right.move_to_joint_position,
        args=(q_right.positions, speed_factor),
    )
    t_left.start()
    t_right.start()
    t_left.join()
    t_right.join()
    del panda_left, panda_right


class FollowerSensor(robot_arm_sensor.RobotArmSensor):
    def __init__(self, arm: arm.Panda):
        self._name = "follower"
        self._arm = arm
        self._follower_joint_state = containers.TwoArmJointStates(left=None, right=None)
        self._observables = {
            self.get_obs_key(
                joint_observations.Observations.JOINT_POS
            ): observable.Generic(self._joint_pos),
            self.get_obs_key(
                joint_observations.Observations.JOINT_VEL
            ): observable.Generic(self._joint_vel),
            self.get_obs_key(
                joint_observations.Observations.JOINT_TORQUES
            ): observable.Generic(self._joint_torques),
            "follower_virtual_joint_torques":observable.Generic(self._virtual_joint_torques)
        }
        for obs in self._observables.values():
            obs.enabled = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind("ipc:///tmp/parti-haptic-sim-obs")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def _joint_pos(self, physics: mjcf.Physics) -> np.ndarray:
        del physics
        if self._follower_joint_state.left is None:
            return np.zeros(7)
        return self._follower_joint_state.left.q.positions

    def _joint_vel(self, physics: mjcf.Physics) -> np.ndarray:
        del physics
        if self._follower_joint_state.left is None:
            return np.zeros(7)
        return self._follower_joint_state.left.dq.velocites

    def _joint_torques(self, physics: mjcf.Physics) -> np.ndarray:
        if self._follower_joint_state.left is None:
            return np.zeros(7)
        return self._follower_joint_state.left.tau_ext.torques
    
    def _virtual_joint_torques(self, physics: mjcf.Physics) -> np.ndarray:
        if self._follower_joint_state.left is None:
            return np.zeros(7)
        return physics.bind(self._arm.joints).qfrc_constraint

    def _run(self) -> None:
        while True:
            try:
                self._follower_joint_state = pickle.loads(self.socket.recv())
            except zmq.error.ContextTerminated:
                break
        self.socket.close()

    def close(self) -> None:
        self.context.term()
