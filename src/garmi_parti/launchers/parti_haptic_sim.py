"""
Haptic interaction demo for the PARTI system.

This demo loads a simulation scene with HIL connection to the
PARTI system that allows the user to manipulate and haptically
interact with the virtual environment.
Alternatively, a teleoperation connection with a two-arm follower
can be established for a model-mediated teleoperation (MMT) scenario.
"""

from __future__ import annotations

import argparse
import logging
import pathlib

import numpy as np
from dm_control import composer
from dm_env import specs
from dm_robotics.agentflow.preprocessors import observation_transforms
from dm_robotics.moma.tasks import run_loop
from dm_robotics.panda import arm_constants, environment
from dm_robotics.panda import parameters as params
from dm_robotics.panda import utils as dmr_panda_utils

from ..parti import haptic_simulation as sim
from ..teleoperation import containers, utils

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("parti_haptic_sim")


XML_PATH = (
    pathlib.Path(pathlib.Path(__file__).parent.parent) / "assets" / "parti_mmt.xml"
)

Q_TELEOP_LEFT = containers.JointPositions([0.02, -1.18, -0.06, -1.47, 0.04, 1.92, 0.75])
Q_TELEOP_RIGHT = containers.JointPositions(
    [-0.04, -1.16, 0.08, -1.57, 0.00, 2.05, 0.84]
)

SPEED_FACTOR = 0.2


def main() -> None:
    """
    Simulation for model-mediated teleoperation with the PARTI system.
    The sim connects to the robots to render haptical feedback.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument(
        "--deactivate-left", action="store_true", help="deactivate left robot"
    )
    parser.add_argument(
        "--deactivate-right", action="store_true", help="deactivate right robot"
    )
    parser.add_argument(
        "--use-ros",
        action="store_true",
        help="use rosbridge connection to update scene.",
    )
    parser.add_argument(
        "--ros-hostname", type=str, help="ROS hostname", default="localhost"
    )
    parser.add_argument("--ros-port", type=int, help="rosbridge port", default=9090)
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, force=True)
    if args.sim_only:
        left_hostname = None
        right_hostname = None
    else:
        left_hostname, right_hostname = utils.get_robot_hostnames()

    if args.deactivate_left:
        left_hostname = None
    if args.deactivate_right:
        right_hostname = None

    arena = composer.Arena(xml_path=XML_PATH)
    left_frame = arena.mjcf_model.find("site", "left")
    right_frame = arena.mjcf_model.find("site", "right")
    plane = arena.mjcf_model.find("joint", "plane")
    obj = [arena.mjcf_model.find("joint", jn) for jn in ["x", "y", "theta"]]

    left = params.RobotParams(
        robot_ip=left_hostname,
        name="left",
        joint_damping=[50, 0, 0, 0, 0, 0, 0],
        has_hand=False,
        joint_positions=Q_TELEOP_LEFT.positions,
        attach_site=left_frame,
        gripper=sim.make_endeffector("right"),
        actuation=arm_constants.Actuation.HAPTIC,
    )
    right = params.RobotParams(
        robot_ip=right_hostname,
        name="right",
        joint_damping=[50, 0, 0, 0, 0, 0, 0],
        has_hand=False,
        joint_positions=Q_TELEOP_RIGHT.positions,
        attach_site=right_frame,
        gripper=sim.make_gripper("left"),
        actuation=arm_constants.Actuation.HAPTIC,
    )
    robot_params = [left, right]

    scene_effector = sim.SceneEffector(plane, obj)

    env_builder = environment.PandaEnvironment(robot_params, arena, 0.016)
    env_builder.add_extra_effectors([scene_effector])

    add_object_obs = observation_transforms.AddObservation(
        "object",
        scene_effector.get_object_observations,
        specs.Array((3,), dtype=np.float32),
    )
    add_virtual_object_obs = observation_transforms.AddObservation(
        "virtual_object",
        scene_effector.get_virtual_object_observations,
        specs.Array((3,), dtype=np.float32),
    )
    add_plane_obs = observation_transforms.AddObservation(
        "plane",
        scene_effector.get_plane_observations,
        specs.Array((1,), dtype=np.float32),
    )
    add_virtual_plane_obs = observation_transforms.AddObservation(
        "virtual_plane",
        scene_effector.get_virtual_plane_observations,
        specs.Array((1,), dtype=np.float32),
    )
    add_updating_obs = observation_transforms.AddObservation(
        "updating",
        scene_effector.get_updating_observations,
        specs.Array((1,), dtype=np.float32),
    )

    agent = sim.TeleopAgent(
        arena,
        specs.Array((19,), dtype=np.float32),
        np.r_[Q_TELEOP_LEFT.positions, Q_TELEOP_RIGHT.positions],
        args.use_ros,
        args.ros_hostname,
        args.ros_port,
    )

    add_force_torque_obs = observation_transforms.AddObservation("force_torque", agent.get_force_torque_obs, specs.Array((6,), dtype=np.float64))

    env_builder.add_extra_sensors([sim.FollowerSensor(env_builder.robots["left"].arm)])
    env_builder.add_timestep_preprocessors(
        [
            add_force_torque_obs,
            add_object_obs,
            add_virtual_object_obs,
            add_plane_obs,
            add_virtual_plane_obs,
            add_updating_obs,
            observation_transforms.RetainObservations(
                [
                    "time",
                    "left_joint_pos",
                    "left_joint_vel",
                    "left_joint_torques",
                    "right_joint_pos",
                    "right_joint_vel",
                    "right_joint_torques",
                    "follower_joint_pos",
                    "follower_joint_vel",
                    "follower_joint_torques",
                    "follower_virtual_joint_torques",
                    "object",
                    "virtual_object",
                    "plane",
                    "virtual_plane",
                    "updating",
                    "force_torque"
                ]
            ),
        ]
    )

    with env_builder.build_task_environment() as env:
        dmr_panda_utils.full_spec(env)
        if not args.testing:
            app = dmr_panda_utils.ApplicationWithPlot()
            app.launch(env, policy=agent.step)
        else:
            run_loop.run(env, agent, [], 100)

    agent.shutdown()
