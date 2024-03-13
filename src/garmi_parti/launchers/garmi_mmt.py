from __future__ import annotations

import argparse
import logging

import numpy as np
from scipy.spatial import transform

from ..garmi import mmt
from ..teleoperation import containers, server, utils

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("garmi")


def main() -> None:
    """
    GARMI teleoperation demo.
    The GARMI system acts as a network server and teleoperation
    follower that accepts connections from teleoperation clients.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=13701)
    args = parser.parse_args()

    left, right = utils.get_robot_hostnames()
    left_params = containers.TeleopParams(
        left,
        transform.Rotation.from_euler(
            "XYZ", [0, 90 / 180 * np.pi, -90 / 180 * np.pi]
        ).inv(),
    )
    right_params = containers.TeleopParams(
        right,
        transform.Rotation.from_euler(
            "XYZ", [0, 90 / 180 * np.pi, 90 / 180 * np.pi]
        ).inv(),
    )
    srv = server.Server(mmt.Follower(left_params, right_params, False, True), args.port)
    server.user_interface(srv)
    srv.shutdown()
