from __future__ import annotations

import argparse
from unittest import mock

import numpy as np
import pytest

# Set the seed for reproducibility (optional)
rng = np.random.default_rng(42)


@pytest.fixture(autouse=True)
def _mock_argparse():
    with mock.patch(
        "argparse.ArgumentParser.parse_args",
        return_value=argparse.Namespace(
            port=rng.integers(low=13701, high=16701),
            debug=False,
            gamepad_port=13702,
            joint_mode=False,
            host="127.0.0.1",
            mode="joint",
            sim_only=True,
            testing=True,
            deactivate_left=False,
            deactivate_right=False,
            use_ros=False,
            ros_hostname="localhost",
            ros_port=9090,
        ),
    ):
        yield


@pytest.fixture(autouse=True)
def _mock_panda():
    with mock.patch("panda_py.Panda") as patch_panda:
        patch_panda.return_value.get_orientation.return_value = np.array([0, 0, 0, 1])
        patch_panda.return_value.get_position.return_value = np.zeros(3)
        patch_panda.return_value.get_state.return_value = mock.MagicMock(
            O_F_ext_hat_K=np.zeros(6),
            q=np.zeros(7),
            dq=np.zeros(7),
            tau_ext_hat_filtered=np.zeros(7),
        )
        yield


@pytest.fixture(autouse=True)
def _mock_gripper():
    with mock.patch("panda_py.libfranka.Gripper"):
        yield


@pytest.fixture(autouse=True)
def _mock_environ():
    with mock.patch.dict(
        "os.environ",
        {
            "PANDA_LEFT": "garmi-left",
            "PANDA_RIGHT": "garmi-right",
            "PANDA": "panda",
        },
    ):
        yield
