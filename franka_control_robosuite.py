"""
A script to collect a batch of human demonstrations.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import json
import os
import shutil
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from user_data import UserData 
import threading

# DDS Setup
ChannelFactoryInitialize()
pub = ChannelPublisher("robot_joint_command", UserData)
pub.Init()

def unwrap_env(env):
    while hasattr(env, 'env'):
        env = env.env
    return env

def collect_human_trajectory(env, device, arm, env_configuration):
    env.reset()
    env.render()
    task_completion_hold_count = -1

    while True:
        active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]

        # Get action and grasp only
        action, grasp = input2action(
            device=device, robot=active_robot, active_arm=arm, env_configuration=env_configuration
        )

        if action is None:
            break

        start_time = time.time()
        env.step(action)
        env.render()
        step_time = time.time() - start_time

        obs = env._get_observations()
        qpos = active_robot.sim.data.qpos[active_robot._ref_joint_pos_indexes]
        gripper_left_joint, gripper_right_joint = obs.get("robot0_gripper_qpos")
        gripper_opening_size = abs(gripper_left_joint - gripper_right_joint)

        msg = UserData(
            string_data=json.dumps({
                "qpos": qpos.tolist(),
                "gripper_qpos": [gripper_opening_size]
            }),
            float_data=time.time()
        )

        pub.Write(msg, 0.1)

        # Check for success hold count
        if task_completion_hold_count == 0:
            break

        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 10
        else:
            task_completion_hold_count = -1

    env.close()

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations"),
    )
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
    parser.add_argument(
        "--controller", type=str, default="OSC_POSE", help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'"
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    parser.add_argument("--num-demos", type=int, default=1, help="Number of demonstrations to collect")

    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True, #True
        has_offscreen_renderer=True,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        control_freq=20, 
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    config["render_camera"] = args.camera  
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity, env=env)
        print(f"[DEBUG] env attached to keyboard: {device.env is not None}")
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    device.start_control()

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    # collect demonstrations
    # Wrap original env just once (for rendering)
    wrapped_env = VisualizationWrapper(env)

    for i in range(args.num_demos):
        tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
        collecting_env = DataCollectionWrapper(wrapped_env, tmp_directory)

        if args.device == "keyboard":
            from robosuite.devices import Keyboard
            device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity, env=collecting_env)
        elif args.device == "spacemouse":
            from robosuite.devices import SpaceMouse
            device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        else:
            raise Exception("Invalid device choice.")

        device.start_control()

        print(f"\nCollecting demonstration {i+1}/{args.num_demos}")
        collect_human_trajectory(collecting_env, device, args.arm, args.config)

        print(f"[INFO] Finished demo {i+1}. npz should be in {tmp_directory}")
        print(f"[INFO] Saving hdf5 to {new_dir}/demo.hdf5")

        collecting_env.close()


