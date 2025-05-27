import threading
import numpy as np
import time
import json
from argparse import ArgumentParser
from frankx import JointMotion, Robot, Gripper
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from user_data import UserData  
from threading import Event

# Synchronization
qpos_event = Event()
gripper_event = Event()
qpos_lock = threading.Lock()

# Shared variables
latest_qpos = None
latest_gripper_qpos = None
last_recv_time = None
recv_count = 0
last_target_qpos = None
last_gripper_width = None

MAX_WIDTH = 0.082566  # Max gripper opening

# DDS subscriber function
def subscribe_loop(subscriber):
    global latest_qpos, latest_gripper_qpos

    while True:
        msg = subscriber.Read()
        if msg is not None:
            try:
                payload = json.loads(msg.string_data)
                qpos = np.array(payload["qpos"])
                gripper_qpos = payload["gripper_qpos"]

                with qpos_lock:
                    latest_qpos = qpos
                    latest_gripper_qpos = gripper_qpos

                print(f"\n New message received @ {time.strftime('%H:%M:%S')}")
                print("    - qpos         :", np.array2string(qpos, precision=4, separator=', '))
                print("    - gripper_qpos :", gripper_qpos)

                qpos_event.set()      # Notify control loop
                gripper_event.set()   # Notify gripper control loop

            except Exception as e:
                print("Error parsing DDS message:", e)

# Control loop for arm movement
def control_loop(robot):
    print("[THREAD] Arm control loop started.")
    MOVEMENT_THRESHOLD = 4e-2  # rad ≈ 0.57°

    while True:
        qpos_event.wait()  # Wait for a new qpos command
        qpos_event.clear()

        with qpos_lock:
            target_qpos = latest_qpos

        if target_qpos is not None:
            try:
                global last_target_qpos

                if (
                    last_target_qpos is not None and
                    np.allclose(target_qpos, last_target_qpos, atol=MOVEMENT_THRESHOLD)
                ):
                    continue  # Skip small movements

                robot.move(JointMotion(target_qpos))
                last_target_qpos = target_qpos.copy()

            except Exception as e:
                print("Arm Control Error:", e)
                if "Reflex" in str(e):
                    print("Reflex mode triggered. Trying to recover...")
                    robot.recover_from_errors()

# Control loop for gripper movement
def gripper_loop(gripper):
    print("[THREAD] Gripper control loop started.")

    while True:
        gripper_event.wait()  # Wait for a new gripper command
        gripper_event.clear()

        with qpos_lock:
            target_gripper_qpos = latest_gripper_qpos

        if target_gripper_qpos is not None:
            try:
                gripper_width = min(abs(target_gripper_qpos[0]), MAX_WIDTH)

                global last_gripper_width
                if last_gripper_width is None or abs(gripper_width - last_gripper_width) > 1e-3:
                    gripper.move(gripper_width)
                    last_gripper_width = gripper_width
                    print(f"Gripper moved to {gripper_width:.4f}")

            except Exception as e:
                print("Gripper Control Error:", e)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", default="172.16.0.2")
    args = parser.parse_args()

    # Connect to the robot
    robot = Robot(args.host)
    robot.set_default_behavior()
    robot.recover_from_errors()
    robot.set_dynamic_rel(0.08)  # Adjust as needed

    gripper = Gripper(args.host)

    # DDS Initialization
    ChannelFactoryInitialize()
    sub = ChannelSubscriber("robot_joint_command", UserData)
    sub.Init()

    # Start subscription thread
    sub_thread = threading.Thread(target=subscribe_loop, args=(sub,))
    sub_thread.daemon = True
    sub_thread.start()

    # Start arm control thread
    arm_thread = threading.Thread(target=control_loop, args=(robot,))
    arm_thread.daemon = True
    arm_thread.start()

    # Start gripper control thread
    gripper_thread = threading.Thread(target=gripper_loop, args=(gripper,))
    gripper_thread.daemon = True
    gripper_thread.start()

    print("[MAIN] Listening for qpos from Robosuite via Unitree SDK DDS...")

    while True:
        time.sleep(1)  # Keep the main thread alive
