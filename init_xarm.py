from xarm.wrapper import XArmAPI
import time

def init_robot():
    arm = XArmAPI("192.168.1.209")
    # arm = XArmAPI(ip)
    arm.motion_enable(enable=True) # Enables motion. Set True so the robot motors don't turn on and off all the time
    print("Arm's current position: ", arm.get_position()) # Curent position of the robot
    # print(arm.get_inverse_kinematics(pose)) # Get the joint angles from teach pendant pose
    print("joint speed limit: ", arm.joint_speed_limit) # minimum is 0.057 deg and max is 180 deg. Start with 10 deg per sec
    arm.set_mode(0)
    arm.set_state(state=0)

    # Test gripper
    arm.set_gripper_position(300)
    time.sleep(1)

    speed = 50  # mm/s
    acc = 60

    # initial position
    init_pos = [206, 0, 110]  # f
    arm.set_position(x=init_pos[0], y=init_pos[1], z=init_pos[2], roll=179.9, pitch=0, yaw=0, speed=speed, mvacc=acc, wait=True)
    return arm

init_robot()