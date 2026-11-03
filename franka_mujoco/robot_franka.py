import numpy as np
import panda_py
from panda_py import libfranka, controllers
import time
import threading
from observations_combine import quat_to_matrix
from filtr import Filtr_CBF

_VEL = 0.8 
DT = 0.02

IP = "172.16.0.2"
panda = panda_py.Panda(IP)
gripper = libfranka.Gripper(IP)         
HOME = np.array([0.0,-0.3,0.0,-2.0,0.0,1.7,0.785])
V_HOME = 0.3

Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
Q_MARGIN = 0.05


finger_time = None
finger_width = None

def reset_scene():
    global finger_time, finger_width, ctrl, gripper_policy_open

    gripper_policy_open = True
    while not gripper_is_open:
        time.sleep(0.01)
    
    if ctrl is not None:
        panda.stop_controller()
        ctrl = None
    panda.recover()
    panda.move_to_joint_position(HOME,speed_factor = 0.2)
    finger_width = None
    finger_time = None


def state():
    
    global finger_time, finger_width
    q_state = panda.get_state()
    q_pos = np.array(q_state.q)
    q_vel = np.array(q_state.dq)
    tcp_pos = np.array(panda.get_position()).flatten()
    tcp_quat = np.array(panda.get_orientation()).flatten()
    gripper_width = gripper.read_once().width
    fingers = gripper_width/2
    now = time.perf_counter()
    fingers_vel = 0.0

    if finger_width is not None:
        fingers_vel = (gripper_width - finger_width)/2/(now - finger_time)
        
    finger_width = gripper_width
    finger_time = now

    return {
        "q": q_pos,
        "q_vel": q_vel,
        "fingers": fingers,
        "fingers_vel": fingers_vel,
        "tcp_pos": tcp_pos,
        "tcp_quat": tcp_quat
    }

    
ctrl = None
q_command = None
next_step = None

filtr = Filtr_CBF(stol_z = 0.006, h_min = 0.015, alfa = 0.5, t_br = 0.18)

def movement(q_goal, gripper_open):
    
    global ctrl,q_command, next_step, gripper_policy_open

    gripper_policy_open = bool(gripper_open)

    if ctrl is None:
        ctrl = controllers.JointPosition()
        panda.start_controller(ctrl)
        q_command = np.array(panda.get_state().q)
        next_step = time.perf_counter()

    step = np.clip(q_goal- q_command, -_VEL*DT,_VEL*DT)

    
    state = panda.get_state()
    q_real = np.array(state.q)
    dq_real = np.array(state.dq)
    step = filtr.filtruj(step, q_real, dq_real, finger_width/2)
    q_command = np.clip(q_command + step, Q_MIN + Q_MARGIN, Q_MAX - Q_MARGIN)
    ctrl.set_control(q_command, np.zeros(7))

    next_step = next_step + DT
    wait = next_step - time.perf_counter()
    if wait >0:
        time.sleep(wait)
    elif wait <-DT:
        next_step = time.perf_counter()
        

gripper_policy_open = True
gripper_is_open = False

def gripper_action():
    global gripper_is_open

    while True:
        if gripper_policy_open != gripper_is_open:
            target = gripper_policy_open
            if target:
                gripper.move(0.08,0.1)
            else:
                gripper.grasp(0.0,0.1, 20.0,0.08,0.08)
            gripper_is_open = target
        
        time.sleep(0.005)

threading.Thread(target = gripper_action, daemon = True).start()

def robot_stop():
    global ctrl 
    if ctrl is not None:
        panda.stop_controller()
        ctrl = None
        