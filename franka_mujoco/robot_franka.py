import json
import os
import numpy as np
import time
import multiprocessing as mp
import chwytak_proces
from filtr import Filtr_CBF

# ustawienia ruchu muszą być takie jak w treningu policy (model nowy_B)
DT = 0.06                    # krok policy [s] (decimation 6)
VEL = 0.3                    # największa prędkość stawu [rad/s]
V_PALCOW = 0.06              # prędkość rozstawu palców przy move/grasp [m/s]
FILTR_CELU = 0.05            # wygładzanie celu w sterowniku (set_filter, co 1 ms; 1 = brak)

IP = "172.16.0.2"
HOME = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.785])
V_HOME = 0.3

Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
Q_MARGIN = 0.05

# chwytak działa w osobnym procesie (chwytak_proces.py), dane wymieniamy przez wspólną pamięć.
# proces startujemy przed importem panda_py (inaczej panda.get_state() się zawiesza)
_mp = mp.get_context("fork")
_cel_otwarty = _mp.Value("b", 1, lock=False)                 # czego chce policy
_jest_otwarty = _mp.Value("b", 0, lock=False)                # stan po ostatniej komendzie
_szerokosc = _mp.Value("d", -1.0, lock=False)                # [m], -1 = brak odczytu
_predkosc = _mp.Value("d", 0.0, lock=False)                  # [m/s] jednego palca
_mp.Process(target=chwytak_proces.petla, args=(IP, _cel_otwarty, _jest_otwarty, _szerokosc, _predkosc, V_PALCOW),
            daemon=True).start()

import panda_py
from panda_py import controllers
panda = panda_py.Panda(IP)
# progi kolizji jak w przykładach libfranka: od 20 Nm / 20 N robot staje
panda.get_robot().set_collision_behavior([10.0] * 7, [20.0] * 7, [10.0] * 6, [20.0] * 6)

# kamery na chwytaku: D435i z kalibracji hand-eye i druga po przeciwnej stronie (obrót o 180 st. wokół osi chwytaka)
T_TCP_CAM = np.array(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "camera", "kalibracja.json")))["T_tcp_cam"])
KAMERY_NA_CHWYTAKU = [T_TCP_CAM, np.diag([-1.0, -1.0, 1.0, 1.0]) @ T_TCP_CAM]
filtr = Filtr_CBF(stol_z = 0.0, h_min = 0.02, alfa = 0.5, t_br = 0.18, kamery = KAMERY_NA_CHWYTAKU, vel = VEL, dt = DT)

ctrl = None
q_command = None
next_step = None
ostatni_blad = None


def chwytak_otwarty():
    return bool(_jest_otwarty.value)


def reset_scene():
    global ctrl

    _cel_otwarty.value = 1
    t0 = time.perf_counter()
    while not chwytak_otwarty():
        if time.perf_counter() - t0 > 5.0:
            raise RuntimeError("chwytak sie nie otworzyl")
        time.sleep(0.01)

    if ctrl is not None:
        panda.stop_controller()
        ctrl = None
    panda.recover()
    q0 = np.array(panda.get_state().q)
    dq = HOME - q0
    T = max(1.875 * np.max(np.abs(dq)) / V_HOME, 1.0)
    move = controllers.JointPosition()
    panda.start_controller(move)

    with panda.create_context(frequency=1000, max_runtime=T + 0.5) as ctx:
        while ctx.ok():
            s = min(move.get_time() / T, 1.0)
            move.set_control(q0 + dq * (10 * s**3 - 15 * s**4 + 6 * s**5), np.zeros(7))
    panda.stop_controller()


def state():
    t0 = time.perf_counter()
    while _szerokosc.value < 0:                              # pierwszy odczyt chwytaka
        if time.perf_counter() - t0 > 3.0:
            raise RuntimeError("brak odczytu szerokosci chwytaka")
        time.sleep(0.01)
    q_state = panda.get_state()
    q_pos = np.array(q_state.q)
    q_vel = np.array(q_state.dq)
    tcp_pos = np.array(panda.get_position()).flatten()
    tcp_quat = np.array(panda.get_orientation()).flatten()

    return {
        "q": q_pos,
        "q_vel": q_vel,
        "fingers": _szerokosc.value/2,
        "fingers_vel": _predkosc.value,
        "tcp_pos": tcp_pos,
        "tcp_quat": tcp_quat
    }


def movement(q_goal, gripper_open):

    global ctrl,q_command, next_step

    _cel_otwarty.value = int(bool(gripper_open))

    if ctrl is None:
        ctrl = controllers.JointPosition()
        ctrl.set_filter(FILTR_CELU)
        panda.start_controller(ctrl)
        q_command = np.array(panda.get_state().q)
        next_step = time.perf_counter()

    step = np.clip(q_goal- q_command, -VEL*DT, VEL*DT)

    state = panda.get_state()
    q_real = np.array(state.q)
    dq_real = np.array(state.dq)
    step = filtr.filtruj(step, q_real, dq_real, _szerokosc.value/2)
    q_command = np.clip(q_command + step, Q_MIN + Q_MARGIN, Q_MAX - Q_MARGIN)
    q_command = np.clip(q_command, q_real - 0.1, q_real + 0.1)
    ctrl.set_control(q_command, np.zeros(7))

    next_step = next_step + DT
    wait = next_step - time.perf_counter()
    if wait >0:
        time.sleep(wait)
    elif wait <-DT:
        next_step = time.perf_counter()
    return robot_ok()


def robot_stop():
    global ctrl
    if ctrl is not None:
        panda.stop_controller()
        ctrl = None


def robot_ok():
    """False, gdy robot stanął (błąd, odruch, stop). panda_py sam nie zgłasza błędów sterownika."""
    global ostatni_blad
    if ctrl is None:
        return True
    try:
        panda.raise_error()
    except RuntimeError as e:
        ostatni_blad = str(e)
        return False
    if not panda.is_moving():
        ostatni_blad = "sterownik przestał działać"
        return False
    return True
