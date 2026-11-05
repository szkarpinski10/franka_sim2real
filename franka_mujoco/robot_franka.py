import json
import os
import numpy as np
import time
import multiprocessing as mp
import chwytak_proces
from observations_combine import quat_to_matrix
from filtr import Filtr_CBF

# zwolnione tempo: krok policy co 0.02*TEMPO s przy tym samym ruchu na krok (0.016 rad), więc robot jedzie TEMPO razy
# wolniej; run_robot.py mnoży prędkości w obserwacji przez TEMPO, żeby wyglądały jak w treningu. MuJoCo (40 układów):
# TEMPO 1: 31/40 wież, stawy 99% < 0.92 rad/s i 15 rad/s2; TEMPO 2: 28/40, 0.45 rad/s, 7.9 rad/s2; TEMPO 3: 28/40.
# 5.11 na robocie pełne tempo (1) szarpało i dało odruch cartesian_reflex. TEMPO = 1 -> jak w treningu.
TEMPO = 3                                                    # ok. prędkość z pierwszej próby (film 5.11), ale równo
_VEL = 0.8 / TEMPO
DT = 0.02 * TEMPO

IP = "172.16.0.2"
# chwytak w osobnym procesie (opis w chwytak_proces.py); start PRZED połączeniem z robotem, gdy nie ma jeszcze wątków
_mp = mp.get_context("fork")
_cel_otwarty = _mp.Value("b", 1, lock=False)                 # czego chce policy
_jest_otwarty = _mp.Value("b", 0, lock=False)                # stan po ostatniej komendzie (na starcie: nieznany -> otwórz)
_szerokosc = _mp.Value("d", -1.0, lock=False)                # [m], -1 = jeszcze brak odczytu
_predkosc = _mp.Value("d", 0.0, lock=False)                  # [m/s] jednego palca
_mp.Process(target=chwytak_proces.petla, args=(IP, _cel_otwarty, _jest_otwarty, _szerokosc, _predkosc, 0.1 / TEMPO),
            daemon=True).start()
# panda_py dopiero PO starcie procesu chwytaka – import panda_py przed fork zawiesza potem panda.get_state()
import panda_py
from panda_py import controllers
panda = panda_py.Panda(IP)
HOME = np.array([0.0,-0.3,0.0,-2.0,0.0,1.7,0.785])
V_HOME = 0.3

Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
Q_MARGIN = 0.05


def chwytak_otwarty():
    """Czy chwytak jest otwarty (po ostatniej wykonanej komendzie)."""
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
    # szerokość chwytaka z procesu chwytaka (wspólna pamięć) – pętla robota nigdy na chwytak nie czeka
    t0 = time.perf_counter()
    while _szerokosc.value < 0:                              # tylko na samym początku
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

    
ctrl = None
q_command = None
next_step = None

# kamery na chwytaku (nie ma ich w modelu MuJoCo): D435i z kalibracji hand-eye, druga kamera lustrzanie po drugiej
# stronie (obrót 180 st. wokół osi chwytaka) – przy przechyle chwytaka kamera potrafi zejść niżej niż palce
T_TCP_CAM = np.array(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "camera", "kalibracja.json")))["T_tcp_cam"])
KAMERY_NA_CHWYTAKU = [T_TCP_CAM, np.diag([-1.0, -1.0, 1.0, 1.0]) @ T_TCP_CAM]
filtr = Filtr_CBF(stol_z = 0.0, h_min = 0.015, alfa = 0.5, t_br = 0.18,   # nowy stół (5.11): blat na wysokości bazy
                  kamery = KAMERY_NA_CHWYTAKU, vel = _VEL, dt = DT)

def movement(q_goal, gripper_open):
    
    global ctrl,q_command, next_step

    _cel_otwarty.value = int(bool(gripper_open))

    if ctrl is None:
        ctrl = controllers.JointPosition()
        panda.start_controller(ctrl)
        q_command = np.array(panda.get_state().q)
        next_step = time.perf_counter()

    step = np.clip(q_goal- q_command, -_VEL*DT,_VEL*DT)

    
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
        

ostatni_blad = None      # opis ostatniego błędu robota (do wypisania w run_robot.py)


def robot_ok():
    """False, gdy robot stanął z błędem / odruchem / stopem. 5.11: po "cartesian_reflex" is_moving() dalej dawało
    True i run_robot jechał dalej – panda_py nie przekazuje błędów sterownika sam, trzeba spytać raise_error()."""
    global ostatni_blad
    if ctrl is None:
        return True
    try:
        panda.raise_error()
    except RuntimeError as e:
        ostatni_blad = str(e)
        return False
    if not panda.is_moving():
        ostatni_blad = "sterownik przestał działać (is_moving = False)"
        return False
    return True