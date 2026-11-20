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
KOSTKA = 0.05                # szerokość kostki [m] - przy zamykaniu szacowana szerokość nie schodzi niżej
# pomiar 19.11 (sam chwytak): move 8 -> 5 cm trwa ok. 1.0 s zamiast 0.5 s, pusty grasp 1.85 s zamiast 1.33 s ->
# każde polecenie ma ok. 0.5 s narzutu (rozruch/hamowanie), prędkość palców zgodna (0.062 m/s)
# Samo opóźnienie startu w szacunku (palce stoją 0.5 s) nie działa: policy uznaje, że chwytak nie reaguje, otwiera
# i odjeżdża (log p1_20261119_122557). Dlatego szacunek rusza od razu, ale wolniej: 8 -> 5 cm w ok. 1 s jak naprawdę.
SKALA_PALCOW = 1.0           # szacowana prędkość palców = V_PALCOW * SKALA_PALCOW
# prawdziwy chwytak dostaje większą prędkość niż w treningu (max Franka Hand 0.1 m/s): z narzutem ok. 0.5 s palce
# dochodzą do kostki po ok. 0.8 s zamiast 1.0 s, a policy 3250 podnosi po 0.6-0.84 s (19.11). Szacunek dla policy
# zostaje z V_PALCOW (jak w treningu).
V_CHWYTAKA = 0.06            # [m/s] prędkość rozstawu palców wysyłana do chwytaka
FILTR_CELU = 0.05            # wygładzanie celu w sterowniku (set_filter, co 1 ms; 1 = brak)
KROKI_ZAMKNIJ = 2            # tyle kroków z rzędu "zamknij", zanim ruszy grasp()
_ile_zamknij = 0

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
_ruch = _mp.Array("d", 4, lock=False)                        # trwa move/grasp: [trwa, t0, szerokość na starcie, cel]
_mp.Process(target=chwytak_proces.petla, args=(IP, _cel_otwarty, _jest_otwarty, _szerokosc, _predkosc, V_CHWYTAKA,
                                               _ruch),
            daemon=True).start()

import panda_py
from panda_py import controllers
panda = panda_py.Panda(IP)
# progi kolizji: od PROG_KOLIZJI Nm / N robot staje (odruch). 20 (jak w przykładach libfranka) zatrzymywało robota przy
# każdym lekkim trąceniu kostki palcem (12.11), a policy tak robi także w MuJoCo przed chwytem. Stół chroni filtr.
PROG_KOLIZJI = 50.0
panda.get_robot().set_collision_behavior([10.0] * 7, [PROG_KOLIZJI] * 7, [10.0] * 6, [PROG_KOLIZJI] * 6)

# kamery na chwytaku: D435i z kalibracji hand-eye i druga po przeciwnej stronie (obrót o 180 st. wokół osi chwytaka)
T_TCP_CAM = np.array(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "camera", "kalibracja.json")))["T_tcp_cam"])
KAMERY_NA_CHWYTAKU = [T_TCP_CAM, np.diag([-1.0, -1.0, 1.0, 1.0]) @ T_TCP_CAM]
# 20.11: kamery wyłączone z filtra (decyzja Szymona). W logach 12.11/19.11 bryła RealSense przy pochylonym chwytaku
# schodziła do 2.1-2.9 cm, czyli do podłogi filtra (2.2 cm) - filtr hamował zjazd i palce zamykały się wyżej.
# Uwaga: prawdziwa kamera dalej jest na chwytaku - bez filtra może dotknąć stołu; najlepiej zdjąć ją do jazdy policy.
KAMERY_W_FILTRZE = False
filtr = Filtr_CBF(stol_z = 0.0, h_min = 0.02, alfa = 0.5, t_br = 0.18,
                  kamery = KAMERY_NA_CHWYTAKU if KAMERY_W_FILTRZE else (), vel = VEL, dt = DT)

ctrl = None
q_command = None
next_step = None
ostatni_blad = None


def chwytak_otwarty():
    return bool(_jest_otwarty.value)


def chwytak_zamkniety():
    """Palce zamknięte albo właśnie się zamykają. _jest_otwarty zmienia się dopiero, gdy grasp() wróci (ok. 1 s po
    zatrzymaniu się palców na kostce) - 12.11 przez ten czas robot uznawał, że nic nie trzyma, policy podniosła kostkę,
    nie zobaczyła jej w chwytaku i ją puściła."""
    zamykanie = _ruch[0] > 0.5 and _ruch[3] < _ruch[2]
    return zamykanie or not _jest_otwarty.value


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


def palce():
    """(szerokość chwytaka [m], prędkość jednego palca [m/s]). W trakcie move/grasp proces chwytaka nie może czytać
    szerokości (12.11: policy widziała 8 cm, a palce się zamykały i przewróciły kostkę) - wtedy liczymy ją z prędkości
    ruchu. Przy zamykaniu nie niżej niż KOSTKA: grasp wraca dopiero ok. 1 s po zatrzymaniu się palców na kostce
    (12.11: szacunek zszedł do 0 cm, policy uznała chwyt za pusty i puściła podniesioną kostkę); pusty chwyt i tak
    da prawdziwy odczyt (0 cm), gdy palce zamkną się do końca."""
    if _ruch[0] > 0.5:
        t0, w0, cel = _ruch[1], _ruch[2], _ruch[3]
        kier = 1.0 if cel > w0 else -1.0
        v = V_PALCOW * SKALA_PALCOW                          # wolniej niż zadane: tyle naprawdę trwa cały ruch
        w = w0 + kier * v * (time.perf_counter() - t0)
        if kier < 0 and w <= max(cel, min(KOSTKA, w0)):      # zamykanie: palce stoją na kostce (albo ją mijają)
            return max(cel, min(KOSTKA, w0)), 0.0
        if (w - cel) * kier >= 0:                            # już dojechał
            return cel, 0.0
        return w, kier * v / 2
    return _szerokosc.value, _predkosc.value


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
    szer, v_palca = palce()

    return {
        "q": q_pos,
        "q_vel": q_vel,
        "fingers": szer/2,
        "fingers_vel": v_palca,
        "tcp_pos": tcp_pos,
        "tcp_quat": tcp_quat
    }


def movement(q_goal, gripper_open):

    global ctrl,q_command, next_step

    # zamykamy dopiero, gdy policy chce zamknąć KROKI_ZAMKNIJ kroków z rzędu (otwieramy od razu).
    # W symulacji krótkie "zamknij" prawie nie rusza palców, a na robocie grasp() jest nieprzerywalny
    # (12.11, logi 141226 i 142110: "zamknij" przez 1 krok na 9-10 cm -> palce zwęziły się nad kostką i stanęły na niej).
    global _ile_zamknij
    _ile_zamknij = 0 if gripper_open else _ile_zamknij + 1
    if gripper_open or _ile_zamknij >= KROKI_ZAMKNIJ:
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
    step = filtr.filtruj(step, q_real, dq_real, palce()[0]/2)
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


def ma_blad():
    """Czy robot ma aktywny błąd (np. po odruchu) – wtedy przed ruchem trzeba recover() (robi to reset_scene)."""
    try:
        panda.raise_error()
    except RuntimeError:
        return True
    return False


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
