"""test_kamery_sim.py – walidacja fuzji kamer w MuJoCo (bez sprzętu).

Policy nie dostaje prawdziwych pozycji kostek, tylko wynik fuzja.py, karmionej udawanymi kamerami:
pomiar co 0.1 s (10 kl/s), opóźniony o 0.1 s, z szumem; kamera na chwytaku widzi tylko, gdy robot jedzie wolno
i kostka jest pod nią; statyw czasem jest zasłonięty (ramię). Wynik porównujemy z robot_p1_p2.py (prawdziwe pozycje).
Uruchom:  python test_kamery_sim.py
"""
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "camera"))
import robot_mujoco as robot
from filtr import Filtr_CBF
from fuzja import Fuzja, poza, trzyma as trzyma_kostke
from observations_combine import obs_combine

SCALE = 0.03
N_STEPS = 1000
POLICY_PATH = "policy1/2026-11-01_16-15-43_800/exported/policy.pt"
CO_ILE = 5                                                   # kroków między klatkami (5 x 20 ms = 10 kl/s)
OPOZNIENIE = 5                                               # kroków opóźnienia obrazu (100 ms)
OPOZNIENIE_ZALOZONE = 5                                      # tyle zakłada kamery.py (na robocie trzeba je zmierzyć)
SZUM_POS, SZUM_YAW = 0.003, np.radians(2)
STATYW_WIDZI = 0.85                                          # szansa, że ramię nie zasłania kostki
V_MAX_REKA = 0.05

policy = torch.jit.load(POLICY_PATH, map_location="cpu").eval()
filtr = Filtr_CBF(h_min=0.006, alfa=0.5, t_br=0.12)
rng = np.random.default_rng(0)
TABLE = robot.model.geom("table").id
ROBOT = ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger")


# te trzy funkcje jak w robot_p1_p2.py (tamten plik przy imporcie otwiera okno MuJoCo)
def cube_2_on_cube_1():
    cubes = robot.cubes_position_table()
    cube_1, cube_2 = cubes["cube_1"][0], cubes["cube_2"][0]
    return np.linalg.norm(cube_2[:2] - cube_1[:2]) < 0.03 and 0.04 < cube_2[2] - cube_1[2] < 0.06


def rand_cubes_on_table(n, seed=0, min_sep=0.1):
    rng_uklad = np.random.default_rng(seed)
    cases = []
    while len(cases) < n:
        x = rng_uklad.uniform(0.35, 0.55, 3)
        y = rng_uklad.uniform(-0.20, 0.20, 3)
        if min(np.hypot(x[0] - x[1], y[0] - y[1]), np.hypot(x[1] - x[2], y[1] - y[2]),
               np.hypot(x[0] - x[2], y[0] - y[2])) < min_sep:
            continue
        yaw = rng_uklad.uniform(-np.pi, np.pi, 3)
        cases.append({f"cube_{i + 1}": [x[i], y[i], 0.025, yaw[i]] for i in range(3)})
    return cases


def table_contact_def():
    for c in robot.data.contact[:robot.data.ncon]:
        cialo1 = robot.model.body(robot.model.geom_bodyid[c.geom1]).name
        cialo2 = robot.model.body(robot.model.geom_bodyid[c.geom2]).name
        if (c.geom1 == TABLE and cialo2 in ROBOT) or (c.geom2 == TABLE and cialo1 in ROBOT):
            return True
    return False


def prawdziwe():
    """Prawdziwe pozycje i obroty kostek (jak policzyłaby je kamera: obrót wokół pionu co 90 st.)."""
    out = {}
    for k in ("cube_1", "cube_2", "cube_3"):
        R = robot.data.body(k).xmat.reshape(3, 3)
        os_pozioma = R[:, np.argmin(np.abs(R[2, :]))]
        out[k] = (robot.data.body(k).xpos.copy(), np.arctan2(os_pozioma[1], os_pozioma[0]) % (np.pi / 2))
    return out


def epizod():
    fuzja = Fuzja()
    historia = []                                            # (kostki, poza TCP) z każdego kroku
    last_action = np.zeros(8, dtype=np.float32)
    gripper_open, kontakt = True, False
    for krok in range(N_STEPS):
        t = krok * robot.DT
        s = robot.state()
        T_tcp = poza(s["tcp_pos"], s["tcp_quat"])
        historia.append((prawdziwe(), T_tcp))
        trzyma = trzyma_kostke(not gripper_open, s["fingers"], s["fingers_vel"])

        if krok % CO_ILE == 0:                               # obie kamery dają klatkę sprzed OPOZNIENIE kroków
            kostki_wtedy, T_wtedy = historia[max(0, krok - OPOZNIENIE)]
            T_zalozona = historia[max(0, krok - OPOZNIENIE_ZALOZONE)][1]   # poza TCP z chwili zdjęcia
            T_przed = historia[max(0, krok - OPOZNIENIE - CO_ILE)][1]
            v_tcp = np.linalg.norm(T_wtedy[:3, 3] - T_przed[:3, 3]) / (CO_ILE * robot.DT)
            for k, (pos, yaw) in kostki_wtedy.items():
                pomiar = (pos + rng.normal(0, SZUM_POS, 3), yaw + rng.normal(0, SZUM_YAW))
                w_chwytaku = k == "cube_2" and trzyma
                pod_kamera = np.linalg.norm(pos[:2] - T_wtedy[:2, 3]) < 0.2 and not w_chwytaku
                if v_tcp < V_MAX_REKA and pod_kamera:
                    fuzja.dodaj("reka", k, *pomiar, t, T_zalozona)
                if rng.random() < (0.6 if w_chwytaku else STATYW_WIDZI):
                    fuzja.dodaj("statyw", k, *pomiar, t, T_zalozona)

        kostki = fuzja.kostki(T_tcp, trzyma, t)
        if kostki is None:                                   # na robocie czekamy w czekaj_na_kostki
            kostki = robot.cubes_position_table()
        obs = obs_combine(s["q"], s["q_vel"], s["fingers"], s["fingers_vel"], last_action, s["tcp_pos"],
                          s["tcp_quat"], kostki)
        with torch.no_grad():
            action = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
        q_goal = s["q"] + SCALE * action[:7]
        gripper_open = action[7] >= 0
        step = np.clip(q_goal - robot.data.ctrl[:7], -robot._VEL * robot.DT, robot._VEL * robot.DT)
        step = filtr.filtruj(step, s["q"], s["q_vel"], s["fingers"])
        robot.movement(robot.data.ctrl[:7] + step, gripper_open)
        kontakt = kontakt or table_contact_def()
        last_action = action.copy()
    return cube_2_on_cube_1(), kontakt, fuzja.n_niezgodnych


if __name__ == "__main__":
    SEED, N = 7, int(sys.argv[1]) if len(sys.argv) > 1 else 100
    n_sukces = n_kontakt = 0
    for i, case in enumerate(rand_cubes_on_table(N, seed=SEED)):
        robot.reset_scene(case)
        sukces, kontakt, niezgodne = epizod()
        n_sukces += sukces
        n_kontakt += kontakt
        print(f"Case {i}: {'sukces' if sukces else 'porazka'}, kontakt ze stolem {'tak' if kontakt else 'nie'}, "
              f"niezgodnosci kamer {niezgodne}", flush=True)
    print(f"\nsukcesow: {n_sukces}/{N}, kontaktow ze stolem: {n_kontakt}/{N}")
