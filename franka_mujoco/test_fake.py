"""TEST robot_franka.py W DOMU na udawanej France (fake_franka.py).

Warunek: w robot_franka.py jest FAKE = True.
Robi to samo co robot_p1_p2.py (ta sama pętla z polityką), tylko:
  - robot = robot_franka (a nie robot_mujoco),
  - czy wieża stoi, sprawdzamy na PRAWDZIWYCH pozycjach kostek z fizyki (na robocie tego nie ma - tam patrzysz oczami),
  - mierzymy, czy pętla trzyma 50 Hz.
Każda próba trwa naprawdę 20 s (czas rzeczywisty, jak na robocie) + jazda do home.
"""
import time
import numpy as np
import torch

import robot_franka as robot
import fake_franka
from observations_combine import obs_combine, quat_to_matrix

SCALE = 0.03
N_STEPS = 1000
N_PROB = 10                                 # ile prób (każda ~25 s)
PRAWDZIWE_KOSTKI = False                     # True = polityka widzi PRAWDZIWE pozycje kostek z fizyki (jak w robot_mujoco)
                                             # False = widzi to, co liczy robot_franka (stałe pozycje + kostka w chwytaku = TCP)

policy = torch.jit.load("nowe_policy/2026-10-26_18-47-17_1300/exported/policy.pt", map_location="cpu").eval()


def wieza_stoi():
    c = fake_franka.true_cubes()
    xy = np.linalg.norm(c["cube_2"][:2] - c["cube_1"][:2])
    z = c["cube_2"][2] - c["cube_1"][2]
    return xy < 0.03 and 0.04 < z < 0.06, xy, z


for proba in range(N_PROB):
    robot.reset_scene()
    last_action = np.zeros(8, dtype=np.float32)
    czasy = []
    zamkniec = 0
    off_hand, yaw_deg, yaw_est = [], [], []                   # pomiar: gdzie NAPRAWDĘ jest czerwona kostka, gdy jest w chwytaku
    t_prev = time.perf_counter()
    for step in range(N_STEPS):
        s = robot.state()
        cubes = fake_franka.true_cubes_pose() if PRAWDZIWE_KOSTKI else robot.cubes_position_table(s)

        # kostka w chwytaku (ten sam warunek co w robot_franka.cubes_position_table)
        w = s["fingers"] * 2
        if not robot.gripper_policy_open and 0.04 < w < 0.06:
            pos2, quat2 = fake_franka.true_cubes_pose()["cube_2"]
            R_hand = quat_to_matrix(s["tcp_quat"])
            off_hand.append(R_hand.T @ (pos2 - s["tcp_pos"]))        # przesunięcie kostki względem TCP, w układzie RĘKI
            x, y, z, ww = quat2
            yaw_deg.append(np.degrees(2 * np.arctan2(z, ww)))      # obrót kostki wokół pionu
            ex, ey, ez, ew = robot.cubes_position_table(s)["cube_2"][1]
            yaw_est.append(np.degrees(2 * np.arctan2(ez, ew)))     # obrót, który poda robot_franka
        obs = obs_combine(s["q"], s["q_vel"], s["fingers"], s["fingers_vel"],
                          last_action, s["tcp_pos"], s["tcp_quat"], cubes)
        with torch.no_grad():
            action = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
        if last_action[7] >= 0 and action[7] < 0:
            zamkniec += 1
        if not robot.movement(s["q"] + SCALE * action[:7], action[7] >= 0):
            print(f"   BŁĄD ROBOTA w kroku {step} - przerywam próbę")
            break
        last_action = action.copy()
        t_now = time.perf_counter()
        czasy.append(t_now - t_prev)
        t_prev = t_now
    robot.stop()

    ok, xy, z = wieza_stoi()
    czasy = np.array(czasy[1:]) * 1000
    print(f"próba {proba}: {'STOI' if ok else 'NIE'}   xy = {xy*100:4.1f} cm  z = {z*100:4.1f} cm   "
          f"| krok pętli: średnio {czasy.mean():5.1f} ms, najdłuższy {czasy.max():5.1f} ms (cel 20 ms)   "
          f"| komend 'zamknij': {zamkniec}")
    if off_hand:
        o = np.mean(off_hand, axis=0) * 100
        print(f"      kostka w chwytaku ({len(off_hand)} kroków): przesunięcie od TCP w układzie ręki "
              f"[{o[0]:5.2f}, {o[1]:5.2f}, {o[2]:5.2f}] cm,  obrót kostki: prawdziwy {np.mean(yaw_deg):6.1f} st., "
              f"z robot_franka {np.mean(yaw_est):6.1f} st.")