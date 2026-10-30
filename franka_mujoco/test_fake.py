"""TEST robot_franka.py W DOMU na udawanej France (fake_franka.py).

Warunek: w robot_franka.py jest FAKE = True.
Robi to samo co robot_p1_p2.py (ta sama pętla z polityką), tylko:
  - robot = robot_franka (a nie robot_mujoco),
  - czy wieża stoi, sprawdzamy na PRAWDZIWYCH pozycjach kostek z fizyki (na robocie tego nie ma - tam patrzysz oczami),
  - mierzymy, czy pętla trzyma 50 Hz.
Każda próba trwa naprawdę 20 s (czas rzeczywisty, jak na robocie) + jazda do home.

Skąd polityka ma pozycje kostek (KOSTKI):
  "kamera"     - jak będzie na robocie z kamerą: JEDEN pomiar zaraz po reset_scene (robot w HOME) udawaną kamerą
                 (fake_franka.zmierz_kostki: prawda + szum SZUM_MM), zamrożony na cały epizod; czerwona w chwytaku
                 z robot_franka (RED_IN_HAND), a po odłożeniu zostaje tam, gdzie była w chwytaku
  "prawdziwe"  - PRAWDZIWE pozycje z fizyki w każdym kroku (jak w robot_mujoco) – górna granica
  "zgadywane"  - to, co liczy sam robot_franka (CUBES_START + kostka w chwytaku), kostki stoją w CUBES_START
Przy "kamera" i "prawdziwe" kostki stoją w losowych układach jak w robot_p1_p2.py (x 0.35-0.55, y +-0.20, dowolny obrót).

Uruchom:  python test_fake.py [kostki] [szum_mm] [ile_prob] [seed]     np.  python test_fake.py kamera 5 20 0
"""
import sys
import time
import numpy as np
import torch

import robot_franka as robot
import fake_franka
from observations_combine import obs_combine, quat_to_matrix

SCALE = 0.03
N_STEPS = 1000
KOSTKI = sys.argv[1] if len(sys.argv) > 1 else "kamera"          # "kamera" / "prawdziwe" / "zgadywane" (opis wyżej)
SZUM_MM = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0       # szum udawanej kamery w poziomie [mm] (w pionie 2 mm, obrót 2 st.)
N_PROB = int(sys.argv[3]) if len(sys.argv) > 3 else 10           # ile prób (każda ~25 s)
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
assert KOSTKI in ("kamera", "prawdziwe", "zgadywane"), KOSTKI

policy = torch.jit.load("nowe_policy/2026-10-26_18-47-17_1300/exported/policy.pt", map_location="cpu").eval()


def rand_cubes_on_table(n, seed=0, min_sep=0.1):
    """Losowe układy kostek – to samo co w robot_p1_p2.py (tam import ładowałby od razu politykę z argumentów)."""
    rng = np.random.default_rng(seed)
    cases = []
    while len(cases) < n:
        x = rng.uniform(0.35, 0.55, 3)
        y = rng.uniform(-0.20, 0.20, 3)
        if min(np.hypot(x[0] - x[1], y[0] - y[1]), np.hypot(x[1] - x[2], y[1] - y[2]),
               np.hypot(x[0] - x[2], y[0] - y[2])) < min_sep:
            continue
        yaw = rng.uniform(-np.pi, np.pi, 3)
        cases.append({f"cube_{i + 1}": [x[i], y[i], 0.025, yaw[i]] for i in range(3)})
    return cases


def wieza_stoi():
    c = fake_franka.true_cubes()
    xy = np.linalg.norm(c["cube_2"][:2] - c["cube_1"][:2])
    z = c["cube_2"][2] - c["cube_1"][2]
    return xy < 0.03 and 0.04 < z < 0.06, xy, z


cases = rand_cubes_on_table(N_PROB, seed=SEED)
rng = np.random.default_rng(SEED + 1000)                     # szum kamery – osobno, żeby układy nie zależały od szumu
print(f"KOSTKI = {KOSTKI}" + (f", szum kamery {SZUM_MM} mm" if KOSTKI == "kamera" else "") + f", prób {N_PROB}, seed {SEED}")
n_stoi = 0
for proba in range(N_PROB):
    robot.reset_scene()
    if KOSTKI != "zgadywane":
        fake_franka.place_cubes(cases[proba])                # zamiast CUBES_START z reset_scene
    if KOSTKI == "kamera":
        zmierzone = fake_franka.zmierz_kostki(SZUM_MM / 1000, 0.002, np.radians(2.0), rng)   # robot stoi w HOME
        blad = [np.linalg.norm(zmierzone[n][0][:2] - fake_franka.true_cubes()[n][:2]) * 1000 for n in zmierzone]
    trzymana = False                                         # czy czerwona była już w chwytaku
    last_action = np.zeros(8, dtype=np.float32)
    czasy = []
    zamkniec = 0
    off_hand, yaw_deg, yaw_est = [], [], []                   # pomiar: gdzie NAPRAWDĘ jest czerwona kostka, gdy jest w chwytaku
    t_prev = time.perf_counter()
    for step in range(N_STEPS):
        s = robot.state()
        if KOSTKI == "prawdziwe":
            cubes = fake_franka.true_cubes_pose()
        elif KOSTKI == "zgadywane":
            cubes = robot.cubes_position_table(s)
        else:
            z_robota = robot.cubes_position_table(s)        # czerwona w chwytaku: TCP + RED_IN_HAND
            trzymana = trzymana or 0.04 < s["fingers"] * 2 < 0.06
            cubes = dict(zmierzone)
            if trzymana:                                     # od pierwszego chwytu czerwona z robot_franka
                cubes["cube_2"] = z_robota["cube_2"]

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
    n_stoi += ok
    czasy = np.array(czasy[1:]) * 1000
    if KOSTKI == "kamera":
        print(f"próba {proba}: błąd kamery w poziomie: " + ", ".join(f"{b:.1f}" for b in blad) + " mm")
    print(f"próba {proba}: {'STOI' if ok else 'NIE'}   xy = {xy*100:4.1f} cm  z = {z*100:4.1f} cm   "
          f"| krok pętli: średnio {czasy.mean():5.1f} ms, najdłuższy {czasy.max():5.1f} ms (cel 20 ms)   "
          f"| komend 'zamknij': {zamkniec}")
    if off_hand:
        o = np.mean(off_hand, axis=0) * 100
        print(f"      kostka w chwytaku ({len(off_hand)} kroków): przesunięcie od TCP w układzie ręki "
              f"[{o[0]:5.2f}, {o[1]:5.2f}, {o[2]:5.2f}] cm,  obrót kostki: prawdziwy {np.mean(yaw_deg):6.1f} st., "
              f"z robot_franka {np.mean(yaw_est):6.1f} st.")

print(f"\nKOSTKI = {KOSTKI}" + (f", szum kamery {SZUM_MM} mm" if KOSTKI == "kamera" else "") +
      f":  STOI {n_stoi}/{N_PROB} ({100 * n_stoi / N_PROB:.0f}%)")
