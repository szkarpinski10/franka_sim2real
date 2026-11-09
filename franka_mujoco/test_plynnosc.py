"""test_plynnosc.py - skuteczność i płynność policy w MuJoCo przy sterowaniu jak na robocie (robot_franka.movement).

Przykład (cała sekwencja, kostki z udawanych kamer, wersja najbliższa robotowi):
    python test_plynnosc.py --policy policy1/2026-11-05_18-04-46_2499/exported/policy.pt \
        --policy2 policy2/2026-11-06_19-01-50_3200/exported/policy.pt --dt 0.06 --vel 0.3 --obs-skala 1 \
        --v-palcow 0.03 --filtr 0.05 --robot --kamery --n 100
Parametry:
    --dt, --vel   krok policy [s] i największa prędkość stawu [rad/s]
    --obs-skala   mnożnik prędkości w obserwacji (1 dla modelu trenowanego w tym tempie)
    --filtr       wygładzanie celu jak set_filter w panda_py (1 = brak, "liniowo" = cel jedzie równo przez krok)
    --robot       niedoskonałości robota: krok dłuższy o 0-15 ms, odczyt chwytaka co 66 ms, opóźniona komenda chwytaka
    --p2          sama policy2: start z czerwoną na niebieskiej
    --policy2     cała sekwencja: --policy, HOME, --policy2
    --kamery      kostki z fuzja.py i udawanych kamer (co 120 ms, opóźnienie 120 ms, szum 3 mm / 2 st.)
Wynik: wieże, kontakt ze stołem, prędkość i przyspieszenie stawów (99%), równość ruchu w kroku (1 = równo).
"""
import argparse
import json
import os

import mujoco
import numpy as np
import torch

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "camera"))
import robot_mujoco as robot
from filtr import Filtr_CBF
from fuzja import Fuzja, poza, trzyma
from observations_combine import obs_combine

ap = argparse.ArgumentParser()
ap.add_argument("--policy", default="policy1/2026-11-05_18-04-46_2499/exported/policy.pt")
ap.add_argument("--dt", type=float, default=0.06)
ap.add_argument("--vel", type=float, default=0.3)
ap.add_argument("--obs-skala", type=float, default=1.0)
ap.add_argument("--filtr", default="0.05")
ap.add_argument("--robot", action="store_true")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--p2", action="store_true", help="policy2: start z wieżą czerwona na niebieskiej, sukces = zielona na czerwonej")
ap.add_argument("--v-palcow", type=float, default=None, help="prędkość jednego palca [m/s] (domyślnie 0.03 jak w treningu nowy_B)")
ap.add_argument("--policy2", default=None, help="cała sekwencja: po --policy ramię do HOME i ta policy")
ap.add_argument("--kamery", action="store_true", help="kostki z fuzji udawanych kamer zamiast prawdziwych")
ap.add_argument("--bez-reki", action="store_true", help="z --kamery: tylko statyw (bez kamery na chwytaku)")
args = ap.parse_args()

SCALE = 0.03
N_STEPS = 1000
P_STEP = int(round(args.dt / robot.SIM_DT))
v_palca = args.v_palcow if args.v_palcow is not None else 0.03            # jak robot_franka.V_PALCOW / 2
policy = torch.jit.load(args.policy, map_location="cpu").eval()
policy2 = torch.jit.load(args.policy2, map_location="cpu").eval() if args.policy2 else None
KAMERA_CO, KAMERA_OPOZN = 2, 2                                # klatka co 2 kroki, opóźnienie 2 kroki (120 ms)
rng_kam = np.random.default_rng(args.seed + 900)
T =np.array(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "camera",
                                         "kalibracja.json")))["T_tcp_cam"])
filtr_stolu = Filtr_CBF(stol_z=0.0, h_min=0.02, alfa=0.5, t_br=0.18, vel=args.vel, dt=args.dt,
                        kamery=[T, np.diag([-1.0, -1.0, 1.0, 1.0]) @ T])
liniowo = args.filtr == "liniowo"
c_sub = 0.0 if liniowo else 1 - (1 - float(args.filtr)) ** (robot.SIM_DT / 0.001)   # na krok fizyki 0.5 ms
rng = np.random.default_rng(args.seed + 500)
TABLE = robot.model.geom("table").id
ROBOT = ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger")


def uklady(n, seed):
    r = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        x, y = r.uniform(0.35, 0.55, 3), r.uniform(-0.2, 0.2, 3)
        if min(np.hypot(x[0] - x[1], y[0] - y[1]), np.hypot(x[1] - x[2], y[1] - y[2]),
               np.hypot(x[0] - x[2], y[0] - y[2])) < 0.1:
            continue
        yaw = r.uniform(-np.pi, np.pi, 3)
        uklad = {f"cube_{i + 1}": [x[i], y[i], 0.025, yaw[i]] for i in range(3)}
        if args.p2:                                           # jak rand_tower_pos_and_cube_3 w treningu policy2:
            if np.hypot(x[0] - x[2], y[0] - y[2]) < 0.12:     # czerwona na niebieskiej (do 1 cm obok), zielona
                continue                                      # co najmniej 12 cm od wieży
            uklad["cube_2"] = [x[0] + r.uniform(-0.01, 0.01), y[0] + r.uniform(-0.01, 0.01), 0.0755, yaw[1]]
        out.append(uklad)
    return out


def kontakt_ze_stolem():
    for c in robot.data.contact[:robot.data.ncon]:
        b1 = robot.model.body(robot.model.geom_bodyid[c.geom1]).name
        b2 = robot.model.body(robot.model.geom_bodyid[c.geom2]).name
        if (c.geom1 == TABLE and b2 in ROBOT) or (c.geom2 == TABLE and b1 in ROBOT):
            return True
    return False


def prawdziwe_kostki():
    """Pozycja i obrót wokół pionu każdej kostki, tak jak liczy je kamera (obrót co 90 st.)."""
    out = {}
    for k in ("cube_1", "cube_2", "cube_3"):
        R = robot.data.body(k).xmat.reshape(3, 3)
        os_pozioma = R[:, np.argmin(np.abs(R[2, :]))]
        out[k] = (robot.data.body(k).xpos.copy(), np.arctan2(os_pozioma[1], os_pozioma[0]) % (np.pi / 2))
    return out


def udawane_kamery(fuzja, historia, krok, trzymana, trzyma_teraz):
    """Wkłada do fuzji pomiary sprzed KAMERA_OPOZN kroków, z szumem, jak kamery.py na robocie."""
    kostki, T_tcp = historia[max(0, krok - KAMERA_OPOZN)]
    T_przed = historia[max(0, krok - KAMERA_OPOZN - KAMERA_CO)][1]
    v_tcp = np.linalg.norm(T_tcp[:3, 3] - T_przed[:3, 3]) / (KAMERA_CO * args.dt)
    t = (krok - KAMERA_OPOZN) * args.dt
    for k, (pos, yaw) in kostki.items():
        pomiar = (pos + rng_kam.normal(0, 0.003, 3), yaw + rng_kam.normal(0, np.radians(2)))
        w_chwytaku = k == trzymana and trzyma_teraz
        if not args.bez_reki and v_tcp < 0.05 and np.linalg.norm(pos[:2] - T_tcp[:2, 3]) < 0.2 and not w_chwytaku:
            fuzja.dodaj("reka", k, *pomiar, t, T_tcp)          # kamera na chwytaku: wolny ruch, kostka pod nią
        if rng_kam.random() < (0.6 if w_chwytaku else 0.85):   # statyw: czasem zasłonięty przez ramię
            fuzja.dodaj("statyw", k, *pomiar, t, T_tcp)


def epizod(policy, trzymana="cube_2"):
    d = robot.data
    fuzja, historia = Fuzja(trzymana), []
    la = np.zeros(8, dtype=np.float32)
    cmd = d.qpos[:7].copy()                                   # komenda (jak q_command w robot_franka)
    d.ctrl[:7] = cmd
    chwytak_cel, chwytak_kolejka = 0.04, []                   # (czas wykonania, cel) - opóźnienie komendy chwytaka
    palec_obs, palec_obs_t, palec_v = float(d.qpos[7]), 0.0, 0.0
    t, kontakt = 0.0, False
    v_zapis, rownosc = [], []
    for krok in range(N_STEPS):
        s = robot.state()
        if args.robot:                                        # szerokość chwytaka tylko co ok. 66 ms
            if t - palec_obs_t >= 0.066:
                palec_v = (float(d.qpos[7]) - palec_obs) / max(t - palec_obs_t, 1e-6)
                palec_obs, palec_obs_t = float(d.qpos[7]), t
            fingers, fingers_vel = palec_obs, palec_v
        else:
            fingers, fingers_vel = s["fingers"], s["fingers_vel"]
        kostki = robot.cubes_position_table()
        if args.kamery:
            T_tcp = poza(s["tcp_pos"], s["tcp_quat"])
            historia.append((prawdziwe_kostki(), T_tcp))
            trzyma_teraz = trzyma(chwytak_cel == 0.0, fingers, fingers_vel)
            if krok % KAMERA_CO == 0:
                udawane_kamery(fuzja, historia, krok, trzymana, trzyma_teraz)
            z_fuzji = fuzja.kostki(T_tcp, trzyma_teraz, krok * args.dt)
            if z_fuzji is not None:                           # na robocie czeka na to czekaj_na_kostki
                kostki = z_fuzji
        obs = obs_combine(s["q"], s["q_vel"] * args.obs_skala, fingers, fingers_vel * args.obs_skala, la,
                          s["tcp_pos"], s["tcp_quat"], kostki)
        with torch.no_grad():
            a = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
        step = np.clip(s["q"] + SCALE * a[:7] - cmd, -args.vel * args.dt, args.vel * args.dt)
        step = filtr_stolu.filtruj(step, s["q"], s["q_vel"], fingers)
        cmd_prev = cmd.copy()
        cmd = np.clip(cmd + step, s["q"] - 0.1, s["q"] + 0.1)
        nowy_chwytak = 0.04 if a[7] >= 0 else 0.0
        if args.robot:
            chwytak_kolejka.append((t + rng.uniform(0.0, 0.1), nowy_chwytak))
        else:
            chwytak_cel = nowy_chwytak
        n_sub = P_STEP + (int(rng.uniform(0, 0.015) / robot.SIM_DT) if args.robot else 0)
        predk = []
        for i in range(n_sub):
            if liniowo:
                d.ctrl[:7] = cmd_prev + (cmd - cmd_prev) * min(1.0, (i + 1) / P_STEP)
            else:
                d.ctrl[:7] = d.ctrl[:7] + c_sub * (cmd - d.ctrl[:7])
            while chwytak_kolejka and chwytak_kolejka[0][0] <= t:
                chwytak_cel = chwytak_kolejka.pop(0)[1]
            d.ctrl[7] = chwytak_cel
            przed = d.qpos[7:9].copy()
            mujoco.mj_step(robot.model, d)
            lim = v_palca * robot.SIM_DT
            d.qpos[7:9] = przed + np.clip(d.qpos[7:9] - przed, -lim, lim)
            d.qvel[7:9] = np.clip(d.qvel[7:9], -v_palca, v_palca)
            t += robot.SIM_DT
            if i % 4 == 0:                                    # co 2 ms
                v_zapis.append(d.qvel[:7].copy())
                predk.append(np.abs(d.qvel[:7]).max())
        predk = np.array(predk)
        if predk.mean() > 0.05:                               # tylko kroki, w których robot się rusza
            rownosc.append(predk.min() / predk.mean())
        kontakt = kontakt or kontakt_ze_stolem()
        la = a.copy()
    sukces = na_sobie("cube_1", "cube_2")
    if args.p2:                                               # cała wieża: dodatkowo zielona na czerwonej
        sukces = sukces and na_sobie("cube_2", "cube_3")
    v = np.array(v_zapis)
    przysp = np.abs(np.diff(v, axis=0)).max(axis=1) / 0.002
    return sukces, kontakt, np.abs(v).max(axis=1), przysp, rownosc


def na_sobie(dolna, gorna):
    c = robot.cubes_position_table()
    d, g = c[dolna][0], c[gorna][0]
    return np.linalg.norm(g[:2] - d[:2]) < 0.03 and 0.04 < g[2] - d[2] < 0.06


def do_home():
    """Ramię i chwytak od razu w HOME, kostki zostają (na robocie: reset_scene w menu)."""
    d = robot.data
    d.qpos[:7], d.qvel[:9], d.qpos[7:9], d.ctrl[:7] = robot.HOME, 0.0, 0.04, robot.HOME
    mujoco.mj_forward(robot.model, d)


wieze = kontakty = p1_udane = 0
V, A, R = [], [], []
for case in uklady(args.n, args.seed):
    robot.reset_scene(case)
    s, k, v, a, r = epizod(policy, "cube_3" if args.p2 else "cube_2")
    if policy2 is not None and s:                             # policy1 postawiła czerwoną -> policy2
        p1_udane += 1
        do_home()
        _, k2, v2, a2, r2 = epizod(policy2, "cube_3")
        s = na_sobie("cube_1", "cube_2") and na_sobie("cube_2", "cube_3")
        k, v, a, r = k or k2, np.concatenate([v, v2]), np.concatenate([a, a2]), r + r2
    wieze += s
    kontakty += k
    V.append(v); A.append(a); R += r
V, A = np.concatenate(V), np.concatenate(A)
if policy2 is not None:
    print(f"policy1: {p1_udane}/{args.n}, policy2 po udanej policy1: {wieze}/{p1_udane}", flush=True)
print(f"dt {args.dt * 1000:.0f} ms, vel {args.vel:.3f}, obs x{args.obs_skala:g}, filtr {args.filtr}, robot {args.robot}, "
      f"kamery {args.kamery}{' (bez ręki)' if args.bez_reki else ''}: wieże {wieze}/{args.n}, kontakt ze stołem {kontakty}/{args.n} | stawy 99%: prędkość "
      f"{np.percentile(V, 99):.2f} rad/s, przyspieszenie {np.percentile(A, 99):.1f} rad/s2 | równość ruchu w kroku "
      f"{np.mean(R):.2f}", flush=True)
