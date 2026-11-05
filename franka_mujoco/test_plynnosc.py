"""test_plynnosc.py – płynność i skuteczność policy w MuJoCo przy sterowaniu jak na prawdziwym robocie (bez sprzętu).

Uruchom (z franka_mujoco/):
    python test_plynnosc.py --policy policy1/2026-11-01_16-15-43_800/exported/policy.pt --dt 0.06 --vel 0.267 \
        --obs-skala 3 --filtr 0.05 --n 30
Parametry:
    --dt          krok policy [s]  (0.02 = jak w treningu; 0.06 = TEMPO 3 albo model trenowany co 60 ms)
    --vel         największa prędkość stawu [rad/s]; ruch na krok = vel * dt (obecny model: 0.016 rad)
    --obs-skala   x prędkości stawów i palców w obserwacji (obecny model w TEMPO 3: 3; model trenowany w nowym
                  tempie: 1)
    --filtr       wygładzanie celu w sterowniku jak JointPosition.set_filter w panda_py: część różnicy do nowego
                  celu, o którą cel przesuwa się co 1 ms (1 = brak, 0.05 ~ 20 ms, 0.02 ~ 50 ms); "liniowo" = cel jedzie
                  równo przez cały krok (wzorzec)
    --robot       niedoskonałości robota: krok pętli dłuższy o 0-15 ms, szerokość chwytaka co ok. 66 ms,
                  komenda chwytaka z opóźnieniem do 0.1 s
Sterowanie jak robot_franka.movement: cel = q + 0.03*a, krok przycięty do vel*dt od poprzedniej komendy, filtr stołu
(blat 0, 1.5 cm, 0.18 s, z bryłami obu kamer), komenda max 0.1 rad przed robotem, prędkość docelowa 0.
Pozycje kostek prawdziwe (tu badamy ruch, nie kamery – kamery: test_kamery_sim.py).
Wynik: wieże, kontakt ze stołem, prędkość i przyspieszenie stawów (99%), "równość" ruchu w kroku
(najmniejsza / średnia prędkość w obrębie kroku policy: 1 = jedzie równo, blisko 0 = zrywami).
"""
import argparse
import json
import os

import mujoco
import numpy as np
import torch

import robot_mujoco as robot
from filtr import Filtr_CBF
from observations_combine import obs_combine

ap = argparse.ArgumentParser()
ap.add_argument("--policy", default="policy1/2026-11-01_16-15-43_800/exported/policy.pt")
ap.add_argument("--dt", type=float, default=0.06)
ap.add_argument("--vel", type=float, default=0.8 / 3)
ap.add_argument("--obs-skala", type=float, default=3.0)
ap.add_argument("--filtr", default="1")
ap.add_argument("--robot", action="store_true")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--v-palcow", type=float, default=None, help="prędkość jednego palca [m/s] (domyślnie 0.05 * vel / 0.8)")
args = ap.parse_args()

SCALE = 0.03
N_STEPS = 1000
P_STEP = int(round(args.dt / robot.SIM_DT))
v_palca = args.v_palcow if args.v_palcow is not None else 0.05 * args.vel / 0.8   # palce zwalniają jak stawy
policy = torch.jit.load(args.policy, map_location="cpu").eval()
T = np.array(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "camera",
                                         "kalibracja.json")))["T_tcp_cam"])
filtr_stolu = Filtr_CBF(stol_z=0.0, h_min=0.015, alfa=0.5, t_br=0.18, vel=args.vel, dt=args.dt,
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
        out.append({f"cube_{i + 1}": [x[i], y[i], 0.025, yaw[i]] for i in range(3)})
    return out


def kontakt_ze_stolem():
    for c in robot.data.contact[:robot.data.ncon]:
        b1 = robot.model.body(robot.model.geom_bodyid[c.geom1]).name
        b2 = robot.model.body(robot.model.geom_bodyid[c.geom2]).name
        if (c.geom1 == TABLE and b2 in ROBOT) or (c.geom2 == TABLE and b1 in ROBOT):
            return True
    return False


def epizod():
    d = robot.data
    la = np.zeros(8, dtype=np.float32)
    cmd = d.qpos[:7].copy()                                   # komenda (jak q_command w robot_franka)
    d.ctrl[:7] = cmd
    chwytak_cel, chwytak_kolejka = 0.04, []                   # (czas wykonania, cel) – opóźnienie komendy chwytaka
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
        obs = obs_combine(s["q"], s["q_vel"] * args.obs_skala, fingers, fingers_vel * args.obs_skala, la,
                          s["tcp_pos"], s["tcp_quat"], robot.cubes_position_table())
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
    c = robot.cubes_position_table()
    c1, c2 = c["cube_1"][0], c["cube_2"][0]
    sukces = np.linalg.norm(c2[:2] - c1[:2]) < 0.03 and 0.04 < c2[2] - c1[2] < 0.06
    v = np.array(v_zapis)
    przysp = np.abs(np.diff(v, axis=0)).max(axis=1) / 0.002
    return sukces, kontakt, np.abs(v).max(axis=1), przysp, rownosc


wieze = kontakty = 0
V, A, R = [], [], []
for case in uklady(args.n, args.seed):
    robot.reset_scene(case)
    s, k, v, a, r = epizod()
    wieze += s
    kontakty += k
    V.append(v); A.append(a); R += r
V, A = np.concatenate(V), np.concatenate(A)
print(f"dt {args.dt * 1000:.0f} ms, vel {args.vel:.3f}, obs x{args.obs_skala:g}, filtr {args.filtr}, robot {args.robot}: "
      f"wieże {wieze}/{args.n}, kontakt ze stołem {kontakty}/{args.n} | stawy 99%: prędkość {np.percentile(V, 99):.2f} rad/s, "
      f"przyspieszenie {np.percentile(A, 99):.1f} rad/s2 | równość ruchu w kroku {np.mean(R):.2f}", flush=True)
