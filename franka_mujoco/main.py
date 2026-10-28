import time
import torch
import numpy as np
import mujoco
import mujoco.viewer
from robot_sim import cube2_contacts, grasp_angles
from robot_sim import (model, data, get_joints, get_finger, get_ee, get_cubes,
                       arm_move, gripper_move, step_function, reset)
from observations_combine import obs_combine



def quat_xyzw_to_mat(q):
    x, y, z, w = q
    s = 2.0 / (x * x + y * y + z * z + w * w)
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w),     s * (x * z + y * w)],
        [s * (x * y + z * w),     1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w),     s * (y * z + x * w),     1 - s * (x * x + y * y)],
    ])
 
 
def rel_yaw4(ee_quat_xyzw, cubes):
    """cubes = {"cube_1": (pos, quat_xyzw), ...} -> [sin, cos] x 3 kostki."""
    R_hand = quat_xyzw_to_mat(ee_quat_xyzw)
    hand_yaw = np.arctan2(R_hand[1, 1], R_hand[0, 1])
    out = []
    for name in ("cube_1", "cube_2", "cube_3"):
        R = quat_xyzw_to_mat(cubes[name][1])
        up = int(np.argmax(np.abs(R[2, :])))
        h = 1 if up == 0 else 0
        d = hand_yaw - np.arctan2(R[1, h], R[0, h])
        out += [np.sin(4 * d), np.cos(4 * d)]
    return np.array(out, dtype=np.float32)


# ===== ustawienia =====
# Wpisz folder treningu, z którego jest film (ten, który "elegancko działa").
# policy.pt powstaje w <folder>/exported/ po uruchomieniu play na tym checkpoincie.
POLICIES = {"baza42": "stack/policy1/exported/policy.pt"}
QUAT_FIX = False     # True = kwaternion chwytaka bez przeskoków znaku (q i -q to ten sam obrót) - test: bez wpływu
RELEASE_GUARD = False # True = "bezpiecznik": nie otwieraj palców nad wieżą, dopóki dłoń jedzie szybciej niż GUARD_V
GUARD_V = 0.05       # [m/s] dłoń musi zwolnić poniżej 5 cm/s, zanim puści kostkę
GUARD_MAX = 15       # najwyżej tyle kroków (0.3 s) czekania, potem i tak puszcza
VEL_MODE = "interp"  # "clip" = stare obcinanie qvel po kroku fizyki, "interp" = cel PD przesuwany z limitem prędkości (jak na Frandze)
REL_YAW4 = False      # True = dokładamy 6 liczb rel_yaw4 na końcu
GRIPPER_POS = True  # False = bez gripper_pos (usunięte w Isaacu). Nowa policy: 72 + 6 = 78 liczb
VIEW = False   # True = okno MuJoCo (wolniej)
DEBUG = False    # True = wydruki kroków 20-45 w każdym epizodzie
PRINT_GRASPS = False # wypisuj każdy moment zamykania chwytaka
DUMP_OBS = False # True = wypisz obserwację i akcję z kroków 0-2 (tylko układ 0) do porównania z Isaakiem
SCALE = 0.03          # = scale w RelativeJointPositionActionCfg
N_STEPS = 1000       # 1000 x 0.02 s = 20 s, jak epizod w treningu

N_CASES = 100
SEED = 0

def random_cases(n, seed=0, min_sep=0.1):
    """Losowe układy jak w treningu: x 0.4-0.6, y ±0.1, yaw ±pi, odstęp >= min_sep."""
    rng = np.random.default_rng(seed)
    cases = []
    while len(cases) < n:
        xy = np.column_stack([rng.uniform(0.4, 0.6, 3), rng.uniform(-0.1, 0.1, 3)])
        d = [np.linalg.norm(xy[i] - xy[j]) for i in range(3) for j in range(i + 1, 3)]
        if min(d) < min_sep:
            continue
        yaw = rng.uniform(-np.pi, np.pi, 3)
        cases.append({f"cube_{k+1}": [xy[k, 0], xy[k, 1], 0.025, yaw[k]] for k in range(3)})
    return cases


def set_cube_yaw(name, yaw):
    """Obraca kostkę wokół osi z (MuJoCo: kwaternion w kolejności w, x, y, z)."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    adr = model.jnt_qposadr[model.body_jntadr[body_id]]
    data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


def reset_case(case):
    reset({k: v[:3] for k, v in case.items()})   # Twój reset (pozycje)
    for name, v in case.items():
        set_cube_yaw(name, v[3])                  # + losowy obrót
    mujoco.mj_forward(model, data)


XY_OK = 0.02      # SUKCES (precyzyjnie): przesunięcie < 2 cm
XY_STOI = 0.03    # "wieża stoi": czerwona leży na niebieskiej, przesunięcie < 3 cm


def check_tower():
    """Czy czerwona (cube_2) stoi na niebieskiej (cube_1)?"""
    c1 = get_cubes()["cube_1"][0]
    c2 = get_cubes()["cube_2"][0]
    d_xy = np.linalg.norm(c2[:2] - c1[:2])
    d_z = c2[2] - c1[2]
    ok = d_xy < XY_OK and 0.04 < d_z < 0.06
    return ok, d_xy, d_z


SEG = [("joint_pos_rel", 9), ("joint_vel_rel", 9), ("actions", 8), ("eef_pos", 3), ("eef_quat", 4)]
if GRIPPER_POS:
    SEG.append(("gripper_pos", 2))
SEG.append(("cubes(object_obs)", 39))
if REL_YAW4:
    SEG.append(("rel_yaw4", 6))
N_OBS = sum(n for _, n in SEG)


def dump(tag, obs, action):
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print(f"\n[{tag}] rozmiar obs = {len(obs)}")
    i = 0
    for name, n in SEG:
        print(f"  {name:18s}: {np.asarray(obs[i:i + n])}")
        i += n
    if i != len(obs):
        print(f"  UWAGA: podział {i} != {len(obs)}")
    print(f"  {'AKCJA':18s}: {np.asarray(action)}")


def robot_touches(cube_name):
    """Które części ROBOTA dotykają danej kostki (pusta lista = żadna)."""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube_name)
    out = set()
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if cid in (b1, b2):
            other = b2 if b1 == cid else b1
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
            if not name.startswith("cube") and name != "world":
                out.add(name)
    return sorted(out)


ETAPY = []    # gdzie kończy się każdy epizod
OTWARCIA = [] # (etap, stan w chwili otwarcia palców nad wieżą)
GRASPS = []   # (dz, dxy, yaw) w chwili przełączenia chwytaka na "zamknij" blisko cube_2
PROBY = []    # (etap, lista prób chwytu w epizodzie)
HOLDS = []    # (etap, dz_hold) głębokość trzymania kostki 5 kroków po podniesieniu
GSTAT = {"wstrzymane_kroki": 0, "puszczenia_z_czekaniem": 0, "limit_czekania": 0}
QSTAT = {"ee_flips": 0, "c2_flips": 0, "w_min": 1.0, "w_max": -1.0, "start": [], "etapy_flip": []}
ZGUBIENIA = [] # (etap, stan w chwili, gdy podniesiona kostka odjechała od palców)

# rodzaj próby chwytu po tym, jak daleko zamknęły się palce (1 palec: 0.04 otwarty, 0.025 = kostka 5 cm)
def typ_proby(min_palce):
    if min_palce >= 0.030:
        return "ZA_KROTKO"   # komenda 'zamknij' cofnięta, zanim palce doszły do kostki
    if min_palce >= 0.020:
        return "W_PALCACH"   # palce zatrzymały się na kostce
    return "PUSTO"           # palce zamknęły się dalej niż szerokość kostki: chybił / złapał za krawędź i wyślizgnęła się


import robot_sim as _rs


def step_interp(q_goal):
    """Jeden krok policy (0.02 s) BEZ nadpisywania prędkości po kroku fizyki.

    Zamiast obcinać data.qvel (niefizyczne: zabiera/dodaje energię, psuje kontakty),
    przesuwamy CEL PD w stronę q_goal najwyżej o _VEL * dt na krok fizyki.
    Tak samo trzeba będzie sterować prawdziwą Franką (interpolator celu z limitem prędkości),
    bo komenda szybsza niż limit przerywa ruch robota.
    Palce: bez zmian (jak w step_function), żeby zmieniać jedną rzecz naraz.
    """
    dt = model.opt.timestep
    max_dq = _rs._VEL * dt
    data.ctrl[7] = _rs._grip_target
    for _ in range(_rs.N_SUB):
        cur = data.ctrl[:7].copy()
        data.ctrl[:7] = cur + np.clip(q_goal - cur, -max_dq, max_dq)
        mujoco.mj_step(model, data)
        raw = np.abs(data.qvel[:7])
        _rs.VEL_STATS["kroki"] += 1
        _rs.VEL_STATS["przyciete"] += int(np.any(raw > _rs._VEL))      # tu: ile kroków robot i tak jechał szybciej niż limit
        _rs.VEL_STATS["max_raw"] = np.maximum(_rs.VEL_STATS["max_raw"], raw)
        data.qvel[7:9] = np.clip(data.qvel[7:9], -_rs._FINGER_VEL, _rs._FINGER_VEL)  # palce jak dotąd


def run_episode(policy, viewer=None, dump_obs=False):
    last_action = np.zeros(8, dtype=np.float32)
    # przebieg epizodu (do klasyfikacji porażek)
    ev = {"proby_chwytu": 0, "podniesiona": False, "krok_podniesienia": None, "max_z": 0.0,
          "nad_wieza": False, "upadla_po_podniesieniu": False,
          "stala_na_wiezy": False, "krok_stania": None, "puszczenia": [],
          "upadek": None, "ostatni_dotyk": None,
          "otwarcie": None,   # stan w chwili, gdy palce NAPRAWDĘ się rozwierają nad wieżą
          "proby": [], "cur": None,   # każda próba chwytu: od komendy 'zamknij' do komendy 'otwórz'
          "zgubienie": None}          # chwila, gdy podniesiona kostka odjechała od palców
    prev_finger = 0.04
    prev_tcp = None
    if VEL_MODE == "interp":
        data.ctrl[:7] = data.qpos[:7]   # cel startuje z aktualnej pozycji robota (nie z poprzedniego epizodu)
    prev_q_raw = None    # surowy kwaternion chwytaka z poprzedniego kroku (do liczenia przeskoków)
    prev_q_used = None   # kwaternion podany polityce w poprzednim kroku
    prev_c2q = None
    prev_tcp_g = None    # do prędkości dłoni (bezpiecznik)
    guard_wait = 0
    for step in range(N_STEPS):
        joint_q, joint_vel = get_joints()
        finger_pos, finger_vel = get_finger()
        tcp_pos, ee_quat = get_ee()
        cubes = get_cubes()

        # --- kwaterniony: czy znak przeskakuje? (q i -q = ten sam obrót, ale dla sieci to inne liczby) ---
        QSTAT["w_min"] = min(QSTAT["w_min"], float(ee_quat[3]))
        QSTAT["w_max"] = max(QSTAT["w_max"], float(ee_quat[3]))
        if step == 0:
            QSTAT["start"].append(ee_quat.round(3).tolist())
        if prev_q_raw is not None and np.dot(ee_quat, prev_q_raw) < 0:
            QSTAT["ee_flips"] += 1
            ev["ee_flip"] = True
        prev_q_raw = ee_quat.copy()
        c2q = np.asarray(cubes["cube_2"][1])
        if prev_c2q is not None and np.dot(c2q, prev_c2q) < 0:
            QSTAT["c2_flips"] += 1
        prev_c2q = c2q.copy()
        if QUAT_FIX and prev_q_used is not None and np.dot(ee_quat, prev_q_used) < 0:
            ee_quat = -ee_quat           # trzymamy ten sam znak co w poprzednim kroku
        prev_q_used = ee_quat.copy()

        obs = obs_combine(joint_q, joint_vel, finger_pos, finger_vel, last_action, tcp_pos, ee_quat, cubes)
        obs = np.asarray(obs, dtype=np.float32)
        if not GRIPPER_POS:
            obs = np.delete(obs, [33, 34])   # gripper_pos leży na miejscach 33-34 (po 9+9+8+3+4 = 33 liczbach)
        if REL_YAW4:
            obs = np.concatenate([obs, rel_yaw4(ee_quat, cubes)])

        with torch.no_grad():
            action = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()

        if dump_obs and step <= 2:
            dump(f"MUJOCO krok {step}", obs, action)

        if DEBUG and 20 <= step <= 45:
            print(f"[MUJOCO {step}] grip_a={action[7]:6.2f}  palce={finger_pos:.4f}  "
                f"cube2={cubes['cube_2'][0].round(3)}  tcp={tcp_pos.round(3)}")
            d_yaw, tilt = grasp_angles()
            print(f"   yaw_roznica={d_yaw:6.1f}°  przechyl={tilt:5.1f}°  kontakty={cube2_contacts()}")

        # --- przebieg epizodu ---
        c1p, c2p = cubes["cube_1"][0], cubes["cube_2"][0]
        ev["max_z"] = max(ev["max_z"], float(c2p[2]))
        if c2p[2] > 0.045 and not ev["podniesiona"]:
            ev["podniesiona"], ev["krok_podniesienia"] = True, step
        if ev["podniesiona"] and c2p[2] < 0.03 and not ev["nad_wieza"]:
            ev["upadla_po_podniesieniu"] = True
        if np.linalg.norm(c2p[:2] - c1p[:2]) < 0.02 and c2p[2] > c1p[2] + 0.045:
            ev["nad_wieza"] = True
        # stoi na wieży (jak check_tower) przy otwartych palcach?
        if (np.linalg.norm(c2p[:2] - c1p[:2]) < 0.02 and 0.04 < c2p[2] - c1p[2] < 0.06
                and finger_pos > 0.035 and not ev["stala_na_wiezy"]):
            ev["stala_na_wiezy"], ev["krok_stania"] = True, step
        # moment, gdy palce faktycznie się rozwierają (przechodzą przez 3.5 cm) nad wieżą
        if ev["otwarcie"] is None and prev_finger < 0.035 <= finger_pos:
            off = float(np.linalg.norm(c2p[:2] - c1p[:2]))
            if off < 0.08 and c2p[2] > c1p[2] + 0.03:
                vadr = model.jnt_dofadr[model.joint("cube_2_joint").id]
                v = data.qvel[vadr:vadr + 3]
                ev["otwarcie"] = {"przesuniecie_cm": off * 100,
                                  "wysokosc_cm": float(c2p[2] - c1p[2]) * 100,
                                  "v_bok_cm_s": float(np.linalg.norm(v[:2])) * 100,
                                  "v_dol_cm_s": float(-v[2]) * 100}

        # moment, w którym postawiona wieża PRZESTAJE stać -> kto ją ruszył?
        if ev["stala_na_wiezy"] and ev["upadek"] is None:
            dot = robot_touches("cube_2") + robot_touches("cube_1")
            if dot and step > ev["krok_stania"]:
                ev["ostatni_dotyk"] = (step, sorted(set(dot)))
            d_xy_t = np.linalg.norm(c2p[:2] - c1p[:2])
            on_top_t = d_xy_t < 0.03 and 0.04 < c2p[2] - c1p[2] < 0.06
            if not on_top_t:
                ev["upadek"] = {
                    "krok": step,
                    "po_ilu_krokach": step - ev["krok_stania"],
                    "robot_dotyka_czerwonej": robot_touches("cube_2"),
                    "robot_dotyka_niebieskiej": robot_touches("cube_1"),
                    "tcp_nad_czerwona_cm": round(float(tcp_pos[2] - c2p[2]) * 100, 1),
                    "tcp_xy_od_czerwonej_cm": round(float(np.linalg.norm(tcp_pos[:2] - c2p[:2])) * 100, 1),
                    "palce_cm": round(float(finger_pos) * 100, 1),
                    "ostatni_dotyk": ev["ostatni_dotyk"],
                }

        # moment PUSZCZENIA nad wieżą: komenda zmienia się z "zamknij" na "otwórz"
        if last_action[7] < 0 and action[7] >= 0 and np.linalg.norm(c2p[:2] - c1p[:2]) < 0.05:
            ev["puszczenia"].append((step, float(c2p[2] - c1p[2]), float(np.linalg.norm(c2p[:2] - c1p[:2]))))

        # pomiar chwytu: TA SAMA definicja co w Isaacu (debug_grasp_moment)
        if last_action[7] >= 0 and action[7] < 0:
            c2 = cubes["cube_2"][0]
            dz = tcp_pos[2] - c2[2]
            dxy = float(np.linalg.norm(tcp_pos[:2] - c2[:2]))
            if dxy < 0.05 and c2[2] < 0.04 and dz < 0.10:
                d_yaw, tilt = grasp_angles()
                GRASPS.append((dz, dxy, d_yaw))
                ev["proby_chwytu"] += 1
                ev["cur"] = {"krok": step, "dz": float(dz), "dxy": dxy, "yaw": float(d_yaw),
                             "min_palce": float(finger_pos), "kroki": 0, "podniosla": False}
                ev["proby"].append(ev["cur"])
                if PRINT_GRASPS:
                    print(f"[MUJOCO CHWYT] krok {step:3d}  dz = {dz*100:5.1f} cm  dxy = {dxy*100:4.1f} cm  "
                          f"yaw = {d_yaw:6.1f}°  przechył kostki = {tilt:4.1f}°")

        # śledzenie trwającej próby chwytu (dopóki komenda = 'zamknij')
        if ev["cur"] is not None:
            if action[7] < 0:
                ev["cur"]["kroki"] += 1
                ev["cur"]["min_palce"] = min(ev["cur"]["min_palce"], float(finger_pos))
                if c2p[2] > 0.045:
                    ev["cur"]["podniosla"] = True
            else:
                ev["cur"] = None

        # chwila ZGUBIENIA: kostka była podniesiona, a teraz odjechała od TCP o > 2 cm
        # względem tego, jak trzymała się w chwili podniesienia
        d_tcp_c2 = float(np.linalg.norm(tcp_pos - c2p))
        if ev["podniesiona"] and "d_lift" not in ev:
            ev["d_lift"] = d_tcp_c2
        # GŁĘBOKOŚĆ TRZYMANIA (to, co mierzy hold_depth): 5 kroków po podniesieniu, kostka w palcach
        if (ev["podniesiona"] and "dz_hold" not in ev and step == ev["krok_podniesienia"] + 5
                and c2p[2] > 0.045 and finger_pos < 0.035):
            ev["dz_hold"] = float(tcp_pos[2] - c2p[2])
        if (ev["podniesiona"] and ev["zgubienie"] is None and not ev["nad_wieza"]
                and d_tcp_c2 > ev["d_lift"] + 0.02):
            v_tcp = 0.0 if prev_tcp is None else float(np.linalg.norm(tcp_pos - prev_tcp)) / 0.02
            udane = [q for q in ev["proby"] if q["podniosla"]]
            ch = udane[-1] if udane else (ev["proby"][-1] if ev["proby"] else None)
            ev["zgubienie"] = {"krok": step,
                               "po_ilu_krokach": step - ev["krok_podniesienia"],
                               # palce > 3 cm = chwytak otwarty (robot sam puścił), < 3 cm = wyślizgnęła się z zamkniętych palców
                               "komenda": "OTWÓRZ" if finger_pos > 0.03 else "zamknij",
                               "palce_cm": float(finger_pos) * 100,
                               "v_tcp_cm_s": v_tcp * 100,
                               "z_kostki_cm": float(c2p[2]) * 100,
                               "dz_przy_chwycie_cm": ch["dz"] * 100 if ch else float("nan"),
                               "dxy_przy_chwycie_cm": ch["dxy"] * 100 if ch else float("nan")}
        prev_tcp = tcp_pos.copy()

        prev_finger = finger_pos
        q_goal = joint_q + SCALE * action[:7]     # akcja WZGLĘDNA
        open_cmd = action[7] >= 0
        # --- bezpiecznik puszczania: kostka w palcach nad wieżą, a dłoń jeszcze jedzie -> trzymaj ---
        v_now = 0.0 if prev_tcp_g is None else float(np.linalg.norm(tcp_pos - prev_tcp_g)) / 0.02
        prev_tcp_g = tcp_pos.copy()
        if RELEASE_GUARD:
            above_tower = (np.linalg.norm(c2p[:2] - c1p[:2]) < 0.03 and c2p[2] > c1p[2] + 0.04
                           and np.linalg.norm(tcp_pos - c2p) < 0.03)
            if open_cmd and finger_pos < 0.035 and above_tower and v_now > GUARD_V:
                if guard_wait < GUARD_MAX:
                    open_cmd = False
                    if guard_wait == 0:
                        GSTAT["puszczenia_z_czekaniem"] += 1
                    guard_wait += 1
                    GSTAT["wstrzymane_kroki"] += 1
                elif guard_wait == GUARD_MAX:
                    GSTAT["limit_czekania"] += 1
                    guard_wait += 1
            if not open_cmd and action[7] < 0:
                guard_wait = 0           # polityka znowu chce trzymać -> nowe puszczenie liczymy od zera
        gripper_move(open_cmd)
        if VEL_MODE == "interp":
            step_interp(q_goal)
        else:
            arm_move(q_goal)
            step_function()
        last_action = action.copy()

        if viewer is not None:
            viewer.sync()
            time.sleep(0.02)

    ok, d_xy, d_z = check_tower()
    stoi = 0.04 < d_z < 0.06 and d_xy < XY_STOI
    if ok:
        etap = "SUKCES"
    elif stoi:
        etap = "S) wieża stoi, ale przesunięta 2-3 cm"
    elif not ev["podniesiona"]:
        etap = "A) nie podniósł kostki"
    elif ev["stala_na_wiezy"]:
        etap = "C1) stała na wieży, potem ją strącił"
    elif ev["nad_wieza"]:
        etap = "C2) była nad wieżą, ale nie ustała (spadła przy odkładaniu)"
    elif ev["upadla_po_podniesieniu"]:
        etap = "B) podniósł, ale zgubił w drodze"
    else:
        etap = "D) inne"
    if ev["puszczenia"]:
        st_, dzp, dxyp = ev["puszczenia"][0]
        print(f"      1. puszczenie nad wieżą: krok {st_}, kostka {dzp*100:4.1f} cm nad niebieską "
              f"(powinno być 5.0), przesunięcie {dxyp*100:4.1f} cm; stała na wieży od kroku {ev['krok_stania']}")
    print(f"      próby chwytu: {ev['proby_chwytu']}  podniesiona: {ev['podniesiona']} "
          f"(krok {ev['krok_podniesienia']})  max z: {ev['max_z']*100:4.1f} cm  -> {etap}")
    if ev["otwarcie"] is not None:
        o = ev["otwarcie"]
        print(f"      palce otwarte nad wieżą: przesunięcie {o['przesuniecie_cm']:4.1f} cm, wysokość {o['wysokosc_cm']:4.1f} cm, "
              f"kostka jedzie w bok {o['v_bok_cm_s']:5.1f} cm/s, w dół {o['v_dol_cm_s']:5.1f} cm/s")
        OTWARCIA.append((etap[:2], o))
    if etap.startswith("C1") and ev["upadek"] is not None:
        u = ev["upadek"]
        print(f"      >>> UPADEK WIEŻY: krok {u['krok']} ({u['po_ilu_krokach']} kroków po postawieniu), "
              f"robot dotyka czerwonej: {u['robot_dotyka_czerwonej'] or 'NIE'}, niebieskiej: {u['robot_dotyka_niebieskiej'] or 'NIE'}, "
              f"TCP {u['tcp_nad_czerwona_cm']} cm nad czerwoną i {u['tcp_xy_od_czerwonej_cm']} cm w bok, palce {u['palce_cm']} cm")
        od = u["ostatni_dotyk"]
        print(f"          ostatni dotyk robota do wieży PO postawieniu: "
              + (f"krok {od[0]} ({u['krok'] - od[0]} kroków przed upadkiem): {od[1]}" if od else "BRAK (robot jej nie dotykał)"))
    if ev["proby"]:
        p = ev["proby"][0]
        typy = [typ_proby(q["min_palce"]) for q in ev["proby"]]
        print(f"      1. chwyt: dz {p['dz']*100:4.1f} cm, dxy {p['dxy']*100:4.1f} cm, yaw {p['yaw']:5.1f}°, "
              f"palce zeszły do {p['min_palce']*100:4.2f} cm, 'zamknij' przez {p['kroki']} kroków -> {typ_proby(p['min_palce'])}"
              f"   | wszystkie próby: W_PALCACH {typy.count('W_PALCACH')}, PUSTO {typy.count('PUSTO')}, ZA_KROTKO {typy.count('ZA_KROTKO')}")
    if ev["zgubienie"] is not None and etap.startswith("B"):
        z = ev["zgubienie"]
        print(f"      >>> ZGUBIENIE: {z['po_ilu_krokach']} kroków po podniesieniu, kostka {z['z_kostki_cm']:4.1f} cm nad stołem, "
              f"komenda: {z['komenda']}, palce {z['palce_cm']:4.2f} cm, TCP jedzie {z['v_tcp_cm_s']:5.1f} cm/s, "
              f"chwyt był: dz {z['dz_przy_chwycie_cm']:4.1f} cm, dxy {z['dxy_przy_chwycie_cm']:4.1f} cm")
    if "dz_hold" in ev:
        print(f"      trzymanie w drodze: TCP {ev['dz_hold']*100:4.1f} cm nad środkiem kostki (0 = głęboko, 2.5 = za krawędź)")
        HOLDS.append((etap[:2], ev["dz_hold"]))
    QSTAT["etapy_flip"].append((etap[:2], ev.get("ee_flip", False)))
    PROBY.append((etap[:2], ev["proby"]))
    if ev["zgubienie"] is not None:
        ZGUBIENIA.append((etap[:2], ev["zgubienie"]))
    ETAPY.append(etap)
    return ok, d_xy, d_z


# ===== kontrola: jakie ustawienia fizyki NAPRAWDĘ są używane =====
import robot_sim
print("robot_sim z pliku:", robot_sim.__file__)
print(f"KP = {robot_sim.KP}   KD = {robot_sim.KD}   twarde kontakty na {robot_sim._hard} geometriach")
print(f"_VEL = {robot_sim._VEL}   cone = {model.opt.cone} (1 = elliptic)   impratio = {model.opt.impratio}")
print(f"TORSION (condim 4) = {robot_sim.TORSION}  na {robot_sim._torsion} geometriach")
# ===== testy =====
cases = random_cases(N_CASES, seed=SEED)

viewer = mujoco.viewer.launch_passive(model, data) if VIEW else None
if viewer is not None:
    time.sleep(3)

wyniki = {}
for name, path in POLICIES.items():
    policy = torch.jit.load(path, map_location="cpu").eval()
    print(f"rozmiar obserwacji w MuJoCo = {N_OBS}  (gripper_pos={GRIPPER_POS}, rel_yaw4={REL_YAW4})")
    print(f"\n===== {name}  ({path}) =====")
    n_ok = 0
    n_stoi = 0
    for i, case in enumerate(cases):
        reset_case(case)
        ok, d_xy, d_z = run_episode(policy, viewer, dump_obs=(DUMP_OBS and i == 0))
        stoi = 0.04 < d_z < 0.06 and d_xy < XY_STOI
        n_ok += ok
        n_stoi += stoi
        znak = "WIEŻA STOI ✅" if ok else ("STOI, przesunięta 🟡" if stoi else "NIE ❌")
        print(f"[układ {i:2d}] {znak}   xy = {d_xy*100:4.1f} cm   z = {d_z*100:4.1f} cm")
    wyniki[name] = (n_ok, n_stoi)

if viewer is not None:
    viewer.close()

# ===== diagnostyka limitu prędkości =====
st = robot_sim.VEL_STATS
print("\n===== LIMIT PRĘDKOŚCI W MUJOCO =====")
print(f"limit _VEL = {robot_sim._VEL} rad/s   tryb: {VEL_MODE}  (interp: liczba = kroki, w których robot i tak jechał szybciej niż limit)")
print(f"kroki fizyki z przycięciem: {st['przyciete']} / {st['kroki']}  ({100*st['przyciete']/max(st['kroki'],1):.2f} %)")
print("max prędkość stawów przed przycięciem [rad/s]:", np.round(st["max_raw"], 2))

if GRASPS:
    g = np.array(GRASPS)
    print(f"\n===== MUJOCO: {len(g)} chwytów | dz mediana {np.median(g[:,0])*100:4.1f} cm "
          f"(min {g[:,0].min()*100:4.1f}, max {g[:,0].max()*100:4.1f}) | "
          f"dxy mediana {np.median(g[:,1])*100:4.1f} cm | |yaw| mediana {np.median(np.abs(g[:,2])):4.1f}° =====")

if ETAPY:
    from collections import Counter
    print("\n===== GDZIE SIĘ PSUJE =====")
    for k, v in sorted(Counter(ETAPY).items()):
        print(f"  {k:60s}: {v}")

if OTWARCIA:
    print("\n===== CHWILA OTWARCIA PALCÓW NAD WIEŻĄ (średnie) =====")
    for grupa in ("SU", "S)", "C1", "C2"):
        r = [o for e, o in OTWARCIA if e == grupa]
        if r:
            print(f"  {grupa}: n={len(r):2d}  przesunięcie {np.mean([o['przesuniecie_cm'] for o in r]):4.1f} cm   "
                  f"wysokość {np.mean([o['wysokosc_cm'] for o in r]):4.1f} cm   "
                  f"prędkość w bok {np.mean([o['v_bok_cm_s'] for o in r]):5.1f} cm/s   "
                  f"w dół {np.mean([o['v_dol_cm_s'] for o in r]):5.1f} cm/s")

if PROBY:
    print("\n===== CHWYT: PIERWSZA PRÓBA W EPIZODZIE (mediany) =====")
    for grupa in ("SU", "A)", "B)", "S)", "C1", "C2"):
        r = [pr[0] for e, pr in PROBY if e == grupa and pr]
        if not r:
            continue
        typy = [typ_proby(p["min_palce"]) for p in r]
        print(f"  {grupa}: n={len(r):2d}  dz {np.median([p['dz'] for p in r])*100:4.1f} cm   "
              f"dxy {np.median([p['dxy'] for p in r])*100:4.1f} cm   |yaw| {np.median([abs(p['yaw']) for p in r]):4.1f}°   "
              f"palce min {np.median([p['min_palce'] for p in r])*100:4.2f} cm   'zamknij' {np.median([p['kroki'] for p in r]):4.0f} kroków   "
              f"| W_PALCACH {typy.count('W_PALCACH')}, PUSTO {typy.count('PUSTO')}, ZA_KROTKO {typy.count('ZA_KROTKO')}")
    print("\n===== CHWYT: WSZYSTKIE PRÓBY (ile jakich) =====")
    for grupa in ("SU", "A)", "B)"):
        r = [p for e, pr in PROBY if e == grupa for p in pr]
        if r:
            typy = [typ_proby(p["min_palce"]) for p in r]
            print(f"  {grupa}: {len(r):4d} prób  W_PALCACH {typy.count('W_PALCACH')}, PUSTO {typy.count('PUSTO')}, "
                  f"ZA_KROTKO {typy.count('ZA_KROTKO')}")

if HOLDS:
    print("\n===== GŁĘBOKOŚĆ TRZYMANIA W DRODZE (TCP nad środkiem kostki, mediana; 0 = głęboko, 2.5 = krawędź) =====")
    for grupa in ("SU", "B)", "S)", "C1", "C2"):
        r = [h for e, h in HOLDS if e == grupa]
        if r:
            print(f"  {grupa}: n={len(r):2d}  {np.median(r)*100:4.1f} cm   (min {min(r)*100:4.1f}, max {max(r)*100:4.1f})")

if ZGUBIENIA:
    r = [z for e, z in ZGUBIENIA if e == "B)"]
    if r:
        n_open = sum(z["komenda"] == "OTWÓRZ" for z in r)
        print(f"\n===== B) ZGUBIENIE W DRODZE: n={len(r)} =====")
        print(f"  palce były OTWARTE (robot sam puścił): {n_open}   palce ZAMKNIĘTE (wyślizgnęła się): {len(r) - n_open}")
        print(f"  mediany: TCP jedzie {np.median([z['v_tcp_cm_s'] for z in r]):5.1f} cm/s, "
              f"po {np.median([z['po_ilu_krokach'] for z in r]):4.0f} krokach od podniesienia, "
              f"chwyt był dz {np.nanmedian([z['dz_przy_chwycie_cm'] for z in r]):4.1f} cm, "
              f"dxy {np.nanmedian([z['dxy_przy_chwycie_cm'] for z in r]):4.1f} cm")

print(f"\n===== BEZPIECZNIK PUSZCZANIA (RELEASE_GUARD = {RELEASE_GUARD}, GUARD_V = {GUARD_V*100:.0f} cm/s) =====")
print(f"  puszczenia wstrzymane: {GSTAT['puszczenia_z_czekaniem']}   kroki czekania razem: {GSTAT['wstrzymane_kroki']}   "
      f"limit {GUARD_MAX} kroków osiągnięty: {GSTAT['limit_czekania']}")

print(f"\n===== KWATERNIONY (QUAT_FIX = {QUAT_FIX}) =====")
print(f"  chwytak: przeskoki znaku (q -> -q między krokami): {QSTAT['ee_flips']}   "
      f"w (4. liczba) od {QSTAT['w_min']:.3f} do {QSTAT['w_max']:.3f}")
print(f"  cube_2 : przeskoki znaku: {QSTAT['c2_flips']}")
print(f"  kwaternion chwytaka w kroku 0 (x, y, z, w), pierwsze 3 układy: {QSTAT['start'][:3]}")
from collections import Counter as _C
_f = _C(e for e, fl in QSTAT["etapy_flip"] if fl)
_a = _C(e for e, fl in QSTAT["etapy_flip"])
print("  epizody z przeskokiem / wszystkie, wg wyniku: "
      + ", ".join(f"{k}: {_f.get(k, 0)}/{v}" for k, v in sorted(_a.items())))

print("\n===== PODSUMOWANIE =====")
for name, (n_ok, n_stoi) in wyniki.items():
    print(f"{name:10s}: SUKCES (xy < 2 cm): {n_ok}/{len(cases)}   |   wieża stoi (xy < 3 cm): {n_stoi}/{len(cases)}")