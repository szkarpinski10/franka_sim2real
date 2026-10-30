"""UDAWANA FRANKA: te same funkcje co panda_py, ale w środku symulacja MuJoCo (z Twojego robot_mujoco.py).

Po co: żeby przetestować robot_real.py W DOMU, bez robota. W robot_real.py ustawiasz FAKE = True
i zamiast prawdziwego panda_py ładuje się ten plik. Kod robot_real.py zostaje BEZ ZMIAN.

Co udaje:
    panda_py.Panda(IP)              -> get_state().q / .dq, get_position(), get_orientation(), get_pose(),
                                       recover(), start_controller(), stop_controller(), create_context(), teaching_mode()
    controllers.JointPosition()     -> set_control(q, dq), get_time()
    libfranka.Gripper(IP)           -> read_once().width, move(width, speed), grasp(...), homing(), stop()

Jak działa: fizyka MuJoCo liczy się w tle w CZASIE RZECZYWISTYM (jak prawdziwy robot - nie czeka na Twój program).
Kontroler JointPosition ustawia cel regulatora PD ramienia (jak w robot_mujoco.py).
Chwytak: prędkość z komendy (speed = prędkość rozstawu, czyli palec jedzie speed/2), komendy CZEKAJĄ aż palce staną
(jak prawdziwy libfranka). Limit prędkości palców działa na PRZESUNIĘCIE w każdym kroku fizyki
(samo obcinanie qvel po kroku nie wystarczało - palce i tak jechały kilka razy za szybko).

Dodatkowo (tylko do testów):
    place_cubes(pozycje)   - postaw kostki w podanych miejscach ([x, y, z] z jednym yaw albo [x, y, z, yaw] osobno)
    true_cubes()           - PRAWDZIWE pozycje kostek z fizyki (do sprawdzenia, czy wieża stoi)
    zmierz_kostki(...)     - UDAWANA KAMERA: prawdziwe pozycje + szum, w formacie jak camera/kamera_kostki.py
"""
import threading
import time
from types import SimpleNamespace

import numpy as np
import mujoco
import robot_mujoco as sim          # Twoja symulacja: model, data, ustawienia fizyki (twarde kontakty, PD, grawitacja...)

VIEW = False                       # True = okno MuJoCo (tylko na zdalnym pulpicie, nie przez SSH)
if VIEW:
    import mujoco.viewer

_lock = threading.Lock()            # fizyka liczy się w osobnym wątku -> odczyty i zapisy pod kłódką
_ctrl_active = None                 # aktualnie działający kontroler JointPosition (None = żaden)
_finger_v = 0.05                    # limit prędkości JEDNEGO palca [m/s] (ustawiany przez komendy chwytaka)
_started = False


# ===================== fizyka w tle (czas rzeczywisty) =====================
def _physics_loop():
    viewer = mujoco.viewer.launch_passive(sim.model, sim.data) if VIEW else None
    t0 = time.perf_counter()
    sim_t = 0.0
    last_view = 0.0
    while True:
        now = time.perf_counter() - t0
        if now - sim_t > 0.1:                       # komputer nie nadąża -> nie goń w nieskończoność
            sim_t = now - 0.1
        with _lock:
            n = 0
            while sim_t < now and n < 80:           # dogoń zegar (najwyżej 80 kroków = 40 ms naraz)
                if _ctrl_active is not None and _ctrl_active._q is not None:
                    sim.data.ctrl[:7] = _ctrl_active._q
                fingers_before = sim.data.qpos[7:9].copy()        # pozycja palców przed krokiem
                mujoco.mj_step(sim.model, sim.data)
                lim = _finger_v * sim.SIM_DT                       # max przesunięcie palca w jednym kroku
                sim.data.qpos[7:9] = fingers_before + np.clip(sim.data.qpos[7:9] - fingers_before, -lim, lim)
                sim.data.qvel[7:9] = np.clip(sim.data.qvel[7:9], -_finger_v, _finger_v)
                sim_t += sim.SIM_DT
                n += 1
            if viewer is not None and now - last_view > 0.03:
                viewer.sync()
                last_view = now
        time.sleep(0.0005)


def _start_physics():
    global _started
    if not _started:
        with _lock:
            sim.data.qpos[:7] = sim.HOME
            sim.data.qpos[7:9] = 0.04
            sim.data.ctrl[:7] = sim.HOME
            sim.data.ctrl[7] = 0.04
            mujoco.mj_forward(sim.model, sim.data)
        threading.Thread(target=_physics_loop, daemon=True).start()
        _started = True


# ===================== ramię =====================
class JointPosition:
    """Udawany controllers.JointPosition: trzyma cel stawów, fizyka ciągnie do niego regulatorem PD."""

    def __init__(self):
        self._q = None
        self._t_start = time.perf_counter()

    def set_control(self, position, velocity=None):
        self._q = np.array(position, dtype=float).copy()

    def get_time(self):
        return time.perf_counter() - self._t_start


class _Context:
    """Udawany panda.create_context(frequency, max_runtime): metronom pętli while ctx.ok()."""

    def __init__(self, frequency, max_runtime):
        self.period = 1.0 / frequency
        self.max_runtime = max_runtime
        self.t_start = time.perf_counter()
        self.t_next = self.t_start

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def ok(self):
        self.t_next += self.period
        wait = self.t_next - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        return (time.perf_counter() - self.t_start) < self.max_runtime


class Panda:
    def __init__(self, ip=None):
        _start_physics()

    def get_state(self):
        with _lock:
            return SimpleNamespace(q=sim.data.qpos[:7].copy(), dq=sim.data.qvel[:7].copy())

    def get_position(self):
        with _lock:
            return sim.data.site_xpos[sim.TCP].copy()

    def get_orientation(self):
        q_wxyz = np.zeros(4)
        with _lock:
            mujoco.mju_mat2Quat(q_wxyz, sim.data.site_xmat[sim.TCP])
        return q_wxyz[[1, 2, 3, 0]]                     # (x, y, z, w) jak panda_py domyślnie

    def get_pose(self):
        with _lock:
            T = np.eye(4)
            T[:3, :3] = sim.data.site_xmat[sim.TCP].reshape(3, 3)
            T[:3, 3] = sim.data.site_xpos[sim.TCP]
        return T

    def recover(self):
        pass

    def teaching_mode(self, on):
        pass

    def start_controller(self, ctrl):
        global _ctrl_active
        with _lock:
            ctrl._q = sim.data.qpos[:7].copy()          # start: cel = tam, gdzie robot stoi
            ctrl._t_start = time.perf_counter()
            _ctrl_active = ctrl

    def stop_controller(self):
        global _ctrl_active
        with _lock:
            sim.data.ctrl[:7] = sim.data.qpos[:7]       # po wyłączeniu robot stoi tam, gdzie jest
            _ctrl_active = None

    def create_context(self, frequency=1000, max_runtime=1.0):
        return _Context(frequency, max_runtime)


# ===================== chwytak =====================
class Gripper:
    def __init__(self, ip=None):
        _start_physics()

    def read_once(self):
        with _lock:
            w = 2.0 * float(sim.data.qpos[7])
        return SimpleNamespace(width=w, max_width=0.08)

    def _go(self, width, speed, timeout=3.0):
        """Ustaw cel palców i CZEKAJ, aż palce staną (jak prawdziwy libfranka)."""
        global _finger_v
        with _lock:
            sim.data.ctrl[7] = float(np.clip(width / 2.0, 0.0, 0.04))
            _finger_v = max(speed / 2.0, 1e-3)
        t0 = time.perf_counter()
        time.sleep(0.03)
        still = 0
        while time.perf_counter() - t0 < timeout:
            with _lock:
                v = abs(float(sim.data.qvel[7]))
            still = still + 1 if v < 1e-3 else 0
            if still >= 10:                             # palce nie ruszają się od ~20 ms -> koniec ruchu
                break
            time.sleep(0.002)
        return self.read_once().width

    def move(self, width, speed):
        self._go(width, speed)
        return True

    def grasp(self, width, speed, force, epsilon_inner=0.005, epsilon_outer=0.005):
        w = self._go(width, speed)
        return (width - epsilon_inner) < w < (width + epsilon_outer)

    def homing(self):
        return self.move(0.08, 0.1)

    def stop(self):
        return True


# panda_py ma moduły "libfranka" i "controllers" - udajemy je tak samo
libfranka = SimpleNamespace(Gripper=Gripper)
controllers = SimpleNamespace(JointPosition=JointPosition)


# ===================== tylko do testów =====================
def place_cubes(cubes: dict, yaw: float = 0.0):
    """Postaw kostki w podanych miejscach, obrócone o yaw [rad] wokół pionu. cubes = {"cube_1": [x, y, z], ...}
    albo każda kostka z własnym obrotem: {"cube_1": [x, y, z, yaw], ...} (jak układy z robot_p1_p2.py)."""
    with _lock:
        for name, pos in cubes.items():
            adr = sim.model.jnt_qposadr[sim.model.joint(name + "_joint").id]
            vadr = sim.model.jnt_dofadr[sim.model.joint(name + "_joint").id]
            a = pos[3] if len(pos) > 3 else yaw
            sim.data.qpos[adr:adr + 3] = pos[:3]
            sim.data.qpos[adr + 3:adr + 7] = [np.cos(a / 2), 0, 0, np.sin(a / 2)]
            sim.data.qvel[vadr:vadr + 6] = 0.0
        mujoco.mj_forward(sim.model, sim.data)


def true_cubes_pose():
    """PRAWDZIWE pozycje i obroty kostek z fizyki, w tym samym formacie co robot_franka.cubes_position_table():
    {"cube_1": (pozycja [x,y,z], kwaternion [x,y,z,w]), ...}"""
    with _lock:
        return {n: (sim.data.body(n).xpos.copy(), sim.data.body(n).xquat[[1, 2, 3, 0]].copy())
                for n in ("cube_1", "cube_2", "cube_3")}


def true_cubes():
    """PRAWDZIWE pozycje kostek z fizyki: {"cube_1": array([x, y, z]), ...}"""
    with _lock:
        return {n: sim.data.body(n).xpos.copy() for n in ("cube_1", "cube_2", "cube_3")}


def zmierz_kostki(szum_xy=0.005, szum_z=0.002, szum_yaw=np.radians(2.0), rng=np.random):
    """UDAWANA KAMERA (jak KameraKostki.zmierz z camera/kamera_kostki.py): prawdziwa pozycja + szum [m] / [rad].
    Obrót tylko wokół pionu, w zakresie [0, 90 st.) – kostka wygląda tak samo co 90 st. (YAW_OD = 0 w kamera_kostki.py).
    Zwraca {"cube_1": (pozycja [x,y,z], kwaternion [x,y,z,w]), ...}"""
    out = {}
    for n, (pos, quat) in true_cubes_pose().items():
        x, y, z, w = quat
        R_x = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)])   # oś x kostki w bazie
        yaw = (np.arctan2(R_x[1], R_x[0]) + rng.normal(0.0, szum_yaw)) % (np.pi / 2)
        pos = pos + rng.normal(0.0, 1.0, 3) * [szum_xy, szum_xy, szum_z]
        out[n] = (pos, np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]))
    return out