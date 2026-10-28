import numpy as np
#import panda_py
#from panda_py import libfranka, controllers
import time
import threading
from observations_combine import quat_to_matrix

FAKE = True          

if FAKE:
    import fake_franka as panda_py
    from fake_franka import libfranka, controllers
else:
    import panda_py
    from panda_py import libfranka, controllers


_VEL = 1.0
DT = 0.02

IP = "172.16.0.2"
panda = panda_py.Panda(IP)
gripper = libfranka.Gripper(IP)          # tylko wątek chwytaka: move/grasp (blokują)
gripper_reader = libfranka.Gripper(IP)   # tylko odczyt szerokości (read_once) z pętli głównej
# gdyby na robocie drugie połączenie z chwytakiem nie działało: gripper_reader = gripper

HOME = np.array([0.0,-0.3,0.0,-2.0,0.0,1.7,0.785])
V_HOME = 0.3

# limity stawów Franka Emika Panda [rad] + zapas, żeby cel nie wyszedł poza zakres (robot stanąłby z błędem)
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
Q_MARGIN = 0.05

prev_width = None
prev_width_t = None


ctrl_move = None
q_command = None
next_step = None

gripper_policy_open = True
gripper_is_open = True


CUBES_START = {
    "cube_1": np.array([0.42,0.08,0.025]),
    "cube_2": np.array([0.42,-0.08,0.025]),
    "cube_3": np.array([0.52,0.00,0.025]),   # trening losował x 0.35-0.55
}

# kostki stawiamy obrócone o 45 st. (równo z palcami w HOME) - przy 0 st. policy ma problem
CUBES_YAW = np.pi/4
QUAT_ZERO = np.array([0.0,0.0,np.sin(CUBES_YAW/2),np.cos(CUBES_YAW/2)])
red_pos = CUBES_START["cube_2"].copy()
red_quat = QUAT_ZERO.copy()
# środek trzymanej kostki względem TCP w układzie ręki [m] (średnia z testów w MuJoCo; na robocie sprawdzić)
RED_IN_HAND = np.array([-0.0085, 0.0, -0.0087])

def state():
    global prev_width, prev_width_t
    state = panda.get_state()
    width = gripper_reader.read_once().width

    # prędkość jednego palca z różnicy szerokości (w treningu policy widzi prędkość palców)
    t = time.perf_counter()
    finger_vel = 0.0
    if prev_width is not None and t > prev_width_t:
        finger_vel = (width - prev_width) / 2 / (t - prev_width_t)
    prev_width, prev_width_t = width, t

    return{
        "q": np.array(state.q),
        "q_vel": np.array(state.dq),
        "fingers": width/2,
        "fingers_vel": finger_vel,
        "tcp_pos": np.array(panda.get_position()).flatten(),
        "tcp_quat":np.array(panda.get_orientation()).flatten(),
    }
    
    
def reset_scene(cubes_pos=None):
    
    global gripper_policy_open, gripper_is_open, red_pos, red_quat, prev_width, prev_width_t
    stop()
    panda.recover()
    q0 = np.array(panda.get_state().q)
    dq = HOME - q0
    T = max(1.875 * np.max(np.abs(dq))/V_HOME, 1.0)

    ctrl = controllers.JointPosition()
    panda.start_controller(ctrl)
    with panda.create_context(frequency = 1000, max_runtime = T+0.5) as ctx:
        while ctx.ok():
            t = ctrl.get_time()
            s = min(t/T,1.0)
            b = 10*s**3-15*s**4+6*s**5
            ctrl.set_control(q0+dq*b)
    panda.stop_controller()

    # otwarcie przez wątek chwytaka (bezpośrednie gripper.move mogłoby się zderzyć z trwającym w wątku grasp)
    gripper_policy_open = True
    t0 = time.perf_counter()
    while not gripper_is_open and time.perf_counter() - t0 < 3.0:
        time.sleep(0.01)
    prev_width, prev_width_t = None, None

    red_pos = CUBES_START["cube_2"].copy()
    red_quat = QUAT_ZERO.copy()
    if FAKE:
        panda_py.place_cubes(CUBES_START, CUBES_YAW)
    
def movement(q_goal,gripper_open):
    global ctrl_move, q_command, next_step, gripper_policy_open

    
    if ctrl_move is None:
        ctrl_move = controllers.JointPosition()
        panda.start_controller(ctrl_move)
        q_command = np.array(panda.get_state().q)
        next_step = time.perf_counter()

    step = np.clip(np.asarray(q_goal)-q_command, -_VEL*DT, _VEL*DT)
    q_new = np.clip(q_command + step, Q_MIN + Q_MARGIN, Q_MAX - Q_MARGIN)
    step = q_new - q_command
    q_command = q_new
    ctrl_move.set_control(q_command, step/DT)

    gripper_policy_open = bool(gripper_open)

    next_step = next_step +DT
    wait = next_step - time.perf_counter()
    if wait >0:
        time.sleep(wait)
    return robot_ok()


def robot_ok():
    """False, gdy robot zatrzymał się z błędem (kolizja / reflex / wciśnięty stop). Nazwy trybów sprawdzić na robocie."""
    mode = getattr(panda.get_state(), "robot_mode", None)
    if mode is None:
        return True
    return not str(mode).endswith(("kReflex", "kUserStopped"))

def stop():
    global ctrl_move
    if ctrl_move is not None:
        panda.stop_controller()
        ctrl_move = None

def gripper_thr():
    global gripper_is_open
    while True:
        if gripper_policy_open != gripper_is_open:
            # komenda zapamiętana PRZED ruchem: move/grasp czekają, aż palce staną, a policy w tym czasie może zmienić zdanie
            target_open = gripper_policy_open
            if target_open:
                gripper.move(0.08,0.1)
            else:
                gripper.grasp(0.0, 0.1, 20.0, 0.08, 0.08)
            gripper_is_open = target_open
        time.sleep(0.005)

threading.Thread(target = gripper_thr, daemon = True).start()

def cubes_position_table(s=None):
    global red_pos, red_quat
    if s is None:
        s = state()
    gripper_width = s["fingers"]*2
    # kostka w chwytaku: palce zaciśnięte na kostce (5 cm). Położenie kostki względem palców jest prawie stałe,
    # więc bierzemy zmierzone przesunięcie RED_IN_HAND zamiast liczyć je od pozycji sprzed chwytu
    # (robot często przesuwa kostkę, zanim ją złapie - wtedy błąd był 4-8 cm)
    if 0.04 < gripper_width < 0.06:
        rotation_tcp = quat_to_matrix(s["tcp_quat"])
        red_pos = s["tcp_pos"].copy() + rotation_tcp @ RED_IN_HAND
        hand_yaw = np.arctan2(rotation_tcp[1,1],rotation_tcp[0,1])
        yaw = (hand_yaw+np.pi/4) % (np.pi/2) - np.pi/4
        red_quat = np.array([0.0,0.0, np.sin(yaw/2),np.cos(yaw/2)])
    return {
        "cube_1": (CUBES_START["cube_1"],QUAT_ZERO),
        "cube_2": (red_pos,red_quat),
        "cube_3": (CUBES_START["cube_3"],QUAT_ZERO),
    }