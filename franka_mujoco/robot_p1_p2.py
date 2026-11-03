import time
import numpy as np
import mujoco
import mujoco.viewer
import torch
import robot_mujoco as robot
#import robot_franka as robot
from filtr import Filtr_CBF
from observations_combine import obs_combine

VIEWER = True
if VIEWER: viewer = mujoco.viewer.launch_passive(robot.model, robot.data)

SCALE = 0.03
N_STEPS = 1000

POLICY_PATH = "policy1/2026-11-01_16-15-43_800/exported/policy.pt"
policy = torch.jit.load(POLICY_PATH,map_location = "cpu").eval()

TABLE = robot.model.geom("table").id
ROBOT = ("link0","link1","link2","link3","link4","link5","link6","link7","hand","left_finger","right_finger")

_FILTR = True
filtr = Filtr_CBF(h_min =0.006, alfa = 0.5, t_br = 0.12)

def cube_2_on_cube_1():
    
    cubes = robot.cubes_position_table()
    cube_1 = cubes["cube_1"][0]
    cube_2 = cubes["cube_2"][0]
    xy = np.linalg.norm(cube_2[:2]- cube_1[:2])
    z = cube_2[2] - cube_1[2]
    return  xy<0.03 and 0.04<z<0.06

def rand_cubes_on_table(n, seed = 0, min_sep = 0.1):
    rng = np.random.default_rng(seed)
    cases = []

    while len(cases)< n:
        x = rng.uniform(0.35,0.55,3)
        y = rng.uniform(-0.20,0.20,3)

        dist_1_2 = np.hypot(x[0]-x[1],y[0]-y[1])
        dist_2_3 = np.hypot(x[1]-x[2],y[1]-y[2])
        dist_1_3 = np.hypot(x[0]-x[2],y[0]-y[2])
        if min(dist_1_2, dist_2_3, dist_1_3)<min_sep:
            continue
        
        yaw = rng.uniform(-np.pi, np.pi, 3)

        cases.append({
            "cube_1": [x[0],y[0], 0.025, yaw[0]],
            "cube_2": [x[1],y[1], 0.025, yaw[1]],
            "cube_3": [x[2],y[2], 0.025, yaw[2]],
        })
    return cases

def table_contact_def():
    for i in robot.data.contact[:robot.data.ncon]:
        cialo1 = robot.model.body(robot.model.geom_bodyid[i.geom1]).name
        cialo2 = robot.model.body(robot.model.geom_bodyid[i.geom2]).name
        if (i.geom1 == TABLE and cialo2 in ROBOT) or (i.geom2 == TABLE and cialo1 in ROBOT):
            return True
    return False

def epizod(policy):
    last_action = np.zeros(8, dtype = np.float32)
    table_contact = False
    for step in range (N_STEPS):
        state = robot.state()
        cubes_pos = robot.cubes_position_table()

        observations = obs_combine(state["q"],state["q_vel"],state["fingers"],
                                            state["fingers_vel"],last_action, state["tcp_pos"],state["tcp_quat"],cubes_pos)
        
        with torch.no_grad():
            action = policy(torch.from_numpy(observations).unsqueeze(0)).squeeze(0).numpy()
        
        q_goal = state["q"] + SCALE * action[:7]
        gripper_open = action[7] >= 0
        step = np.clip(q_goal - robot.data.ctrl[:7], -robot._VEL * robot.DT, robot._VEL * robot.DT)
        if _FILTR:
            step = filtr.filtruj(step, state["q"], state["q_vel"], state["fingers"])
        robot.movement(robot.data.ctrl[:7] + step, gripper_open)
        if table_contact_def(): table_contact = True
        if VIEWER:
            viewer.sync()
            time.sleep(robot.DT)

        last_action = action.copy()

    return cube_2_on_cube_1(), table_contact

if __name__ == "__main__":
    SEED = 7
    N_ITERATIONS = 100
    rand_cases = rand_cubes_on_table(N_ITERATIONS, seed = SEED)
    n_success = 0
    n_table_contact = 0
    for i, case in enumerate(rand_cases):
        robot.reset_scene(case)
        success, table_contact = epizod(policy)
        n_success += success
        n_table_contact+= table_contact
        print(f"Case {i}: {"sukces" if success else "porazka"}, kontakty ze stolem {"tak" if table_contact else "nie"} ")

    print(f"\n sukcesow: {n_success}/{N_ITERATIONS}, kontakow ze stolem: {n_table_contact}/{N_ITERATIONS}")