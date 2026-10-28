import numpy as np
import torch
import robot_mujoco as robot
from observations_combine import obs_combine

SCALE = 0.03
N_STEPS = 1000

#path = "nowe_policy/2026-10-25_20-23-50/exported/policy.pt"
#path = "stack/policy1/exported/policy.pt"
import sys
# python robot_p1_p2.py <policy.pt> <prędkość palców> <seed układów>
path = sys.argv[1] if len(sys.argv) > 1 else "nowe_policy/2026-10-26_18-47-17/exported/policy.pt"
if len(sys.argv) > 2:
    robot._FINGER_VEL = float(sys.argv[2])
policy = torch.jit.load(path,map_location = "cpu").eval()

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


def epizod(policy):
    last_action = np.zeros(8, dtype = np.float32)
    
    for step in range (N_STEPS):
        state = robot.state()
        cubes_pos = robot.cubes_position_table()

        observations = obs_combine(state["q"],state["q_vel"],state["fingers"],
                                            state["fingers_vel"],last_action, state["tcp_pos"],state["tcp_quat"],cubes_pos)
        
        with torch.no_grad():
            action = policy(torch.from_numpy(observations).unsqueeze(0)).squeeze(0).numpy()
        
        q_goal = state["q"] + SCALE * action[:7]
        gripper_open = action[7] >= 0
        robot.movement(q_goal,gripper_open)

        last_action = action.copy()

    return cube_2_on_cube_1()

if __name__ == "__main__":
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    cases = rand_cubes_on_table(100, seed=seed)
    n_stoi = 0
    for i, case in enumerate(cases):
        robot.reset_scene(case)
        stoi = epizod(policy)
        n_stoi += stoi
        print(f"układ {i:2d}: {'STOI' if stoi else 'NIE'}")
    print(f"\nwieża stoi: {n_stoi}/100   ({path}, palce {robot._FINGER_VEL}, seed {seed})")