import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("franka_emika_panda/franka_stack.xml")
data = mujoco.MjData(model)

# pozycja home
HOME = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.785])
_VEL = 1.0
_FINGER_VEL = 0.05
KP = 800.0
KD = 120.0
SIM_DT = 0.0005
P_STEP = 40
TCP = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
CUBES = ["cube_1", "cube_2", "cube_3"]


model.opt.timestep = SIM_DT
model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
model.opt.impratio = 10.0

for i in range(7):
    model.actuator_gainprm[i, 0 ] = KP
    model.actuator_biasprm[i,:3] = [0,-KP,-KD]

model.dof_armature[:7] =0.0

model.actuator_gainprm[7,0] = 4000
model.actuator_biasprm[7,:3] = [0,-4000,-200]
model.actuator_ctrlrange[7] = [0, 0.04]
model.actuator_forcerange[7] = [-400,400]

for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i ) or ""
    if name and name != "world" and not name.startswith("cube"):
        model.body_gravcomp[i] = 1.0

for i in range(model.ngeom):
    name = mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[i]) or ""
    if name.startswith("cube") or name in ("left_finger","right_finger"):
        model.geom_solref[i] = [0.002,1.0]
        model.geom_solimp[i] = [0.95, 0.99, 0.001, 0.5, 2.0 ]


def reset_scene(cubes_pos):
    mujoco.mj_resetData(model,data)
    data.qpos[:7] = HOME
    data.qpos[7:9] = 0.04
    for cube in CUBES:
        x,y,z,yaw = cubes_pos[cube]
        adr = model.jnt_qposadr[model.joint(cube+"_joint").id]
        data.qpos[adr:adr+3] = [x, y, z]
        data.qpos[adr+3:adr+7] = [np.cos(yaw/2),0,0, np.sin(yaw/2)]
    data.ctrl[:7] = HOME
    data.ctrl[7] = 0.04
    mujoco.mj_forward(model,data)

def state():
    q = data.qpos[:7].copy()
    q_vel = data.qvel[:7].copy()
    fingers = float(data.qpos[7])
    finger_vel = float(data.qvel[7])
    tcp_pos = data.site_xpos[TCP].copy()
    q_wxyz = np.zeros(4)
    mujoco.mju_mat2Quat(q_wxyz, data.site_xmat[TCP])
    tcp_quat = q_wxyz[[1,2,3,0]]
    return {"q": q, "q_vel": q_vel, "fingers": fingers, "fingers_vel": finger_vel, "tcp_pos":tcp_pos, "tcp_quat": tcp_quat}

def cubes_position_table():
    cubes_out = {}
    for cube in CUBES:
        pos = data.body(cube).xpos.copy()
        quat = data.body(cube).xquat[[1,2,3,0]].copy()
        cubes_out[cube] = (pos,quat)
    return cubes_out
    
    
def movement(q_goal, gripper_open):
    data.ctrl[7] = 0.04 if gripper_open else 0.0
    max_step = _VEL * SIM_DT
    max_finger = _FINGER_VEL*SIM_DT
    for _ in range(P_STEP):
        goal = data.ctrl[:7].copy()
        data.ctrl[:7] = goal + np.clip(q_goal- goal, -max_step, max_step)

        fingers_before = data.qpos[7:9].copy()
        mujoco.mj_step(model,data)
        move = data.qpos[7:9] - fingers_before
        data.qpos[7:9] = fingers_before + np.clip(move, -max_finger, max_finger)
        data.qvel[7:9] = np.clip(data.qvel[7:9],-_FINGER_VEL,_FINGER_VEL)
        

# test
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    reset_scene({
        "cube_1": [0.45, 0.05, 0.025, 0.0],
        "cube_2": [0.55, -0.05, 0.025, 0.5],
        "cube_3": [0.42, -0.08, 0.025, 1.0],
    })
    print("stan po resecie:", state())
    print("kostki:", cubes_position_table())
    cel = HOME.copy()
    cel[0] += 0.3
    for _ in range(50):
        movement(cel, gripper_open=True)
    print("staw 1 po 1 s:", round(state()["q"][0], 3), "(cel 0.3)")