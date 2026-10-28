### Plik do połączenia obserwacji na prawdziwego robota

import numpy as np

DEFAULT_Q = np.array([0.0,-0.3,0.0,-2.0,0.0,1.7,0.785])
FINGERS_OPEN = 0.04
CUBES = ["cube_1","cube_2","cube_3"]

def quat_to_matrix(q):
    x,y,z,w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])

def ee_cube_yaw(ee_quat, cubes:dict):
    
    rotation_hand = quat_to_matrix(ee_quat)
    hand_yaw = np.arctan2(rotation_hand[1,1],rotation_hand[0,1]) 

    out:list = []
    
    for i in CUBES:
        rotation_cubes = quat_to_matrix(cubes[i][1])
        vertical_idx = int(np.argmax(np.abs(rotation_cubes[2,:]))) 
        horizontal_idx = 1 if vertical_idx == 0 else 0
        horizontal_axis = rotation_cubes[:, horizontal_idx] 
        cube_yaw = np.arctan2(horizontal_axis[1],horizontal_axis[0]) 
        diff = hand_yaw - cube_yaw
        out = out + [np.sin(4.0*diff),np.cos(4.0*diff)] 
    
    return np.array(out)

def obs_combine(joint_q,joint_vel,finger,finger_vel,last_action,ee_pos,ee_quat,cubes: dict):
    """
    joint_pos_rel (9) - pozycje 7 stawów i 2 palców 
    joint_vel_rel (9) - prędkości 7 stawów i 2 palców
    last_action  (8) - ostatnia wykonana akcja 
    eef_pos (3) - pozycja ee 
    eef_quat (4) - orientacja ee 
    object_pos (39) - 
    cube_1 pos,
        cube_1 quat,
        cube_2 pos,
        cube_2 quat,
        cube_3 pos,
        cube_3 quat,
        gripper to cube_1,
        gripper to cube_2,
        gripper to cube_3,
        cube_1 to cube_2,
        cube_2 to cube_3,
        cube_1 to cube_3,
    ee_cube_yaw (6) 
    """
    cube_1_pos, cube_1_quat = cubes["cube_1"] 
    cube_2_pos, cube_2_quat = cubes["cube_2"]
    cube_3_pos, cube_3_quat = cubes["cube_3"]

    joint_pos_rel = np.concatenate([joint_q - DEFAULT_Q, [finger- FINGERS_OPEN], [finger-FINGERS_OPEN]])
    joint_vel_rel = np.concatenate([joint_vel, [finger_vel], [finger_vel]])

    object_pos = np.concatenate([
        cube_1_pos, cube_1_quat,
        cube_2_pos, cube_2_quat,
        cube_3_pos, cube_3_quat,
        cube_1_pos - ee_pos,
        cube_2_pos - ee_pos,
        cube_3_pos - ee_pos,
        cube_1_pos - cube_2_pos,
        cube_2_pos - cube_3_pos,
        cube_1_pos - cube_3_pos,
        ])

    
    
    
    obs_comb = np.concatenate([joint_pos_rel,joint_vel_rel,last_action,ee_pos,ee_quat, object_pos, ee_cube_yaw(ee_quat, cubes)]).astype(np.float32)
    return obs_comb