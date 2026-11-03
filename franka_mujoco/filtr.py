import os
import numpy as np
import mujoco

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "franka_emika_panda", "franka_stack.xml")


class Filtr_CBF:
    def __init__(self, stol_z=0.0, h_min=0.006, alfa=0.2, t_br=0.12, vel=0.8, dt=0.02):
        self.m = mujoco.MjModel.from_xml_path(XML)
        self.d = mujoco.MjData(self.m)
        self.stol_z, self.h_min, self.alfa, self.t_br, self.lim = stol_z, h_min, alfa, t_br, vel * dt
        self.n_aktywny = 0
        self.ciala = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
                      for n in ("left_finger", "right_finger", "hand", "link5", "link6", "link7")]
        # punkty ciała (w układzie ciała): wierzchołki wszystkich jego siatek kolizyjnych
        self.pkt = {}
        for b in self.ciala:
            punkty = []
            for g in range(self.m.ngeom):
                if self.m.geom_bodyid[g] == b and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH \
                        and self.m.geom_contype[g] | self.m.geom_conaffinity[g]:
                    mid = self.m.geom_dataid[g]
                    v = self.m.mesh_vert[self.m.mesh_vertadr[mid]:self.m.mesh_vertadr[mid] + self.m.mesh_vertnum[mid]]
                    R = np.zeros(9); mujoco.mju_quat2Mat(R, self.m.geom_quat[g])
                    punkty.append(self.m.geom_pos[g] + v @ R.reshape(3, 3).T)
            self.pkt[b] = np.vstack(punkty)

    def filtruj(self, dq, q, dq_zmierzone, palec):
        # 1. kinematyka prosta dla faktycznej pozycji robota
        self.d.qpos[:7] = q; self.d.qpos[7:9] = palec
        mujoco.mj_kinematics(self.m, self.d); mujoco.mj_comPos(self.m, self.d)
        for b in self.ciala:
            # 2. najniższy punkt ciała i jego Jakobian (jak zmiana stawów zmienia jego wysokość)
            pts = self.d.xpos[b] + self.pkt[b] @ self.d.xmat[b].reshape(3, 3).T
            p = pts[np.argmin(pts[:, 2])]
            jac = np.zeros((3, self.m.nv)); mujoco.mj_jac(self.m, self.d, jac, None, p, b)
            J = jac[2, :7]
            # 3. odległość od blatu minus droga hamowania
            v_dol = max(0.0, -(J @ dq_zmierzone))
            h = p[2] - self.stol_z - self.h_min - self.t_br * v_dol
            # 4. jeśli ruch obniża palec za szybko, usuwamy tylko tę część ruchu
            if J @ dq < -self.alfa * h:
                dq = dq + (-self.alfa * h - J @ dq) * J / (J @ J)
                self.n_aktywny += 1
        # 5. nie przekraczamy limitu prędkości (skalujemy cały wektor)
        return dq / max(1.0, np.max(np.abs(dq)) / self.lim)
