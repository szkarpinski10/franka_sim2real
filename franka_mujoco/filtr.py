import os
import numpy as np
import mujoco

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "franka_emika_panda", "franka_stack.xml")


TCP_W_DLONI = 0.1034                                         # TCP = dłoń + 10.34 cm wzdłuż osi z (ta sama orientacja)


def punkty_kamery(T_tcp_cam, wymiary=(0.12, 0.04, 0.035)):
    """Punkty prostopadłościanu wokół kamery (z zapasem na uchwyt) w układzie DŁONI. Kamera: x w bok, y w dół,
    z = kierunek patrzenia; obiektyw na przedniej ścianie, korpus za nim (z ujemne)."""
    sx, sy, sz = wymiary
    x, y, z = np.meshgrid(np.linspace(-sx / 2, sx / 2, 7), np.linspace(-sy / 2, sy / 2, 3), np.linspace(-sz, 0.005, 3))
    p = np.c_[x.ravel(), y.ravel(), z.ravel(), np.ones(x.size)]
    T_dlon_tcp = np.eye(4)
    T_dlon_tcp[2, 3] = TCP_W_DLONI
    return (T_dlon_tcp @ np.asarray(T_tcp_cam) @ p.T)[:3].T


class Filtr_CBF:
    def __init__(self, stol_z=0.0, h_min=0.006, alfa=0.2, t_br=0.12, vel=0.8, dt=0.02, kamery=()):
        """kamery = poza każdej kamery na chwytaku względem TCP (4 x 4) – ich bryły dochodzą do punktów dłoni."""
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
            if b == mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "hand"):
                punkty += [punkty_kamery(T) for T in kamery]  # kamer nie ma w modelu MuJoCo
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
