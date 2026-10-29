"""kamera_kostki.py – pozycje i obroty kostek w układzie bazy robota z kamery na chwytaku.

Jako moduł (dla kodu robota):
    from kamera_kostki import KameraKostki
    kam = KameraKostki()                       # wczytuje kalibracja.json, włącza kamerę
    kostki = kam.zmierz(poza_tcp(panda))       # robot MUSI stać; {"cube_1": (pos, quat), ...} tylko widoczne kostki
    kam.zamknij()
Jako podgląd (przy robocie, robot stoi lub jest przesuwany ręką):  python kamera_kostki.py
    pokazuje na żywo pozycje kostek w układzie bazy – połóż kostkę w zmierzonym linijką miejscu i porównaj.
    python kamera_kostki.py --bez-robota   – bez robota, pozycje względem TCP (tylko do sprawdzenia wykrywania).

Kostki jak w Isaacu: cube_1 = niebieska (tagi 0-5), cube_2 = czerwona (6-11), cube_3 = zielona (12-17).
Środek kostki = środek tagu przesunięty o pół boku w głąb kostki (oś z tagu). Działa z każdej widocznej ściany.
Obrót: tylko wokół pionu (kostka leży na stole), jako kwaternion x, y, z, w (jak w robot_franka.py).
Wymaga:  kalibracja.json (z kalibruj_handeye.py),  pip install pyrealsense2 opencv-python pupil-apriltags
"""
import argparse
import json
import os

import cv2
import numpy as np
import pupil_apriltags
import pyrealsense2 as rs

DIR = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1280, 720, 30                                    # ten sam tryb co przy kalibracji (zbierz_handeye.py)
EXPOSURE = None                                              # None = automatyczna
CUBE_TAG = 0.035                                             # bok CZARNEGO kwadratu tagu na kostce [m] – WPISZ ZMIERZONY
HALF = 0.025                                                 # pół boku kostki [m]
MIN_MARGIN = 30                                              # słabsze wykrycia odrzucamy (pewność wykrycia tagu)
KOSTKI = {"cube_1": range(0, 6), "cube_2": range(6, 12), "cube_3": range(12, 18)}
KOLOR = {"cube_1": "niebieska", "cube_2": "czerwona", "cube_3": "zielona"}
# kostka wygląda tak samo co 90 st., więc obrót podajemy w zakresie [YAW_OD, YAW_OD + 90 st.).
# YAW_OD = 0 -> kostka postawiona pod 45 st. (CUBES_YAW) daje ok. 45 st., jak QUAT_ZERO w robot_franka.py.
# Kostka postawiona blisko 0 st. będzie skakać między ~0 a ~90 st. – stawiaj pod 45 st.
YAW_OD = 0.0


def poza_tcp(panda):
    """Świeża poza TCP (4 x 4). Bez działającej pętli sterowania panda.get_pose() potrafi oddawać stary stan,
    więc wtedy czytamy prosto z libfranka (read_once)."""
    try:
        return np.array(panda.get_robot().read_once().O_T_EE).reshape(4, 4).T
    except Exception:                                        # działa pętla sterowania -> jej stan jest aktualny
        return np.array(panda.get_pose())


def yaw_to_quat(yaw):
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])


def srodek_i_yaw(T_base_cam, R_cam_tag, t_cam_tag):
    """Środek kostki [m] i jej obrót wokół pionu [rad] w układzie bazy z jednego tagu."""
    srodek_cam = np.asarray(t_cam_tag).flatten() + R_cam_tag[:, 2] * HALF   # oś z tagu wchodzi w kostkę
    srodek = T_base_cam[:3, :3] @ srodek_cam + T_base_cam[:3, 3]
    R = T_base_cam[:3, :3] @ R_cam_tag                       # osie tagu w układzie bazy
    os_pozioma = R[:, np.argmin(np.abs(R[2, :]))]            # najbardziej pozioma oś tagu = krawędź kostki
    yaw = np.arctan2(os_pozioma[1], os_pozioma[0])
    return srodek, (yaw - YAW_OD) % (np.pi / 2) + YAW_OD


def usrednij_yaw(yaws):
    """Średnia kątów, które powtarzają się co 90 st. (średnia na kole dla 4 * kąt)."""
    yaws = np.asarray(yaws)
    y = np.arctan2(np.sin(4 * yaws).mean(), np.cos(4 * yaws).mean()) / 4
    return (y - YAW_OD) % (np.pi / 2) + YAW_OD


class KameraKostki:
    def __init__(self, kalibracja=os.path.join(DIR, "kalibracja.json")):
        if kalibracja is None:
            self.T_tcp_cam = np.eye(4)                       # bez kalibracji: wyniki w układzie kamery
        else:
            with open(kalibracja) as f:
                self.T_tcp_cam = np.array(json.load(f)["T_tcp_cam"])
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
        profile = self.pipe.start(cfg)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.cam_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)
        if EXPOSURE is not None:
            sensor = profile.get_device().first_color_sensor()
            sensor.set_option(rs.option.enable_auto_exposure, 0)
            sensor.set_option(rs.option.exposure, EXPOSURE)
        self.detector = pupil_apriltags.Detector(families="tag36h11", nthreads=4)
        for _ in range(15):                                  # pierwsze klatki bywają ciemne (auto-ekspozycja)
            self.pipe.wait_for_frames(5000)

    def klatka(self):
        while True:
            color = self.pipe.wait_for_frames(5000).get_color_frame()
            if color:
                return np.asanyarray(color.get_data())

    def wykryj(self, img):
        """Tagi z kostek (ID 0-17) wykryte wystarczająco pewnie."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return [d for d in self.detector.detect(gray, estimate_tag_pose=True, camera_params=self.cam_params,
                                                tag_size=CUBE_TAG)
                if d.tag_id < 18 and d.decision_margin >= MIN_MARGIN]

    def pomiary(self, tagi, T_base_tcp):
        """{nazwa kostki: [(środek, yaw), ...]} – jeden wpis na każdy widoczny tag tej kostki."""
        T_base_cam = np.asarray(T_base_tcp) @ self.T_tcp_cam
        out = {}
        for d in tagi:
            nazwa = next(k for k, ids in KOSTKI.items() if d.tag_id in ids)
            out.setdefault(nazwa, []).append(srodek_i_yaw(T_base_cam, d.pose_R, d.pose_t))
        return out

    def zmierz(self, T_base_tcp, n_klatek=10):
        """Pozycje kostek z kilku klatek (mediana). Robot musi stać – T_base_tcp = panda.get_pose()."""
        zebrane = {}
        for _ in range(n_klatek):
            for nazwa, lista in self.pomiary(self.wykryj(self.klatka()), T_base_tcp).items():
                zebrane.setdefault(nazwa, []).extend(lista)
        wynik = {}
        for nazwa, lista in zebrane.items():
            if len(lista) < n_klatek // 2:                   # widoczna tylko na części klatek – niepewna
                continue
            pos = np.median([p for p, _ in lista], axis=0)
            wynik[nazwa] = (pos, yaw_to_quat(usrednij_yaw([y for _, y in lista])))
        return wynik

    def zamknij(self):
        self.pipe.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bez-robota", action="store_true", help="bez robota i kalibracji: pozycje względem kamery")
    args = ap.parse_args()

    panda = None
    if not args.bez_robota:
        import panda_py
        panda = panda_py.Panda("172.16.0.2")                 # adres robota (jak w robot_franka.py)
    kam = KameraKostki(None if args.bez_robota else os.path.join(DIR, "kalibracja.json"))
    uklad = "względem kamery" if args.bez_robota else "w układzie bazy"
    try:
        while True:
            img = kam.klatka()
            tagi = kam.wykryj(img)
            T_base_tcp = np.eye(4) if panda is None else poza_tcp(panda)
            view = img.copy()
            for d in tagi:
                cv2.polylines(view, [d.corners.astype(int).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
                cv2.putText(view, str(d.tag_id), tuple(d.center.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)
            linie = [f"kostki {uklad} (x, y, z [cm], obrot [st.])   q = koniec"]
            for nazwa, lista in sorted(kam.pomiary(tagi, T_base_tcp).items()):
                pos = np.mean([p for p, _ in lista], axis=0)
                yaw = usrednij_yaw([y for _, y in lista])
                linie.append(f"{nazwa} {KOLOR[nazwa]:9s} {np.round(pos * 100, 1)}  {np.degrees(yaw):5.1f}  "
                             f"(scian: {len(lista)})")
            for i, txt in enumerate(linie):
                cv2.putText(view, txt, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("kostki", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        kam.zamknij()
        cv2.destroyAllWindows()
