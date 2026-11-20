"""pole_robocze.py - podgląd z Logitecha z zaznaczonym polem roboczym (gdzie stawiać kostki) i pozycjami kostek.

Uruchom:  python pole_robocze.py                         (zakres z treningu: x 0.35-0.55 m, y -0.20..0.20 m)
          python pole_robocze.py --x 0.30 0.60 --y -0.25 0.25   (inny zakres, np. po zmianie w treningu)
Pole robocze to prostokąt na stole w układzie robota (x do przodu, y w lewo od środka podstawy), rysowany na obrazie
z kalibracji statywu (statyw.json). Kostka w polu: zielona, poza polem: czerwona; obok pozycja x, y [cm].
Klawisze: q = koniec.   Wymaga: logitech.json, statyw.json (kalibruj_logitech.py, kalibruj_statyw.py)
"""
import argparse
import json
import os

import cv2
import numpy as np

from kamera_kostki import CUBE_TAG, KOSTKI, srodek_i_yaw, tag_wiarygodny
from logitech import Logitech

DIR = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--x", type=float, nargs=2, default=[0.35, 0.55], help="zakres x kostek [m]")
ap.add_argument("--y", type=float, nargs=2, default=[-0.20, 0.20], help="zakres y kostek [m]")
args = ap.parse_args()

with open(os.path.join(DIR, "statyw.json")) as f:
    T_base_cam = np.array(json.load(f)["T_base_cam"])
T_cam_base = np.linalg.inv(T_base_cam)
cam = Logitech()
if cam.cam_params is None:
    raise SystemExit("Brak logitech.json - najpierw kalibruj_logitech.py")
fx, fy, cx, cy = cam.cam_params
KOLOR = {"cube_1": "niebieska", "cube_2": "czerwona", "cube_3": "zielona"}


def na_obraz(p_base):
    """Punkty w układzie robota [m] (N x 3) -> piksele (obraz już wyprostowany, więc bez zniekształceń)."""
    p = (T_cam_base[:3, :3] @ np.asarray(p_base, float).T).T + T_cam_base[:3, 3]
    return np.c_[fx * p[:, 0] / p[:, 2] + cx, fy * p[:, 1] / p[:, 2] + cy].astype(int)


def linia(img, a, b, kolor, grubosc=2, n=20):
    """Odcinek w układzie robota (dzielony na kawałki, więc dobrze wygląda też przy perspektywie)."""
    pts = na_obraz(np.linspace(a, b, n))
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, kolor, grubosc)


(x0, x1), (y0, y1) = args.x, args.y
rogi = [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)]
try:
    while True:
        img = cam.klatka()
        view = img.copy()
        # pole robocze na stole (z = 0) i kropkowana siatka co 5 cm
        for i in range(4):
            linia(view, rogi[i], rogi[(i + 1) % 4], (0, 200, 255), 3)
        for x in np.arange(x0, x1 + 1e-6, 0.05):
            for y in np.arange(y0, y1 + 1e-6, 0.05):
                cv2.circle(view, tuple(na_obraz([[x, y, 0.0]])[0]), 2, (0, 200, 255), -1)
        # podstawa robota i kierunki osi
        o = na_obraz([[0, 0, 0]])[0]
        cv2.circle(view, tuple(o), 6, (255, 255, 255), -1)
        linia(view, (0, 0, 0), (0.15, 0, 0), (0, 0, 255), 2)
        linia(view, (0, 0, 0), (0, 0.15, 0), (0, 255, 0), 2)
        cv2.putText(view, "x", tuple(na_obraz([[0.17, 0, 0]])[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(view, "y", tuple(na_obraz([[0, 0.17, 0]])[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # kostki: środek z każdej widocznej ściany, uśredniony
        kostki = {}
        for d in cam.wykryj(img, CUBE_TAG):
            nazwa = next((k for k, ids in KOSTKI.items() if d.tag_id in ids), None)
            if nazwa is not None and tag_wiarygodny(T_base_cam, d.pose_R):
                kostki.setdefault(nazwa, []).append(srodek_i_yaw(T_base_cam, d.pose_R, d.pose_t)[0])
        for nazwa, lista in kostki.items():
            p = np.mean(lista, axis=0)
            w_polu = x0 <= p[0] <= x1 and y0 <= p[1] <= y1
            kolor = (0, 255, 0) if w_polu else (0, 0, 255)
            u, v = na_obraz([[p[0], p[1], p[2]]])[0]
            cv2.circle(view, (u, v), 18, kolor, 3)
            cv2.putText(view, f"{KOLOR[nazwa]} {p[0] * 100:.0f}, {p[1] * 100:.0f} cm" + ("" if w_polu else " POZA"),
                        (u + 22, v), cv2.FONT_HERSHEY_SIMPLEX, 0.6, kolor, 2)
        cv2.putText(view, f"pole: x {x0 * 100:.0f}-{x1 * 100:.0f} cm, y {y0 * 100:.0f}..{y1 * 100:.0f} cm   q = koniec",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow("pole robocze", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cam.zamknij()
    cv2.destroyAllWindows()
