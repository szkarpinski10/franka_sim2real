"""kalibruj_statyw.py – gdzie stoi Logitech względem bazy robota (kalibracja kamery na statywie).

Pomysł: tag 20 leży na stole i widzą go OBIE kamery naraz. Kamera na chwytaku jest już skalibrowana (kalibracja.json),
więc z niej wiemy, gdzie są narożniki tagu w układzie BAZY. Logitech widzi te same narożniki w SWOIM układzie.
Szukamy obrotu i przesunięcia, które najlepiej nakłada jedne punkty na drugie (metoda Kabscha, jak dopasowanie
dwóch chmur punktów). Każde położenie tagu daje 4 narożniki; 3-5 różnych położeń na stole daje dokładniejszy wynik.

Uruchom (robot stoi, bez sterowania; Logitech ostatecznie ustawiony – po kalibracji statywu NIE ruszać):
    python kalibruj_statyw.py
    połóż tag 20 tak, żeby widziały go obie kamery -> s (zbiera ~1 s klatek z obu kamer),
    przesuń tag w inne miejsce (robota można przestawić ręką, żeby kamera na chwytaku go widziała) -> s, ...
    k = policz i zapisz,  q = koniec.
Wynik: statyw.json (T_base_cam 4 x 4) – używa go kamery.py.
Wymaga:  kalibracja.json, logitech.json, panda_py, pyrealsense2, opencv-python, pupil-apriltags
"""
import json
import os

import cv2
import numpy as np
import panda_py

from kamera_kostki import KameraKostki, poza_tcp
from logitech import Logitech

DIR = os.path.dirname(os.path.abspath(__file__))
WYNIK = os.path.join(DIR, "statyw.json")
IP = "172.16.0.2"
CALIB_TAG = 0.070                                            # bok czarnego kwadratu tagu 20 [m]
N_KLATEK = 10                                                # klatek na jedno położenie tagu

# narożniki tagu w jego własnym układzie (środek tagu = 0, tag leży w płaszczyźnie z = 0)
s = CALIB_TAG / 2
NAROZNIKI = np.array([[-s, -s, 0, 1], [s, -s, 0, 1], [s, s, 0, 1], [-s, s, 0, 1]]).T


def macierz(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(t).flatten()
    return T


def tag20(detector, gray, cam_params):
    """Poza tagu 20 względem kamery albo None."""
    for d in detector.detect(gray, estimate_tag_pose=True, camera_params=cam_params, tag_size=CALIB_TAG):
        if d.tag_id == 20 and d.decision_margin >= 30:
            return macierz(d.pose_R, d.pose_t)
    return None


def kabsch(A, B):
    """Obrót R i przesunięcie t takie, że R @ a + t ≈ b dla par punktów (kolumny A i B)."""
    ca, cb = A.mean(axis=1, keepdims=True), B.mean(axis=1, keepdims=True)
    U, _, Vt = np.linalg.svd((B - cb) @ (A - ca).T)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt      # det: żeby wyszedł obrót, a nie odbicie lustrzane
    return macierz(R, cb - R @ ca)


panda = panda_py.Panda(IP)
reka = KameraKostki()
logi = Logitech()
if logi.cam_params is None:
    raise SystemExit("Brak logitech.json – najpierw kalibruj_logitech.py")
pkt_baza, pkt_logi = [], []                                  # narożniki tagu: w bazie (z ręki) i względem Logitecha

try:
    while True:
        img_r, img_l = reka.klatka(), logi.klatka()
        widzi_r = tag20(reka.detector, cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY), reka.cam_params) is not None
        widzi_l = tag20(logi.detector, cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY), logi.cam_params) is not None
        view = np.hstack([cv2.resize(img_r, (640, 360)), cv2.resize(img_l, (640, 360))])
        cv2.putText(view, f"chwytak: {'WIDZI' if widzi_r else 'nie widzi'}   statyw: {'WIDZI' if widzi_l else 'nie widzi'}"
                          f"   polozen: {len(pkt_baza) // 4}   s / k / q", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)
        cv2.imshow("kalibracja statywu", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            T_base_tcp = poza_tcp(panda)                     # robot stoi -> jedna poza na całe zbieranie
            T_base_cam_r = T_base_tcp @ reka.T_tcp_cam
            z_reki, z_logi = [], []
            for _ in range(N_KLATEK):
                Tr = tag20(reka.detector, cv2.cvtColor(reka.klatka(), cv2.COLOR_BGR2GRAY), reka.cam_params)
                Tl = tag20(logi.detector, cv2.cvtColor(logi.klatka(), cv2.COLOR_BGR2GRAY), logi.cam_params)
                if Tr is not None and Tl is not None:
                    z_reki.append((T_base_cam_r @ Tr @ NAROZNIKI)[:3])
                    z_logi.append((Tl @ NAROZNIKI)[:3])
            if len(z_reki) < N_KLATEK // 2:
                print(f"Tag widoczny w obu kamerach tylko na {len(z_reki)} / {N_KLATEK} klatkach – nie zapisuję")
                continue
            pkt_baza.extend(np.median(z_reki, axis=0).T)     # mediana klatek: odporna na pojedyncze złe wykrycia
            pkt_logi.extend(np.median(z_logi, axis=0).T)
            print(f"położenie {len(pkt_baza) // 4}: środek tagu w bazie "
                  f"{np.round(np.mean(pkt_baza[-4:], axis=0) * 100, 1)} cm")
        if key == ord("k"):
            if not pkt_baza:
                print("Najpierw zbierz co najmniej jedno położenie tagu (s)")
                continue
            A, B = np.array(pkt_logi).T, np.array(pkt_baza).T
            T_base_cam = kabsch(A, B)
            blad = np.linalg.norm(T_base_cam[:3, :3] @ A + T_base_cam[:3, 3:] - B, axis=0)
            print(f"Położeń tagu: {len(pkt_baza) // 4}   błąd narożników: średni {blad.mean() * 1000:.1f} mm, "
                  f"max {blad.max() * 1000:.1f} mm   (dobrze: średnio < 5 mm)")
            print(f"Logitech w bazie: {np.round(T_base_cam[:3, 3] * 100, 1)} cm   (sprawdź miarką, czy to ma sens)")
            with open(WYNIK, "w") as f:
                json.dump({"T_base_cam": T_base_cam.tolist(), "polozen": len(pkt_baza) // 4,
                           "blad_sredni_mm": float(blad.mean() * 1000), "blad_max_mm": float(blad.max() * 1000)},
                          f, indent=2)
            print(f"Zapisano {WYNIK}")
finally:
    reka.zamknij()
    logi.zamknij()
    cv2.destroyAllWindows()
