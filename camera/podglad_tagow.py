"""podglad_tagow.py – podgląd z kamery RealSense z wykrytymi AprilTagami (bez robota).

Uruchom:  python podglad_tagow.py
Okno pokazuje obraz z kamery, obrysy tagów, ich ID i odległość od kamery.
Klawisze:  s = zapisz klatkę do zdjecia/,  q = koniec.
Po co: sprawdzić, czy tagi na kostkach są wykrywane z pozycji HOME i z jakiej odległości przestają być widoczne.
Przed uruchomieniem zamknij RealSense Viewer (kamerę może używać tylko jeden program naraz).
Wymaga:  pip install pyrealsense2 opencv-python pupil-apriltags
"""
import os                                                    # folder na zapisane klatki
import time                                                  # licznik klatek/s

import cv2                                                   # obraz, okno, rysowanie
import numpy as np
import pupil_apriltags                                       # wykrywanie AprilTagów
import pyrealsense2 as rs                                    # sterownik kamery

W, H, FPS = 1280, 720, 30                                    # tryb koloru (sprawdź w profiles.py, czy działa; inaczej 640 x 480)
EXPOSURE = None                                              # None = automatyczna; liczba (np. 100) = stała, krótka -> mniej rozmyć w ruchu
# tagi tag36h11, każda ściana kostki ma inne ID:  niebieska 0-5, czerwona 6-11, zielona 12-17;  20 = duży tag do kalibracji
CUBE_TAG = 0.035                                             # bok CZARNEGO kwadratu tagu na kostce [m] – WPISZ ZMIERZONY
CALIB_TAG = 0.070                                            # bok CZARNEGO kwadratu tagu ID 20 [m] – WPISZ ZMIERZONY
TAG_SIZE = {i: CUBE_TAG for i in range(18)}
TAG_SIZE[20] = CALIB_TAG
CUBE_NAME = {i: ("niebieska", "czerwona", "zielona")[i // 6] + f" ściana {i % 6}" for i in range(18)}
CUBE_NAME[20] = "kalibracja"
OUT_DIR = "zdjecia"

# ---------- kamera ----------
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)   # sam kolor – głębia do tagów niepotrzebna
profile = pipe.start(cfg)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
cam_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)          # parametry kamery z fabrycznej kalibracji RealSense
print(f"Kamera: {W} x {H} @ {FPS}   fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.ppx:.1f} cy={intr.ppy:.1f}   "
      f"zniekształcenie: {intr.model} {np.round(intr.coeffs, 4)}")
if EXPOSURE is not None:
    sensor = profile.get_device().first_color_sensor()
    sensor.set_option(rs.option.enable_auto_exposure, 0)
    sensor.set_option(rs.option.exposure, EXPOSURE)

detector = pupil_apriltags.Detector(families="tag36h11", nthreads=4)
sizes = sorted(set(TAG_SIZE.values()))                       # pupil-apriltags liczy pozę dla jednego rozmiaru naraz


def detect(gray):
    """Zwraca wykryte tagi ze znanych ID, każdy z pozą policzoną dla jego rozmiaru."""
    out = {}
    for s in sizes:
        for d in detector.detect(gray, estimate_tag_pose=True, camera_params=cam_params, tag_size=s):
            if TAG_SIZE.get(d.tag_id) == s:
                out[d.tag_id] = d
    return list(out.values())


os.makedirs(OUT_DIR, exist_ok=True)
n_saved, t_prev, fps_view = 0, time.time(), 0.0
try:
    while True:
        frames = pipe.wait_for_frames(5000)
        color = frames.get_color_frame()
        if not color:
            continue
        img = np.asanyarray(color.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tags = detect(gray)

        view = img.copy()
        for d in tags:
            c = d.corners.astype(int)
            cv2.polylines(view, [c.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            t = d.pose_t.flatten()                           # pozycja tagu względem kamery [m]
            txt = f"ID {d.tag_id} {CUBE_NAME.get(d.tag_id, '')}  {np.linalg.norm(t) * 100:.1f} cm  m={d.decision_margin:.0f}"
            cv2.putText(view, txt, (c[0][0], c[0][1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        now = time.time()
        fps_view = 0.9 * fps_view + 0.1 / max(now - t_prev, 1e-3)
        t_prev = now
        cv2.putText(view, f"{fps_view:4.1f} kl/s   tagi: {sorted(d.tag_id for d in tags)}   s = zapisz, q = koniec",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("AprilTagi", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            path = os.path.join(OUT_DIR, f"klatka_{n_saved:03d}.png")
            cv2.imwrite(path, img)                           # zapisujemy obraz BEZ rysunków
            n_saved += 1
            print(f"zapisano {path}   tagi: " + ", ".join(
                f"ID {d.tag_id}: {np.round(d.pose_t.flatten() * 100, 1)} cm" for d in tags))
finally:
    pipe.stop()
    cv2.destroyAllWindows()
