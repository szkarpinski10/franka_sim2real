"""zbierz_handeye.py – zbieranie danych do kalibracji hand-eye (gdzie jest kamera względem TCP).

Uruchom (przy robocie, Desk: stawy odblokowane, FCI włączone):  python zbierz_handeye.py
Przygotowanie: tag ID 20 przyklejony NIERUCHOMO na stole.
Co robić: ustaw robota w 15-20 różnych pozycjach (inna wysokość, wyraźnie inne nachylenie i obrót chwytaka,
tag zawsze w kadrze). W każdej pozycji puść robota, poczekaj aż stoi i naciśnij SPACJĘ.
Zapisuje się para: zdjęcie + poza TCP robota (+ poza tagu z kamery) pod tym samym numerem.
Klawisze:  spacja = zapisz pozycję,  q = koniec.
Wynik: folder dane_handeye/ (zdjęcia, pozy.npz, kamera.json). Kalibrację liczymy potem w domu.
Wymaga:  pip install pyrealsense2 opencv-python pupil-apriltags  + panda_py pasujące do robota
"""
import json                                                  # zapis parametrów kamery
import os

import cv2
import numpy as np
import panda_py                                              # robot
import pupil_apriltags
import pyrealsense2 as rs

IP = "172.16.0.2"                                            # adres robota (jak w robot_franka.py)
W, H, FPS = 1280, 720, 30                                    # tryb koloru (ten sam co w podglad_tagow.py)
EXPOSURE = None                                              # None = automatyczna
CALIB_ID = 20                                                # tag na stole (ID 0-17 są na kostkach)
CALIB_SIZE = 0.070                                           # bok CZARNEGO kwadratu tagu ID 20 [m] – WPISZ ZMIERZONY
TEACHING = True                                              # True = robota można przesuwać ręką (tryb nauczania)
OUT_DIR = "dane_handeye"

# ---------- robot ----------
panda = panda_py.Panda(IP)
if TEACHING:
    try:
        panda.teaching_mode(True)
        print("Tryb nauczania WŁĄCZONY – robota można przesuwać ręką.")
    except Exception as e:                                   # gdyby nie działał: ustawiaj robota z Desk
        print(f"Tryb nauczania nie zadziałał ({e}) – ustawiaj robota z Desk.")

# ---------- kamera ----------
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
profile = pipe.start(cfg)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
cam_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)
if EXPOSURE is not None:
    sensor = profile.get_device().first_color_sensor()
    sensor.set_option(rs.option.enable_auto_exposure, 0)
    sensor.set_option(rs.option.exposure, EXPOSURE)

def stan_robota():
    """Świeży stan robota: (poza TCP 4 x 4, q, dq).
    Po przerwaniu trybu nauczania (np. wciśnięty stop) panda.get_state()/get_pose() oddają ciągle ten sam stary stan,
    więc gdy żadna pętla sterowania nie działa, czytamy prosto z libfranka (read_once)."""
    try:
        s = panda.get_robot().read_once()                    # działa tylko, gdy nie działa pętla sterowania
        return np.array(s.O_T_EE).reshape(4, 4).T, np.array(s.q), np.array(s.dq)
    except Exception:                                        # tryb nauczania działa -> stan z jego pętli jest aktualny
        s = panda.get_state()
        return np.array(panda.get_pose()), np.array(s.q), np.array(s.dq)


os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "kamera.json"), "w") as f:
    json.dump({"width": W, "height": H, "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
               "model": str(intr.model), "coeffs": list(intr.coeffs), "calib_tag_id": CALIB_ID,
               "calib_tag_size": CALIB_SIZE}, f, indent=2)

detector = pupil_apriltags.Detector(families="tag36h11", nthreads=4)

T_base_tcp, R_cam_tag, t_cam_tag, q_joints = [], [], [], []  # zapisywane pary
try:
    while True:
        frames = pipe.wait_for_frames(5000)
        color = frames.get_color_frame()
        if not color:
            continue
        img = np.asanyarray(color.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tag = [d for d in detector.detect(gray, estimate_tag_pose=True, camera_params=cam_params,
                                          tag_size=CALIB_SIZE) if d.tag_id == CALIB_ID]

        view = img.copy()
        if tag:
            d = tag[0]
            cv2.polylines(view, [d.corners.astype(int).reshape(-1, 1, 2)], True, (0, 255, 0), 3)
            msg = f"tag {CALIB_ID}: {np.linalg.norm(d.pose_t) * 100:.1f} cm  m={d.decision_margin:.0f}   SPACJA = zapisz"
            colr = (0, 255, 0)
        else:
            msg, colr = f"NIE WIDZĘ tagu {CALIB_ID}", (0, 0, 255)
        cv2.putText(view, f"zapisanych: {len(T_base_tcp)}   {msg}   q = koniec", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colr, 2)
        cv2.imshow("hand-eye", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            if not tag:
                print("Tag niewidoczny – ustaw robota tak, żeby tag był w kadrze.")
                continue
            T_tcp, q, dq = stan_robota()
            if np.abs(dq).max() > 0.01:                      # zdjęcie i poza muszą być z tej samej chwili
                print(f"Robot jeszcze się rusza ({np.abs(dq).max():.3f} rad/s) – poczekaj, aż stanie.")
                continue
            if q_joints and np.abs(q - q_joints[-1]).max() < 1e-4:
                print("NIE ZAPISANO: pozycja robota taka sama jak przy poprzednim zdjęciu. Przesuń robota; "
                      "jeśli przesunąłeś, a to się powtarza – robot nie podaje nowej pozycji (zakończ q i uruchom od nowa).")
                continue
            i = len(T_base_tcp)
            cv2.imwrite(os.path.join(OUT_DIR, f"zdjecie_{i:02d}.png"), img)
            T_base_tcp.append(T_tcp)                         # poza TCP w układzie bazy (4 x 4)
            q_joints.append(q)
            R_cam_tag.append(tag[0].pose_R)
            t_cam_tag.append(tag[0].pose_t.flatten())
            np.savez(os.path.join(OUT_DIR, "pozy.npz"), T_base_tcp=np.array(T_base_tcp), q=np.array(q_joints),
                     R_cam_tag=np.array(R_cam_tag), t_cam_tag=np.array(t_cam_tag))   # zapis po każdej pozycji
            print(f"[{i:02d}] zapisano   TCP {np.round(T_base_tcp[-1][:3, 3] * 100, 1)} cm   "
                  f"tag od kamery {np.linalg.norm(t_cam_tag[-1]) * 100:.1f} cm")
finally:
    pipe.stop()
    cv2.destroyAllWindows()
    if TEACHING:
        try:
            panda.teaching_mode(False)
        except Exception:
            pass
    print(f"Koniec. Zapisanych pozycji: {len(T_base_tcp)} (potrzeba 15-20) w folderze {OUT_DIR}/")
