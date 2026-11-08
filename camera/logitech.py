"""logitech.py - kamera Logitech na statywie: obraz bez zniekształceń i wykrywanie tagów.

Podgląd:  python logitech.py   (+/- ostrość, s zapis klatki, q koniec)
Pomiar opóźnienia obrazu:  python logitech.py --opoznienie   (ekran laptopa przed kamerą miga czarny/biały,
wynik wpisać do OPOZNIENIE_STATYW w kamery.py)
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import pupil_apriltags

DIR = os.path.dirname(os.path.abspath(__file__))
KALIBRACJA = os.path.join(DIR, "logitech.json")
# stała nazwa urządzenia (numer /dev/videoN zmienia się po przepięciu)
INDEKS = "/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_23B315FF-video-index0"
W, H, FPS = 1280, 720, 30
FOCUS = 0                                                    # stała ostrość, nie zmieniać po kalibracji obiektywu
MIN_MARGIN = 30                                              # słabsze wykrycia odrzucamy (jak w kamera_kostki.py)


def otworz():
    """Otwiera kamerę ze stałą ostrością (autofokus zmieniałby parametry obiektywu w trakcie pracy)."""
    cap = cv2.VideoCapture(INDEKS, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # bez MJPG 1280 x 720 idzie tylko w kilku kl/s
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                      # zawsze najnowsza klatka, nie stara z kolejki
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, FOCUS)
    ok, img = cap.read()
    if not ok:
        raise RuntimeError(f"Nie da się otworzyć kamery {INDEKS} - sprawdź INDEKS (ls /dev/video*)")
    if img.shape[1] != W or img.shape[0] != H:
        raise RuntimeError(f"Kamera daje {img.shape[1]} x {img.shape[0]}, a nie {W} x {H}")
    return cap


class Logitech:
    def __init__(self, kalibracja=KALIBRACJA):
        self.cap = otworz()
        self.detector = pupil_apriltags.Detector(families="tag36h11", nthreads=4)
        self.cam_params = None                               # bez kalibracji: tylko ID tagów, bez pozy
        if kalibracja is not None and os.path.exists(kalibracja):
            with open(kalibracja) as f:
                k = json.load(f)
            if k["focus"] != FOCUS:
                print(f"UWAGA: kalibracja zrobiona przy FOCUS = {k['focus']}, a teraz FOCUS = {FOCUS}")
            K, dist = np.array(k["K"]), np.array(k["dist"])
            # mapa prostowania liczona raz; potem każda klatka to szybkie remap
            self.mapa = cv2.initUndistortRectifyMap(K, dist, None, K, (W, H), cv2.CV_16SC2)
            self.cam_params = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    def klatka(self):
        ok, img = self.cap.read()
        if not ok:
            raise RuntimeError("Kamera przestała dawać obraz")
        if self.cam_params is not None:
            img = cv2.remap(img, *self.mapa, cv2.INTER_LINEAR)
        return img

    def wykryj(self, img, tag_size):
        """Tagi wykryte wystarczająco pewnie; z pozą (pose_R, pose_t), jeśli obiektyw jest skalibrowany."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.cam_params is None:
            tagi = self.detector.detect(gray)
        else:
            tagi = self.detector.detect(gray, estimate_tag_pose=True, camera_params=self.cam_params,
                                        tag_size=tag_size)
        return [d for d in tagi if d.decision_margin >= MIN_MARGIN]

    def zamknij(self):
        self.cap.release()


def zmierz_opoznienie(n_zmian=12):
    cap = otworz()
    # stała ekspozycja na czas pomiaru (auto-ekspozycja wyrównuje jasność czarnego i białego ekranu)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)                   # V4L2: 1 = ręczna, 3 = automatyczna
    cap.set(cv2.CAP_PROP_EXPOSURE, 150)                      # ok. 15 ms (jednostki 0.1 ms)
    cv2.namedWindow("miganie", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("miganie", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    wyniki, bialy = [], False
    for _ in range(30):                                      # rozgrzewka: ekspozycja się ustawia
        cap.read()

    def jasnosc_ekranu(kolor):
        """Średnia jasność obrazu, gdy ekran od 1 s pokazuje dany kolor."""
        cv2.imshow("miganie", np.full((H, W), kolor, np.uint8))
        cv2.waitKey(1000)
        return np.mean([cap.read()[1].mean() for _ in range(10)])

    # próg jasności między czarnym a białym ekranem (ekran nie musi wypełniać kadru)
    czarny, bialy_poziom = jasnosc_ekranu(0), jasnosc_ekranu(255)
    if bialy_poziom - czarny < 5:
        raise SystemExit(f"Kamera prawie nie widzi różnicy czarny/biały ({czarny:.0f} / {bialy_poziom:.0f}) "
                         "- przybliż ekran laptopa do kamery")
    prog = (czarny + bialy_poziom) / 2
    print(f"jasność: czarny ekran {czarny:.0f}, biały {bialy_poziom:.0f}, próg {prog:.0f}")
    for _ in range(n_zmian):
        bialy = not bialy
        cv2.imshow("miganie", np.full((H, W), 255 if bialy else 0, np.uint8))
        cv2.waitKey(1)
        t_zmiany = time.perf_counter()
        while time.perf_counter() - t_zmiany < 1.0:          # czekamy, aż kamera zobaczy nową jasność
            ok, img = cap.read()
            jasnosc = img.mean()
            if (bialy and jasnosc > prog) or (not bialy and jasnosc < prog):
                wyniki.append(time.perf_counter() - t_zmiany)
                break
        cv2.waitKey(300)                                     # obraz stoi chwilę, zanim zmienimy
    cap.release()
    cv2.destroyAllWindows()
    if len(wyniki) < n_zmian // 2:
        raise SystemExit("Kamera nie widzi migania - przybliż ekran laptopa do kamery")
    print(f"Opóźnienie: mediana {np.median(wyniki) * 1000:.0f} ms, min {min(wyniki) * 1000:.0f}, "
          f"max {max(wyniki) * 1000:.0f} ms  ({len(wyniki)} pomiarów) -> OPOZNIENIE_STATYW w kamery.py")


if __name__ == "__main__" and "--opoznienie" in sys.argv:
    zmierz_opoznienie()
elif __name__ == "__main__":
    CUBE_TAG, CALIB_TAG = 0.035, 0.070                       # jak w podglad_tagow.py
    cam = Logitech()
    if cam.cam_params is None:
        print("Brak logitech.json - pokazuję tylko ID tagów (odległość po kalibruj_logitech.py)")
    focus = FOCUS
    os.makedirs(os.path.join(DIR, "zdjecia"), exist_ok=True)
    t0, n = time.time(), 0
    try:
        while True:
            img = cam.klatka()
            n += 1
            view = img.copy()
            for d in cam.wykryj(img, CUBE_TAG):
                txt = f"{d.tag_id} m={d.decision_margin:.0f}"
                if cam.cam_params is not None:
                    # odległość rośnie proporcjonalnie do rozmiaru tagu, więc tag 20 przeliczamy z CUBE_TAG
                    rozmiar = CALIB_TAG if d.tag_id == 20 else CUBE_TAG
                    txt += f" {np.linalg.norm(d.pose_t) * rozmiar / CUBE_TAG * 100:.0f} cm"
                cv2.polylines(view, [d.corners.astype(int).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
                cv2.putText(view, txt, tuple(d.center.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            fps = n / (time.time() - t0)
            cv2.putText(view, f"{fps:.1f} kl/s   focus {focus}   (+/- ostrosc, s zapis, q koniec)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("logitech", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("+"), ord("=")) or key == ord("-"):
                focus = int(np.clip(focus + (5 if key != ord("-") else -5), 0, 255))
                cam.cap.set(cv2.CAP_PROP_FOCUS, focus)
                print(f"focus = {focus}   (wpisz do FOCUS w logitech.py, jeśli tagi są najostrzejsze)")
            if key == ord("s"):
                nazwa = os.path.join(DIR, "zdjecia", f"logitech_{time.strftime('%H%M%S')}.png")
                cv2.imwrite(nazwa, img)
                print("zapisano", nazwa)
    finally:
        cam.zamknij()
        cv2.destroyAllWindows()
