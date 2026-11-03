"""kalibruj_logitech.py – kalibracja obiektywu Logitecha szachownicą (ogniskowa, środek obrazu, zniekształcenia).

Uruchom:  python kalibruj_logitech.py
    s = zapisz klatkę (tylko gdy szachownica wykryta – zielone punkty),  k = policz kalibrację,  q = koniec.
    Zbierz 15-25 zdjęć: szachownica w różnych miejscach obrazu (też w rogach), pod różnymi kątami (do ok. 45 st.),
    blisko i daleko. Szachownica musi być płaska (naklejona na sztywną płytę).
Tylko liczenie z już zapisanych zdjęć:  python kalibruj_logitech.py --licz
Wynik: logitech.json (K, dist) – używa go logitech.py.
WAŻNE: kalibracja jest ważna tylko dla tej ostrości (FOCUS w logitech.py) i tej rozdzielczości.
Wymaga:  pip install opencv-python
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

import logitech

DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(DIR, "dane_logitech")
SZACHOWNICA = (9, 6)                                         # WEWNĘTRZNE narożniki (pola 10 x 7 -> 9 x 6) – WPISZ SWOJE
POLE = 0.025                                                 # bok jednego pola [m] – ZMIERZ suwmiarką

ap = argparse.ArgumentParser()
ap.add_argument("--licz", action="store_true", help="bez kamery: policz z zapisanych zdjęć")
args = ap.parse_args()

# narożniki szachownicy w jej własnym układzie (płaska, z = 0)
obj = np.zeros((SZACHOWNICA[0] * SZACHOWNICA[1], 3), np.float32)
obj[:, :2] = np.mgrid[0:SZACHOWNICA[0], 0:SZACHOWNICA[1]].T.reshape(-1, 2) * POLE


def narozniki(gray):
    ok, c = cv2.findChessboardCorners(gray, SZACHOWNICA, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    kryterium = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), kryterium)   # dokładność poniżej piksela


def licz():
    pliki = sorted(glob.glob(os.path.join(FOLDER, "*.png")))
    obj_pts, img_pts, uzyte = [], [], []
    for p in pliki:
        c = narozniki(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY))
        if c is not None:
            obj_pts.append(obj)
            img_pts.append(c)
            uzyte.append(p)
    print(f"Zdjęć z wykrytą szachownicą: {len(uzyte)} / {len(pliki)}")
    if len(uzyte) < 10:
        raise SystemExit("Za mało zdjęć (potrzeba co najmniej 10, najlepiej 15-25).")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (logitech.W, logitech.H), None, None)
    print(f"Błąd reprojekcji: {rms:.3f} px   (dobrze < 0.5 px, źle > 1 px)")
    print("Błąd każdego zdjęcia (duży = rozmazane / źle wykryte – usuń plik i policz jeszcze raz):")
    for p, o, i, r, t in zip(uzyte, obj_pts, img_pts, rvecs, tvecs):
        rzut, _ = cv2.projectPoints(o, r, t, K, dist)
        blad = np.sqrt(np.mean(np.sum((rzut - i) ** 2, axis=2)))
        print(f"  {os.path.basename(p)}  {blad:.3f} px" + ("   <-- podejrzane" if blad > 2 * rms else ""))
    print(f"fx = {K[0, 0]:.1f}  fy = {K[1, 1]:.1f}  cx = {K[0, 2]:.1f}  cy = {K[1, 2]:.1f}")
    print(f"zniekształcenia: {np.round(dist.flatten(), 4)}")
    with open(logitech.KALIBRACJA, "w") as f:
        json.dump({"K": K.tolist(), "dist": dist.flatten().tolist(), "rms": float(rms), "zdjec": len(uzyte),
                   "W": logitech.W, "H": logitech.H, "focus": logitech.FOCUS}, f, indent=2)
    print(f"Zapisano {logitech.KALIBRACJA}")


if args.licz:
    licz()
    raise SystemExit

os.makedirs(FOLDER, exist_ok=True)
cap = logitech.otworz()                                      # surowy obraz – kalibrujemy właśnie zniekształcenia
n = len(glob.glob(os.path.join(FOLDER, "*.png")))
try:
    while True:
        ok, img = cap.read()
        if not ok:
            raise RuntimeError("Kamera przestała dawać obraz")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # szybkie sprawdzenie na zmniejszonym obrazie, żeby podgląd nie zwalniał
        maly = cv2.resize(gray, None, fx=0.5, fy=0.5)
        ok_maly, c = cv2.findChessboardCorners(maly, SZACHOWNICA, cv2.CALIB_CB_FAST_CHECK)
        view = img.copy()
        if ok_maly:
            cv2.drawChessboardCorners(view, SZACHOWNICA, c * 2, True)
        cv2.putText(view, f"zdjec: {n}   s = zapisz, k = licz, q = koniec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        cv2.imshow("kalibracja logitech", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            if narozniki(gray) is None:
                print("Szachownica nie wykryta – nie zapisuję")
                continue
            cv2.imwrite(os.path.join(FOLDER, f"zdjecie_{n:02d}.png"), img)
            n += 1
            print(f"zapisano zdjęcie {n}")
        if key == ord("k"):
            licz()
finally:
    cap.release()
    cv2.destroyAllWindows()
