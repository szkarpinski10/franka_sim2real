"""kalibruj_handeye.py – kalibracja hand-eye: gdzie jest kamera względem TCP robota.

Uruchom (po zbierz_handeye.py, robot niepotrzebny):  python kalibruj_handeye.py
                                    albo:            python kalibruj_handeye.py --pomin 3,7   (bez pozycji 3 i 7)
Bierze dane_handeye/pozy.npz (poza TCP + poza tagu ID 20 z kamery) i liczy T_tcp_cam:
metodą Parka-Martina (AX = XB), potem dopasowaniem, w którym tag 20 ma wyjść zawsze w tym samym miejscu (AX = YB).
Liczone samym numpy – w OpenCV 5 nie ma już cv2.calibrateHandEye.
Sprawdzenie: tag 20 leży nieruchomo na stole, więc z każdej pozycji robota powinien wyjść w TYM SAMYM miejscu
w układzie bazy. Rozrzut tego miejsca = błąd kalibracji (dobrze: średnio < 5 mm, źle: > 10 mm).
Pozycje z dużym błędem można pominąć (--pomin) albo zebrać dane jeszcze raz.
Wynik: kalibracja.json (T_tcp_cam 4 x 4 – używa go kamera_kostki.py).
Wymaga:  numpy
"""
import argparse
import json
import os

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("dane", nargs="?", default=os.path.join(DIR, "dane_handeye"), help="folder z pozy.npz")
ap.add_argument("--pomin", default="", help="numery pozycji do pominięcia, np. 3,7")
ap.add_argument("--wynik", default=os.path.join(DIR, "kalibracja.json"))
args = ap.parse_args()


def macierz(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(t).flatten()
    return T


def kat(R):
    """Kąt obrotu macierzy R [st.]."""
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def log_R(R):
    """Obrót R jako wektor (oś * kąt) [rad]."""
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if a < 1e-9:
        return np.zeros(3)
    return a / (2 * np.sin(a)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def najblizszy_obrot(M):
    """Macierz obrotu najbliższa M (średnia obrotów = suma macierzy rzutowana na obroty)."""
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt


def park(T_base_tcp, T_cam_tag):
    """Park-Martin: A X = X B dla par pozycji; A = ruch TCP, B = ruch tagu widziany z kamery."""
    A, B = [], []
    for i in range(len(T_base_tcp)):
        for j in range(i + 1, len(T_base_tcp)):
            A.append(np.linalg.inv(T_base_tcp[i]) @ T_base_tcp[j])
            B.append(T_cam_tag[i] @ np.linalg.inv(T_cam_tag[j]))
    M = sum(np.outer(log_R(b[:3, :3]), log_R(a[:3, :3])) for a, b in zip(A, B))
    R = najblizszy_obrot(np.linalg.inv(M.T @ M + 1e-12 * np.eye(3)) @ M.T)   # (M^T M)^(-1/2) M^T, potem rzut
    C = np.vstack([a[:3, :3] - np.eye(3) for a in A])
    d = np.concatenate([R @ b[:3, 3] - a[:3, 3] for a, b in zip(A, B)])
    return macierz(R, np.linalg.lstsq(C, d, rcond=None)[0])


def dopasuj(X, T_base_tcp, T_cam_tag, iteracje=200):
    """Poprawia X tak, żeby tag (Y = poza tagu w bazie) wychodził z każdej pozycji w tym samym miejscu."""
    for _ in range(iteracje):
        Ys = [Tb @ X @ Tc for Tb, Tc in zip(T_base_tcp, T_cam_tag)]
        Y = macierz(najblizszy_obrot(sum(y[:3, :3] for y in Ys)), np.mean([y[:3, 3] for y in Ys], axis=0))
        Xs = [np.linalg.inv(Tb) @ Y @ np.linalg.inv(Tc) for Tb, Tc in zip(T_base_tcp, T_cam_tag)]
        X = macierz(najblizszy_obrot(sum(x[:3, :3] for x in Xs)), np.mean([x[:3, 3] for x in Xs], axis=0))
    return X


def tag_w_bazie(T_tcp_cam, T_base_tcp, R_cam_tag, t_cam_tag):
    """Poza tagu w układzie bazy policzona osobno z każdej pozycji robota."""
    return np.array([Tb @ T_tcp_cam @ macierz(R, t) for Tb, R, t in zip(T_base_tcp, R_cam_tag, t_cam_tag)])


# ---------- dane ----------
d = np.load(os.path.join(args.dane, "pozy.npz"))
T_base_tcp, R_cam_tag, t_cam_tag = d["T_base_tcp"], d["R_cam_tag"], d["t_cam_tag"]
numery = np.arange(len(T_base_tcp))
pomin = {int(x) for x in args.pomin.split(",") if x.strip()}
keep = np.array([i not in pomin for i in numery])
T_base_tcp, R_cam_tag, t_cam_tag, numery = T_base_tcp[keep], R_cam_tag[keep], t_cam_tag[keep], numery[keep]
n = len(numery)
print(f"Pozycji: {n}" + (f"  (pominięte: {sorted(pomin)})" if pomin else ""))
if n < 5:
    raise SystemExit("Za mało pozycji (potrzeba co najmniej 5, najlepiej 15-20).")

# czy pozycje są wystarczająco różne – hand-eye potrzebuje obrotów wokół różnych osi
obroty = [kat(T_base_tcp[i][:3, :3].T @ T_base_tcp[j][:3, :3]) for i in range(n) for j in range(i + 1, n)]
print(f"Największa różnica orientacji chwytaka między pozycjami: {max(obroty):.0f} st.")
if max(obroty) < 30:
    print("  UWAGA: pozycje mało się różnią obrotem (< 30 st.) – kalibracja może być niedokładna.")

# ---------- kalibracja ----------
T_cam_tag = np.array([macierz(R, t) for R, t in zip(R_cam_tag, t_cam_tag)])
X_park = park(T_base_tcp, T_cam_tag)
wyniki = {}
for nazwa, T_tcp_cam in (("PARK", X_park), ("DOPASOWANIE", dopasuj(X_park, T_base_tcp, T_cam_tag))):
    tagi = tag_w_bazie(T_tcp_cam, T_base_tcp, R_cam_tag, t_cam_tag)
    p = tagi[:, :3, 3]
    blad = np.linalg.norm(p - np.median(p, axis=0), axis=1)  # odległość od "środka" wszystkich pomiarów [m]
    wyniki[nazwa] = (T_tcp_cam, blad, tagi)
    print(f"{nazwa:11s} błąd tagu: średni {blad.mean() * 1000:5.1f} mm   max {blad.max() * 1000:5.1f} mm   "
          f"kamera od TCP {np.round(T_tcp_cam[:3, 3] * 100, 1)} cm")

# ---------- najlepsza metoda ----------
nazwa = min(wyniki, key=lambda k: wyniki[k][1].mean())
T_tcp_cam, blad, tagi = wyniki[nazwa]
p = tagi[:, :3, 3]
tag_srodek = np.median(p, axis=0)
obrot_tagu = [kat(tagi[0][:3, :3].T @ T[:3, :3]) for T in tagi]
print(f"\nNajlepsza: {nazwa}")
print("Błąd każdej pozycji (duży błąd = podejrzana pozycja, można ją pominąć przez --pomin):")
for i, b in zip(numery, blad):
    print(f"  [{i:02d}] {b * 1000:5.1f} mm" + ("   <-- podejrzana" if b > max(2 * np.median(blad), 0.008) else ""))
print(f"Tag 20 w układzie bazy: {np.round(tag_srodek * 100, 1)} cm  (z = wysokość blatu względem podstawy robota)")
print(f"Rozrzut obrotu tagu: max {max(obrot_tagu):.1f} st.")
print(f"Kamera względem TCP: {np.round(T_tcp_cam[:3, 3] * 100, 1)} cm   (sprawdź linijką, czy to ma sens)")

if blad.mean() < 0.005:
    print("OCENA: dobra kalibracja.")
elif blad.mean() < 0.010:
    print("OCENA: średnia – spróbuj pominąć podejrzane pozycje albo dozbierać dane.")
else:
    print("OCENA: zła – zbierz dane jeszcze raz (sprawdź rozmiar tagu, czy kamera jest sztywno, czy robot stał).")

with open(args.wynik, "w") as f:
    json.dump({"T_tcp_cam": T_tcp_cam.tolist(), "metoda": nazwa, "pozycji": int(n), "pominiete": sorted(pomin),
               "blad_sredni_mm": float(blad.mean() * 1000), "blad_max_mm": float(blad.max() * 1000),
               "tag20_w_bazie": tag_srodek.tolist(), "dane": os.path.abspath(args.dane)}, f, indent=2)
print(f"Zapisano {args.wynik}")
