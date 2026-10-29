"""sprawdz_kalibracje.py – robot POWOLI jedzie nad kostki wykryte kamerą; sprawdzasz na oko, czy trafia.

Uruchom (przy robocie, Desk: stawy odblokowane, FCI włączone, PRZYCISK AWARYJNY POD RĘKĄ):
    python sprawdz_kalibracje.py
Przebieg: HOME -> pomiar kostek kamerą -> dla każdej kostki (po Enterze) zjazd NAD_KOSTKA nad jej górę
-> patrzysz, czy palce są nad środkiem (zmierz linijką przesunięcie) -> pomiar z bliska dla porównania -> HOME.
Kostki stawiaj w zakresie z treningu: x 0.35-0.55 m, y -0.20..0.20 m, obrócone o 45 st.
Każdy ruch czeka na Enter. Cel poza bezpiecznym obszarem (OBSZAR) jest odrzucany.
Ruch jak w do_home.py: własny sterownik stawów, najszybszy staw max V_MAX. Kąty stawów do celu liczymy krokami od
aktualnej pozycji (ik_blisko), bo panda_py.ik / move_to_pose potrafią wybrać zupełnie inne ułożenie ramienia
(np. staw 1 i 3 obrócone o ok. 2 rad) – wtedy robot jechałby wielkim łukiem.
Wymaga:  kalibracja.json (z kalibruj_handeye.py), panda_py pasujące do robota
"""
import time

import numpy as np
import panda_py

from panda_py import controllers

from kamera_kostki import HALF, KOLOR, KameraKostki, poza_tcp

IP = "172.16.0.2"                                            # adres robota (jak w robot_franka.py)
HOME = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.785])     # jak w robot_franka.py
V_MAX = 0.1                                                  # największa prędkość stawu [rad/s] (jak w do_home.py)
MAX_ZMIANA = 1.2                                             # cel wymagający większego obrotu stawu [rad] odrzucamy
NAD_KOSTKA = 0.03                                            # wysokość TCP nad górą kostki [m]
OBSZAR = {"x": (0.25, 0.70), "y": (-0.35, 0.35), "z": (0.07, 0.60)}   # dozwolone cele TCP [m] (góra kostki: ok. 0.05)
# limity stawów Pandy [rad] (jak w robot_franka.py)
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
Q_MARGIN = 0.05


def zapytaj(tekst):
    """Enter = dalej, s = pomiń, q = koniec."""
    odp = input(f"{tekst}  [Enter = dalej, s = pomiń, q = koniec] ").strip().lower()
    if odp == "q":
        raise KeyboardInterrupt
    return odp != "s"


def w_obszarze(p):
    return all(lo <= v <= hi for v, (lo, hi) in zip(p, OBSZAR.values()))


def pokaz(kostki):
    for nazwa, (pos, quat) in sorted(kostki.items()):
        yaw = np.degrees(2 * np.arctan2(quat[2], quat[3]))
        print(f"  {nazwa} {KOLOR[nazwa]:9s} {np.round(pos * 100, 1)} cm   obrót {yaw:5.1f} st.")


def log_R(R):
    """Obrót R jako wektor (oś * kąt) [rad]."""
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if a < 1e-9:
        return np.zeros(3)
    return a / (2 * np.sin(a)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def blad(T, T_cel):
    """Błąd pozy: 3 x pozycja [m] + 3 x obrót [rad], w układzie bazy."""
    return np.concatenate([T_cel[:3, 3] - T[:3, 3], log_R(T_cel[:3, :3] @ T[:3, :3].T)])


def ik_blisko(T_cel, q0, iteracje=300):
    """Kąty stawów dla pozy T_cel, liczone małymi krokami od q0 (rozwiązanie zostaje blisko q0)."""
    q = np.array(q0, dtype=float)
    for _ in range(iteracje):
        T = panda_py.fk(q)
        e = blad(T, T_cel)
        if np.linalg.norm(e) < 1e-5:
            break
        J = np.zeros((6, 7))                                 # jakobian liczony numerycznie z fk
        for j in range(7):
            dq = np.zeros(7)
            dq[j] = 1e-6
            J[:, j] = -blad(panda_py.fk(q + dq), T) / 1e-6
        q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e), Q_MIN + Q_MARGIN, Q_MAX - Q_MARGIN)
    return q, np.linalg.norm(blad(panda_py.fk(q), T_cel))


def jedz(q_cel):
    """Płynny ruch stawów do q_cel (jak do_home.py / reset_scene w robot_franka.py)."""
    q0 = np.array(panda.get_robot().read_once().q)
    dq = np.asarray(q_cel) - q0
    T = max(1.875 * np.max(np.abs(dq)) / V_MAX, 1.0)
    print(f"  ruch: największy obrót stawu {np.degrees(np.max(np.abs(dq))):.0f} st., potrwa {T:.0f} s")
    ctrl = controllers.JointPosition()
    panda.start_controller(ctrl)
    try:
        with panda.create_context(frequency=1000, max_runtime=T + 0.5) as ctx:
            while ctx.ok():
                s = min(ctrl.get_time() / T, 1.0)
                ctrl.set_control(q0 + dq * (10 * s**3 - 15 * s**4 + 6 * s**5))
    finally:
        panda.stop_controller()
    time.sleep(0.5)                                          # robot ma stać przy zdjęciu


def do_home():
    jedz(HOME)


panda = panda_py.Panda(IP)
panda.recover()
kam = KameraKostki()
try:
    if zapytaj("Robot pojedzie POWOLI do HOME."):
        do_home()
    kostki = kam.zmierz(poza_tcp(panda))
    print(f"Kamera widzi z HOME {len(kostki)} kostki:")
    pokaz(kostki)
    if not kostki:
        raise SystemExit("Nie widać żadnej kostki – sprawdź podglad_tagow.py / kamera_kostki.py.")

    for nazwa, (pos, quat) in sorted(kostki.items()):
        T = poza_tcp(panda)                                  # chwytak w dół jak w HOME, zmieniamy pozycję i obrót wokół pionu
        T[:3, 3] = pos + [0.0, 0.0, HALF + NAD_KOSTKA]
        # palce równolegle do ścianek kostki: obrót dłoni wokół pionu o najmniejszy kąt (kostka wygląda tak samo co 90 st.)
        hand_yaw = np.arctan2(T[1, 1], T[0, 1])              # jak w robot_franka.py / observations_combine.py
        cube_yaw = 2 * np.arctan2(quat[2], quat[3])
        obrot = (cube_yaw - hand_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        c, s = np.cos(obrot), np.sin(obrot)
        T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ T[:3, :3]
        print(f"\n{nazwa}: kostka {np.degrees(cube_yaw):.1f} st., chwytak {np.degrees(hand_yaw):.1f} st. "
              f"-> obracam chwytak o {np.degrees(obrot):+.1f} st.")
        if not w_obszarze(T[:3, 3]):
            print(f"{nazwa}: cel {np.round(T[:3, 3] * 100, 1)} cm poza OBSZAR – pomijam.")
            continue
        q_teraz = np.array(panda.get_robot().read_once().q)
        q_cel, err = ik_blisko(T, q_teraz)
        if err > 1e-3 or np.max(np.abs(q_cel - q_teraz)) > MAX_ZMIANA:
            print(f"{nazwa}: nie umiem tam dojechać bez dużego obrotu stawów (błąd {err * 1000:.1f} mm, "
                  f"obrót {np.max(np.abs(q_cel - q_teraz)):.2f} rad) – pomijam.")
            continue
        if not zapytaj(f"\n{nazwa} ({KOLOR[nazwa]}): jadę do {np.round(T[:3, 3] * 100, 1)} cm "
                       f"({NAD_KOSTKA * 100:.0f} cm nad kostką)."):
            continue
        jedz(q_cel)
        tcp = poza_tcp(panda)[:3, 3]
        print(f"  TCP stoi w {np.round(tcp * 100, 1)} cm. Sprawdź na oko/linijką, czy środek palców jest nad środkiem "
              f"kostki. Przesunięcie > 1 cm = kalibracja do poprawy.")
        z_bliska = kam.zmierz(poza_tcp(panda))              # druga opinia: ta sama kostka widziana z bliska
        if nazwa in z_bliska:
            roznica = (z_bliska[nazwa][0] - pos) * 100
            print(f"  Pomiar z bliska różni się od pomiaru z HOME o {np.round(roznica, 1)} cm "
                  f"(|xy| = {np.linalg.norm(roznica[:2]):.1f} cm).")
        else:
            print("  Z bliska kamera nie widzi tej kostki.")
        input("  Enter = dalej ")
    if zapytaj("\nPowrót do HOME."):
        do_home()
except KeyboardInterrupt:
    print("\nPrzerwano.")
finally:
    kam.zamknij()
