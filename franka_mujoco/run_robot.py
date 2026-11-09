"""run_robot.py - policy1 / policy2 na prawdziwym robocie, z menu.

Uruchom (Desk: stawy odblokowane, FCI włączone, PRZYCISK AWARYJNY W RĘCE):  python run_robot.py
Menu:
    h  - powrót do HOME
    1  - policy1: czerwona na niebieską
    2  - policy2: zielona na czerwoną (czerwona musi już stać na niebieskiej)
    3  - policy1, HOME, policy2 (cała wieża)
    p  - tryb w powietrzu wł./wył. (filtr udaje stół 5 cm wyżej, robot nie dotknie kostek)
    q  - koniec
Ctrl+C w trakcie policy przerywa tylko ten epizod. Log każdego epizodu: logi/<policy>_<czas>.npz.
"""
import os
import sys
import time

import numpy as np
import torch

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(DIR, "..", "camera"))
import robot_franka as robot
from fuzja import Fuzja, trzyma as trzyma_kostke
from kamery import Kamery
from observations_combine import obs_combine

SCALE = 0.03
N_STEPS = 1000                     # kroków policy jak w treningu (60 s przy kroku 60 ms)
KONIEC_PO = 25                     # tyle kroków wieża musi stać z otwartym chwytakiem, żeby skończyć epizod
POWIETRZE = 0.05                   # [m] o tyle filtr podnosi stół w trybie w powietrzu

# obie policy mają te same ustawienia ruchu (stałe w robot_franka.py)
POLICY = {
    "p1": dict(plik="policy1/2026-11-05_18-04-46_2499/exported/policy.pt", dolna="cube_1", gorna="cube_2"),
    "p2": dict(plik="policy2/2026-11-06_19-01-50_3200/exported/policy.pt", dolna="cube_2", gorna="cube_3"),
}
for p in POLICY.values():
    p["model"] = torch.jit.load(os.path.join(DIR, p["plik"]), map_location="cpu").eval()


def pytaj(tekst):
    return input(f"{tekst} [t/n] ").strip().lower() == "t"


def trzyma(s):
    return trzyma_kostke(not robot.chwytak_otwarty(), s["fingers"], s["fingers_vel"])


def na_sobie(kostki, dolna, gorna, xy=0.02):
    d, g = kostki[dolna][0], kostki[gorna][0]
    return np.linalg.norm(g[:2] - d[:2]) < xy and 0.04 < g[2] - d[2] < 0.06


def wieza_gotowa(kostki, s, dolna, gorna):
    """Górna na dolnej, chwytak otwarty i podniesiony nad wieżę."""
    return (na_sobie(kostki, dolna, gorna) and robot.chwytak_otwarty()
            and s["tcp_pos"][2] > kostki[gorna][0][2] + 0.06)


def czekaj_na_kostki(timeout=10.0):
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        s = robot.state()
        kostki = kamery.cubes_position_table(s["tcp_pos"], s["tcp_quat"], False)
        if kostki is not None:
            return kostki
        time.sleep(0.02)
    print(f"Kamery nie widzą wszystkich kostek: widziane {sorted(kamery.fuzja.ostatnie) or 'żadna'}, klatek: "
          f"chwytak {kamery.n_klatek['reka']}, statyw {kamery.n_klatek['statyw']} (0 = kamera nie działa)")
    return None


def w_home():
    return np.abs(robot.state()["q"] - robot.HOME).max() < 0.05


def do_home():
    if pytaj("Jechać do HOME? (sprawdź, czy nic nie stoi na drodze)"):
        robot.reset_scene()                                  # otwiera chwytak, kasuje błąd robota, jedzie do HOME
        return True
    return False


def zapisz_log(log, nazwa):
    n = min(len(v) for v in log.values())                   # przerwany krok mógł dopisać tylko część pól
    if n == 0:
        return
    os.makedirs(os.path.join(DIR, "logi"), exist_ok=True)
    plik = os.path.join(DIR, "logi", f"{nazwa}_{time.strftime('%Y%m%d_%H%M%S')}.npz")
    np.savez(plik, **{k: np.array(v[:n]) for k, v in log.items()})
    print(f"Zapisano log {plik}")


def epizod(nazwa):
    """Jedna policy od HOME do postawionej wieży (albo końca czasu). Zwraca True, gdy wieża stoi."""
    p = POLICY[nazwa]
    if (robot.ma_blad() or not w_home()) and not do_home():  # błąd po odruchu kasuje dopiero reset_scene
        return False
    kamery.fuzja = Fuzja(p["gorna"])                       # nowa fuzja: tylko świeże pomiary, właściwa kostka w chwytaku
    kostki = czekaj_na_kostki()
    if kostki is None:
        return False
    for k, (pos, _) in kostki.items():
        print(f"  {k}: {np.round(pos * 100, 1)} cm")
    if nazwa == "p2" and not na_sobie(kostki, "cube_1", "cube_2", xy=0.03):
        print("Uwaga: kamery nie widzą czerwonej na niebieskiej")
    if not pytaj(f"Start {nazwa}? (przycisk awaryjny w ręce)"):
        return False

    log = {k: [] for k in ("t", "q", "dq", "fingers", "tcp_pos", "action", "kostki", "trzyma")}
    last_action = np.zeros(8, dtype=np.float32)
    gotowa, sukces = 0, False
    t0 = time.perf_counter()
    try:
        for krok in range(N_STEPS):
            s = robot.state()
            kostki = kamery.cubes_position_table(s["tcp_pos"], s["tcp_quat"], trzyma(s))
            obs = obs_combine(s["q"], s["q_vel"], s["fingers"], s["fingers_vel"], last_action, s["tcp_pos"],
                              s["tcp_quat"], kostki)
            with torch.no_grad():
                action = p["model"](torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
            if not robot.movement(s["q"] + SCALE * action[:7], action[7] >= 0):
                print(f"Robot stanął w kroku {krok}: {robot.ostatni_blad}")
                break
            last_action = action.copy()
            for k, v in (("t", time.perf_counter() - t0), ("q", s["q"]), ("dq", s["q_vel"]),
                         ("fingers", s["fingers"]), ("tcp_pos", s["tcp_pos"]), ("action", action),
                         ("trzyma", trzyma(s)), ("kostki", np.concatenate([np.concatenate(kostki[c]) for c in
                                                                            ("cube_1", "cube_2", "cube_3")]))):
                log[k].append(v)
            gotowa = gotowa + 1 if wieza_gotowa(kostki, s, p["dolna"], p["gorna"]) else 0
            if gotowa >= KONIEC_PO:
                sukces = True
                print(f"Wieża postawiona po {krok * robot.DT:.1f} s")
                break
        else:
            print("Koniec czasu epizodu")
    except KeyboardInterrupt:
        print("\nPrzerwano epizod (Ctrl+C)")
    finally:
        robot.robot_stop()                                   # robot zostaje w miejscu
        kroki = np.diff(log["t"])
        if len(kroki):
            print(f"Krok pętli: średnio {kroki.mean() * 1000:.1f} ms, max {kroki.max() * 1000:.1f} ms "
                  f"(powinno być {robot.DT * 1000:.0f})")
        print(f"Niezgodności kamer: {kamery.fuzja.n_niezgodnych}, odrzucone skoki: {kamery.fuzja.n_odrzuconych}")
        zapisz_log(log, nazwa)
    return sukces


MENU = "\n[h] HOME   [1] policy1   [2] policy2   [3] policy1 + policy2   [p] w powietrzu: {}   [q] koniec\n> "

kamery = Kamery()
powietrze = False
try:
    while True:
        wybor = input(MENU.format("WŁ." if powietrze else "wył.")).strip().lower()
        if wybor == "h":
            do_home()
        elif wybor == "1":
            epizod("p1")
        elif wybor == "2":
            epizod("p2")
        elif wybor == "3":
            if epizod("p1") and do_home():
                epizod("p2")
        elif wybor == "p":
            powietrze = not powietrze
            robot.filtr.stol_z += POWIETRZE if powietrze else -POWIETRZE
            print(f"Filtr: palce i kamery najniżej {(robot.filtr.stol_z + robot.filtr.h_min) * 100:.1f} cm nad bazą")
        elif wybor == "q":
            break
finally:
    robot.robot_stop()
    kamery.zamknij()
