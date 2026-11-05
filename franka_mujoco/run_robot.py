"""run_robot.py – policy1 na prawdziwym robocie: kostki z kamer, ruch przez robot_franka (z filtrem stołu).

Uruchom (Desk: stawy odblokowane, FCI włączone, PRZYCISK AWARYJNY W RĘCE):  python run_robot.py
Jeden epizod: HOME -> czekanie, aż kamery zobaczą wszystkie kostki -> Enter -> policy -> po postawieniu wieży
(albo po N_STEPS) powrót do HOME. Log każdego kroku: logi/epizod_<czas>.npz.
Pierwsze uruchomienie:  python run_robot.py --powietrze
    filtr udaje stół 5 cm wyżej: palce i kamery zostają ok. 6.5 cm nad blatem, czyli NAD kostkami – robot wykonuje
    ruchy policy, ale niczego nie dotyka (test pętli, przycisku stop i płynności ruchu).
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "camera"))
import robot_franka as robot
from fuzja import trzyma as trzyma_kostke
from kamery import Kamery
from observations_combine import obs_combine

SCALE = 0.03
N_STEPS = 1000                                               # kroków policy jak w treningu (20 s x robot.TEMPO)
POLICY_PATH = "policy1/2026-11-01_16-15-43_800/exported/policy.pt"
KONIEC_PO = 25                                               # tyle kroków (0.5 s x TEMPO) wieża musi stać, zanim wrócimy do HOME

policy = torch.jit.load(POLICY_PATH, map_location="cpu").eval()


def trzyma(state):
    return trzyma_kostke(not robot.chwytak_otwarty(), state["fingers"], state["fingers_vel"])


def wieza_gotowa(kostki, state):
    """Czerwona na niebieskiej, chwytak otwarty i podniesiony nad wieżę."""
    niebieska, czerwona = kostki["cube_1"][0], kostki["cube_2"][0]
    na_niebieskiej = np.linalg.norm(czerwona[:2] - niebieska[:2]) < 0.02 and 0.04 < czerwona[2] - niebieska[2] < 0.06
    return na_niebieskiej and robot.chwytak_otwarty() and state["tcp_pos"][2] > czerwona[2] + 0.06


def czekaj_na_kostki(kamery, timeout=10.0):
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        s = robot.state()
        kostki = kamery.cubes_position_table(s["tcp_pos"], s["tcp_quat"], False)
        if kostki is not None:
            return kostki
        time.sleep(0.02)
    widziane = sorted(kamery.fuzja.ostatnie)
    raise RuntimeError(f"Kamery nie widzą wszystkich trzech kostek: widziane {widziane or 'żadna'}; klatek z kamer: "
                       f"chwytak {kamery.n_klatek['reka']}, statyw {kamery.n_klatek['statyw']} "
                       f"(0 klatek = kamera nie działa / zajęta przez inny program)")


if "--powietrze" in sys.argv:
    robot.filtr.stol_z += 0.05
    print(f"TRYB W POWIETRZU: filtr zatrzyma palce i kamery ok. {(robot.filtr.stol_z + robot.filtr.h_min) * 100:.1f} cm "
          f"nad bazą (kostki mają 5 cm) – robot nie dotknie stołu ani kostek")

kamery = Kamery()
log_zapisany = False


def zapisz_log():
    global log_zapisany
    if log_zapisany or not log["t"]:
        return
    katalog = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logi")
    os.makedirs(katalog, exist_ok=True)
    plik = os.path.join(katalog, f"epizod_{time.strftime('%Y%m%d_%H%M%S')}.npz")
    n = min(len(v) for v in log.values())                    # przerwany krok mógł dopisać tylko część pól
    np.savez(plik, **{k: np.array(v[:n]) for k, v in log.items()})
    log_zapisany = True
    print(f"Zapisano log {plik}")


log = {k: [] for k in ("t", "q", "dq", "fingers", "tcp_pos", "action", "kostki", "trzyma")}
try:
    robot.reset_scene()
    kostki = czekaj_na_kostki(kamery)
    for k, (pos, _) in kostki.items():
        print(f"{k}: {np.round(pos * 100, 1)} cm")
    input("Enter = start policy (przycisk awaryjny w ręce)")

    last_action = np.zeros(8, dtype=np.float32)
    gotowa = 0
    t0 = time.perf_counter()
    for krok in range(N_STEPS):
        s = robot.state()
        kostki = kamery.cubes_position_table(s["tcp_pos"], s["tcp_quat"], trzyma(s))
        # zwolnione tempo: prędkości x TEMPO, żeby wyglądały jak w treningu (opis w robot_franka.py)
        obs = obs_combine(s["q"], s["q_vel"] * robot.TEMPO, s["fingers"], s["fingers_vel"] * robot.TEMPO, last_action, s["tcp_pos"],
                          s["tcp_quat"], kostki)
        with torch.no_grad():
            action = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
        if not robot.movement(s["q"] + SCALE * action[:7], action[7] >= 0):
            print(f"Robot zatrzymał się (odruch / stop) w kroku {krok} – przerywam")
            print("Błąd robota:", robot.ostatni_blad)
            break
        last_action = action.copy()

        for k, v in (("t", time.perf_counter() - t0), ("q", s["q"]), ("dq", s["q_vel"]), ("fingers", s["fingers"]),
                     ("tcp_pos", s["tcp_pos"]), ("action", action), ("trzyma", trzyma(s)),
                     ("kostki", np.concatenate([np.concatenate(kostki[k]) for k in ("cube_1", "cube_2", "cube_3")]))):
            log[k].append(v)

        gotowa = gotowa + 1 if wieza_gotowa(kostki, s) else 0
        if gotowa >= KONIEC_PO:
            print(f"Wieża postawiona po {krok * robot.DT:.1f} s")
            break
    else:
        print("Koniec czasu epizodu")

    kroki = np.diff(log["t"])
    if len(kroki):                                           # robot mógł stanąć już w pierwszych krokach
        print(f"Krok pętli: średnio {kroki.mean() * 1000:.1f} ms, max {kroki.max() * 1000:.1f} ms   (powinno być {robot.DT * 1000:.0f})")
    print(f"Niezgodności kamer: {kamery.fuzja.n_niezgodnych}, odrzuconych pojedynczych skoków: {kamery.fuzja.n_odrzuconych}")
    zapisz_log()                                             # zanim ktoś przerwie program przy pytaniu niżej
    input("Enter = powrót do HOME (sprawdź, czy nic nie stoi na drodze)")
    robot.reset_scene()                                      # otwiera chwytak i wraca do HOME
finally:
    robot.robot_stop()
    # log PRZED zamykaniem kamer: 5.11 drugie Ctrl+C przerwało zamykanie kamer i log się nie zapisał
    zapisz_log()
    kamery.zamknij()
