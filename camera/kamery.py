"""kamery.py – obie kamery w tle + fuzja; daje pozycje kostek dla policy.

Jako moduł (pętla robota, run_robot.py):
    from kamery import Kamery
    kamery = Kamery()
    kostki = kamery.cubes_position_table(tcp_pos, tcp_quat, trzyma)   # {"cube_1": (pos, quat), ...} albo None
    kamery.zamknij()
Każda kamera ma swój wątek: bierze klatkę, wykrywa tagi, przelicza kostki na układ bazy i wkłada do fuzji.
Pętla robota (50 Hz) nigdy nie czeka na kamerę (ok. 10 kl/s) – bierze ostatni wynik fuzji.
Kamera na chwytaku potrzebuje pozy TCP w chwili zdjęcia: dostaje ją z pętli robota (cubes_position_table
zapisuje pozę przy każdym wywołaniu), więc wątek kamery sam nie rozmawia z robotem.

Sprawdzenie przy robocie (robot stoi, bez sterowania):  python kamery.py
    co 0.5 s wypisuje każdą kostkę z obu kamer osobno i różnicę między nimi (dobrze: < 1.5 cm).
Wymaga:  kalibracja.json, logitech.json, statyw.json
"""
import json
import os
import threading
import time
from collections import deque

import numpy as np

from fuzja import Fuzja, KOSTKI, poza, usrednij_yaw

DIR = os.path.dirname(os.path.abspath(__file__))
CUBE_TAG = 0.035
V_MAX_REKA = 0.05                                            # [m/s] szybciej -> zdjęcie z chwytaka rozmyte/nieaktualne
POZA_SWIEZA = 0.05                                           # [s] w historii nie ma pozy tak blisko chwili zdjęcia -> pomijamy
# opóźnienie obrazu = od zdjęcia do odebrania klatki. Wynik dostaje pozę TCP z CHWILI ZDJĘCIA (z historii), bo przy
# czerwonej w chwytaku 100 ms różnicy to kilka cm (MuJoCo: opóźnienie 40 ms -> 26/30 wież, 100 ms -> 18/30).
OPOZNIENIE_REKA = 0.06                                       # [s] tylko gdy RealSense nie poda czasu zdjęcia
OPOZNIENIE_STATYW = 0.10                                     # [s] ZMIERZ: python logitech.py --opoznienie


class Kamery:
    def __init__(self, reka=True, statyw=True):
        self.fuzja = Fuzja()
        self.historia = deque(maxlen=150)                    # (czas, poza TCP) z ostatnich ~3 s pętli robota
        self.dziala = True
        self.n_klatek = {"reka": 0, "statyw": 0}
        self.opoznienie = {"reka": OPOZNIENIE_REKA, "statyw": OPOZNIENIE_STATYW}
        self.watki = []
        if reka:
            self.watki.append(threading.Thread(target=self._watek_reka, daemon=True))
        if statyw:
            self.watki.append(threading.Thread(target=self._watek_statyw, daemon=True))
        for w in self.watki:
            w.start()

    def ustaw_poze(self, T_base_tcp):
        self.historia.append((time.perf_counter(), T_base_tcp))

    def poza(self, t):
        """Poza TCP najbliższa chwili t (z historii) albo None, jeśli takiej nie ma (np. pętla robota nie działa)."""
        historia = list(self.historia)                       # kopia: pętla robota dopisuje w tym samym czasie
        if not historia:
            return None
        t_najblizsza, T = min(historia, key=lambda x: abs(x[0] - t))
        return T if abs(t_najblizsza - t) < POZA_SWIEZA else None

    def cubes_position_table(self, tcp_pos, tcp_quat, trzyma):
        T = poza(tcp_pos, tcp_quat)
        self.ustaw_poze(T)
        return self.fuzja.kostki(T, trzyma, time.perf_counter())

    def _wloz(self, kamera, pomiary, t, T_base_tcp):
        for nazwa, lista in pomiary.items():                 # lista = [(środek, yaw)] – po jednym z każdej ściany
            pos = np.mean([p for p, _ in lista], axis=0)
            yaw = usrednij_yaw([y for _, y in lista])
            self.fuzja.dodaj(kamera, nazwa, pos, yaw, t, T_base_tcp)

    def _watek_reka(self):
        import pyrealsense2 as rs
        from kamera_kostki import KameraKostki
        kam = KameraKostki()
        try:
            while self.dziala:
                color = kam.pipe.wait_for_frames(5000).get_color_frame()
                if not color:
                    continue
                img = np.asanyarray(color.get_data())
                # RealSense podaje czas zdjęcia w czasie komputera [ms] -> prawdziwe opóźnienie tej klatki
                if color.get_frame_timestamp_domain() == rs.timestamp_domain.global_time:
                    self.opoznienie["reka"] = time.time() - color.get_timestamp() / 1000
                t = time.perf_counter() - self.opoznienie["reka"]   # chwila zdjęcia
                self.n_klatek["reka"] += 1
                T, T_przed = self.poza(t), self.poza(t - 0.05)
                if T is None or T_przed is None:
                    continue
                # robot jechał szybko -> zdjęcie rozmyte, a mały błąd czasu to duży błąd pozycji
                if np.linalg.norm(T[:3, 3] - T_przed[:3, 3]) / 0.05 > V_MAX_REKA:
                    continue
                self._wloz("reka", kam.pomiary(kam.wykryj(img), T), t, T)
        finally:
            kam.zamknij()

    def _watek_statyw(self):
        from kamera_kostki import KOSTKI as TAGI, srodek_i_yaw
        from logitech import Logitech
        with open(os.path.join(DIR, "statyw.json")) as f:
            T_base_cam = np.array(json.load(f)["T_base_cam"])
        kam = Logitech()
        try:
            while self.dziala:
                img = kam.klatka()
                t = time.perf_counter() - self.opoznienie["statyw"]   # chwila zdjęcia
                self.n_klatek["statyw"] += 1
                pomiary = {}
                for d in kam.wykryj(img, CUBE_TAG):
                    nazwa = next((k for k, ids in TAGI.items() if d.tag_id in ids), None)
                    if nazwa is not None:
                        pomiary.setdefault(nazwa, []).append(srodek_i_yaw(T_base_cam, d.pose_R, d.pose_t))
                self._wloz("statyw", pomiary, t, self.poza(t))  # poza TCP – tylko dla czerwonej w chwytaku
        finally:
            kam.zamknij()

    def zamknij(self):
        self.dziala = False
        for w in self.watki:
            w.join(timeout=2.0)


if __name__ == "__main__":
    import panda_py
    from kamera_kostki import poza_tcp

    panda = panda_py.Panda("172.16.0.2")
    kamery = Kamery()
    try:
        while True:
            for _ in range(10):                              # w pętli robota pozę wpisuje cubes_position_table
                kamery.ustaw_poze(poza_tcp(panda))
                time.sleep(0.05)
            teraz = time.perf_counter()
            print(f"--- klatki: chwytak {kamery.n_klatek['reka']}, statyw {kamery.n_klatek['statyw']}   opóźnienie "
                  f"chwytak {kamery.opoznienie['reka'] * 1000:.0f} ms, statyw {kamery.opoznienie['statyw'] * 1000:.0f} ms")
            for k in KOSTKI:
                m = kamery.fuzja.swieze(k, teraz)
                txt = "   ".join(f"{kam}: {np.round(p * 100, 1)} cm {np.degrees(y):5.1f} st." for kam, (_, p, y, _) in m.items())
                if len(m) == 2:
                    txt += f"   różnica {np.linalg.norm(m['reka'][1] - m['statyw'][1]) * 100:.1f} cm"
                print(f"{k}: {txt or 'nie widać'}")
    except KeyboardInterrupt:
        pass
    finally:
        kamery.zamknij()
