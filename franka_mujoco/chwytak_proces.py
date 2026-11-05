"""chwytak_proces.py – chwytak Pandy w OSOBNYM PROCESIE (uruchamiany z robot_franka.py).

Dlaczego proces, a nie wątek: funkcje chwytaka z panda_py (read_once, move, grasp) czekają na chwytak, NIE puszczając
GIL-a – cały Python stoi (5.11: drugi wątek dostawał 2% czasu; read_once średnio 57 ms, max 100 ms, grasp ok. 0.5 s).
W pętli robota dawało to 10-15 kroków/s zamiast 50 (drgania), a w wątku w tle – kamery bez ani jednej klatki.
Osobny proces ma własny GIL; z pętlą robota wymienia tylko liczby we wspólnej pamięci (multiprocessing.Value).
Jedno połączenie z chwytakiem: drugie Gripper(IP) odcina pierwsze.
"""
import time


def petla(ip, cel_otwarty, jest_otwarty, szerokosc, predkosc, v_palcow=0.1):
    """cel_otwarty: czego chce policy (1/0); jest_otwarty: stan po ostatniej komendzie; szerokosc [m]; predkosc
    jednego palca [m/s]; v_palcow = prędkość rozstawu palców przy move/grasp [m/s] (0.1 / TEMPO). Jedna komenda albo jeden odczyt naraz – opóźnienie komendy najwyżej jeden odczyt (ok. 0.1 s)."""
    # panda_py ładujemy dopiero TU, w procesie potomnym: jeśli proces rodzica załadował panda_py przed fork,
    # panda.get_state() w rodzicu zawiesza się na zawsze (sprawdzone 5.11)
    from panda_py import libfranka
    gripper = libfranka.Gripper(ip)
    prev_width, prev_t = None, None
    while True:
        try:
            cel = bool(cel_otwarty.value)
            if cel != bool(jest_otwarty.value):
                if cel:
                    gripper.move(0.08, v_palcow)
                else:
                    gripper.grasp(0.0, v_palcow, 20.0, 0.08, 0.08)
                jest_otwarty.value = int(cel)
                continue
            width = gripper.read_once().width
        except Exception as e:
            print("Błąd chwytaka:", e)
            time.sleep(0.5)
            continue
        now = time.perf_counter()
        if prev_width is not None and now > prev_t:
            predkosc.value = (width - prev_width) / 2 / (now - prev_t)
        prev_width, prev_t = width, now
        szerokosc.value = width
