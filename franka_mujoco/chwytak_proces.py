"""chwytak_proces.py - chwytak w osobnym procesie (uruchamiany z robot_franka.py).

Funkcje chwytaka z panda_py blokują cały Python na czas odczytu/ruchu, więc w tym samym procesie spowalniały
pętlę robota i kamery. Z pętlą robota wymieniamy tylko liczby we wspólnej pamięci.
W czasie move/grasp szerokości nie da się czytać, więc zapisujemy początek ruchu (ruch = (t0, szerokość, cel)),
a robot_franka.state() liczy z tego szerokość w trakcie ruchu (palce jadą ze stałą prędkością).
"""
import time

SILA_CHWYTU = 20.0      # [N] 10 N za słabo: kostka wyślizgiwała się przy podnoszeniu (12.11); 20 N zadziałało


def petla(ip, cel_otwarty, jest_otwarty, szerokosc, predkosc, v_palcow=0.1, ruch=None):
    # panda_py importujemy dopiero tutaj, w procesie potomnym
    from panda_py import libfranka
    gripper = libfranka.Gripper(ip)
    prev_width, prev_t = None, None
    while True:
        try:
            cel = bool(cel_otwarty.value)
            if cel != bool(jest_otwarty.value):
                if ruch is not None:                         # [trwa, t0, szerokość na starcie, szerokość docelowa]
                    ruch[1], ruch[2], ruch[3] = time.perf_counter(), szerokosc.value, 0.08 if cel else 0.0
                    ruch[0] = 1.0
                if cel:
                    gripper.move(0.08, v_palcow)
                else:
                    gripper.grasp(0.0, v_palcow, SILA_CHWYTU, 0.08, 0.08)
                jest_otwarty.value = int(cel)
                width = gripper.read_once().width            # od razu prawdziwa szerokość po ruchu
                szerokosc.value = width
                predkosc.value = 0.0
                prev_width, prev_t = width, time.perf_counter()
                if ruch is not None:
                    ruch[0] = 0.0
                continue
            width = gripper.read_once().width
        except Exception as e:
            if ruch is not None:
                ruch[0] = 0.0
            print("Błąd chwytaka:", e)
            time.sleep(0.5)
            continue
        now = time.perf_counter()
        if prev_width is not None and now > prev_t:
            predkosc.value = (width - prev_width) / 2 / (now - prev_t)
        prev_width, prev_t = width, now
        szerokosc.value = width
