"""chwytak_proces.py - chwytak w osobnym procesie (uruchamiany z robot_franka.py).

Funkcje chwytaka z panda_py blokują cały Python na czas odczytu/ruchu, więc w tym samym procesie spowalniały
pętlę robota i kamery. Z pętlą robota wymieniamy tylko liczby we wspólnej pamięci.
"""
import time


def petla(ip, cel_otwarty, jest_otwarty, szerokosc, predkosc, v_palcow=0.1):
    # panda_py importujemy dopiero tutaj, w procesie potomnym
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
