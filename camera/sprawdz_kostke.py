"""sprawdz_kostke.py - robot jako linijka: ile mylą się kamery w pozycji kostki.

Uruchom (robot NIE jest sterowany, tylko odczyt):  python sprawdz_kostke.py          (czerwona)
                                                     python sprawdz_kostke.py cube_1   (niebieska)
Wciśnij stop (białe światło) i prowadząc robota ręką ustaw chwytak DOKŁADNIE nad środkiem kostki: palce otwarte,
równo po obu stronach kostki, kilka cm nad nią. Wtedy pozycja chwytaka (z robota, dokładna) = prawdziwa pozycja kostki.
Co 0.5 s: chwytak, kostka z każdej kamery i różnica w poziomie (x, y). Ctrl+C = koniec.
Różnica ok. 0.5 cm lub mniej: kamery dobrze; 1.5-2 cm: kamery się mylą (kalibracja do poprawy).
"""
import sys
import time

import numpy as np
import panda_py

from kamera_kostki import poza_tcp
from kamery import Kamery

KOSTKA = sys.argv[1] if len(sys.argv) > 1 else "cube_2"
NAZWA = {"cube_1": "niebieska", "cube_2": "czerwona", "cube_3": "zielona"}[KOSTKA]

panda = panda_py.Panda("172.16.0.2")
kamery = Kamery()
print(f"Ustaw chwytak nad środkiem kostki: {NAZWA}. Ctrl+C = koniec.")
try:
    while True:
        for _ in range(10):                                  # historia póz dla kamery na chwytaku
            T = poza_tcp(panda)
            kamery.ustaw_poze(T)
            time.sleep(0.05)
        tcp = T[:3, 3]
        m = kamery.fuzja.swieze(KOSTKA, time.perf_counter())
        linia = f"chwytak [{tcp[0] * 100:5.1f} {tcp[1] * 100:5.1f}] cm (z {tcp[2] * 100:4.1f})"
        if not m:
            linia += f" | {NAZWA}: nie widać"
        for kam, (_, p, _, _) in m.items():
            d = (p[:2] - tcp[:2]) * 100
            linia += f" | {kam}: [{p[0] * 100:5.1f} {p[1] * 100:5.1f}]  różnica dx {d[0]:+4.1f} dy {d[1]:+4.1f} = {np.linalg.norm(d):.1f} cm"
        print(linia, flush=True)
except KeyboardInterrupt:
    pass
finally:
    kamery.zamknij()
