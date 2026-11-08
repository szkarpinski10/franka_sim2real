"""analiza_logu.py - analiza epizodu z robota (logi z run_robot.py).

Uruchom:  python analiza_logu.py                       (najnowszy log)
          python analiza_logu.py logi/p1_20261105_131500.npz
Pokazuje równość pętli, prędkości i przyspieszenia stawów, zmiany akcji policy i pracę chwytaka.
"""
import glob
import os
import sys

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
plik = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob(os.path.join(DIR, "logi", "*.npz")), key=os.path.getmtime)
d = np.load(plik)
t, q, dq, a = d["t"], d["q"], d["dq"], d["action"]
fingers, trzyma = d["fingers"], d["trzyma"]
n = len(t)
kroki = np.diff(t)
dt = np.median(kroki)
print(f"{os.path.basename(plik)}: {n} kroków, {t[-1]:.1f} s")

print("\nPĘTLA")
print(f"  krok: mediana {dt * 1000:.1f} ms, 99% < {np.percentile(kroki, 99) * 1000:.1f} ms, max {kroki.max() * 1000:.1f} ms,"
      f" kroków dłuższych niż 1.5 x mediana: {(kroki > 1.5 * dt).sum()}")

print("\nRUCH STAWÓW (z q_vel robota)")
v = np.abs(dq).max(axis=1)
acc = np.abs(np.diff(dq, axis=0)).max(axis=1) / kroki
print(f"  prędkość: 99% < {np.percentile(v, 99):.2f} rad/s, max {v.max():.2f} rad/s")
print(f"  przyspieszenie (co krok policy): 99% < {np.percentile(acc, 99):.1f} rad/s2, max {acc.max():.1f} rad/s2")
ruch = (np.abs(dq[1:]) > 0.05) & (np.abs(dq[:-1]) > 0.05)
zmiany = ((np.sign(dq[1:]) != np.sign(dq[:-1])) & ruch).sum()
print(f"  zmiany kierunku ruchu stawów: {zmiany / t[-1]:.1f} na s   (MuJoCo, pełne tempo: ok. 7 na s)")

print("\nPOLICY")
da = np.abs(np.diff(a[:, :7], axis=0)).mean(axis=1)
print(f"  średnia |zmiana akcji| między krokami: {da.mean():.3f}   (MuJoCo: ok. 0.27 normalnie, 0.36 w trybie --powietrze)")
print(f"  akcje stawów: średnio |a| {np.abs(a[:, :7]).mean():.2f}, max {np.abs(a[:, :7]).max():.2f}")

print("\nCHWYTAK")
zamkniecia = int(((a[1:, 7] < 0) & (a[:-1, 7] >= 0)).sum())
print(f"  komend 'zamknij': {zamkniecia}, kroków z kostką w chwytaku: {int(trzyma.sum())}, "
      f"szerokość min {fingers.min() * 200:.1f} cm, max {fingers.max() * 200:.1f} cm")

print("\nNAJGWAŁTOWNIEJSZE CHWILE (przyspieszenie stawu)")
for i in np.argsort(acc)[-5:][::-1]:
    staw = int(np.argmax(np.abs(dq[i + 1] - dq[i])))
    print(f"  t = {t[i + 1]:5.1f} s: {acc[i]:5.1f} rad/s2 (staw {staw + 1}), prędkość {v[i + 1]:.2f} rad/s, "
          f"TCP z = {d['tcp_pos'][i + 1][2] * 100:.1f} cm")
