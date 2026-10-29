"""do_home.py – powolny ruch robota do HOME (jak reset_scene w robot_franka.py, tylko wolniej i z potwierdzeniem).

Uruchom (Desk: stawy odblokowane, FCI włączone, PRZYCISK AWARYJNY W RĘCE):  python do_home.py
Najpierw wypisuje, o ile obróci się każdy staw i ile potrwa ruch – rusza dopiero po wpisaniu "tak".
Wszystkie stawy jadą razem i kończą w tej samej chwili; najszybszy staw osiąga najwyżej V_MAX.
Płynny start i stop (wielomian 5. stopnia), więc średnia prędkość jest ok. 2 razy mniejsza niż V_MAX.
Wymaga:  panda_py pasujące do robota
"""
import numpy as np
import panda_py
from panda_py import controllers

IP = "172.16.0.2"
HOME = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.785])     # jak w robot_franka.py
V_MAX = 0.1                                                  # największa prędkość stawu [rad/s]

panda = panda_py.Panda(IP)
q0 = np.array(panda.get_state().q)
dq = HOME - q0
T = max(1.875 * np.max(np.abs(dq)) / V_MAX, 1.0)             # 1.875 = szczyt prędkości wielomianu 5. stopnia

print("staw   teraz [st.]   HOME [st.]   obrót [st.]")
for i in range(7):
    print(f"  {i + 1}    {np.degrees(q0[i]):8.1f}    {np.degrees(HOME[i]):8.1f}    {np.degrees(dq[i]):+8.1f}")
print(f"Ruch potrwa {T:.0f} s (najszybszy staw max {V_MAX} rad/s).")
if input('Wpisz "tak", żeby ruszyć: ').strip().lower() != "tak":
    raise SystemExit("Anulowano – robot się nie ruszył.")

panda.recover()
ctrl = controllers.JointPosition()
panda.start_controller(ctrl)
try:
    sekunda = 0
    with panda.create_context(frequency=1000, max_runtime=T + 0.5) as ctx:
        while ctx.ok():
            t = ctrl.get_time()
            s = min(t / T, 1.0)
            b = 10 * s**3 - 15 * s**4 + 6 * s**5
            ctrl.set_control(q0 + dq * b)
            if t >= sekunda:                                 # postęp co 5 s
                print(f"  {t:4.0f} s / {T:.0f} s")
                sekunda += 5
finally:
    panda.stop_controller()

q = np.array(panda.get_state().q)
print(f"Koniec. Największa odchyłka od HOME: {np.degrees(np.max(np.abs(q - HOME))):.2f} st.")
print(f"TCP w HOME: get_position() = {np.round(np.array(panda.get_position()).flatten(), 4)}   "
      f"get_orientation() = {np.round(np.array(panda.get_orientation()).flatten(), 4)}   (do pomiary.txt)")
