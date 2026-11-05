"""fuzja.py – łączy pomiary kostek z dwóch kamer w jedną pozycję każdej kostki (sam numpy, bez kamer).

Kamery: "reka" = D435i na chwytaku, "statyw" = Logitech obok stołu. Każda kamera wkłada pomiary przez dodaj(),
policy bierze wynik z kostki(). Czas podajemy jawnie (teraz, t), więc ten sam kod działa na robocie
(time.perf_counter) i w symulacji (czas symulacji).

Zasady:
  obie kamery mają świeży pomiar  -> średnia (przy niezgodności > 1.5 cm wygrywa kamera na chwytaku, bo jest bliżej)
  świeży pomiar ma jedna kamera    -> ta jedna
  żadna                            -> ostatnia znana pozycja (kostka nie znika z obserwacji)
  czerwona w chwytaku              -> jedzie razem z TCP: trzymamy jej położenie WZGLĘDEM TCP (offset);
                                      każdy świeży pomiar po chwycie poprawia offset
Kostki jak w Isaacu: cube_1 = niebieska, cube_2 = czerwona, cube_3 = zielona. Kwaterniony x, y, z, w.
"""
import threading

import numpy as np

KOSTKI = ("cube_1", "cube_2", "cube_3")
CZERWONA = "cube_2"
SWIEZY = 0.3                                                 # [s] pomiar starszy od tego nie jest już "świeży"
NIEZGODNE = 0.015                                            # [m] tyle mogą się różnić kamery, zanim uznamy, że coś nie tak
PO_CHWYCIE = 0.1                                             # [s] po zamknięciu palców kostka jeszcze się układa
SKOK = 0.03                                                  # [m] dalej od ostatniej pozycji -> czekamy na potwierdzenie
# 5.11 na robocie: pojedyncze złe klatki (niejednoznaczny AprilTag, częściowo zasłonięty tag) dawały 1.5-5 cm błędu,
# a w następnej klatce pomiar był znów dobry. Przesunięta kostka zostaje w nowym miejscu, więc skok przyjmujemy dopiero,
# gdy drugi, NOWSZY pomiar potwierdzi go z dokładnością do NIEZGODNE.
# kostka wygląda tak samo co 90 st., więc kamera zna obrót tylko w jakimś przedziale 90 st. Policy patrzy na kwaternion
# kostki wprost i działa najlepiej z przedziałem [45, 135) st. (MuJoCo, 60 układów: 50/60; [0, 90): 41/60)
YAW_OD = np.radians(45)


def trzyma(zamkniety, palec, v_palca):
    """Chwytak trzyma kostkę: zamknięty, a palce STANĘŁY na szerokości kostki (5 cm). Sama szerokość nie wystarczy –
    przy zamykaniu palce przechodzą przez 5 cm, zanim dotkną kostki. palec = pół szerokości chwytaka [m]."""
    return zamkniety and 0.04 < 2 * palec < 0.06 and abs(v_palca) < 0.005


def yaw_to_quat(yaw):
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])


def usrednij_yaw(yaws):
    """Średnia kątów, które powtarzają się co 90 st. (kostka wygląda tak samo po obrocie o 90 st.)."""
    yaws = np.asarray(yaws)
    y = np.arctan2(np.sin(4 * yaws).mean(), np.cos(4 * yaws).mean()) / 4
    return (y - YAW_OD) % (np.pi / 2) + YAW_OD


def quat_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R):
    """Macierz obrotu -> kwaternion x, y, z, w."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])                    # znaki z elementów poza przekątną
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    return np.array([x, y, z, w])


def poza(pos, quat):
    """Pozycja + kwaternion -> macierz 4 x 4."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = quat_to_matrix(quat), pos
    return T


class Fuzja:
    def __init__(self):
        self.lock = threading.Lock()                         # kamery piszą z innych wątków niż pętla robota
        self.pomiary = {"reka": {}, "statyw": {}}            # kamera -> kostka -> (t, pos, yaw, T_base_tcp)
        self.ostatnie = {}                                   # kostka -> (pos, quat) – ostatni wynik
        self.offset = None                                   # czerwona w układzie TCP (4 x 4), gdy trzymana
        self.t_chwytu = None
        self.n_niezgodnych = 0                               # do logów: ile razy kamery się nie zgodziły
        self.kandydat = {}                                   # kostka -> (t, pos) – skok czekający na potwierdzenie
        self.n_odrzuconych = 0                               # do logów: ile pojedynczych skoków odrzucono
        self.t_puszczenia = None                             # chwila puszczenia czerwonej (skok z "w chwytaku" na pomiar)

    def dodaj(self, kamera, kostka, pos, yaw, t, T_base_tcp=None):
        """Jeden pomiar kostki. T_base_tcp = poza TCP w chwili zdjęcia (potrzebna do czerwonej w chwytaku)."""
        with self.lock:
            self.pomiary[kamera][kostka] = (t, np.asarray(pos, dtype=float), float(yaw), T_base_tcp)

    def swieze(self, kostka, teraz, od=None):
        """Świeże pomiary kostki z obu kamer: {kamera: (t, pos, yaw, T_base_tcp)}."""
        out = {}
        for kamera, kostki in self.pomiary.items():
            m = kostki.get(kostka)
            if m is not None and teraz - m[0] < SWIEZY and (od is None or m[0] >= od):
                out[kamera] = m
        return out

    def kostki(self, T_base_tcp, trzyma, teraz):
        """Pozycje wszystkich kostek {"cube_1": (pos, quat), ...}; None, dopóki któraś nie była ani razu widziana.
        T_base_tcp = obecna poza TCP (4 x 4), trzyma = czy chwytak trzyma czerwoną kostkę."""
        with self.lock:
            for kostka in KOSTKI:
                if kostka == CZERWONA and trzyma and CZERWONA in self.ostatnie:
                    self.ostatnie[kostka] = self._czerwona_w_chwytaku(T_base_tcp, teraz)
                    continue
                if kostka == CZERWONA:
                    if self.offset is not None:                  # właśnie puszczona
                        self.t_puszczenia = teraz
                    self.offset, self.t_chwytu = None, None
                m = self.swieze(kostka, teraz)
                zgodne_obie = False                              # obie kamery widzą to samo -> skok bez potwierdzenia
                if len(m) == 2:
                    zgodne_obie = np.linalg.norm(m["reka"][1] - m["statyw"][1]) <= NIEZGODNE
                    if np.linalg.norm(m["reka"][1] - m["statyw"][1]) > NIEZGODNE:
                        self.n_niezgodnych += 1
                        # wygrywa pomiar bliższy ostatniej znanej pozycji (kostka nie skacze o kilka cm); 5.11 na robocie
                        # myliła się raz kamera na chwytaku (obrót, 4-5 cm), raz statyw (częściowo zasłonięta, 2.5 cm)
                        if abs(m["reka"][0] - m["statyw"][0]) > 0.15:    # z różnych chwil -> nowszy (starszy mógł
                            kamera = max(m, key=lambda k: m[k][0])      # być złą klatką albo kostka się przesunęła)
                        elif kostka in self.ostatnie:
                            stara = self.ostatnie[kostka][0]
                            kamera = min(m, key=lambda k: np.linalg.norm(m[k][1] - stara))
                        else:
                            kamera = "reka"
                        m = {kamera: m[kamera]}
                if m:
                    pos = np.mean([p for _, p, _, _ in m.values()], axis=0)
                    yaw = usrednij_yaw([y for _, _, y, _ in m.values()])
                    t_pomiaru = max(t for t, _, _, _ in m.values())
                    tuz_po_puszczeniu = (kostka == CZERWONA and self.t_puszczenia is not None
                                         and teraz - self.t_puszczenia < 1.0)
                    podejrzany = not zgodne_obie and not tuz_po_puszczeniu    # skok widziany przez jedną kamerę
                    if podejrzany and kostka in self.ostatnie and np.linalg.norm(pos - self.ostatnie[kostka][0]) > SKOK:
                        kand = self.kandydat.get(kostka)
                        if kand is None or t_pomiaru <= kand[0] or np.linalg.norm(pos - kand[1]) > NIEZGODNE:
                            if kand is None or t_pomiaru > kand[0]:  # nowy (inny) skok – zapamiętaj, nie przyjmuj
                                if kand is not None:
                                    self.n_odrzuconych += 1
                                self.kandydat[kostka] = (t_pomiaru, pos)
                            continue
                    elif kostka in self.kandydat:                # był skok, ale kolejny pomiar wrócił na stare miejsce
                        self.n_odrzuconych += 1
                    self.kandydat.pop(kostka, None)
                    self.ostatnie[kostka] = (pos, yaw_to_quat(yaw))
            if any(k not in self.ostatnie for k in KOSTKI):
                return None
            return dict(self.ostatnie)

    def _czerwona_w_chwytaku(self, T_base_tcp, teraz):
        if self.offset is None:                              # chwila chwytu: kostka tam, gdzie była widziana ostatnio
            self.t_chwytu = teraz
            self.offset = np.linalg.inv(T_base_tcp) @ poza(*self.ostatnie[CZERWONA])
        m = self.swieze(CZERWONA, teraz, od=self.t_chwytu + PO_CHWYCIE)
        m = [x for x in m.values() if x[3] is not None]
        if m:                                                # świeży pomiar po chwycie -> poprawiamy offset
            t, pos, yaw, T_tcp_zdjecie = max(m, key=lambda x: x[0])
            self.offset = np.linalg.inv(T_tcp_zdjecie) @ poza(pos, yaw_to_quat(yaw))
        self.offset[1, 3] = 0.0                              # palce zaciskają się wzdłuż osi y TCP -> kostka na środku
        T = T_base_tcp @ self.offset                         # kostka jedzie razem z TCP
        return T[:3, 3].copy(), matrix_to_quat(T[:3, :3])
