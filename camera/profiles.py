"""profiles.py – jakie tryby obrazu daje kamera przy obecnym połączeniu i które z nich naprawdę działają.

Uruchom:  python profiles.py
1. Wypisuje tryby, które kamera zgłasza (rozdzielczość, klatki/s, format). Na USB 2 lista jest krótsza niż na USB 3.
2. Po kolei włącza głębię i kolor RAZEM w tych samych trybach (tak jak nasze skrypty) i liczy,
   ile par klatek naprawdę przychodzi w ciągu kilku sekund.
Przed uruchomieniem zamknij RealSense Viewer i inne skrypty: kamerę może używać tylko jeden program naraz.
Działa na Windowsie i na Linuksie.
"""
import time                                                  # pomiar czasu
from collections import defaultdict                          # słownik z domyślną wartością

import pyrealsense2 as rs                                    # sterownik kamery

TEST_S = 3.0                                                 # ile sekund liczymy klatki w jednym trybie
FIRST_MS = 5000                                              # ile czekamy na pierwszą klatkę [ms]


def name(x):
    """'stream.depth' -> 'depth', 'format.z16' -> 'z16'."""
    return str(x).split('.')[-1]


ctx = rs.context()
devs = ctx.query_devices()                                   # podłączone kamery
if len(devs) == 0:
    print("Nie widzę kamery. Sprawdź kabel i czy RealSense Viewer jest zamknięty.")
    raise SystemExit(1)
dev = devs[0]
info = lambda k: dev.get_info(k) if dev.supports(k) else '?'
serial = info(rs.camera_info.serial_number)
print(f"Kamera: {info(rs.camera_info.name)}   firmware: {info(rs.camera_info.firmware_version)}   "
      f"USB: {info(rs.camera_info.usb_type_descriptor)}   pyrealsense2: {getattr(rs, '__version__', '?')}")

# ---------- 1. Tryby zgłaszane przez kamerę ----------
modes = defaultdict(lambda: defaultdict(set))                # strumień -> (w, h) -> zbiór klatek/s
formats = defaultdict(set)                                   # strumień -> zbiór formatów
supported = set()                                            # (strumień, format, w, h, fps) – do testu poniżej
for sensor in dev.query_sensors():
    for p in sensor.get_stream_profiles():
        if not p.is_video_stream_profile():
            continue                                         # pomijamy żyroskop i akcelerometr
        v = p.as_video_stream_profile()
        s, f = name(p.stream_type()), name(p.format())
        modes[s][(v.width(), v.height())].add(p.fps())
        formats[s].add(f)
        supported.add((s, f, v.width(), v.height(), p.fps()))

print("\nTryby zgłaszane przez kamerę:")
for s in ('depth', 'color', 'infrared'):
    if s not in modes:
        continue
    print(f"  {s} (formaty: {', '.join(sorted(formats[s]))})")
    for (w, h) in sorted(modes[s], key=lambda wh: -wh[0] * wh[1]):
        print(f"     {w:>4} x {h:<4}  klatki/s: {', '.join(str(x) for x in sorted(modes[s][(w, h)]))}")

# ---------- 2. Test: głębia + kolor razem ----------
both = sorted({(w, h, fps) for (s, f, w, h, fps) in supported if s == 'depth' and f == 'z16'
               and ('color', 'bgr8', w, h, fps) in supported},
              key=lambda m: (-m[0] * m[1], -m[2]))           # od największej rozdzielczości


def run(w, h, fps):
    """Włącza głębię i kolor w trybie w x h @ fps. Zwraca (klatki/s, które naprawdę przyszły, komunikat błędu)."""
    pipe = rs.pipeline(ctx)
    cfg = rs.config()
    cfg.enable_device(serial)                                # ta sama kamera co wyżej
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    try:
        pipe.start(cfg)
    except RuntimeError as e:                                # np. "Couldn't resolve requests"
        return 0.0, str(e)
    n, msg = 0, ''
    try:
        pipe.wait_for_frames(FIRST_MS)                       # pierwsza klatka (kamera się rozgrzewa)
        t0 = time.time()
        while time.time() - t0 < TEST_S:
            fs = pipe.wait_for_frames(FIRST_MS)
            if fs.get_depth_frame() and fs.get_color_frame():
                n += 1                                       # liczymy tylko pełne pary: głębia + kolor
        rate = n / (time.time() - t0)
    except RuntimeError as e:                                # np. "Frame didn't arrive within 5000"
        rate, msg = 0.0, str(e)
    finally:
        try:
            pipe.stop()
        except RuntimeError:
            pass
    return rate, msg


print(f"\nTest głębia + kolor razem ({TEST_S:.0f} s na tryb):")
if not both:
    print("  Kamera nie zgłasza żadnego wspólnego trybu głębi (z16) i koloru (bgr8).")
best = None
for (w, h, fps) in both:
    rate, msg = run(w, h, fps)
    ok = rate >= 0.8 * fps                                   # przychodzi co najmniej 80% obiecanych klatek
    note = "OK" if ok else ("BRAK KLATEK" if rate == 0 else "za mało klatek")
    print(f"  {w:>4} x {h:<4} @ {fps:>2} kl/s:  przychodzi {rate:5.1f} kl/s   {note}"
          + (f"   ({msg})" if msg else ""))
    if ok and best is None:
        best = (w, h, fps)                                   # pierwszy działający = największa rozdzielczość

print()
if best:
    print(f"Najlepszy działający tryb: {best[0]} x {best[1]} @ {best[2]} kl/s")
else:
    print("Żaden tryb głębia + kolor nie dał klatek.")
