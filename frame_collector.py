"""Save real frames from the mirror window while you play, for training the
ball-proposal network (``train_ball_net.py``) on the real game.

    python frame_collector.py [--every 2] [--minutes 30] [--title "AirPlay"]

Reads the mirror window's own picture only (like ``shot_recorder.py``); it
never clicks, aims or sends anything to the game. A frame is saved when a
table is found and the picture has changed since the last saved frame, at
most one every ``--every`` seconds, to ``outputs/real/<date-time>/``.
``outputs/`` is not in git: these pictures show Miniclip's game and player
profiles and must never be committed.

What helps most: several tables (cloth colours), slow aiming with the cue
stick lying across balls, the cue ball against a cushion, racks and the
break, the black 8 and balls the colour of the cloth.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
# Mean absolute difference (0-255) of a 64x36 grey thumbnail that counts as
# a new picture (a stick moving over the table is ~1-3; a still frame ~0.3).
CHANGE = 1.2


def thumbnail(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--every", type=float, default=2.0, help="seconds between saved frames at most")
    parser.add_argument("--minutes", type=float, default=0, help="stop after this long (0: Ctrl+C)")
    parser.add_argument("--title", default=None, help="part of the mirror window's title")
    parser.add_argument("--out", default=str(HERE / "outputs" / "real"))
    args = parser.parse_args(argv)
    import window_capture as wc
    from detector import find_table_bounds
    from shot_recorder import find_mirror
    hwnd = find_mirror(args.title)
    if hwnd is None:
        print("No mirror window found. Start the AirPlay receiver (or pass --title).")
        return 1
    out = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving frames from \"{wc._title(hwnd)}\" to {out}. Play normally; Ctrl+C to stop.")
    capture = wc.WindowCapture(hwnd)
    started = last_save = time.monotonic()
    last_thumb = None
    saved = no_table = 0
    try:
        while not args.minutes or time.monotonic() - started < args.minutes * 60:
            time.sleep(.25)
            if time.monotonic() - last_save < args.every:
                continue
            client = wc._client_rect(hwnd)
            if client is None:
                print("The mirror window is gone.")
                break
            frame = capture.grab({"left": client[0], "top": client[1], "width": client[2], "height": client[3]})
            if frame is None:
                continue
            thumb = thumbnail(frame)
            if last_thumb is not None and float(np.mean(np.abs(thumb - last_thumb))) < CHANGE:
                continue
            if find_table_bounds(frame) is None:
                no_table += 1
                continue
            cv2.imwrite(str(out / f"{datetime.now().strftime('%H%M%S-%f')[:-3]}.png"), frame)
            saved += 1
            last_save, last_thumb = time.monotonic(), thumb
            if saved % 25 == 0:
                print(f"  {saved} frames saved")
    except KeyboardInterrupt:
        pass
    finally:
        capture.close()
    print(f"{saved} frames saved in {out} ({no_table} changed pictures without a table skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
