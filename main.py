"""Pool Practice Overlay: read-only screen analysis and a transparent visual guide.

Start the app with ``python main.py`` (or run.bat). The code is split into:

- ``detector.py``      table, balls, cue ball, Miniclip's aim graphics, shot prediction (no Qt)
- ``capture.py``       guide drawing, overlay un-blending, screen-coordinate helpers
- ``overlay.py``       live guide window, control panel, cue tracking, background analysis
- ``simulator_ui.py``  shot simulator and screenshot tester

Every name is re-exported here, so ``import main`` keeps working for the tests and tools.
"""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from detector import *  # noqa: F401,F403
from detector import (_RIM_ANGLES, _first_obstacle, _rim_scores, _ring_kernel,  # noqa: F401
                      _run_peaks, _select_target_on_ray)
from capture import *  # noqa: F401,F403
from capture import _debug_dump, _screen_scale  # noqa: F401
from simulator_ui import *  # noqa: F401,F403
from overlay import *  # noqa: F401,F403
from overlay import Control


def run():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    control = Control(); control.show(); control.raise_(); control.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
