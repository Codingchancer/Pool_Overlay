# Pool Practice Overlay — notes for Claude

Windows overlay (PySide6 + OpenCV + mss) for Miniclip 8 Ball Pool mirrored from an iPhone through an AirPlay receiver. Two people work on this repo, each with their own Claude Code session (some run in the cloud on Linux and can't run the Windows overlay; real-screen testing happens on the Windows PCs).

## Working together (every session follows this)
1. At the start: `git checkout main && git pull`, then read `TASKS.md`.
2. Claim a task before starting: move it to "In progress" in `TASKS.md` with your name and branch, commit only that edit to `main` and push it. If someone else already holds that task or area, pick another or ask.
3. Work on your own short branch, named `<name>/<task>` (e.g. `sam/ball-detection`). Never commit code directly to `main`, never force-push `main`, never rewrite shared history.
4. Only touch files you own: `TASKS.md` → "Who owns what" says which files belong to which person (Jartartarta: detector/precision; Codingchancer: overlay + simulator). Ask the user which of the two they are if it isn't clear. Need a change in the other person's files? Open a GitHub issue instead of editing them. Big moves — splitting or renaming files, reformatting — get their own task and the other person is told first.
5. Commit small, push often, and pull `main` into your branch regularly.
6. Before merging, run the checks below; all must pass.
7. Open a pull request, merge it within hours rather than days, then move the task to "Done" in `TASKS.md`.

## Hard rule
It is a **visual practice aid only**. Never add code that clicks, aims, moves the mouse, sends input, or takes shots in the game.

## Files
- `main.py` — entry point only (`python main.py`); re-exports every name from the modules below so `import main` keeps working. New code goes in the modules, not here.
- `detector.py` — area **detector**: table / ball / cue / aim / ghost detection, `detect`, `predict_shot`, path tracing. Pure OpenCV/numpy, no Qt.
- `capture.py` — area **overlay**: guide drawing (`paint_guide`, `GUIDE_ALPHA`, `GuideStyle`: colours/thickness/bounce, saved via QSettings; never add white/grey colours or raise the alpha), `render_guide_layer`, `remove_overlay`, `PAINT_SETTLE`, DPI helpers.
- `overlay.py` — area **overlay** (window) / **simulator-ui** (control panel): `Overlay` guide window, `Control` panel, `CueTracker`, background `AnalysisTask`; `GuideStabilizer` keeps the drawn guide still while the shot is unchanged (tolerances from measured frame noise).
- `pool_rules.py` — area **simulator-ui**: `EightBallGame` (turns, groups, fouls, ball in hand, win/lose), pure Python; the physics reports `first_contact`, `rail_after_contact`, `pocket_order` for it.
- `simulator_ui.py` — area **simulator-ui**: `ShotSimulator`, `SimCanvas`, `ScreenshotTester`.
- `pool_physics.py` — the simulator's approximate 2D physics: follow/draw/english with the sliding/rolling model; no swerve, squirt, throw, jump or 8-ball rules (the UI and README must keep saying so).
- `fake_game.py` — synthetic table renderer with exact truth (a test fixture, not a game).
- `stress_test.py` — random synthetic shots vs truth (`--count 300`, `--save-failures DIR`).
- `live_check.py` — runs the real app against the synthetic table window through UI Automation.
- `tests/` — unit tests. Screenshot-based tests skip when `outputs/*.png` is missing. The real screenshots are deliberately NOT in git (public repo: Miniclip's image + a player profile) — never commit images from `outputs/`.
- `setup.bat` / `run.bat` — build `.venv` with the newest Python 3.15…3.11 and start the app. `.venv` is never committed.

## How the detector works (short)
- Table: four corner pockets (dark blobs next to cloth), cushion nose = pocket centre inset by `POCKET_TO_CUSHION`·span, then each edge snapped to the cushion→cloth brightness drop. Cloth-colour fit is the fallback.
- Balls: Hough circles on a cloth-difference image and a grey image, refined to sub-pixel on the rim, rejected when the inside is mostly cloth.
- Cue ball = the white ball whose aim ray ends in Miniclip's hollow ghost-ball reticle. Aim = polar ray scan + sub-pixel ghost.
- Target = ball touching the ghost; object/cue lines are fitted on the game's own short white lines when visible.

## Capture feedback (important)
The guide is visible in screenshots/screen shares (`WDA_NONE`). To stop it feeding back into detection, every stroke is drawn at ≤60% alpha and the exact rendered layer is un-blended from each capture (`remove_overlay`); the control panel is greyed out in captures; analysis waits `PAINT_SETTLE` after a repaint. Any change to drawing must keep these working — test the detect → draw → capture loop, not just window visibility (`tests/test_aim_alignment_and_capture.py`). `POOL_OVERLAY_DEBUG_DIR` dumps capture/layer/cleaned frames.

## Checks after a change
```
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python -m py_compile main.py detector.py capture.py overlay.py simulator_ui.py pool_physics.py fake_game.py stress_test.py live_check.py
.venv\Scripts\python stress_test.py --count 300 --quiet
```
Last known results: 45 tests pass; 283/300 held-out synthetic shots pass every check; live check 23/23.

## Known limits
- Ghost ring is hidden when the struck ball nearly touches the cue ball.
- Balls boxed in inside a rack can be off by ~0.5 radius.
- Not yet verified on a real live AirPlay mirror or at Windows display scaling other than 100%.
