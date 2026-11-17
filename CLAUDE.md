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
- `capture.py` — area **overlay**: guide drawing (`paint_guide`, `GUIDE_ALPHA`, `GuideStyle`: colours/thickness/bounce, saved via QSettings; never add white/grey colours or raise the alpha), `render_guide_layer`, `remove_overlay`, `shown_layer` (which recent drawing a capture shows), DPI helpers.
- `overlay.py` — area **overlay** (window) / **simulator-ui** (control panel): `Overlay` guide window, `Control` panel, `CueTracker`, `grab_clean` + background `AnalysisTask` (capture, clean and analyse back to back off the window thread); `GuideStabilizer` keeps the drawn guide still while the shot is unchanged (tolerances from measured frame noise).
- `tracking.py` — area **overlay**: `ShotTracker` reuses the table and balls of the last full `detect` while they stay still and only re-runs `predict_shot` (live speed).
- `window_capture.py` — area **overlay**: `WindowCapture` copies the mirror window's own picture (no guide in it), `window_under` finds the mirror under the guide (known titles win). GDI objects made once per worker and released by `close` (issue #13).
- `analysis_process.py` — area **overlay**: capture + `ShotTracker` loop in its own process (`AnalysisProcess`); exits when the app is gone, even if killed.
- `pool_rules.py` — area **simulator-ui**: `EightBallGame` (turns, groups, fouls, ball in hand, win/lose), pure Python; the physics reports `first_contact`, `rail_after_contact`, `pocket_order` for it.
- `simulator_ui.py` — area **simulator-ui**: `ShotSimulator`, `SimCanvas`, `ScreenshotTester`.
- `pool_physics.py` — the simulator's approximate 2D physics: follow/draw/english with the sliding/rolling model; no swerve, squirt, throw, jump or 8-ball rules (the UI and README must keep saying so).
- `fake_game.py` — synthetic table renderer with exact truth (a test fixture, not a game).
- `stress_test.py` — random synthetic shots vs truth (`--count 300`, `--save-failures DIR`).
- `live_check.py` — runs the real app against the synthetic table window through UI Automation.
- `sim_game.py` / `play_test.py` — area **detector** (tests): headless 8-ball test game (physics + rules + synthetic frames with truth; `shot_stream` gives a whole shot as the mirror shows it) and the overlay checked on whole games.
- `shot_recorder.py` / `recorder_check.py` — area **detector**: logs real shots from the mirror window (slider % at release, cue/struck-ball tracks, where balls stopped; reads the picture only) to `outputs/shots/`, and its check on synthetic games.
- `tests/` — unit tests. Screenshot-based tests skip when `outputs/*.png` is missing. The real screenshots are deliberately NOT in git (public repo: Miniclip's image + a player profile) — never commit images from `outputs/`.
- `setup.bat` / `run.bat` — build `.venv` with the newest Python 3.15…3.11 and start the app. `.venv` is never committed.

## How the detector works (short)
- Table: four corner pockets (dark blobs next to cloth), cushion nose = pocket centre inset by `POCKET_TO_CUSHION`·span, then each edge snapped to the cushion→cloth brightness drop. Cloth-colour fit is the fallback.
- Balls: Hough circles on a cloth-difference image and a grey image, refined to sub-pixel on the rim, rejected when the inside is mostly cloth. Of two object-ball circles closer than 0.9 radii the better rim stays (`drop_second_circles`: a second circle on a striped ball's white cap).
- Cue ball = the white ball whose aim ray ends in Miniclip's hollow ghost-ball reticle. Aim = polar ray scan + sub-pixel ghost.
- Struck ball right next to the cue ball ("near" shots): the ghost ring lies on the cue ball and hides the ray. `find_ghost_ring` finds the ring anywhere (thin bright stroke between dark outlines all round, white, ~0.9 ball radii; fitted on its own stroke, 0.05 px); only the cue ball can overlap it (`ball_under_ring`; the struck ball, which the game's object line leaves, only touches it). A cue circle on the ring itself, or none, is found again with the ring painted out (`paint_out_ring`, `ball_beside_ring`). A ghost on the cue ball is placed on the ring in `_predict_paths`, and the aim there comes from the stick (`stick_direction_near`: its band of off-cloth shade behind the cue ball, else its Hough edges; 0.16 deg at the 95th percentile).
- Target = ball touching the ghost; object/cue lines are fitted on the game's own short white lines when visible (`game_line_direction`: first the narrow band through the ball with the most stroke pixels, then its principal axis; long runs along the table border, the cushion's lit edge, are cleared from the line mask by `_clear_border_runs`).
- White cloth: lines read as thin ridges between their dark outlines (`white_stroke_masks`); cue ball and stripes fitted on the lit-side rim dip (`_WhiteCloth`), away from the shadow direction the balls show. A cue ball circle finding rejected as cloth is taken back (`white_cloth_missed_balls`: lit-side rim, drop shadow, lit face, no ball within 3 radii).
- Grey cloth: lettering (one darker shade of the cloth) counts as cloth in the ball test, as on light cloths (not on the near-black Lucky Shot cloth). Light tinted cloths (beige, sky, yellow, lime): the cue ball is fitted where the cloth's tint begins on its lit side (`pale_cloth_cue_centre`).
- Light grey cloth: where 3% or more of the cloth passes the white test (`LIT_CLOTH_SHARE`), strokes must be 10 brighter than the cloth around them, as on white. The cue ball is fitted on its lit side where its shading meets the cloth's level (`pale_cloth_cue_centre` with `KNEE_RISE`), also with the ring painted out in near shots. Under the ring, a ball as white as a cue ball is the cue ball whatever lines leave it (`CUE_WHITE`); a short-ray "ghost" that fails the strict ring test is dropped when the ring lies on the cue ball (`SHORT_RAY`).

## Capture feedback (important)
The guide is visible in screenshots/screen shares (`WDA_NONE`). To stop it feeding back into detection, every stroke is drawn at ≤60% alpha and the exact rendered layer is un-blended from each capture (`remove_overlay`); the control panel is greyed out in captures; right after a repaint the screen may still show an older drawing, so the overlay keeps its recent drawings and `shown_layer` picks the one the capture really shows (no fixed wait). With window capture (the default when a mirror window is under the guide) the guide is not in the frames at all; screen capture + removal is the fallback. Any change to drawing must keep these working — test the detect → draw → capture loop, not just window visibility (`tests/test_aim_alignment_and_capture.py`). `POOL_OVERLAY_DEBUG_DIR` dumps capture/layer/cleaned frames.

## Checks after a change
```
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python -m py_compile main.py detector.py capture.py overlay.py simulator_ui.py pool_physics.py fake_game.py stress_test.py live_check.py
.venv\Scripts\python stress_test.py --count 300 --quiet
```
Last known results: 258 tests pass (19 skip without the real screenshots); 290/300 synthetic shots pass every check; 2602/2850 on the 19-cloth sweep (5 kinds x 30) and near shots 1641/1900; live check 12/12.

## Known limits
- Struck ball nearly touching the cue ball: the aim comes from the stick; where no stick shows behind the cue ball (~a quarter, e.g. against a cushion) it comes from the cue ball's centre, only good to ~1 px (whole pixels, rim half hidden), so the aim can be degrees off; white and pale cloths (white, light grey, beige, sky) still fail a quarter to a third of these shots, mostly on the cue ball's position (light grey: the lit arc beside the ring and the struck ball is short, and the fit gives up).
- White and light grey cloth: a striped ball's white caps blend into the cloth; its centre can be ~0.5 radius off, and on the table's lettering it can be missed.
- Balls boxed in inside a rack can be off by ~0.5 radius.
- Not yet verified on a real live AirPlay mirror or at Windows display scaling other than 100%.
