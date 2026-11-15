# Tasks

Shared to-do list for everyone (and every Claude session) working on this repo.
How to use it is in CLAUDE.md under "Working together". In short: pull, claim a task here in its own small commit to `main`, work on your own branch, open a pull request, the other person looks at it and merges, then move the task to Done.

## Who owns what
Each person only edits the files they own. That way both of you can work at the same time without merge conflicts.

| Owner | Area | Files |
|---|---|---|
| **Jartartarta** (+ his Claude, run on his Windows PC next to the real game) | **detector / precision**: everything that reads the game image | `detector.py`, `fake_game.py`, `stress_test.py`, `tests/test_synthetic_table.py` |
| **Codingchancer** (+ his Claude) | **overlay + simulator**: everything that draws, captures, or simulates | `capture.py`, `overlay.py`, `simulator_ui.py`, `pool_physics.py`, `pool_rules.py`, `live_check.py`, `tracking.py`, `window_capture.py`, `analysis_process.py`, `tests/test_realtime.py`, `tests/test_window_capture.py`, `tests/test_analysis_process.py`, `tests/test_pool_physics.py`, `tests/test_pool_rules.py`, `tests/test_guide_settings.py`, `tests/test_steady_guide.py` |
| shared (tell the other first) | **repo** | `main.py`, `README.md`, `CLAUDE.md`, `TASKS.md`, `setup.bat`, `run.bat`, `requirements.txt`, the other test files |

Border rules:
- The `Detection` class in `detector.py` is what the overlay draws. Adding a field is fine; renaming or removing one breaks the overlay, so tell the other person first.
- Need a change in the other person's files? Open a GitHub issue or message them instead of editing it.
- Adding new tests to a shared test file is fine; don't rewrite the other person's tests.

## In progress
| Task | Area | Who | Branch |
|---|---|---|---|

## Next — Jartartarta (detector / precision)
- [ ] **Fast aim reader** (IDEAS.md idea 2, handed over by Codingchancer 15 Nov): read only the aim change per frame (angles within ~±2° of the last aim, ghost ball in a small box) in ~3-5 ms, with the full `predict_shot` every few frames to correct drift. Reading the aim is now the limit (16-18 updates/s). You may edit Codingchancer's `tracking.py` and `tests/test_realtime.py` for this task; keep the results within the `GuideStabilizer` tolerances and check `live_check.py`.
- [ ] **Power needed for a shot:** record ~20-30 real shots (slider % at release + where the balls stop) to fit Miniclip's friction and the %-to-speed scale per cue; then compute the minimum power to pot and add it to `Detection` for Codingchancer to mark on the slider. Also confirm the slider reading on a mid-drag and a full-power frame.
- [ ] **Real-game precision:** play with the guide on; wherever the guide's aim line, target ball or object/cue path does not sit exactly on the game's own white lines, save the frame (set `POOL_OVERLAY_DEBUG_DIR`, keep `capture.png`/`cleaned.png`) and have your Claude fix the detector. Keep real screenshots out of git (`outputs/` is ignored; the repo is public).
- [ ] **Real-game tests:** turn the saved bad frames into tests that skip when the image is missing (like the existing screenshot tests), so fixes stay fixed.
- [ ] Balls boxed in inside a rack can be off by ~0.5 radius.
- [ ] Raise the synthetic pass rate (`stress_test.py --count 300`) from 283/300 towards 297+/300 without making real-game results worse.
- [ ] White cloth precision: cue-ball centre (3-7 px off; outline, rise and shadow-free fits all tried and lost to the drop shadow's edge) and striped balls on pure white tables; check on a live white table.

## Next — Codingchancer (overlay + simulator)
- [ ] **Display scaling:** check Fit table and the guide at Windows scaling 125% and 150%, fix what is off.

## Done
- [x] Stress test everything (Jartartarta, #26): wide sweep (19 cloths x 5 shot kinds), near-cue, harsh mirror and hold-out seeds. Hold-out 473 -> 491/684, harsh mirror 366 -> 398/570, near-cue cue ball 86 -> 121/152, no aim invented on the opponent's turn (11 -> 0/100), blue stress 286/300 unchanged. New `fake_game.harsh()` / `stress_test.py --harsh` (stream scale, 4:2:0 colour, display tone).
- [x] Every cloth colour (Jartartarta, #25): pale cloths work. Synthetic, 20 shots each: white 1 -> 10, light grey 2 -> 15, beige 0 -> 15, sky 6 -> 11; other colours unchanged, blue stress 286/300. Real beige frame now picks the right cue ball. Left: on white cloth the cue centre can be 4-6 px off and some stripes are missed; not tried on a live white table.
- [x] Guide follows a turning stick (Jartartarta, #24, one-off in Codingchancer's `overlay.py`): the stabilizer draws every frame while the aim keeps turning one way (sweep simulation, slow aiming: 0.20 -> 0.06 deg behind). Faster turning: ~one analysis' delay left.
- [x] Stick crossing balls (Jartartarta, #23, one-off in Codingchancer's `tracking.py`): a thin stripe of change across a ball is the stick, so tracking carries on instead of a full detection per frame.
- [x] Window capture + analysis in its own process (Codingchancer, #21; merged by Jartartarta after checks: tests, stress 286/300, live 8/8).
- [x] Smoother guide, detector side (Jartartarta, #22): same results, full detect 184 -> ~115 ms, `predict_shot` ~15 ms median on shots that hit a ball. Stick crossing balls forces full detections in `tracking.py`: noted on #21 for Codingchancer.
- [x] Ball outline where a path meets a cushion (Jartartarta, #20, one-off in Codingchancer's `capture.py`): the paths follow the ball's centre, the outline shows the ball touching the cushion.
- [x] Ghost ring on top of the cue ball + stick rings (Jartartarta, #19): aim found from the game's object line when the ring is hidden (synthetic near shots: struck ball's path right on 28/40, was 0); rings centred on the stick's axis are no longer balls (stress 286/300).
- [x] Draw the paths through other balls (Jartartarta, #18, one-off in Codingchancer's `capture.py` / `tracking.py`): orange dashes for the ball in the way, dash-dot for the deflection; tracking mode now passes pot, stick pull, power slider and these paths too (the `tracking.py` part of #11).
- [x] Prediction: paths through other balls (Jartartarta, #17): `Detection.knock_paths` / `deflect_paths` (where a ball in the way goes, where the moving ball deflects to; synthetic median 1.4 deg). Drawn since #18.
- [x] Table fit with a covered corner pocket (Jartartarta, #16/#17): three corners + the middle pocket fix the table, so Fit table no longer shrinks the overlay. Also: the neon table is found (dark rail bands cut off the pockets), no cushion rebound from a pocket mouth.
- [x] Freeze fix (Jartartarta, #15, one-off in `overlay.py` while Codingchancer was away): one screen capture per worker thread instead of a new one per frame; GDI objects stay at ~26 instead of hitting Windows' 10,000 limit after ~2 min. `tests/test_capture_reuse.py` guards it. Closes #13.
- [x] Freeze + polish (Jartartarta, #14): every ball drawn at one size (shadows made circles up to 22% too big), cue ball centre no longer pulled by its shadow; stress test 285/300. Freeze found: `overlay.py` leaks a screen capture per frame until Windows' GDI limit (~2 min on the synthetic table); tested patch in issue #13 for Codingchancer.
- [x] Grey / black cloth (Jartartarta, #12): the Lucky Shot table's colourless near-black cloth is found (real frame: table within 2 px, cue / gold ball / aim / ghost found). Synthetic black 56/100 (8 ball is the weak spot), purple 84 (was 66).
- [x] Shot power, reading (Jartartarta, #12): `Detection.power_percent` (0-100) and `power_slider` from Miniclip's power slider (needs the capture to include it), `stick_pull` from the cue stick. Real frames read 0% at rest; % scale not yet confirmed mid-drag.
- [x] Prediction: pot or miss (Jartartarta, #10): new `Detection.pot` ("pot" / "jaw" / None) and `pot_pocket`; on 200 synthetic pot shots all 152 clear calls match the exact truth. Drawing it and passing it through `tracking.py`: issue #11 for Codingchancer.
- [x] Any cloth colour (Jartartarta, #9): the detector measured only blue (0/40 shots on green/red/purple); now it measures the cloth hue. Synthetic, 100 shots each: blue 95, green 91, red 86, brown 92, teal 92, purple 66 (test purple = ball 4's colour). Not yet checked on a real non-blue table.
- [x] Prediction: more cushion bounces (Jartartarta, #9): live paths follow up to 3 cushions (`PATH_BOUNCES`), checked against exact truth (`fake_game.truth_path`); simulator keeps 1. Also fixed the drawn cue path ignoring the corrected cue direction (up to 7 deg off).
- [x] Real-game precision, first round (Jartartarta, #9): guide no longer reads its own 1 px threads as the game's line, stick-butt rings are not balls, cue path rebounds off an empty cushion. Not yet re-checked live; keep saving bad frames (task stays under Next).
- [x] Real-time overlay: ~15 guide updates/s while aiming instead of 3-5 (tracking mode, no settle wait, faster ghost search with identical results; live 12/12). Jartartarta: pull main, `detector.py` changed (speed only).
- [x] Set up (Jartartarta): `setup.bat` + `run.bat` work on Jartartarta's Windows PC (Python 3.15.1; 71 tests OK, 19 skipped without screenshots; stress test 280/300).
- [x] Steadier guide: on a still shot the guide is redrawn on 0 of 144 noisy frames instead of 132; real changes show on the next frame.
- [x] Guide settings: colours, thickness, bounce on/off, saved between runs; feedback removal tested for every style.
- [x] Simulator UI: power bar with break zone, two-player scoreboard, aim by clicking the cloth.
- [x] Stronger break: power above 80% is break power (a full break scatters ~12 of 15 balls instead of 2); normal shots unchanged.
- [x] Simulator 8-ball game mode: break, groups, turns, fouls, ball in hand, win/lose (`pool_rules.py`).
- [x] Simulator spin: follow, draw and english (sliding/rolling model), spin ball control, physics-based cue-ball preview.
- [x] Split `main.py` into `detector.py`, `capture.py`, `overlay.py`, `simulator_ui.py` (no behaviour change: 45 tests, identical stress-test output, live 8/8).
- [x] Detector rework, capture-safe guide, synthetic table + stress/live tests (283/300 synthetic, 23/23 live).
- [x] Public repo without personal screenshots.
