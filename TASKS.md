# Tasks

Shared to-do list for everyone (and every Claude session) working on this repo.
How to use it is in CLAUDE.md under "Working together". In short: pull, claim a task here in its own small commit to `main`, work on your own branch, open a pull request, the other person looks at it and merges, then move the task to Done.

## Who owns what
Each person only edits the files they own. That way both of you can work at the same time without merge conflicts.

| Owner | Area | Files |
|---|---|---|
| **Jartartarta** (+ his Claude, run on his Windows PC next to the real game) | **detector / precision**: everything that reads the game image | `detector.py`, `fake_game.py`, `stress_test.py`, `tests/test_synthetic_table.py` |
| **Codingchancer** (+ his Claude) | **overlay + simulator**: everything that draws, captures, or simulates | `capture.py`, `overlay.py`, `simulator_ui.py`, `pool_physics.py`, `pool_rules.py`, `live_check.py`, `tracking.py`, `tests/test_realtime.py`, `tests/test_pool_physics.py`, `tests/test_pool_rules.py`, `tests/test_guide_settings.py`, `tests/test_steady_guide.py` |
| shared (tell the other first) | **repo** | `main.py`, `README.md`, `CLAUDE.md`, `TASKS.md`, `setup.bat`, `run.bat`, `requirements.txt`, the other test files |

Border rules:
- The `Detection` class in `detector.py` is what the overlay draws. Adding a field is fine; renaming or removing one breaks the overlay, so tell the other person first.
- Need a change in the other person's files? Open a GitHub issue or message them instead of editing it.
- Adding new tests to a shared test file is fine; don't rewrite the other person's tests.

## In progress
| Task | Area | Who | Branch |
|---|---|---|---|
| Prediction: paths through other balls | detector | Jartartarta | `jartartarta/paths-through-balls` |
| Table fit with a covered corner pocket (overlay resized to a too-small table when the cue ball sat on a pocket) | detector | Jartartarta | `jartartarta/covered-pocket` |

## Next — Jartartarta (detector / precision)
- [ ] **Power needed for a shot:** record ~20-30 real shots (slider % at release + where the balls stop) to fit Miniclip's friction and the %-to-speed scale per cue; then compute the minimum power to pot and add it to `Detection` for Codingchancer to mark on the slider. Also confirm the slider reading on a mid-drag and a full-power frame.
- [ ] **Real-game precision:** play with the guide on; wherever the guide's aim line, target ball or object/cue path does not sit exactly on the game's own white lines, save the frame (set `POOL_OVERLAY_DEBUG_DIR`, keep `capture.png`/`cleaned.png`) and have your Claude fix the detector. Keep real screenshots out of git (`outputs/` is ignored; the repo is public).
- [ ] **Real-game tests:** turn the saved bad frames into tests that skip when the image is missing (like the existing screenshot tests), so fixes stay fixed.
- [ ] Ghost ring hidden when the struck ball nearly touches the cue ball: find another cue for that case.
- [ ] Balls boxed in inside a rack can be off by ~0.5 radius.
- [ ] Raise the synthetic pass rate (`stress_test.py --count 300`) from 283/300 towards 297+/300 without making real-game results worse.
- [ ] **Prediction: paths through other balls.** Today a path stops at the first ball in the way (orange dotted ring). Predict what happens next: the blocking ball's direction after the hit and where the original ball deflects to (same ghost-ball/90° rule as the main shot), at least one level deep. Keep `blockers` as it is; add new fields for the extra paths and tell Codingchancer so the overlay can draw them.

## Next — Codingchancer (overlay + simulator)
- [ ] **Display scaling:** check Fit table and the guide at Windows scaling 125% and 150%, fix what is off.

## Done
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
