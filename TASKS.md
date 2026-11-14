# Tasks

Shared to-do list for everyone (and every Claude session) working on this repo.
How to use it is in CLAUDE.md under "Working together". In short: pull, claim a task here in its own small commit to `main`, work on your own branch, open a pull request, the other person looks at it and merges, then move the task to Done.

## Who owns what
Each person only edits the files they own. That way both of you can work at the same time without merge conflicts.

| Owner | Area | Files |
|---|---|---|
| **Jartartarta** (+ his Claude, run on his Windows PC next to the real game) | **detector / precision**: everything that reads the game image | `detector.py`, `fake_game.py`, `stress_test.py`, `tests/test_synthetic_table.py` |
| **Codingchancer** (+ his Claude) | **overlay + simulator**: everything that draws, captures, or simulates | `capture.py`, `overlay.py`, `simulator_ui.py`, `pool_physics.py`, `pool_rules.py`, `live_check.py`, `tests/test_pool_physics.py`, `tests/test_pool_rules.py`, `tests/test_guide_settings.py`, `tests/test_steady_guide.py` |
| shared (tell the other first) | **repo** | `main.py`, `README.md`, `CLAUDE.md`, `TASKS.md`, `setup.bat`, `run.bat`, `requirements.txt`, the other test files |

Border rules:
- The `Detection` class in `detector.py` is what the overlay draws. Adding a field is fine; renaming or removing one breaks the overlay, so tell the other person first.
- Need a change in the other person's files? Open a GitHub issue or message them instead of editing it.
- Adding new tests to a shared test file is fine; don't rewrite the other person's tests.

## In progress
| Task | Area | Who | Branch |
|---|---|---|---|
| Real-game precision | detector | Jartartarta | `jartartarta/real-game-precision` |

## Next — Jartartarta (detector / precision)
- [ ] **Real-game precision:** play with the guide on; wherever the guide's aim line, target ball or object/cue path does not sit exactly on the game's own white lines, save the frame (set `POOL_OVERLAY_DEBUG_DIR`, keep `capture.png`/`cleaned.png`) and have your Claude fix the detector. Keep real screenshots out of git (`outputs/` is ignored; the repo is public).
- [ ] **Real-game tests:** turn the saved bad frames into tests that skip when the image is missing (like the existing screenshot tests), so fixes stay fixed.
- [ ] Ghost ring hidden when the struck ball nearly touches the cue ball: find another cue for that case.
- [ ] Balls boxed in inside a rack can be off by ~0.5 radius.
- [ ] Raise the synthetic pass rate (`stress_test.py --count 300`) from 283/300 towards 297+/300 without making real-game results worse.
- [ ] **Prediction: more cushion bounces.** The object-ball and cue-ball paths now stop after one bounce (`max_bounces=1` in `collision_paths` / `predict_shot`). Follow them through 2-3 bounces (losing a little angle accuracy each time), stopping at a pocket or the first ball in the way. Check on the synthetic table that the extra segments match `fake_game`'s truth.
- [ ] **Prediction: pot or miss.** Say whether the object ball drops: does its path reach a pocket mouth inside the jaws, or hit the jaw/cushion point? Add the result to `Detection` as a new field (e.g. `pot: "pot" | "jaw" | None` and the pocket), not by renaming fields. Drawing it (e.g. a green/red ring on the pocket) is in `capture.py`, so open a GitHub issue for Codingchancer with the field name when it's ready.
- [ ] **Prediction: paths through other balls.** Today a path stops at the first ball in the way (orange dotted ring). Predict what happens next: the blocking ball's direction after the hit and where the original ball deflects to (same ghost-ball/90° rule as the main shot), at least one level deep. Keep `blockers` as it is; add new fields for the extra paths and tell Codingchancer so the overlay can draw them.

## Next — Codingchancer (overlay + simulator)
- [ ] **Display scaling:** check Fit table and the guide at Windows scaling 125% and 150%, fix what is off.

## Done
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
