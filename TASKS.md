# Tasks

Shared to-do list for everyone (and every Claude session) working on this repo.
How to use it is in CLAUDE.md under "Working together". In short: pull, claim a task here in its own small commit to `main`, work on your own branch, open a pull request, the other person looks at it and merges, then move the task to Done.

## Who owns what
Each person only edits the files they own. That way both of you can work at the same time without merge conflicts.

| Owner | Area | Files |
|---|---|---|
| **Jartartarta** (+ his Claude, run on his Windows PC next to the real game) | **detector / precision**: everything that reads the game image | `detector.py`, `fake_game.py`, `stress_test.py`, `tests/test_synthetic_table.py` |
| **Codingchancer** (+ his Claude) | **overlay + simulator**: everything that draws, captures, or simulates | `capture.py`, `overlay.py`, `simulator_ui.py`, `pool_physics.py`, `live_check.py`, `tests/test_pool_physics.py` |
| shared (tell the other first) | **repo** | `main.py`, `README.md`, `CLAUDE.md`, `TASKS.md`, `setup.bat`, `run.bat`, `requirements.txt`, the other test files |

Border rules:
- The `Detection` class in `detector.py` is what the overlay draws. Adding a field is fine; renaming or removing one breaks the overlay, so tell the other person first.
- Need a change in the other person's files? Open a GitHub issue or message them instead of editing it.
- Adding new tests to a shared test file is fine; don't rewrite the other person's tests.

## In progress
| Task | Area | Who | Branch |
|---|---|---|---|
| | | | |

## Next — Jartartarta (detector / precision)
- [ ] **Set up:** install the Claude desktop app, open the Code tab in your `Pool_Overlay` folder (not a cloud session), run `setup.bat`, then `run.bat` with the phone mirrored. Your Claude can then run the app next to the real game.
- [ ] **Real-game precision:** play with the guide on; wherever the guide's aim line, target ball or object/cue path does not sit exactly on the game's own white lines, save the frame (set `POOL_OVERLAY_DEBUG_DIR`, keep `capture.png`/`cleaned.png`) and have your Claude fix the detector. Keep real screenshots out of git (`outputs/` is ignored; the repo is public).
- [ ] **Real-game tests:** turn the saved bad frames into tests that skip when the image is missing (like the existing screenshot tests), so fixes stay fixed.
- [ ] Ghost ring hidden when the struck ball nearly touches the cue ball: find another cue for that case.
- [ ] Balls boxed in inside a rack can be off by ~0.5 radius.
- [ ] Raise the synthetic pass rate (`stress_test.py --count 300`) from 283/300 towards 297+/300 without making real-game results worse.

## Next — Codingchancer (overlay + simulator)
- [ ] **Simulator 8-ball rules:** break, solids/stripes, turns, fouls (scratch, wrong ball first, no rail), ball in hand, win/lose on the 8.
- [ ] **Simulator UI:** place the cue ball by hand, a power bar, show whose turn it is and which group they have.
- [ ] **Display scaling:** check Fit table and the guide at Windows scaling 125% and 150%, fix what is off.
- [ ] **Guide settings:** line colours, thickness and length, saved between runs.
- [ ] **Steadier guide:** less flicker between frames when the shot hasn't changed.

## Done
- [x] Simulator spin: follow, draw and english (sliding/rolling model), spin ball control, physics-based cue-ball preview.
- [x] Split `main.py` into `detector.py`, `capture.py`, `overlay.py`, `simulator_ui.py` (no behaviour change: 45 tests, identical stress-test output, live 8/8).
- [x] Detector rework, capture-safe guide, synthetic table + stress/live tests (283/300 synthetic, 23/23 live).
- [x] Public repo without personal screenshots.
