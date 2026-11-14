# Tasks

Shared to-do list for everyone (and every Claude session) working on this repo.
How to use it is in CLAUDE.md under "Working together". In short: pull, claim a task here in its own small commit to `main`, work on your own branch, merge through a pull request, then move the task to Done.

Areas, so two people don't edit the same code at once:
- **detector** — table / ball / cue / aim / ghost detection, `predict_shot`, `fake_game.py`, `stress_test.py`
- **overlay** — guide window, capture, feedback removal (`remove_overlay`), DPI, `live_check.py`
- **simulator-ui** — shot simulator, `pool_physics.py`, screenshot tester, control panel
- **repo** — README, CLAUDE.md, setup/run scripts, file layout

Files per area: detector → `detector.py`, `fake_game.py`, `stress_test.py`; overlay → `capture.py`, the `Overlay` class in `overlay.py`, `live_check.py`; simulator-ui → `simulator_ui.py`, `pool_physics.py`, the `Control` class in `overlay.py`; repo → docs, scripts, `main.py`. Only one open branch at a time per area, and keep branches short (hours, not days).

## In progress
| Task | Area | Who | Branch |
|---|---|---|---|
| | | | |

## Next
- [ ] Try the overlay on the real AirPlay mirror and note what goes wrong (needs Windows + iPhone). — area: overlay
- [ ] Check Fit table and the guide at Windows display scaling 125% / 150%. — area: overlay
- [ ] Ghost ring hidden when the struck ball nearly touches the cue ball: find another cue for that case. — area: detector
- [ ] Balls boxed in inside a rack can be off by ~0.5 radius. — area: detector

## Done
- [x] Split `main.py` into `detector.py`, `capture.py`, `overlay.py`, `simulator_ui.py` (no behaviour change: 45 tests, identical stress-test output, live 8/8).
- [x] Detector rework, capture-safe guide, synthetic table + stress/live tests (283/300 synthetic, 23/23 live).
- [x] Public repo without personal screenshots.
