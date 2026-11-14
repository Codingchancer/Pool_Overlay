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
| Real-game testing: play 8 Ball Pool on the iPhone mirrored to his Windows PC with the overlay on, and report every wrong line, missed ball or bad table fit (how: see below) | testing (no code) | Jartartarta | — |
| Precision: make aim, target and object/cue paths match the game more closely, fixing what the real-game reports show | detector | Codingchancer (Claude) | `codingchancer/precision` |

## Next
- [ ] Check Fit table and the guide at Windows display scaling 125% / 150%. — area: overlay
- [ ] Ghost ring hidden when the struck ball nearly touches the cue ball: find another cue for that case. — area: detector
- [ ] Balls boxed in inside a rack can be off by ~0.5 radius. — area: detector

## How to report a real-game problem (Jartartarta)
1. Run `run.bat`, mirror the phone with AirPlay, press **Fit table**, then **Enable guide**.
2. When the guide is wrong, take a screenshot right away (Win+Shift+S) and also note: what was wrong (aim line, target ball, object/cue path, table edges, a ball not seen), and roughly by how much.
3. Open a GitHub issue in this repo (Issues → New issue), paste the screenshot and the note. One issue per problem. The repo is public, so crop out your player name/avatar first.
4. Optional, most useful: set `POOL_OVERLAY_DEBUG_DIR` to a folder before starting the app; then attach that folder's `capture.png` and `cleaned.png` from the bad moment too.

## Done
- [x] Split `main.py` into `detector.py`, `capture.py`, `overlay.py`, `simulator_ui.py` (no behaviour change: 45 tests, identical stress-test output, live 8/8).
- [x] Detector rework, capture-safe guide, synthetic table + stress/live tests (283/300 synthetic, 23/23 live).
- [x] Public repo without personal screenshots.
