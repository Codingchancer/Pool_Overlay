# Pool Practice Overlay

A Windows practice overlay for **Miniclip 8 Ball Pool** mirrored through an AirPlay receiver. The overlay only observes the screen and draws visual guidance; it does not click, aim, or otherwise control the game.

## Setup (any PC, Python 3.11 – 3.15)

1. Install Python 3.11 or newer from python.org if it is not installed yet (3.15 works; keep the `py` launcher option on).
2. Double-click **`setup.bat`** once. It:
   - picks the newest installed Python 3.11+ (3.15 first, then 3.14 … 3.11);
   - creates a private `.venv` in this folder and installs `requirements.txt`;
   - adds a **Pool Practice Overlay** shortcut to the desktop.
3. Start the app from that shortcut or with **`run.bat`**. `run.bat` runs setup first if the packages are missing.

A `.venv` only works on the PC where it was made. If this folder is copied to another PC, run `setup.bat` there: it sees that the old `.venv` points at a Python that isn't on that PC (for example `did not find executable at 'C:\Python315\python.exe'`) and rebuilds it. If the folder is moved, run `setup.bat` again so the desktop shortcut points at the new place.

Tested setups: Python 3.15 with PySide6 6.12.0, OpenCV 4.14.0.94, numpy 2.5.4 and mss 10.2.0 (the original PC). This copy's `setup.bat` and all 45 tests were run on Python 3.14 with those same packages; the main copy's tests ran on 3.11.

To run by hand instead: `py -3.15 -m venv .venv`, `.venv\Scripts\python -m pip install -r requirements.txt`, `.venv\Scripts\python main.py`.

## Use

1. Start the app (above).
2. **Fit table** runs automatically at start. It finds the four corner pockets, places the table edge on the cushion noses, and sizes the transparent guide to the table plus a small margin, so every analysis frame can see the pockets. If needed, drag the guide's body or bottom-right corner.
3. Select **Enable guide**. Use **Re-detect** if the table view changes. Open **Shot simulator** to practise without the phone mirror, or **Test a screenshot** to check the detector on a saved image.
4. To see the live overlay working without a phone, open **Synthetic test table**, then press **Fit table** and **Enable guide**. A new shot appears every 4 seconds (Space or → for the next one).

The guide starts disabled. Once enabled it reads the mirror window's own picture (Windows `BitBlt`/`PrintWindow`, ~1-5 ms), so the guide on top is never in the analysed frames; if that window gives no picture or no table, it falls back to a screen capture with the guide removed. Capture and analysis run frame after frame in a separate process (`analysis_process.py`), about 16-18 updates per second on the synthetic aiming table (the panel shows the rate and whether it uses window or screen capture), and frames where nothing changed are skipped. A full detection of the table and all balls takes 150-250 ms, so it runs only when a ball moves, the aim line is lost for a few frames, every 3 seconds, or on Re-detect; in between, only the aim graphics are re-read against the known balls (`tracking.py`, 15-50 ms depending on the shot). `POOL_OVERLAY_NO_PROCESS=1` keeps the analysis in the app; `POOL_OVERLAY_SCREEN_CAPTURE=1` forces screen capture. While the shot stays the same (cue ball, ghost ball, target and aim agree with what is drawn, within the detector's frame-to-frame noise), the guide is not redrawn, so it stays perfectly still instead of shimmering; if one frame misses the aim line, the last guide stays up for up to half a second. Any real change is drawn on the next frame.

## Screenshots and screen sharing

The guide is **visible** in screenshots, recordings and screen shares. Windows capture exclusion is not used.

Because the guide is visible, every analysis frame also contains the guide's own lines. To keep them from being read as Miniclip's white aim line, the app:

- draws every guide stroke at no more than about 60% opacity;
- renders the exact guide layer that is on screen, and solves `capture = guide + game × (1 − alpha)` for the game pixels before analysing (pixels come back within a few brightness levels);
- waits about 70 ms after each guide repaint before reading the screen, so the subtracted layer is the one the compositor is showing;
- greys out its own control panel (and its shadow) in captured frames, because the dark panel can merge with a nearby pocket.

To inspect this on a real screen, set `POOL_OVERLAY_DEBUG_DIR` to a folder before starting the app. It then keeps overwriting `capture.png` (what the screen showed), `layer.png` (the guide layer that was subtracted) and `cleaned.png` (what the detector analysed). On a live Windows 11 desktop, 99% of the pixels under the guide came back within 4 brightness levels of a clean capture.

Tests in `tests/test_aim_alignment_and_capture.py` draw the guide over two real game screenshots and run the full detect → draw → capture loop. Without subtraction the next frame goes wrong: the aim jumps to 100° or −160°, or the wrong ball is taken as the cue. With subtraction the table, cue, ghost ball and aim angle stay stable over repeated frames.

## What it detects

- **Table:** the four corner pockets give a rough fit (pockets must open onto the blue cushions, so dark wallpaper and panels on a multi-monitor desktop are ignored). Each side is then snapped onto the cushion nose, where brightness drops sharply from cushion to cloth. The cloth-colour fit remains as a fallback when the pockets are out of view.
- **Balls:** circles are proposed on both a grey image and a "difference from the cloth colour" image (the second finds blue balls on blue cloth). Each proposal is snapped onto the ball's edge to a fraction of a pixel and kept only when its inside is mostly not cloth, which rejects the hollow ghost ring, watermark rings, lettering and shadows. Balls resting against a cushion are kept. Circles on the cue stick behind the cue ball are dropped unless they are full ball size with a clear round edge (stick joints are smaller).
- **Cue ball:** the white ball that the game's aim ray starts from and that ends in a ghost ball. The cue stick only breaks ties, because near a rail most of it is off the table.
- **Aim:** a scan outward from the cue ball in every direction (0.5° steps) finds the white ray, even when it is short or crossed by other lines. Of the candidate directions, the one that ends in Miniclip's hollow **ghost-ball reticle** is used. The reticle centre (found to a fraction of a pixel) is where the cue ball will be at impact, and the angle is fitted on the ray's own pixels through it. When the target is so close that no ray is visible, the ghost ring is searched for all around the cue ball.
- **Target:** the ball that touches the ghost ball. When the ghost rests on a cushion and a ball lies just beside it, that ball only counts if the game draws an object-ball line from it.
- **After impact:** Miniclip draws short white lines for the object ball and the cue ball. When they are visible, the app fits them directly and extends them through a cushion bounce. When the object line is hidden (it runs across other balls), the struck ball's centre is fitted one diameter from the ghost centre and the direction follows from that; the cue ball leaves at 90° (no spin). Paths stop at the first ball in the way, which gets an orange dotted ring.

- **White cloth:** the game's white lines are no brighter than the cloth there and show only as their dark outlines. The aim ray, the object and cue lines and (when needed) the ghost ring are read as thin bright ridges between those outlines. The cue ball's and striped balls' rims are shaded darker than the cloth and run into their drop shadows. So they are fitted on the dip at the rim on their lit side, away from the shadow that the other balls show.

Default colours: cyan = cue ball to the ghost ball, yellow = object ball, magenta dashes = cue ball after impact, orange dots = a ball in the way. The game's own guide stays white.

**Guide style…** (in the control panel) changes the colours of the three lines, the line thickness (thin to extra thick), and whether paths continue after a cushion bounce. Changes show at once and are kept for the next start. Opacity stays fixed and white is not offered, so the guide can still be removed from every capture and never looks like the game's own white lines; the screenshot tester draws in the same style.

If the cue ball or aim ray cannot be identified confidently, the guide says so instead of guessing. The live overlay sees a still frame, so it cannot know the shot power or spin setting. Paths beyond the game's own short lines are estimates.

## Shot simulator and screenshot tester

The shot simulator holds a full numbered rack. Click the cloth to aim (hold and move to fine-tune), drag balls to place them, set power on the bar left of the table, set spin, then shoot (power above 80% is break power: up to about four times harder, to scatter a rack) to see ball-to-ball collisions, cushion rebounds, cloth drag and pocketing. Click the white spin ball where the cue tip should hit (up = follow, down = draw, left/right = english; double-click resets). Before the shot, pink dashes show where the cue ball goes after hitting the target ball with that spin.

It is **approximate 2D physics**:
- an equal-mass impulse model with small energy losses;
- spin with the standard sliding/rolling ball model: a centre hit slides, then rolls; top spin follows through, back spin draws back; side spin (english) changes the angle off cushions;
- not modelled: swerve/masse, squirt, throw between balls, jump shots;
- 8-ball rules only in game mode, and simplified (see below).

Tick **8-ball game (2 players)** to play a game instead of free practice. A scoreboard above the table shows each player's group and balls left, and highlights whose turn it is. Player 1 breaks from behind the dashed line. Balls potted on the break keep the turn but the table stays open; the 8 on the break is spotted back. After the break, the first ball potted legally gives that player solids or stripes. You keep shooting after potting one of your own balls. Fouls give the other player ball in hand anywhere: potting the cue ball, hitting nothing, hitting the wrong ball first (or the 8 first on an open table), or no ball potted and no ball reaching a cushion. Potting the 8 after clearing your group wins; potting it earlier or with a foul loses. In game mode only the cue ball can be moved, and only with ball in hand. Not included: called pockets, push-outs, the three-foul rule.

Use it to practise angles and cue-ball control, not to predict exact outcomes. The same limits are shown in the simulator window.

The screenshot tester opens with a clean table from [Miniclip's aiming guide](https://support.miniclip.com/hc/en-us/articles/203747546-How-to-Aim-with-the-Cue-8-Ball-Pool). **Earlier table sample** loads a full screenshot of a real table. **Open screenshot…** checks any saved image without AirPlay. It draws exactly what the live overlay would draw.

## Testing on a synthetic table

`fake_game.py` draws generic pool-table frames in the visual style the detector relies on: blue cloth with lighter cushions, dark pockets, shaded numbered solid and striped balls, a decorated cue stick with round joints, and a white aim guide (ray, hollow ghost ball, short object-ball and cue-ball lines), plus faint table decorations, letterbox bars, JPEG compression and blur. Its colours were measured on the real screenshot. It is a test fixture, not a game. Because every frame is drawn from known numbers, the detector can be checked against the exact truth.

- `python stress_test.py --count 300` checks the table edges, every ball, the cue ball, ghost ball, aim, target, and the object- and cue-ball directions on random shots, and lists every failed check. `--save-failures DIR` writes annotated images.
- `python live_check.py` runs the whole app live on Windows. It opens the synthetic table in a window, starts the overlay, presses Fit table and Enable guide on the app's own buttons, and compares each analysed frame with the exact image and truth.
- `tests/test_synthetic_table.py` keeps the accuracy and each fixed flaw under test.

Results on 300 random shots that were not used while tuning: 283 passed every check. Table edges were within 1.2 px, aim within 0.11° at the 95th percentile (and never more than 1 px off where it meets the ghost ball), and object-ball direction within 0.7° at the 95th percentile. On average 0.02 balls per frame were missed. Live through the Windows compositor, 23/23 scenes matched the truth; under the visible guide, the captured pixels were off by up to ~190 brightness levels, and after subtraction within 1 level (99th percentile).

Known limits found this way:
- When the struck ball nearly touches the cue ball, the ghost ring lies under the cue ball and cannot be read.
- A ball boxed in on all sides inside a rack can have its centre off by up to about half a radius. It does not affect the struck ball, which is fitted against the ghost.

The synthetic table approximates the real game's look. It cannot replace a check on the real AirPlay mirror.

## Recording shots (for fitting the game's power)

`shot_recorder.py` logs real shots so the game's power and friction can be fitted. It only reads the mirror window's picture: it never clicks, aims or sends input.

1. Start the AirPlay receiver and the game as usual. The overlay may run at the same time.
2. Run `.venv\Scripts\python shot_recorder.py` (add `--title "part of the window title"` if it does not find the mirror window).
3. Play normal shots, about 20–30, at many different powers. Hold the slider still for a moment before letting go. Ctrl+C stops.

Each shot gets a JSON log and two PNGs (the release and the rest frame) in `outputs/shots/<date-time>/`. That folder is not in git, because the frames show the game and a player's profile. A log has:
- the slider reading at release;
- every ball before the shot, and the aim the game showed;
- the cue ball's and the struck ball's positions frame by frame;
- where everything stopped, and which balls were potted.

The opponent's shots are logged with power `null`. Moving the cue ball by hand (ball in hand) is not logged. Spin is not read.

`python recorder_check.py --all-cloths` checks the recorder on whole synthetic games against the exact truth. The test: 18 cloths, 30 fps, the slider snapping back over two frames, and every 5th shot without a slider.
- **Every check:** 227 of 288 shots were logged right.
- **What the power fit uses:** 256 were right in the power at release (within 0.4%), the cue ball's and the struck ball's tracks, and where they stopped or went in. On most cloths the cue ball's movement is within 1 px.
- **The struck ball touching the cue ball:** 19 shots. The detector misses that ball, so the log says the struck ball is unknown rather than guessing.
- **The rest:**
  - the cue ball's position on white, sky and beige cloth, 3–5 px off;
  - end spots of balls close to the cloth's colour;
  - a few balls not followed by the recorder, matched by colour to the wrong ball afterwards.

## Tests

The real game screenshots (`outputs/example_pool_screenshot.png` and `outputs/miniclip_example_aiming.png`) are not in this public repository: one is Miniclip's image and the other shows a player's profile. Without them the 19 screenshot tests are skipped and the two sample buttons in the screenshot tester say the sample is missing. To run those tests, put your own 8 Ball Pool screenshots under those names in `outputs/` (the exact expected positions in the tests only match the original frames). The synthetic-table tests need no screenshots.

```
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python -m py_compile main.py detector.py capture.py overlay.py simulator_ui.py pool_physics.py fake_game.py stress_test.py live_check.py
```
