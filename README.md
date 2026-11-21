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

- **Table:** the four corner pockets give a rough fit (pockets must open onto the blue cushions, so dark wallpaper and panels on a multi-monitor desktop are ignored). With one corner covered, three corners and a middle pocket do; with both corners of a short side lit up (the game rings pockets in red, leaving only a small dark core), the other two corners and the two middle pockets do, mirrored about the middle pockets. Each side is then snapped onto the cushion nose, where brightness steps down from cushion to cloth. A thin bright line on the cloth (the guide's own table border if a capture still shows it, or the game's aim line, the stick and the cue ball along a cushion) drops just as sharply but back to the cloth's level, so it is not taken for the nose. The cloth-colour fit remains as a fallback when the pockets are out of view.
- **Balls:** circles are proposed on both a grey image and a "difference from the cloth colour" image (the second finds blue balls on blue cloth). Each proposal is snapped onto the ball's edge to a fraction of a pixel and kept only when its inside is mostly not cloth, which rejects the hollow ghost ring, watermark rings, lettering and shadows. A ball the colour of the cloth (the purple 4 on a purple table) looks like cloth inside, and as the balls roll its number is mostly turned away. It is kept when it shows the glossy highlight that every ball has at the same spot (measured on the frame's other balls; upper left on real frames), brighter and paler than the cloth and round, not part of a line such as the game's guide or the cue stick. Balls resting against a cushion are kept. Of two circles on one ball (a second one locks onto a striped ball's white cap), the one whose rim lies better on the ball's edge stays. On a pale cloth (white, light grey, beige, sky) a striped ball's white caps blend into the cloth, so it is fitted on the two opposite arcs where its coloured band meets the cloth. Circles on the cue stick behind the cue ball are dropped unless they are full ball size with a clear round edge (stick joints are smaller).
- **Cue ball:** the white ball that the game's aim ray starts from and that ends in a ghost ball. The cue stick only breaks ties, because near a rail most of it is off the table. A striped ball turned end-on (a white face ringed by its colour) is as white as the cue ball, but the cue ball is white right out to its rim (apart from its small red dot), so a ball with a coloured rim is not taken for it. When no ray leads to a ghost (the struck ball right next to the cue ball hides it), the ghost ring is looked for on its own anywhere on the table: a thin bright stroke between dark outlines all the way round, of a ghost ring's size. Only the cue ball can overlap the ghost ring (the ring marks where the cue ball will be at impact; the struck ball only touches it), so the ball under it is the cue ball. If circle finding locked onto the ring or missed the ball under it, the ball is found again on the picture with the ring painted out.
- **Cue ball beside a pocket:** with the cue ball against a cushion next to a corner pocket and aimed into it, the game draws the ghost ring in the pocket's mouth, past the cushion line, and the aim line lies almost entirely on the cue ball. The ring is looked for on the table with a margin round it, so it is found there too; with no ghost on a ray, a ring a short way from the cue ball at the cushion line is taken as its ghost, and the nearest white ball beside it as the cue ball. The stick is only used for the aim when it lies on the cloth behind the cue ball (along a cushion the cushion's straight edge was read as the stick, 5 degrees off).
- **Aim:** a scan outward from the cue ball in every direction (0.5° steps) finds the white ray, even when it is short or crossed by other lines. Of the candidate directions, the one that ends in Miniclip's hollow **ghost-ball reticle** is used. The reticle centre (found to a fraction of a pixel) is where the cue ball will be at impact, and the angle is fitted on the ray's own pixels through it. When the target is so close that the ring lies on the cue ball, the ring is fitted on its own stroke (0.05 px on synthetic frames) and gives the ghost. The aim then comes from the cue stick behind the cue ball: from its centre to a ghost only a few pixels away the angle would turn by degrees per pixel. The stick's band of darker or lighter shade than the cloth (or else its straight edges) gives the direction to a fraction of a degree.
- **Target:** the ball that touches the ghost ball. When the ghost rests on a cushion and a ball lies just beside it, that ball only counts if the game draws an object-ball line from it.
- **After impact:** Miniclip draws short white lines for the object ball and the cue ball. When they are visible, the app fits them directly and extends them through a cushion bounce. Each line is first found as the narrow straight band through the ball (or ghost) that holds most of its pixels, so other white in its way does not tilt it: the cushion's lit edge, a striped ball's cap, or the table's faint head string. When the object line is hidden (it runs across other balls), the struck ball's centre is fitted one diameter from the ghost centre and the direction follows from that; the cue ball leaves at 90° (no spin). Paths stop at the first ball in the way, which gets an orange dotted ring.

- **White cloth:** the game's white lines are no brighter than the cloth there and show only as their dark outlines. The aim ray, the object and cue lines and (when needed) the ghost ring are read as thin bright ridges between those outlines. The cue ball's and striped balls' rims are shaded darker than the cloth and run into their drop shadows. So they are fitted on the dip at the rim on their lit side, away from the shadow that the other balls show. The cue ball can look like cloth to circle finding (white on white, no number badge). A circle it rejected counts after all when it has a ball's lit-side rim, drop shadow and lit face, and no other ball lies near it.
- **Light grey cloth:** its brightly lit middle passes the white test itself, so a guide stroke must be brighter than the cloth around it, as on white cloth. The cue ball's rim is no darker than the cloth there and has no tint. So it is fitted where the ball's shading meets the cloth's level on its lit side, also with the ghost ring painted out in near shots.
- **Grey and light tinted cloths:** on grey cloth the table's watermark lettering is a darker shade of the cloth and is not taken for balls (as on light cloths). On a light tinted cloth (beige, sky, yellow, lime) the cue ball's shaded rim is as light as the cloth, and its drop shadow loses colour like the white ball. So the cue ball is fitted where the cloth's tint begins along its lit side.

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

`fake_game.py` draws generic pool-table frames in the visual style the detector relies on: blue cloth with lighter cushions, dark pockets, shaded numbered solid and striped balls turned at random as the game's balls are (the one number badge mostly off-centre, at the rim or round the back; a stripe is a belt round the ball; in the headless test game the balls roll along their paths), a decorated cue stick with round joints, and a white aim guide (ray, hollow ghost ball, short object-ball and cue-ball lines), plus faint table decorations, letterbox bars, JPEG compression and blur. Its colours were measured on the real screenshot. It is a test fixture, not a game. Because every frame is drawn from known numbers, the detector can be checked against the exact truth.

- `python stress_test.py --count 300` checks the table edges, every ball, the cue ball, ghost ball, aim, target, and the object- and cue-ball directions on random shots, and lists every failed check. `--save-failures DIR` writes annotated images.
- `python live_check.py` runs the whole app live on Windows. It opens the synthetic table in a window, starts the overlay, presses Fit table and Enable guide on the app's own buttons, and compares each analysed frame with the exact image and truth.
- `tests/test_synthetic_table.py` keeps the accuracy and each fixed flaw under test.

Results on 300 random shots that were not used while tuning: 283 passed every check. Table edges were within 1.2 px, aim within 0.11° at the 95th percentile (and never more than 1 px off where it meets the ghost ball), and object-ball direction within 0.7° at the 95th percentile. On average 0.02 balls per frame were missed. Live through the Windows compositor, 23/23 scenes matched the truth; under the visible guide, the captured pixels were off by up to ~190 brightness levels, and after subtraction within 1 level (99th percentile).

Known limits found this way:
- When the struck ball nearly touches the cue ball, the ghost ring lies on the cue ball. The ring is found and the aim is read from the stick, but where the stick does not show behind the cue ball (about a quarter of synthetic near shots, for example against a cushion) the aim comes from the cue ball's own centre, which is then only good to about a pixel, so it can be a few degrees off. On white and pale cloths (white, light grey, beige, sky) about a quarter to a third of these shots still fail a check, mostly on the cue ball's position. Synthetic "near" shots, 100 per cloth on 19 cloths: 1651 of 1900 pass every check (570 before the ring was found on its own).
- On a pale cloth a striped ball frozen against a cushion, or packed beside other balls, keeps the centre that circle finding gave it, which can be up to about half a radius off. About one in a hundred striped balls is never proposed by circle finding there (for example one on the table's lettering) and is missed.
- A ball boxed in on all sides inside a rack can have its centre off by up to about half a radius. It does not affect the struck ball, which is fitted against the ghost.
- The table fit needs two corner pockets of one short side, or both of one long side and a third corner, to show as dark holes; with three or more corners lit or covered it falls back to the cloth colour. A table border that a capture still shows lying exactly on a cushion nose moves that edge 1–2.5 px inwards (the guide is normally not in the frames).

The synthetic table approximates the real game's look. It cannot replace a check on the real AirPlay mirror.

## Neural-network ball finder (experiment, off)

`ball_net.py` is a small trained network (60k weights, `models/balls.onnx`, 0.24 MB) that can propose where the balls are instead of circle finding. It looks at the table picture shrunk to 640x320 and marks every ball centre (cue ball, solid, 8, stripe) in 6 ms on the CPU, using OpenCV's own `cv2.dnn`, so the app needs nothing new. Its marks are only proposals: each is snapped onto the ball's rim and goes through every check and fit described above, exactly like a circle-finding proposal. It does not aim, click or touch the game.

It is trained on the synthetic table and on real screenshots (`train_ball_net.py`: `render` draws synthetic frames, `import-real` and `split-real` bring in a public set of real screenshots with hand-drawn ball boxes, `train` and `export` run in a separate PyTorch environment, see `requirements-train.txt`). All training pictures stay in `outputs/` and are never committed; only the model is. Switch it on with `set POOL_OVERLAY_NN_BALLS=only+trust` before starting the app or a test.

The real screenshots come from the public "8 ball detector" dataset by "8ball" on Roboflow Universe (https://universe.roboflow.com/8ball/8-ball-detector-jcbfd), CC BY 4.0. They are frames of a few recorded games, so they were split by scene: 489 frames from 10 scenes for training and 218 frames from 34 other scenes for testing, never trained on.

What it changes, measured frame by frame against the detector without it:
- Real screenshots of that set (218 test frames, 2334 balls, shown at mirror size): balls found 94.5% -> 97.1%, false balls 108 -> 14, frames with every ball right 128 -> 182.
- Synthetic table (19 cloths, frames it never saw): 2384 -> 2736 of 2850 frames pass every check; near shots 1498 -> 1723 of 1900; 300 random shots 281 -> 287 (old flat balls: 2630 -> 2769, 1658 -> 1721, 291 -> 293). The black 8 on the near-black cloth is found in 242 of 250 frames instead of 69, the purple 4 on purple cloth in 248 instead of 199, and packed racks pass 541 of 570 instead of 437.
- Our own real screenshots (`outputs/bad`, at mirror resolution): ball centres are as good as before (about 1 px from the ball's dark outline, fewer large misses), but it still misses balls that the cue stick lies across and takes one of the stick's round ornaments for a ball. The public set rarely shows the stick across a ball. A cue ball pressed against a cushion beside the ghost ring is found by the network but still dropped by a later check (the detector without it picks the wrong cue ball there too).

So it stays off until frames with the stick across balls are in its training. `frame_collector.py` saves such frames from the mirror window while you play (it only reads the picture).

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

`python recorder_check.py --all-cloths --fps 30 --snap 2 --no-slider-every 5` checks the recorder on whole synthetic games against the exact truth: 18 cloths, 3 games each, 30 fps, the slider snapping back over two frames, every 5th shot without a slider. The balls turn as they roll, as the game's do (`--flat` gives the old look: number up, straight band).
- **Every check:** 389 of 648 shots were logged right (521 with the old flat look).
- **What the power fit uses:** 490 were right in the power at release (within 1.5%), the cue ball's and the struck ball's tracks, and where they stopped or went in (582 flat). The struck ball is followed as a disc on the cloth, not by its picture, since its number and stripe turn as it rolls.
- **The struck ball touching the cue ball:** 43 shots. The detector misses that ball, so the log says the struck ball is unknown rather than guessing.
- **The rest:**
  - the struck ball's movement 1.5–3 px off its true movement, mostly on white and pale cloths and against cushions or pocket jaws;
  - end spots of balls the recorder does not follow, from the detector, more than 3 px off;
  - on the break, a solid and a stripe of one colour swapped in the potted list (in the packed rack a stripe can show mostly white).

## Tests

The real game screenshots (`outputs/example_pool_screenshot.png` and `outputs/miniclip_example_aiming.png`) are not in this public repository: one is Miniclip's image and the other shows a player's profile. Without them the 19 screenshot tests are skipped and the two sample buttons in the screenshot tester say the sample is missing. To run those tests, put your own 8 Ball Pool screenshots under those names in `outputs/` (the exact expected positions in the tests only match the original frames). The synthetic-table tests need no screenshots.

```
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python -m py_compile main.py detector.py capture.py overlay.py simulator_ui.py pool_physics.py fake_game.py stress_test.py live_check.py ball_net.py train_ball_net.py frame_collector.py
```
