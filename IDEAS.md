# Ideas

## True 60 guide updates per second

Today (after PR #8, still true after #20) the live guide updates about 15 times per second while aiming. Per frame (15 Nov: tracked frame ~15 ms, full detection ~150 ms on the synthetic table):

| Step | Time |
|---|---|
| Screen capture (mss) | ~8 ms |
| Working out which guide drawing is on screen, and removing it | ~6 ms |
| Reading the aim (`predict_shot` via `tracking.py`) | 15–30 ms |

60 updates per second leaves 16.7 ms for everything, so speeding up this loop a little more won't get there. It needs three changes. Check step 0 first.

### 0. Measure the mirror's own frame rate first

AirPlay mirroring often sends only 30 frames per second, depending on the phone and the receiver. Analysing more frames than the mirror shows only repeats work, and the guide can't be faster than the game. Count how many *different* frames the UxPlay window shows per second while aiming. If it's 30, the goal is "every frame the mirror shows", not 60.

### 1. Capture the AirPlay window, not the screen (Codingchancer, `overlay.py`)

Use Windows Graphics Capture to record only the UxPlay window. Windows on top of it are left out, so our own guide is never in the frames. Candidate library: `windows-capture` (MIT licence); check the licence and that it works with UxPlay before using it.

- Capture drops from ~8 ms to 1–2 ms.
- Removing the guide and checking which drawing is on screen (`remove_overlay`, `shown_layer`) are no longer needed in the live loop. Keep them as the fallback when only screen capture works.
- The app is told each time the mirror has a new frame, instead of checking over and over.
- The guide still shows in screenshots and screen shares (`WDA_NONE` stays).
- Check: on Windows 11 the yellow capture border can be turned off. Capture should keep working when the window is covered.
- Open the capture session once and reuse it, and close it when the guide is turned off. Issue #13 (fixed in #15): opening a new mss capture on every frame leaked Windows GDI objects until the guide froze. `tests/test_capture_reuse.py` guards this for mss. Write the same kind of test for the new capture.

### 2. Fast aim reader (Codingchancer, `tracking.py`)

While aiming, only the angle changes, and only slightly per frame. Talk to Jartartarta before starting: his `jartartarta/faster` task (faster detection per frame, same results) works on the same time budget from the detector side.

- Try only angles within about ±2° of the last aim, sampling pixels along each candidate line with `cv2.remap` (~1 ms).
- Look for the ghost ball only in a small box around where it should be on the new line (< 1 ms).
- Work out the object-ball and cue-ball paths from that geometry.
- Run the full `predict_shot` every few frames in the background to correct drift and read the game's own white lines. Use it at once if the fast reader loses the aim.
- Target: 3–5 ms per frame. Keep the results within the `GuideStabilizer` tolerances of the full reading, and check this on synthetic aiming sweeps.

### 3. Overlap frames, in a separate process (Codingchancer, `overlay.py`)

Once the guide is no longer in the frames, frames don't depend on what was drawn before.

- Capture frame N+1 while frame N is being analysed.
- Run capture and analysis in their own process and pass the results back (shared memory or a queue), so Python running one thread at a time can't slow down drawing.

### Expected result

- 60 updates per second if the mirror sends 60 fps, with ~10–15 ms from the game frame to the drawn guide.
- With a 30 fps mirror, the guide follows every frame the mirror shows.

### Not planned

Guessing where the aim will be between real frames to fake 60. The guide would sometimes show lines the game isn't showing.
