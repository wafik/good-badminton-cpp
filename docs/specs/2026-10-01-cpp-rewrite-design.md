# Good-Badminton C++ Rewrite — Design

Date: 2026-10-01. Status: approved (user: "lanjutkan sampai jadi").

## Goal

Full C++ port of the Good-Badminton CLI pipeline (approach A). Same outputs as
Python: annotated video (court frames only), detections.jsonl, metadata.json,
court_annotations.txt, audio mux. Heatmaps stay Python (post-process jsonl).

## Location

Sibling of the Python project: `C:/Users/Ulin/Documents/kerjaan/riset/Good-Badminton-Cpp/`.

## Stack

- MSVC 14.44 (VS2022 Community), CMake 4.4, x64.
- OpenCV 4.10 prebuilt (`third_party/opencv/build`).
- ONNX Runtime 1.19.2 CPU (`third_party/onnxruntime-win-x64-1.19.2`).
- nlohmann/json single header (`third_party/nlohmann_json.hpp`).
- ffmpeg/ffprobe on PATH (audio detect + mux), same commands as Python.

## Models

- Pose: `yolo11n-pose.pt` → `yolo11n-pose.onnx`, imgsz 960 fixed, conf 0.15, COCO-17.
- Ball: `yolo11s-ball.pt` → `yolo11s-ball.onnx`, imgsz 640 dynamic (prod runs ROI-scaled, typically 320), conf 0.18.
- Exported once via `tools/export_models.py` (ultralytics in Python venv). Artifacts in `weights/`.

## Architecture (src/)

```
main.cpp            CLI: input video, --out dir, --annotations path
system.cpp          process_video loop + RallyState (faithful port)
court/ detector.cpp mapper.cpp reference.cpp   (OpenCV only)
detect/ onnx_session.cpp yolo_pose.cpp shuttlecock.cpp
track/  player_tracker.cpp
viz/    stats_panel.cpp player_skeleton.cpp court_trajectory.cpp
io/     jsonl_writer.cpp video_io.cpp ffmpeg.cpp
```

## Pipeline semantics (must match Python exactly)

1. Court gate: grayscale matchTemplate TM_CCOEFF_NORMED vs template >= 0.75.
   Non-court: consecutive_non_court_frames++, >=5 force-ends rally, frame NOT
   written to output, no jsonl record. frame counter always increments;
   detect_frame only on court frames.
2. Court frame: ROI crop -> pose ONNX -> ball ONNX (ROI + adaptive imgsz) ->
   shuttlecock filter/state machine -> RallyState.observe(frame_count/fps, seen)
   -> PlayerTracker.update (4 slots, gate 150px, occupied-first) -> jsonl ->
   overlays -> write frame.
3. RallyState: 3 hits / 2.0s window starts; 4.0s quiet ends; court cut force_end.
   start -> tracker.start_new_rally(); end -> clear trajectory.
4. Cleanup: close jsonl/writer, ffprobe audio check, ffmpeg mux (libx264 crf 20,
   aac 160k, -shortest when audio), fallback `-an`, delete temp.

## JSONL schema (identical)

{"schema_version":"1.0","frame":int,"time_sec":float|null,"detect_frame":int,
 "players":{upper|lower|upper2|lower2:{image:[x,y]|null,court:[x,y]|null,
 speed":float|null,"hands":{"left":[x,y]|null,"right":[x,y]|null}}},
 "shuttlecock":{"image":[x,y]|null}}

Zero positions -> null. Same rounding (time_sec 6 dp; floats via shortest repr
matters less — parity check uses tolerance).

## Court annotation

Reuse `court_annotations.txt` when present (same keys: corners, roi_corners,
mid_height). Headless auto-detect fallback writes preview PNG. No interactive
GUI in v1 (manual annotations edited by hand if auto fails).

## Parity gate (definition of done)

`tests/compare_parity.py`: run test4 through both pipelines, join jsonl on
(frame, detect_frame), assert player court coords within 1.0 m tolerance on
>=95% of records, rally count within +/-2, shuttle image within 15 px on
>=90% of records where both see it. Plus `tests/test_rally.cpp` ported
assertions pass.

## Error handling

- Model/session load failure -> clear message, exit 2.
- ffmpeg missing/fail -> fallback re-encode -an; second failure -> keep temp, exit 3.
- Auto-court fail -> keep manual annotations requirement message, exit 4.

## Out of scope (v1)

webui, interactive court annotation, heatmaps (Python side), zh overlay
(EN-only overlay; zh fonts via freetype later), multithreaded pipeline.
