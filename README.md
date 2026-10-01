# Good-Badminton-Cpp

Rewrite **C++17** dari pipeline analisis video bulu tangkis [Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton) (Python) — untuk 1 video panjang yang dulu bisa >10 menit di Python, versi ini jauh lebih ringan (headless, tanpa overhead interpreter) sambil **mempertahankan paritas CLI**: coverage, court detect, shuttle detect, dan rally count identik (17=17 pada test set).

GUI-nya: [Good-Badminton-Studio](https://github.com/wafik/good-badminton-studio) (Tauri 2) — memanggil binary ini sebagai subprocess.

## Stack

- C++17 (MSVC / VS2022), CMake ≥ 3.20
- OpenCV 4.10 (`third_party/opencv`, lokal, tidak ikut ter-commit)
- ONNX Runtime 1.19.2 (`third_party/onnxruntime-win-x64-1.19.2`)
- Model: `yolo11s-ball.onnx` (shuttlecock), `yolo11n-pose-dyn.onnx` (pose)

## Build

```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
# hasil: build/Release/gb_cpp.exe
```

`third_party/` harus sudah berisi OpenCV + ONNX Runtime (lihat `CMakeLists.txt` untuk path persisnya).

## CLI

```
gb_cpp [input_video] [options]
  --out DIR                  output directory (default: outputs/<name>)
  --annotations PATH         court annotations file
  --template PATH            court template image (default: auto — a frame from the video)
  --ball-model PATH          shuttlecock ONNX
  --yolo-pose-model PATH     pose ONNX
  --audio true|false         keep original audio (default: true)
  --display true|false       show video window (default: false, headless)
  --language zh|en|id        stats panel language (default: zh; zh renders EN text)
  --performance-stats ...    per-5s frame timings (default: true)

  # 6 display toggle — default true = paritas Python (main.py)
  --skeletons true|false
  --player-trajectories true|false
  --court-trajectory true|false
  --shuttlecock-trajectory true|false
  --player-stats true|false
  --pose-roi true|false

  --progress-json            emit {"frame":N,"total":M} per frame (opt-in)

Exit codes: 2 model load, 3 ffmpeg, 4 court annotation, 1 lainnya.
```

Output per run: `outputs/<name>/` — video `detect_<nama_video>.mp4`, `detections.jsonl`, `court_annotations.txt`.

## Test

```powershell
build/Release/test_rally.exe   # harus cetak "ok" — rally state self-check
```

CI di GitHub Actions menjalankan check yang sama (tanpa OpenCV — cukup cl + `src/core/rally_state.cpp`).

## Credits

- Upstream: [yo-WASSUP/Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton) — proyek Python asli yang menjadi acuan rewrite ini (juga [Good-Tennis](https://github.com/yo-WASSUP/Good-Tennis), [Good-Pickleball](https://github.com/yo-WASSUP/Good-Pickleball)).
- Pipeline Python lokal `Good-Badminton` — acuan paritas (argumen, default, format output, perilaku exit code).

## Lisensi

Ikuti lisensi upstream ([yo-WASSUP/Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton)).
