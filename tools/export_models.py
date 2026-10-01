# One-shot: export .pt weights to ONNX for the C++ pipeline.
# Run from Good-Badminton: .venv/Scripts/python.exe tools/export_models.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO  # noqa: E402

W = Path(__file__).resolve().parents[2] / "Good-Badminton" / "weights"

# pose: fixed imgsz 960 (production); ball: dynamic, prod runs ROI-scaled 320..640
YOLO(str(W / "yolo11n-pose.pt")).export(format="onnx", imgsz=960, opset=17)
YOLO(str(W / "yolo11s-ball.pt")).export(format="onnx", imgsz=640, dynamic=True, opset=17)
print("EXPORT_DONE")
