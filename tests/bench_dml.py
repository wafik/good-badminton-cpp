# DirectML vs CPU inference benchmark (run with a venv that has
# onnxruntime-directml). Feeds representative letterboxed tensors:
#   pose: test4 ROI 400x850 @imgsz960 rect -> (1,3,960,480)
#   ball: court crop @imgsz640 rect         -> (1,3,640,384)
import time
import numpy as np
import onnxruntime as ort

W = r"C:/Users/Ulin/Documents/kerjaan/riset/Good-Badminton/weights/"
CASES = [
    ("pose", W + "yolo11n-pose-dyn.onnx", (1, 3, 960, 480)),
    ("ball", W + "yolo11s-ball.onnx", (1, 3, 640, 384)),
]
DML = "DmlExecutionProvider" in ort.get_available_providers()


def bench(path, shape, providers, label, warmup=5, iters=30):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, so, providers=providers)
    name = sess.get_inputs()[0].name
    x = np.random.rand(*shape).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {name: x})
    t0 = time.perf_counter()
    for _ in range(iters):
        sess.run(None, {name: x})
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"{label}: {ms:.1f} ms")
    return ms


print("providers:", ort.get_available_providers())
for label, path, shape in CASES:
    cpu = bench(path, shape, ["CPUExecutionProvider"], f"{label:4} CPU")
    if DML:
        dml = bench(path, shape, ["DmlExecutionProvider", "CPUExecutionProvider"], f"{label:4} DML")
        print(f"      -> DML speedup x{cpu / dml:.2f}")
