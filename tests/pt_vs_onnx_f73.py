# frame-73 forensics: .pt vs .onnx detection sets (conf=0.15, imgsz=960, ROI 400x850)
import numpy as np
import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(r"C:/Users/Ulin/Documents/kerjaan/riset/Good-Badminton/videos/test4.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 72)
ok, frame = cap.read()
cap.release()
print("frame ok:", ok, frame.shape if ok else None)
roi = frame[0:850, 0:400]

w = r"C:/Users/Ulin/Documents/kerjaan/riset/Good-Badminton/weights/"
res = {}
for name in ("yolo11n-pose.pt", "yolo11n-pose.onnx"):
    m = YOLO(w + name)
    r = m(roi, conf=0.15, imgsz=960, device="cpu", verbose=False)[0]
    if r.keypoints is not None and len(r.boxes):
        xy = r.keypoints.xy.cpu().numpy()
        bc = r.boxes.conf.cpu().numpy()
        print(name, "n=", len(r.boxes), "box_conf=", np.round(bc, 4).tolist())
        for i in range(len(r.boxes)):
            feet = (xy[i][15] + xy[i][16]) / 2
            print("   feet=", np.round(feet, 1).tolist())
        res[name] = [(float(bc[i]), tuple(np.round((xy[i][15] + xy[i][16]) / 2, 1))) for i in range(len(r.boxes))]
    else:
        print(name, "n=0")
        res[name] = []

# also probe just-below-threshold with conf=0.10 to see flip margin
for name in ("yolo11n-pose.pt", "yolo11n-pose.onnx"):
    m = YOLO(w + name)
    r = m(roi, conf=0.10, imgsz=960, device="cpu", verbose=False)[0]
    n = len(r.boxes) if r.keypoints is not None else 0
    bc = r.boxes.conf.cpu().numpy() if n else np.array([])
    print(name, "@conf0.10 n=", n, "conf=", np.round(bc, 4).tolist())
