// Port of detection/yolo_pose.py (YOLOPoseProcessor) to ONNX Runtime.
//
// ONNX layout verified against weights/yolo11n-pose.onnx with the project venv
// (onnxruntime InferenceSession): input images (1,3,960,960) fixed; output0
// (1,56,18900) channel-major, raw (no NMS baked in):
//   per anchor: [cx,cy,w,h, cls_conf, (kpt_x,kpt_y,kpt_conf) x 17]
//   56 = 4 box + 1 nc + 17*3  (class score at channel 4 — the task brief's
//   "56 = 4+17*3" arithmetic was off; there IS an extra class channel).
// 18900 = 120^2 + 60^2 + 30^2 anchors for 960 input.
// Byte-exact parity with ultralytics predict(conf=0.15, imgsz=960) was
// verified on demo.mp4 frame 100 (6 boxes, keypoints match to <0.1 px).
// ponytail: no device selection (Python auto-picks CUDA via torch) — ORT CPU
// EP only; bundled ORT has no CUDA provider. The Python TypeError fallback
// for odd model() call signatures is an ultralytics quirk, N/A in C++.
#include "gb/detect.h"

#include <algorithm>
#include <iostream>

#include "onnx_util.h"

namespace gb {

namespace {
constexpr int64_t kPoseChannels = 56;  // 4 + nc(1) + 17*3
constexpr float kNmsIou = 0.7f;        // ultralytics predict default args.iou
constexpr int kMaxDet = 300;           // ultralytics default max_det
}  // namespace

struct YoloPoseProcessor::Impl {
    double conf = 0.15;
    int imgsz = 960;
    detect_detail::OnnxModel model;
    bool ok = false;
    bool warned = false;
};

YoloPoseProcessor::YoloPoseProcessor(const std::string& model_path, double conf,
                                     int imgsz)
    : impl_(std::make_unique<Impl>()) {
    impl_->conf = conf;
    impl_->imgsz = imgsz;
    // Python: print(f"Initializing YOLO pose model (model: ..., device: ...)")
    std::cout << "Initializing YOLO pose model (model: " << model_path
              << ", device: cpu)" << std::endl;
    std::string err;
    if (!impl_->model.load(model_path, err)) {
        std::cerr << "YOLO pose model failed to load: " << err << std::endl;
        return;
    }
    if (!impl_->model.dynamic_hw && (impl_->model.static_h != imgsz ||
                                     impl_->model.static_w != imgsz)) {
        std::cerr << "YOLO pose model input " << impl_->model.static_h << "x"
                  << impl_->model.static_w << " != imgsz " << imgsz << std::endl;
        return;
    }
    impl_->ok = true;
}

YoloPoseProcessor::~YoloPoseProcessor() = default;

bool YoloPoseProcessor::ok() const { return impl_->ok; }

std::vector<PersonKeypoints> YoloPoseProcessor::process_frame(
    const cv::Mat& frame) {
    std::vector<PersonKeypoints> out;
    if (!impl_->ok || frame.empty()) return out;

    // Ultralytics disables LetterBox auto for static-input ONNX
    // (pre_transform: ... and (dynamic or pt)); dynamic -> stride-32 min pad.
    const detect_detail::LetterboxResult lb = detect_detail::letterbox(
        frame, impl_->imgsz, impl_->imgsz, impl_->model.dynamic_hw);

    cv::Mat blob;
    cv::dnn::blobFromImage(lb.img, blob, 1.0 / 255.0, lb.img.size(),
                           cv::Scalar(), /*swapRB=*/true, /*crop=*/false);

    std::vector<float> raw;
    std::vector<int64_t> odims;
    std::string err;
    if (!impl_->model.run(reinterpret_cast<const float*>(blob.data),
                          {1, 3, lb.fed_h, lb.fed_w}, raw, odims, err)) {
        if (!impl_->warned) {
            std::cerr << "YOLO pose inference failed: " << err << std::endl;
            impl_->warned = true;
        }
        return out;
    }
    if (odims.size() != 3 || odims[0] != 1 || odims[1] != kPoseChannels ||
        odims[2] <= 0) {
        if (!impl_->warned) {
            std::cerr << "YOLO pose unexpected output shape (want 1x56xN)"
                      << std::endl;
            impl_->warned = true;
        }
        return out;
    }
    const int64_t A = odims[2];
    const size_t As = static_cast<size_t>(A);

    struct Det {
        int64_t a;
        float conf;
        float cx, cy, w, h;
    };
    std::vector<Det> dets;
    dets.reserve(64);
    for (int64_t a = 0; a < A; ++a) {
        // Ultralytics NMS: amax(class chans) > conf, then best-class conf >
        // conf — identical for nc=1 (channel 4 is the only class score).
        const float c = raw[4 * As + a];
        if (c > impl_->conf) {
            dets.push_back({a, c, raw[0 * As + a], raw[1 * As + a],
                            raw[2 * As + a], raw[3 * As + a]});
        }
    }
    if (dets.empty()) return out;

    std::stable_sort(dets.begin(), dets.end(),
                     [](const Det& x, const Det& y) { return x.conf > y.conf; });

    // NMS in input (letterboxed) space — same as ultralytics postprocess,
    // which NMS-es before scaling boxes/keypoints back.
    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    boxes.reserve(dets.size());
    scores.reserve(dets.size());
    for (const Det& d : dets) {
        boxes.emplace_back(d.cx - d.w / 2, d.cy - d.h / 2, d.w, d.h);
        scores.push_back(d.conf);
    }
    const std::vector<int> keep =
        detect_detail::nms_xywh(boxes, scores, kNmsIou, kMaxDet);
    if (keep.empty()) return out;

    const detect_detail::ScaleBack sb = detect_detail::scale_back(
        lb.fed_h, lb.fed_w, frame.rows, frame.cols);

    out.reserve(keep.size());
    for (int idx : keep) {
        const Det& d = dets[idx];
        PersonKeypoints pk;
        for (int k = 0; k < 17; ++k) {
            const float kx = raw[(5 + 3 * k) * As + d.a];
            const float ky = raw[(6 + 3 * k) * As + d.a];
            pk.xy[k] = detect_detail::scale_point(kx, ky, sb);  // clip to frame
            pk.conf[k] = raw[(7 + 3 * k) * As + d.a];  // conf channel unscaled
        }
        out.push_back(pk);
    }
    return out;
}

}  // namespace gb
