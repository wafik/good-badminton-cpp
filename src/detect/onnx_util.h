#pragma once
// Internal helpers for src/detect — mirrors Ultralytics predict-path
// preprocessing/postprocessing exactly (verified against ultralytics source in
// the Good-Badminton venv + onnxruntime sessions on the exported .onnx files).
// Not installed; only included by src/detect/*.cpp.
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifdef _WIN32
#include <windows.h>
#endif

namespace gb {
namespace detect_detail {

// Python round(): nearest, ties-to-even (banker's). Ultralytics LetterBox /
// scale_boxes use it for pad math — std::round (half-away) differs on .5 ties.
inline int pyRound(double v) {
    double f = std::floor(v);
    double diff = v - f;
    if (diff > 0.5) return static_cast<int>(f) + 1;
    if (diff < 0.5) return static_cast<int>(f);
    long fi = static_cast<long>(f);
    return (fi % 2 == 0) ? static_cast<int>(fi) : static_cast<int>(fi) + 1;
}

struct LetterboxResult {
    cv::Mat img;      // BGR, padded to fed_h x fed_w
    int fed_h = 0;
    int fed_w = 0;
};

// Python: ultralytics LetterBox(new_shape=(th,tw), auto=..., stride=32,
// center=True, scaleup=True).__call__ — INTER_LINEAR resize to
// round(orig*r), centered gray-114 pad; auto (dynamic-input models only)
// reduces the pad to `mod stride` (minimum-rectangle; e.g. 1280x720@640 ->
// fed 640x384, verified by spying ORT inputs Ultralytics feeds).
inline LetterboxResult letterbox(const cv::Mat& bgr, int th, int tw, bool auto_pad) {
    LetterboxResult r;
    const int h = bgr.rows, w = bgr.cols;
    const double ratio = std::min(static_cast<double>(th) / h, static_cast<double>(tw) / w);
    // Python: round(shape[1]*r), round(shape[0]*r)
    int nw = pyRound(w * ratio);
    int nh = pyRound(h * ratio);
    int dw = tw - nw, dh = th - nh;
    if (auto_pad) {
        dw %= 32;  // Python np.mod — dw,dh >= 0 here
        dh %= 32;
    }
    const int top = pyRound(dh / 2.0 - 0.1);
    const int bottom = pyRound(dh / 2.0 + 0.1);
    const int left = pyRound(dw / 2.0 - 0.1);
    const int right = pyRound(dw / 2.0 + 0.1);

    cv::Mat resized = bgr;
    if (w != nw || h != nh) {
        cv::resize(bgr, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    }
    cv::copyMakeBorder(resized, r.img, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    r.fed_h = r.img.rows;
    r.fed_w = r.img.cols;
    return r;
}

// Python: ops.scale_boxes/scale_coords(img1_shape=fed_shape, ..., img0_shape).
// gain/pad are recomputed FROM THE ACTUAL FED TENSOR SHAPE (not the stored
// letterbox params): gain = min(fedH/srcH, fedW/srcW),
// pad = round((fed - round(src*gain))/2 - 0.1).
struct ScaleBack {
    double gain = 1.0;
    int pad_x = 0;
    int pad_y = 0;
    int src_h = 0;
    int src_w = 0;
};

inline ScaleBack scale_back(int fed_h, int fed_w, int src_h, int src_w) {
    ScaleBack s;
    s.gain = std::min(static_cast<double>(fed_h) / src_h,
                      static_cast<double>(fed_w) / src_w);
    s.pad_x = pyRound((fed_w - pyRound(src_w * s.gain)) / 2.0 - 0.1);
    s.pad_y = pyRound((fed_h - pyRound(src_h * s.gain)) / 2.0 - 0.1);
    s.src_h = src_h;
    s.src_w = src_w;
    return s;
}

// Python: ops.scale_boxes(xyxy path) — subtract pad, divide gain, clip.
inline cv::Rect2f scale_xyxy(const cv::Rect2f& b, const ScaleBack& s) {
    float x1 = static_cast<float>((b.x - s.pad_x) / s.gain);
    float y1 = static_cast<float>((b.y - s.pad_y) / s.gain);
    float x2 = static_cast<float>((b.x + b.width - s.pad_x) / s.gain);
    float y2 = static_cast<float>((b.y + b.height - s.pad_y) / s.gain);
    x1 = std::min(std::max(x1, 0.0f), static_cast<float>(s.src_w));
    x2 = std::min(std::max(x2, 0.0f), static_cast<float>(s.src_w));
    y1 = std::min(std::max(y1, 0.0f), static_cast<float>(s.src_h));
    y2 = std::min(std::max(y2, 0.0f), static_cast<float>(s.src_h));
    return {x1, y1, x2 - x1, y2 - y1};
}

// Python: ops.scale_coords keypoint path (clip x,y to frame; conf untouched).
inline cv::Point2f scale_point(float x, float y, const ScaleBack& s) {
    float sx = static_cast<float>((x - s.pad_x) / s.gain);
    float sy = static_cast<float>((y - s.pad_y) / s.gain);
    sx = std::min(std::max(sx, 0.0f), static_cast<float>(s.src_w));
    sy = std::min(std::max(sy, 0.0f), static_cast<float>(s.src_h));
    return {sx, sy};
}

// Greedy NMS equivalent to torchvision.ops.nms(iou_thres): sort score-desc,
// keep a box when IoU with every already-kept box <= thr, then cap at max_det.
// Input boxes are xywh (YOLO raw); returns kept indices in score-desc order.
// ponytail: hand-rolled instead of cv2.dnn.NMSBoxes — that API takes xywh and
// has threshold-boundary quirks; a 20-line loop keeps torchvision semantics
// (suppress iff IoU > thr, strict) explicit and dependency-free.
inline std::vector<int> nms_xywh(std::vector<cv::Rect2f>& boxes,
                                 std::vector<float>& scores, float iou_thr,
                                 int max_det) {
    std::vector<int> order(boxes.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return scores[a] > scores[b]; });
    std::vector<int> keep;
    keep.reserve(std::min<size_t>(order.size(), static_cast<size_t>(max_det)));
    for (int idx : order) {
        if (static_cast<int>(keep.size()) >= max_det) break;
        const cv::Rect2f& b = boxes[idx];
        bool overlapped = false;
        for (int k : keep) {
            const cv::Rect2f& kb = boxes[k];
            const float x1 = std::max(b.x, kb.x);
            const float y1 = std::max(b.y, kb.y);
            const float x2 = std::min(b.x + b.width, kb.x + kb.width);
            const float y2 = std::min(b.y + b.height, kb.y + kb.height);
            const float iw = std::max(0.0f, x2 - x1);
            const float ih = std::max(0.0f, y2 - y1);
            const float inter = iw * ih;
            if (inter <= 0.0f) continue;
            const float uni = b.width * b.height + kb.width * kb.height - inter;
            if (uni > 0.0f && inter / uni > iou_thr) {  // suppress iff IoU > thr
                overlapped = true;
                break;
            }
        }
        if (!overlapped) keep.push_back(idx);
    }
    return keep;
}

// GPU auto-detect (Windows): bundle memakai onnxruntime build DirectML —
// DLL-nya mengekspor fungsi ini; build CPU (dev/CI) tidak mengekspor →
// GetProcAddress NULL → tetap CPU. Env GB_FORCE_CPU=1 memaksa CPU
// (parity/debug). Provider gagal daftar? lepas status → CPU.
// Return true = EP terpasang (belum tentu jalan — bisa gagal saat inferensi;
// lihat fallback di OnnxModel::run).
inline bool maybe_add_gpu(Ort::SessionOptions& opts) {
#ifdef _WIN32
    char force[4] = {};
    if (GetEnvironmentVariableA("GB_FORCE_CPU", force, sizeof(force)) > 0) return false;
    using DmlFn = OrtStatus*(ORT_API_CALL*)(OrtSessionOptions*, int);
    static DmlFn fn = []() -> DmlFn {
        HMODULE h = GetModuleHandleA("onnxruntime.dll");
        return h ? reinterpret_cast<DmlFn>(
                       GetProcAddress(h, "OrtSessionOptionsAppendExecutionProvider_DML"))
                 : nullptr;
    }();
    if (!fn) return false;
    if (OrtStatus* st =
            fn(static_cast<OrtSessionOptions*>(opts), 0)) {
        Ort::GetApi().ReleaseStatus(st);
        return false;
    }
    static bool logged = false;
    if (!logged) {
        std::cout << "[gpu] DirectML execution provider aktif (GPU terdeteksi otomatis)\n";
        logged = true;
    }
    return true;
#else
    (void)opts;
    return false;
#endif
}

// Minimal ORT session wrapper: one input / one output, float32 NCHW.
struct OnnxModel {
    std::optional<Ort::Session> session;
    std::string input_name;
    std::string output_name;
    bool dynamic_hw = false;
    int static_h = 0;
    int static_w = 0;
    std::string path_;  // disimpan utk recreate session saat fallback CPU
    bool gpu_ = false;  // EP DML terpasang di session skrg (gagal runtime → false)

    static Ort::Env& env() {
        static Ort::Env e{ORT_LOGGING_LEVEL_WARNING, "gb_detect"};
        return e;
    }

    // Emplace session dari path_. with_gpu=true pasang DirectML (Windows only).
    bool create(bool with_gpu, std::string& err) {
        try {
            Ort::SessionOptions opts;
            opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            gpu_ = with_gpu && maybe_add_gpu(opts);
#ifdef _WIN32
            // ORTCHAR_T is wchar_t on Windows; model_path arrives as UTF-8.
            int wn = MultiByteToWideChar(CP_UTF8, 0, path_.c_str(),
                                         static_cast<int>(path_.size()), nullptr, 0);
            std::wstring wpath(wn, L'\0');
            MultiByteToWideChar(CP_UTF8, 0, path_.c_str(),
                                static_cast<int>(path_.size()), &wpath[0], wn);
            session.emplace(env(), wpath.c_str(), opts);
#else
            session.emplace(env(), path_.c_str(), opts);
#endif
            return true;
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
    }

    bool load(const std::string& path, std::string& err) {
        path_ = path;
        if (!create(true, err)) return false;
        try {
            if (session->GetInputCount() != 1 || session->GetOutputCount() < 1) {
                err = "expected exactly 1 input";
                session.reset();
                return false;
            }
            Ort::AllocatorWithDefaultOptions alloc;
            auto in = session->GetInputNameAllocated(0, alloc);
            auto out = session->GetOutputNameAllocated(0, alloc);
            input_name = in.get();
            output_name = out.get();
            auto ti = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
            auto dims = ti.GetShape();
            if (dims.size() != 4) {
                err = "input rank != 4";
                session.reset();
                return false;
            }
            if (dims[1] > 0 && dims[1] != 3) {
                err = "input channels != 3";
                session.reset();
                return false;
            }
            if (dims[2] > 0 && dims[3] > 0) {
                dynamic_hw = false;
                static_h = static_cast<int>(dims[2]);
                static_w = static_cast<int>(dims[3]);
            } else {
                dynamic_hw = true;
            }
            return true;
        } catch (const std::exception& e) {
            err = e.what();
            session.reset();
            return false;
        }
    }

    // data: NCHW float32, dims e.g. {1,3,H,W}. Returns raw output + its dims.
    bool run(const float* data, const std::vector<int64_t>& dims,
             std::vector<float>& out, std::vector<int64_t>& out_dims,
             std::string& err) {
        try {
            auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            auto tensor = Ort::Value::CreateTensor<float>(
                mem, const_cast<float*>(data), /*count=*/dims_size(dims),
                dims.data(), dims.size());
            const char* in_names[] = {input_name.c_str()};
            const char* out_names[] = {output_name.c_str()};
            Ort::RunOptions ro;
            auto outs = session->Run(ro, in_names, &tensor, 1, out_names, 1);
            out_dims = outs[0].GetTensorTypeAndShapeInfo().GetShape();
            size_t n = 1;
            for (int64_t d : out_dims) {
                if (d < 0) { err = "dynamic output dim"; return false; }
                n *= static_cast<size_t>(d);
            }
            const float* p = outs[0].GetTensorData<float>();
            out.assign(p, p + n);
            return true;
        } catch (const std::exception& e) {
            // DirectML bisa gagal di runtime (ops/shape tak didukung GPU tertentu)
            // — jangan diam2 jadi nol-detection. Rebuild CPU, ulangi run yang sama.
            if (gpu_) {
                gpu_ = false;
                std::string ce;
                if (create(false, ce)) {
                    std::cout << "[gpu] DirectML gagal saat inferensi → fallback CPU "
                                 "(hasil tetap paritas)\n";
                    return run(data, dims, out, out_dims, err);
                }
                err = ce;
                return false;
            }
            err = e.what();
            return false;
        }
    }

private:
    static size_t dims_size(const std::vector<int64_t>& dims) {
        size_t n = 1;
        for (int64_t d : dims) n *= static_cast<size_t>(d);
        return n;
    }
};

}  // namespace detect_detail
}  // namespace gb
