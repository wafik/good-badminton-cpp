// Port of detection/shuttlecock.py (ShuttlecockTracker) to ONNX Runtime.
//
// ONNX layout verified against weights/yolo11s-ball.onnx (project venv):
//   input  ['batch',3,'height','width'] (dynamic — production feeds stride-32
//          minimum-rectangle letterbox, e.g. 1280x720@imgsz640 -> 1x3x384x640;
//          confirmed by spying the ORT inputs ultralytics itself feeds)
//   output0 ['batch',5,'anchors'] = [cx,cy,w,h, conf] (nc=1), raw — no NMS.
// Decode parity with ultralytics predict(conf=0.18, imgsz=640) on demo.mp4
// frame 100: box (487.89,309.09,17.85,18.18) conf 0.819 — exact.
// ponytail: no ultra_device / torch CUDA probe (Python ctor) — CPU EP only.
// When the ONNX session fails to load, detect_ball degrades to "no detection"
// ([0,0] + empty state) instead of raising like Python would.
#include "gb/detect.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>

#include "onnx_util.h"

namespace gb {
namespace {

constexpr int64_t kBallChannels = 5;  // 4 box + nc(1)
constexpr float kNmsIou = 0.7f;       // ultralytics predict default args.iou
constexpr int kMaxDet = 300;          // ultralytics default max_det

struct Candidate {
    cv::Point2i point;
    float confidence;
    float area_ratio;
    float aspect_ratio;
};

double point_distance(const cv::Point2i& a, const cv::Point2i& b) {
    return std::hypot(static_cast<double>(a.x - b.x),
                      static_cast<double>(a.y - b.y));
}

}  // namespace

struct ShuttlecockTracker::Impl {
    Params p;
    detect_detail::OnnxModel model;
    bool ok = false;
    bool warned = false;

    // Python state: deque(maxlen=trajectory_length), last_valid_position,
    // last_candidate, last_detection dict, missing_frames.
    std::vector<cv::Point2i> traj;
    std::optional<cv::Point2i> last_valid_position;  // write-only in Python too
    std::optional<Candidate> last_candidate;         // write-only in Python too
    Detection last_detection;
    int missing_frames = 0;

    bool point_in_roi(const cv::Point2i& pt,
                      const std::optional<RoiCorners>& roi) const {
        if (!roi.has_value()) return true;
        const int x1 = (*roi)[0].x, y1 = (*roi)[0].y;
        const int x2 = (*roi)[1].x, y2 = (*roi)[1].y;
        const int padding =
            static_cast<int>(std::max(x2 - x1, y2 - y1) * p.roi_padding_ratio);
        return (x1 - padding) <= pt.x && pt.x <= (x2 + padding) &&
               (y1 - padding) <= pt.y && pt.y <= (y2 + padding);
    }

    // Python _predict_next_position: constant-velocity extrapolation.
    cv::Point2i predict_next() const {
        if (traj.size() < 2) return traj.back();
        const cv::Point2i& last = traj[traj.size() - 1];
        const cv::Point2i& prev = traj[traj.size() - 2];
        return {last.x + (last.x - prev.x), last.y + (last.y - prev.y)};
    }

    bool is_outlier(const cv::Point2i& pt) const {
        if (traj.empty()) return false;
        const bool strict_gate = missing_frames <= p.max_missing_frames;
        if (point_distance(pt, traj.back()) > p.max_jump_pixels && strict_gate) {
            return true;
        }
        if (point_distance(pt, predict_next()) > p.prediction_gate_pixels &&
            strict_gate) {
            return true;
        }
        return false;
    }

    void append_valid(const cv::Point2i& pt) {
        traj.push_back(pt);
        // Python deque(maxlen=...) drop-oldest.
        if (static_cast<int>(traj.size()) > p.trajectory_length) {
            traj.erase(traj.begin());
        }
        last_valid_position = pt;
        missing_frames = 0;
    }

    void record_missing() {
        ++missing_frames;
        if (missing_frames > p.max_missing_frames) {
            last_valid_position.reset();  // Python: last_valid_position = None
        }
    }

    void mark_rejected() {
        last_detection.accepted = false;
        last_detection.image.reset();
    }

    static Detection empty_detection_state() { return Detection{}; }
};

ShuttlecockTracker::ShuttlecockTracker(const std::string& model_path,
                                       Params params)
    : impl_(std::make_unique<Impl>()) {
    impl_->p = params;
    impl_->last_detection = Impl::empty_detection_state();
    std::string err;
    if (!impl_->model.load(model_path, err)) {
        std::cerr << "shuttlecock model failed to load: " << err << std::endl;
        return;
    }
    impl_->ok = true;
}

ShuttlecockTracker::~ShuttlecockTracker() = default;

bool ShuttlecockTracker::ok() const { return impl_->ok; }

std::array<double, 2> ShuttlecockTracker::detect_ball(
    cv::Mat& frame, double conf,
    const std::optional<RoiCorners>& roi_corners) {
    Impl& im = *impl_;
    auto t0 = std::chrono::steady_clock::now();

    // --- Python detect_ball head: padded court crop + adaptive imgsz ---
    // ponytail: detect only on the padded court crop instead of the full
    // frame. imgsz scales with the crop so pixels-per-shuttlecock matches
    // full-frame@640 (area/aspect/ROI filters below stay valid); gain =
    // (crop_max_dim/frame_max_dim)^2.
    cv::Mat img = frame;
    int ox = 0, oy = 0, imgsz = 640;
    if (roi_corners.has_value() && !frame.empty()) {
        const int fh = frame.rows, fw = frame.cols;
        const int rx1 = (*roi_corners)[0].x, ry1 = (*roi_corners)[0].y;
        const int rx2 = (*roi_corners)[1].x, ry2 = (*roi_corners)[1].y;
        const int pad = static_cast<int>(
            std::max(rx2 - rx1, ry2 - ry1) * im.p.roi_padding_ratio);
        const int cx1 = std::max(0, rx1 - pad), cy1 = std::max(0, ry1 - pad);
        const int cx2 = std::min(fw, rx2 + pad), cy2 = std::min(fh, ry2 + pad);
        if (cx2 - cx1 >= 64 && cy2 - cy1 >= 64 &&
            (cx2 - cx1 < fw || cy2 - cy1 < fh)) {
            img = frame(cv::Rect(cx1, cy1, cx2 - cx1, cy2 - cy1));
            ox = cx1;
            oy = cy1;
            const double ratio = std::max(cx2 - cx1, cy2 - cy1) /
                                 static_cast<double>(std::max(fw, fh));
            // Python: max(320, min(640, int(640*ratio) // 32 * 32 or 320))
            int v = static_cast<int>(640.0 * ratio) / 32 * 32;
            if (v == 0) v = 320;
            imgsz = std::max(320, std::min(640, v));
        }
    }

    std::vector<Candidate> candidates;
    bool ran = false;

    if (im.ok && !img.empty()) {
        // Dynamic model input -> Ultralytics LetterBox auto (stride-32 min pad).
        const detect_detail::LetterboxResult lb = detect_detail::letterbox(
            img, imgsz, imgsz, /*auto_pad=*/im.model.dynamic_hw);

        cv::Mat blob;
        cv::dnn::blobFromImage(lb.img, blob, 1.0 / 255.0, lb.img.size(),
                               cv::Scalar(), /*swapRB=*/true, /*crop=*/false);

        std::vector<float> raw;
        std::vector<int64_t> odims;
        std::string err;
        ran = im.model.run(reinterpret_cast<const float*>(blob.data),
                           {1, 3, lb.fed_h, lb.fed_w}, raw, odims, err);

        if (im.p.show_performance_stats) {
            const double sec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              t0)
                    .count();
            std::printf("YOLO shuttlecock inference took %.2f sec\n", sec);
        }
        if (!ran) {
            if (!im.warned) {
                std::cerr << "shuttlecock inference failed: " << err
                          << std::endl;
                im.warned = true;
            }
        } else if (odims.size() == 3 && odims[0] == 1 &&
                   odims[1] == kBallChannels && odims[2] > 0) {
            // --- decode: conf filter -> NMS -> scale to crop space ---
            const size_t A = static_cast<size_t>(odims[2]);
            struct Det {
                int64_t a;
                float conf;
                float cx, cy, w, h;
            };
            std::vector<Det> dets;
            dets.reserve(32);
            for (int64_t a = 0; a < static_cast<int64_t>(A); ++a) {
                const float c = raw[4 * A + a];
                if (c > conf) {
                    dets.push_back({a, c, raw[0 * A + a], raw[1 * A + a],
                                    raw[2 * A + a], raw[3 * A + a]});
                }
            }
            if (!dets.empty()) {
                std::stable_sort(
                    dets.begin(), dets.end(),
                    [](const Det& x, const Det& y) { return x.conf > y.conf; });
                std::vector<cv::Rect2f> boxes;
                std::vector<float> scores;
                boxes.reserve(dets.size());
                scores.reserve(dets.size());
                for (const Det& d : dets) {
                    boxes.emplace_back(d.cx - d.w / 2, d.cy - d.h / 2, d.w,
                                       d.h);
                    scores.push_back(d.conf);
                }
                const std::vector<int> keep =
                    detect_detail::nms_xywh(boxes, scores, kNmsIou, kMaxDet);

                const detect_detail::ScaleBack sb = detect_detail::scale_back(
                    lb.fed_h, lb.fed_w, img.rows, img.cols);
                // Python: frame_shape is the FULL frame (area denominator
                // stays full-frame even though boxes are crop-space) — the
                // adaptive imgsz above keeps this ratio comparable.
                const int fh = frame.rows, fw = frame.cols;
                const double frame_area =
                    std::max(1.0, static_cast<double>(fh) * fw);

                candidates.reserve(keep.size());
                for (int idx : keep) {
                    const Det& d = dets[idx];
                    // Python uses boxes.xywh of the ultralytics result =
                    // scaled+clipped xyxy re-derived as center/size.
                    const cv::Rect2f s = detect_detail::scale_xyxy(
                        cv::Rect2f(d.cx - d.w / 2, d.cy - d.h / 2, d.w, d.h),
                        sb);
                    const float w = s.width, h = s.height;
                    if (w <= 0 || h <= 0) continue;  // Python width/height guard

                    const cv::Point2i point{
                        static_cast<int>(s.x + w / 2) + ox,
                        static_cast<int>(s.y + h / 2) + oy};
                    const float area_ratio =
                        static_cast<float>((w * h) / frame_area);
                    const float aspect_ratio = std::max(w / h, h / w);
                    if (area_ratio > im.p.max_box_area_ratio ||
                        aspect_ratio > im.p.max_aspect_ratio) {
                        continue;
                    }
                    if (!im.point_in_roi(point, roi_corners)) continue;

                    candidates.push_back({point, d.conf, area_ratio,
                                          aspect_ratio});
                }
            }
        }
    }

    // --- Python _select_candidate ---
    std::optional<Candidate> selected;
    if (!candidates.empty()) {
        if (im.traj.empty()) {
            selected = candidates[0];  // max() keeps first on ties
            for (size_t i = 1; i < candidates.size(); ++i) {
                if (candidates[i].confidence > selected->confidence) {
                    selected = candidates[i];
                }
            }
        } else {
            const cv::Point2i predicted = im.predict_next();
            selected = candidates[0];
            double best = -std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < candidates.size(); ++i) {
                const Candidate& c = candidates[i];
                // Python score: conf*1000 - dist*1.4 - area_ratio*4000
                const double score =
                    c.confidence * 1000.0 -
                    point_distance(c.point, predicted) * 1.4 -
                    c.area_ratio * 4000.0;
                if (i == 0 || score > best) {
                    best = score;
                    selected = c;
                }
            }
        }
    }

    im.last_candidate = selected;

    // Python rebuilds the detection dict on every detect_ball call:
    // visible / accepted=False / image / confidence / candidate_count.
    Detection st;
    st.visible = selected.has_value();
    st.accepted = false;
    st.candidate_count = static_cast<int>(candidates.size());
    if (selected.has_value()) {
        st.image = cv::Point2f(static_cast<float>(selected->point.x),
                               static_cast<float>(selected->point.y));
        st.confidence = selected->confidence;
    }
    im.last_detection = st;

    if (selected.has_value()) {
        return {static_cast<double>(selected->point.x),
                static_cast<double>(selected->point.y)};
    }
    return {0.0, 0.0};
}

std::array<double, 2> ShuttlecockTracker::update_trajectory(
    const std::array<double, 2>& ball,
    const std::optional<RoiCorners>& roi_corners) {
    Impl& im = *impl_;

    // Python: if ball_position == [0,0] or None -> missing + rejected.
    if (ball[0] == 0.0 && ball[1] == 0.0) {
        im.record_missing();
        im.mark_rejected();
        return {0.0, 0.0};
    }
    // Python: point = tuple(ball_position) — values are integral by
    // construction (detect_ball returns int-truncated pixel coords), so the
    // Point2i cast is lossless on the production path.
    // ponytail: if a caller ever hands update_trajectory non-integral
    // coordinates, the stored trajectory truncates (Python would keep the
    // float); only detect_ball-fed points are integral by construction.
    const cv::Point2i point{static_cast<int>(ball[0]), static_cast<int>(ball[1])};

    if (!im.point_in_roi(point, roi_corners)) {
        im.record_missing();
        im.mark_rejected();
        return {0.0, 0.0};
    }
    if (im.is_outlier(point)) {
        im.record_missing();
        im.mark_rejected();
        return {0.0, 0.0};
    }

    im.append_valid(point);
    im.last_detection.accepted = true;
    im.last_detection.image =
        cv::Point2f(static_cast<float>(point.x), static_cast<float>(point.y));
    return {ball[0], ball[1]};  // Python returns list(point) == input values
}

void ShuttlecockTracker::draw_trajectory(cv::Mat& frame) {
    Impl& im = *impl_;
    if (im.traj.empty()) return;

    auto t0 = std::chrono::steady_clock::now();
    const cv::Scalar color(87, 108, 255);
    const size_t n = im.traj.size();
    for (size_t i = 0; i < n; ++i) {
        const int radius = static_cast<int>(3 + (static_cast<double>(i) / n) * 4);
        cv::circle(frame, im.traj[i], radius, color, cv::FILLED, cv::LINE_AA);
    }
    cv::circle(frame, im.traj.back(), 6, cv::Scalar(0, 165, 255), cv::FILLED,
               cv::LINE_AA);

    if (im.p.show_performance_stats) {
        const double sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          t0)
                .count();
        std::printf("Drawing shuttlecock trajectory took %.2f sec\n", sec);
    }
}

void ShuttlecockTracker::handle_visualization(cv::Mat& frame) {
    Impl& im = *impl_;
    if (im.p.show_trajectory && !im.traj.empty()) {
        draw_trajectory(frame);
    }
}

void ShuttlecockTracker::clear_trajectory() {
    Impl& im = *impl_;
    im.traj.clear();
    im.last_valid_position.reset();
    im.last_candidate.reset();
    im.last_detection = Impl::empty_detection_state();
    im.missing_frames = 0;
}

const std::vector<cv::Point2i>& ShuttlecockTracker::trajectory() const {
    return impl_->traj;
}

ShuttlecockTracker::Detection ShuttlecockTracker::last_detection() const {
    return impl_->last_detection;
}

}  // namespace gb
