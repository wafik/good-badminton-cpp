#pragma once
// Inference + detection-side logic. Port sources:
//   detection/yolo_pose.py, detection/shuttlecock.py, visualization/player_pose.py
//   (detect_players / _normalize_people / _is_on_court — detection-adjacent).
// Only ONE pose backend in v1: whichever class system.py actually constructs.
// Production (main.py --pose-family default 'yolo-pose', webui default same)
// constructs YOLOPoseProcessor(conf=0.15, imgsz=960) — RTMPose not ported.
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "gb/types.h"

namespace gb {

class CourtMapper;

struct PersonKeypoints {
    std::array<cv::Point2f, 17> xy{};   // ROI-relative (input frame of process_frame)
    std::array<float, 17> conf{};
};

// Python: YOLOPoseProcessor(model_path, device, conf=0.15, imgsz=960).
// ONNX Runtime session over the exported .onnx; ok() false when load fails.
// (C++ change vs contract draft: dtor + deleted copies for pimpl; no other
//  public signature change.)
class YoloPoseProcessor {
public:
    YoloPoseProcessor(const std::string& model_path, double conf = 0.15, int imgsz = 960);
    ~YoloPoseProcessor();
    YoloPoseProcessor(const YoloPoseProcessor&) = delete;
    YoloPoseProcessor& operator=(const YoloPoseProcessor&) = delete;
    bool ok() const;
    // Python: process_frame(frame) -> (N,17,2) keypoints + conf; [] when none
    std::vector<PersonKeypoints> process_frame(const cv::Mat& frame);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Port of PlayerPoseVisualizer.detect_players + current_pose state.
// feet-midpoint (+10 y) centroid, hip fallback, court-bounds filter, hands
// from keypoints 9/10 keyed by centroid.
class PoseAnalyzer {
public:
    // court_filter_margin: Python player_pose __init__ default is 0.75
    // (system.py does not override it) — contract draft had 1.0; corrected.
    explicit PoseAnalyzer(YoloPoseProcessor& pose, double court_filter_margin = 0.75);
    // roi = court crop (full-frame offset x1,y1); mapper null -> falls back to
    // set_court_mapper() store (Python: active = court_mapper or visualizer
    // .court_mapper); both null -> no filter.
    PoseObs detect(const cv::Mat& roi, int x1, int y1, const CourtMapper* mapper);
    // Python: player_pose_visualizer.court_mapper = ... (system.py sets it
    // before detect_players). Added to mirror Python's stored-mapper fallback.
    void set_court_mapper(const CourtMapper* mapper);
    // Python: current_pose_data. NOTE: keypoints are stored FULL-frame here
    // (offset_x/offset_y kept informational) per types.h/viz.h contract;
    // Python stores ROI-relative + applies offsets while drawing — same pixels.
    std::optional<PoseDrawData> current_pose() const;
private:
    YoloPoseProcessor& pose_;
    double margin_;
    const CourtMapper* stored_mapper_ = nullptr;
    std::optional<PoseDrawData> current_;
};

// Python: ShuttlecockTracker(...) — all ctor params carry Python defaults;
// model_path prepended (ONNX ball model, dynamic imgsz 320..640).
class ShuttlecockTracker {
public:
    struct Params {
        int trajectory_length = 30;
        bool show_trajectory = true;
        bool show_performance_stats = false;
        double max_jump_pixels = 220;
        double prediction_gate_pixels = 260;
        int max_missing_frames = 5;
        double roi_padding_ratio = 0.08;
        double max_box_area_ratio = 0.004;
        double max_aspect_ratio = 4.0;
    };
    // ponytail: no `= {}` default — nested-class NSDMI in a default argument
    // is a hard error on gcc/clang (MSVC-only leniency); sole caller passes
    // Params explicitly (system.cpp).
    ShuttlecockTracker(const std::string& model_path, Params params);
    ~ShuttlecockTracker();
    ShuttlecockTracker(const ShuttlecockTracker&) = delete;
    ShuttlecockTracker& operator=(const ShuttlecockTracker&) = delete;
    bool ok() const;

    // Python: detect_ball(frame, conf=0.18, roi_corners=None) -> [x,y] or [0,0]
    // (defaults added to match Python; contract values unchanged when passed).
    std::array<double, 2> detect_ball(cv::Mat& frame, double conf = 0.18,
                                      const std::optional<RoiCorners>& roi_corners = std::nullopt);
    // Python: update_trajectory(ball_position, roi_corners=None) -> accepted or [0,0]
    std::array<double, 2> update_trajectory(const std::array<double, 2>& ball,
                                            const std::optional<RoiCorners>& roi_corners = std::nullopt);
    void draw_trajectory(cv::Mat& frame);
    void handle_visualization(cv::Mat& frame);  // if show_trajectory && !empty
    void clear_trajectory();
    const std::vector<cv::Point2i>& trajectory() const;
    struct Detection {
        bool visible = false;
        bool accepted = false;
        std::optional<cv::Point2f> image;
        std::optional<float> confidence;
        int candidate_count = 0;
    };
    Detection last_detection() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gb
