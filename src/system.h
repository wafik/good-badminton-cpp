#pragma once
// BadmintonAnalysisSystem — port of badminton_analysis/system.py + the CLI
// surface of main.py. Owned by the system/IO port; see
// docs/specs/2026-10-01-cpp-rewrite-design.md for pipeline semantics.
#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "gb/court.h"
#include "gb/detect.h"
#include "gb/io.h"
#include "gb/rally_state.h"
#include "gb/track.h"
#include "gb/types.h"
#include "gb/viz.h"

namespace gb {

// Coded pipeline errors -> process exit codes (spec "Error handling"):
// 2 model load failure, 3 ffmpeg total failure, 4 court annotation failure.
// Anything else (missing input/template, unreadable video) falls to exit 1.
struct CodedError : std::runtime_error {
    int code;
    CodedError(int exit_code, const std::string& what)
        : std::runtime_error(what), code(exit_code) {}
};

struct SystemOptions {
    std::string video_path = "videos/demo.mp4";       // main.py --video-path default
    std::string output_dir;                           // empty -> outputs/<video_name>
    std::string annotations_path;                     // empty -> <output_dir>/court_annotations.txt
    std::string template_path = "templates/demo.png"; // main.py --template-path default
    std::string ball_model_path;                      // main.cpp resolves the .onnx default
    std::string pose_model_path;
    bool keep_audio = true;          // main.py --audio default true
    bool show_display = false;       // main.py default true; spec: OFF by default (headless)
    std::string language = "zh";     // main.py --language default (viz v1 is EN-only)
    bool show_performance_stats = true;  // main.py --performance-stats default true
    bool progress_json = false;      // studio: emit {"frame":N,"total":M}/line to stdout

    // ponytail: main.py's remaining toggles (--skeletons, --player-trajectories,
    // --court-trajectory, --shuttlecock-trajectory, --player-stats, --pose-roi,
    // --save-images, --visualize-positions) are pinned to their Python defaults
    // (all true except save-images / visualize-positions) to keep the CLI to the
    // spec's minimal surface; heatmaps stay Python-side per spec.
};

class BadmintonAnalysisSystem {
public:
    explicit BadmintonAnalysisSystem(SystemOptions options);
    ~BadmintonAnalysisSystem();

    BadmintonAnalysisSystem(const BadmintonAnalysisSystem&) = delete;
    BadmintonAnalysisSystem& operator=(const BadmintonAnalysisSystem&) = delete;

    // Python: process_video(progress_callback=None)
    void process_video(std::function<void(int, int)> progress_callback = nullptr);

    // Python: is_court_view(frame, template_gray, threshold=0.75) —
    // grayscale matchTemplate TM_CCOEFF_NORMED, max score >= threshold.
    static bool is_court_view(const cv::Mat& gray_frame, const cv::Mat& template_gray,
                              double threshold = 0.75);

    // Python: draw_court_roi (commented out of the main loop; kept for parity).
    cv::Mat draw_court_roi(const cv::Mat& frame, const std::vector<cv::Point2f>& corners,
                           const RoiCorners& roi_corners) const;

    // Python: analyze_shuttlecock — disabled until migrated to detections.jsonl.
    void analyze_shuttlecock(const RoiCorners& roi_corners,
                             const std::vector<cv::Point2f>& corners) const;

private:
    // court_annotations.txt triple; mid_is_int preserves "680" vs "680.0" on
    // rewrite; corners_float is flat, 2 flags per point (token had '.'/exp).
    struct CourtAnnotation {
        std::vector<cv::Point2f> corners;
        std::vector<bool> corners_float;
        RoiCorners roi{};
        double mid_height = 0.0;
        bool mid_is_int = true;
    };

    std::string get_template_path() const;
    std::pair<cv::Mat, cv::Mat> load_template(const std::string& path) const;
    cv::VideoWriter setup_video_writer();
    CourtAnnotation setup_court_annotation(const cv::Mat& template_color);
    void write_metadata(double fps, int total_frames, double video_duration,
                        const std::string& template_path, const CourtAnnotation& ann) const;
    void process_frame(cv::Mat& frame, const cv::Mat& template_gray,
                       const CourtAnnotation& ann, int frame_count, int& detect_frame_count);
    void cleanup(cv::VideoCapture& cap);  // mutates writer/writer-closing state

    SystemOptions opts_;

    std::string video_path_;
    std::string video_name_;
    std::string save_dir_;
    std::string images_save_dir_;
    std::string metadata_path_;
    std::string detections_path_;
    std::string output_video_path_;
    std::string temp_output_video_path_;
    std::string annotations_path_;

    bool show_display_ = false;
    bool show_pose_roi_ = true;             // main.py --pose-roi default true
    bool show_court_trajectory_ = true;     // default true
    bool show_player_stats_ = true;         // default true
    bool keep_audio_ = true;
    bool save_images_ = false;              // no CLI flag in v1 (see SystemOptions)
    bool show_performance_stats_ = true;
    std::string language_;

    std::chrono::steady_clock::time_point start_time_{};
    std::chrono::steady_clock::time_point end_time_{};
    double fps_ = 30.0;
    int frame_width_ = 0;
    int frame_height_ = 0;
    int consecutive_non_court_frames_ = 0;
    static constexpr int kNonCourtFramesThreshold = 5;  // system.py non_court_frames_threshold
    RallyState rally_state_;

    int last_stats_update_frame_ = 0;
    int stats_update_interval_frames_ = 0;
    MovementStats cached_movement_stats_;
    int performance_log_interval_frames_ = 150;

    // Component declaration order = reverse destruction order; the writer is
    // declared first so the tracker (which may hold its pointer) dies first.
    // Model components are unique_ptrs: they are constructed in the ctor body
    // *after* the input/model file checks so failures map to exit code 2.
    std::unique_ptr<JsonlDetectionWriter> detection_writer_;
    std::unique_ptr<PlayerTracker> player_tracker_;
    std::unique_ptr<CourtMapper> court_mapper_;
    std::unique_ptr<StatsVisualizer> stats_visualizer_;
    std::unique_ptr<YoloPoseProcessor> pose_processor_;  // before pose_analyzer_
    std::unique_ptr<PoseAnalyzer> pose_analyzer_;        // references pose_processor_
    std::unique_ptr<ShuttlecockTracker> shuttlecock_tracker_;
    CourtTrajectoryVisualizer court_trajectory_visualizer_;
    PlayerSkeletonRenderer skeleton_renderer_;
    cv::VideoWriter video_writer_;

    std::vector<cv::Point2f> court_corners_;
    RoiCorners court_roi_corners_{};
};

}  // namespace gb
