// Port of badminton_analysis/system.py (BadmintonAnalysisSystem) — the CLI
// pipeline. Parity-critical behaviours are marked "Python:" at call sites;
// "ponytail:" comments mark deliberate shortcuts vs the Python source.
#include "system.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

#include "nlohmann_json.hpp"

namespace gb {
namespace fs = std::filesystem;
using ordered_json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

namespace {

constexpr const char* kSchemaVersion = "1.0";  // data/writer.py SCHEMA_VERSION

constexpr const char* kCourtIncompleteMsg =
    "Court annotation is incomplete: click 4 court corners in order. "
    "ROI is generated automatically.";

double elapsed_sec(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

bool is_integral(double v) { return std::isfinite(v) && v == std::trunc(v); }

// --- court_annotations.txt: mirror Python eval()/str() number formatting ---

std::string fmt_num(double v, bool was_float) {
    if (!was_float && is_integral(v)) {
        return std::to_string(static_cast<long long>(v));  // Python int
    }
    return ordered_json(v).dump();  // repr-like: 426.5 / 385.0
}

// Python literal: [(x, y), (x, y), ...] — flags is flat, 2 per point (token
// carried '.'/exponent -> keep float spelling on rewrite).
std::string fmt_points(const std::vector<cv::Point2f>& pts,
                       const std::vector<bool>& flags) {
    std::string s = "[";
    for (size_t i = 0; i < pts.size(); ++i) {
        if (i) s += ", ";
        const bool fx = flags.size() == 2 * pts.size() ? flags[2 * i] : false;
        const bool fy = flags.size() == 2 * pts.size() ? flags[2 * i + 1] : false;
        s += "(" + fmt_num(pts[i].x, fx) + ", " + fmt_num(pts[i].y, fy) + ")";
    }
    s += "]";
    return s;
}

// roi_corners line: compute_expanded_roi output is always int pairs.
std::string fmt_roi(const RoiCorners& roi) {
    return "[(" + std::to_string(roi[0].x) + ", " + std::to_string(roi[0].y) + "), (" +
           std::to_string(roi[1].x) + ", " + std::to_string(roi[1].y) + ")]";
}

// Pull every number token out of a Python literal such as
// "[(426, 385), (861, 382)]" or "[[426, 385], ...]" (eval() accepts both).
// Second of each pair = token had '.'/exponent (float in Python).
std::vector<std::pair<double, bool>> extract_numbers(const std::string& s) {
    std::vector<std::pair<double, bool>> out;
    size_t i = 0;
    while (i < s.size()) {
        const char c = s[i];
        if (!(std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+' || c == '.')) {
            ++i;
            continue;
        }
        const size_t start = i;
        if (s[i] == '-' || s[i] == '+') ++i;
        bool digits = false;
        bool dot = false;
        bool exp = false;
        while (i < s.size()) {
            const char d = s[i];
            if (std::isdigit(static_cast<unsigned char>(d))) {
                digits = true;
                ++i;
            } else if (d == '.' && !dot && !exp) {
                dot = true;
                ++i;
            } else if ((d == 'e' || d == 'E') && digits && !exp && i + 1 < s.size() &&
                       (std::isdigit(static_cast<unsigned char>(s[i + 1])) || s[i + 1] == '-' ||
                        s[i + 1] == '+')) {
                exp = true;
                ++i;
                if (s[i] == '-' || s[i] == '+') ++i;
            } else {
                break;
            }
        }
        if (!digits) {  // lone '.', '+', "e..." fragment — not a number
            i = start + 1;
            continue;
        }
        const std::string tok = s.substr(start, i - start);
        out.emplace_back(std::strtod(tok.c_str(), nullptr), dot || exp);
    }
    return out;
}

ordered_json point_meta(const cv::Point2f& p, bool was_float) {
    if (!was_float && is_integral(p.x) && is_integral(p.y)) {
        return ordered_json::array({static_cast<long long>(p.x), static_cast<long long>(p.y)});
    }
    return ordered_json::array({static_cast<double>(p.x), static_cast<double>(p.y)});
}

}  // namespace

// ---------------------------------------------------------------------------
// construction / teardown
// ---------------------------------------------------------------------------

BadmintonAnalysisSystem::BadmintonAnalysisSystem(SystemOptions options)
    : opts_(std::move(options)),
      video_path_(opts_.video_path),
      show_display_(opts_.show_display),
      show_pose_roi_(opts_.show_pose_roi),
      show_court_trajectory_(opts_.show_court_trajectory),
      show_player_stats_(opts_.show_player_stats),
      keep_audio_(opts_.keep_audio),
      show_performance_stats_(opts_.show_performance_stats),
      language_(opts_.language) {
    // Python __init__ order: input checks -> model loads -> paths/mkdirs.
    std::error_code ec;
    if (!fs::exists(video_path_, ec)) {
        throw std::runtime_error("Input video not found: " + video_path_ +
                                 "\nPass a valid video file as the input video argument.");
    }
    if (opts_.ball_model_path.empty() || !fs::exists(opts_.ball_model_path, ec)) {
        throw CodedError(2, "Ball detection model not found: " + opts_.ball_model_path +
                                "\nPlace yolo11s-ball.onnx in Good-Badminton/weights or pass "
                                "its path with --ball-model.");
    }
    if (opts_.pose_model_path.empty() || !fs::exists(opts_.pose_model_path, ec)) {
        throw CodedError(2, "Pose model not found: " + opts_.pose_model_path +
                                "\nPlace yolo11n-pose-dyn.onnx in Good-Badminton/weights or pass "
                                "its path with --yolo-pose-model.");
    }

    pose_processor_ = std::make_unique<YoloPoseProcessor>(opts_.pose_model_path);
    if (!pose_processor_->ok()) {
        throw CodedError(2, "Failed to load pose model: " + opts_.pose_model_path);
    }
    // Python court_filter_margin default is 0.75 (player_pose.py); the contract
    // default of 1.0 differs — pass 0.75 explicitly for parity.
    pose_analyzer_ = std::make_unique<PoseAnalyzer>(*pose_processor_, 0.75);
    skeleton_renderer_.show_skeletons = opts_.show_skeletons;
    skeleton_renderer_.show_player_trajectories = opts_.show_player_trajectories;

    ShuttlecockTracker::Params sp;
    sp.trajectory_length = 30;            // Python trajectory_length=30
    sp.show_trajectory = opts_.show_shuttlecock_trajectory;
    sp.show_performance_stats = false;
    shuttlecock_tracker_ =
        std::make_unique<ShuttlecockTracker>(opts_.ball_model_path, sp);
    if (!shuttlecock_tracker_->ok()) {
        throw CodedError(2, "Failed to load ball model: " + opts_.ball_model_path);
    }

    // Python: video_name = basename(path)[:-4]; save_dir = output_dir or outputs/<name>
    const std::string fname = fs::path(video_path_).filename().string();
    video_name_ = fname.size() >= 4 ? fname.substr(0, fname.size() - 4) : std::string();
    save_dir_ = opts_.output_dir.empty() ? (fs::path("outputs") / video_name_).string()
                                         : opts_.output_dir;
    fs::create_directories(save_dir_, ec);
    images_save_dir_ = (fs::path(save_dir_) / "detect_images").string();
    fs::create_directories(images_save_dir_, ec);  // Python always creates it
    metadata_path_ = (fs::path(save_dir_) / "metadata.json").string();
    detections_path_ = (fs::path(save_dir_) / "detections.jsonl").string();
    output_video_path_ = (fs::path(save_dir_) / ("detect_" + video_name_ + ".mp4")).string();
    annotations_path_ = opts_.annotations_path.empty()
                             ? (fs::path(save_dir_) / "court_annotations.txt").string()
                             : opts_.annotations_path;
}

BadmintonAnalysisSystem::~BadmintonAnalysisSystem() = default;

// ---------------------------------------------------------------------------
// pipeline
// ---------------------------------------------------------------------------

std::string BadmintonAnalysisSystem::get_template_path() const {
    if (opts_.template_path.empty()) {
        // Python opens a tkinter file picker here; headless per spec.
        throw std::runtime_error(
            "No court template image selected. Pass --template to run headless.");
    }
    std::error_code ec;
    if (!fs::exists(opts_.template_path, ec)) {
        throw std::runtime_error("Court template image not found: " + opts_.template_path);
    }
    return opts_.template_path;
}

std::pair<cv::Mat, cv::Mat> BadmintonAnalysisSystem::load_template(
    const std::string& path) const {
    // Python _load_template: imread gray + color, resize both to frame size.
    cv::Mat gray = cv::imread(path, cv::IMREAD_GRAYSCALE);
    cv::Mat color = cv::imread(path, cv::IMREAD_COLOR);
    if (gray.empty() || color.empty()) {
        throw std::runtime_error("Unable to read court template image: " + path);
    }
    cv::resize(gray, gray, cv::Size(frame_width_, frame_height_));
    cv::resize(color, color, cv::Size(frame_width_, frame_height_));
    return {gray, color};
}

cv::VideoWriter BadmintonAnalysisSystem::setup_video_writer() {
    // Python _setup_video_writer: temp mp4v file, transcoded in cleanup().
    temp_output_video_path_ =
        (fs::path(save_dir_) / ("temp_detect_" + video_name_ + ".mp4")).string();
    video_writer_ =
        gb::setup_video_writer(frame_width_, frame_height_, fps_, temp_output_video_path_);
    return cv::VideoWriter{};  // keep the Python-ish return shape; member holds it
}

BadmintonAnalysisSystem::CourtAnnotation BadmintonAnalysisSystem::setup_court_annotation(
    const cv::Mat& template_color) {
    CourtAnnotation ann;
    const auto court_error = [&]() {
        return CodedError(4, std::string(kCourtIncompleteMsg) +
                                 "\nHeadless: edit " + annotations_path_ +
                                 " manually (corners=[(x, y), ...], "
                                 "roi_corners=[(x, y), (x, y)], mid_height=<int>) or improve "
                                 "the template so auto-detection succeeds.");
    };

    if (fs::exists(annotations_path_)) {
        // Python _setup_court_annotation read branch:
        //   corners = eval(line1.split('=')[1]); f.readline()  # roi recomputed
        //   mid_height = eval(line3.split('=')[1])
        std::ifstream in(annotations_path_);
        std::string line1, line2, line3;
        if (!std::getline(in, line1) || !std::getline(in, line2) || !std::getline(in, line3)) {
            throw CodedError(4, "Court annotation file is malformed (expected 3 lines): " +
                                    annotations_path_);
        }
        const auto after_eq = [](const std::string& s) -> std::string {
            const auto pos = s.find('=');
            return pos == std::string::npos ? std::string{} : s.substr(pos + 1);
        };
        const std::string corners_lit = after_eq(line1);
        const std::string mid_lit = after_eq(line3);
        if (corners_lit.empty() || mid_lit.empty()) {
            throw CodedError(4, "Court annotation file is malformed (missing '='): " +
                                    annotations_path_);
        }
        const auto corner_nums = extract_numbers(corners_lit);
        if (corner_nums.empty() || corner_nums.size() % 2 != 0) {
            throw court_error();
        }
        for (size_t i = 0; i + 1 < corner_nums.size(); i += 2) {
            ann.corners.emplace_back(static_cast<float>(corner_nums[i].first),
                                     static_cast<float>(corner_nums[i + 1].first));
            ann.corners_float.push_back(corner_nums[i].second);
            ann.corners_float.push_back(corner_nums[i + 1].second);
        }
        const auto mid_nums = extract_numbers(mid_lit);
        if (mid_nums.size() != 1) {
            throw CodedError(4, "Court annotation file is malformed (mid_height): " +
                                    annotations_path_);
        }
        ann.mid_height = mid_nums[0].first;
        ann.mid_is_int = !mid_nums[0].second;
        // Python always recomputes roi from corners (never trusts the file line).
        ann.roi = compute_expanded_roi(ann.corners, template_color.size());
    } else {
        // Python auto branch: corners, roi, mid = annotate_court(...) — headless
        // here = resolve_court_corners' 3-tuple (contract mirrors Python), plus
        // the preview PNG annotate_court would write (no interactive accept).
        const CourtResolution res = resolve_court_corners(template_color);
        if (!res.corners || !res.roi || !res.mid_height || res.corners->size() != 4) {
            throw court_error();
        }
        ann.corners = *res.corners;
        ann.corners_float.assign(ann.corners.size() * 2, false);  // auto corners are ints
        ann.roi = *res.roi;
        ann.mid_height = *res.mid_height;
        ann.mid_is_int = true;
        const std::string preview_path = (fs::path(save_dir_) / "auto_court_preview.png").string();
        const cv::Mat preview =
            render_auto_court_preview(template_color, ann.corners, ann.roi);
        if (!cv::imwrite(preview_path, preview)) {
            throw std::runtime_error("Unable to write court preview: " + preview_path);
        }
        std::cout << "Auto court preview saved: " << preview_path << std::endl;
    }

    if (ann.corners.size() != 4) {
        throw court_error();
    }

    // Python always rewrites court_annotations.txt after either branch.
    std::ofstream out(annotations_path_, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Unable to write court annotations: " + annotations_path_);
    }
    out << "corners=" << fmt_points(ann.corners, ann.corners_float) << "\n";
    out << "roi_corners=" << fmt_roi(ann.roi) << "\n";
    out << "mid_height=" << fmt_num(ann.mid_height, !ann.mid_is_int) << "\n";
    return ann;
}

void BadmintonAnalysisSystem::write_metadata(double fps, int total_frames,
                                             double video_duration,
                                             const std::string& template_path,
                                             const CourtAnnotation& ann) const {
    // Python _write_metadata — same keys, same insertion order (ordered_json).
    ordered_json meta;
    meta["schema_version"] = kSchemaVersion;

    ordered_json video;
    video["path"] = video_path_;
    video["name"] = video_name_;
    video["fps"] = fps;
    video["total_frames"] = total_frames;
    video["duration_sec"] = video_duration;
    video["width"] = frame_width_;
    video["height"] = frame_height_;
    meta["video"] = video;

    ordered_json models;
    models["shuttlecock"] = opts_.ball_model_path;
    meta["models"] = models;

    ordered_json corners = ordered_json::array();
    for (size_t i = 0; i < ann.corners.size(); ++i) {
        const bool fx = ann.corners_float.size() == 2 * ann.corners.size()
                            ? ann.corners_float[2 * i]
                            : false;
        const bool fy = ann.corners_float.size() == 2 * ann.corners.size()
                            ? ann.corners_float[2 * i + 1]
                            : false;
        corners.push_back(point_meta(ann.corners[i], fx || fy));
    }
    ordered_json roi = ordered_json::array({ordered_json::array(
                                                {static_cast<long long>(ann.roi[0].x),
                                                 static_cast<long long>(ann.roi[0].y)}),
                                            ordered_json::array(
                                                {static_cast<long long>(ann.roi[1].x),
                                                 static_cast<long long>(ann.roi[1].y)})});
    ordered_json coord;
    coord["unit"] = "meter";
    coord["width"] = 6.1;
    coord["length"] = 13.4;

    ordered_json court;
    court["template_path"] = template_path;
    court["corners"] = corners;
    court["roi_corners"] = roi;
    court["mid_height"] = ann.mid_is_int
                              ? ordered_json(static_cast<long long>(ann.mid_height))
                              : ordered_json(ann.mid_height);
    court["coordinate_system"] = coord;
    meta["court"] = court;

    ordered_json outputs;
    outputs["video"] = output_video_path_;
    outputs["detections"] = detections_path_;
    meta["outputs"] = outputs;

    write_json(metadata_path_,
               meta.dump(2, ' ', false, ordered_json::error_handler_t::replace));
}

void BadmintonAnalysisSystem::process_video(std::function<void(int, int)> progress_callback) {
    start_time_ = Clock::now();

    cv::VideoCapture cap(video_path_);
    if (!cap.isOpened()) {
        throw std::runtime_error("Unable to open video: " + video_path_);
    }
    const double fps = cap.get(cv::CAP_PROP_FPS);
    const int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    if (fps <= 0) {
        throw std::runtime_error("Unable to read FPS from video: " + video_path_);
    }
    const double video_duration = total_frames / fps;

    fps_ = fps;
    performance_log_interval_frames_ = std::max(1, static_cast<int>(fps * 5));

    frame_width_ = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    frame_height_ = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    const std::string template_path = get_template_path();
    const auto [template_gray, template_color] = load_template(template_path);

    setup_video_writer();  // temp mp4v under save_dir
    const CourtAnnotation ann = setup_court_annotation(template_color);
    court_corners_ = ann.corners;
    court_roi_corners_ = ann.roi;

    write_metadata(fps, total_frames, video_duration, template_path, ann);

    detection_writer_ = std::make_unique<JsonlDetectionWriter>(detections_path_);
    court_mapper_ = std::make_unique<CourtMapper>(ann.corners);
    player_tracker_ = std::make_unique<PlayerTracker>(ann.corners, ann.mid_height, 30,
                                                      detection_writer_.get(), fps);
    stats_visualizer_ =
        std::make_unique<StatsVisualizer>(frame_width_, frame_height_, language_);

    int frame_count = 0;
    int detect_frame_count = 0;
    cv::Mat frame;
    while (cap.read(frame)) {
        ++frame_count;  // Python: frame counter advances even for skipped frames
        process_frame(frame, template_gray, ann, frame_count, detect_frame_count);
        if (opts_.progress_json) {
            std::printf("{\"frame\":%d,\"total\":%d}\n", frame_count, total_frames);
            std::fflush(stdout);
        }
        if (progress_callback) progress_callback(frame_count, total_frames);
    }

    end_time_ = Clock::now();
    const double processing_time = elapsed_sec(start_time_);
    std::printf("\nProcessing complete:\n"
                "Original video duration: %.2f seconds\n"
                "Processing time: %.2f seconds\n",
                video_duration, processing_time);
    if (video_duration > 0) {  // ponytail: Python would ZeroDivisionError here
        std::printf("Processing speed ratio: %.2fx\n", processing_time / video_duration);
    }

    cleanup(cap);
}

bool BadmintonAnalysisSystem::is_court_view(const cv::Mat& gray_frame,
                                            const cv::Mat& template_gray, double threshold) {
    // Python is_court_view: matchTemplate(TM_CCOEFF_NORMED), max >= threshold.
    cv::Mat result;
    cv::matchTemplate(gray_frame, template_gray, result, cv::TM_CCOEFF_NORMED);
    double min_val = 0.0;
    double max_val = 0.0;
    cv::minMaxLoc(result, &min_val, &max_val, nullptr, nullptr);
    return max_val >= threshold;
}

void BadmintonAnalysisSystem::process_frame(cv::Mat& frame, const cv::Mat& template_gray,
                                            const CourtAnnotation& ann, int frame_count,
                                            int& detect_frame_count) {
    cv::Mat gray_frame;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

    const bool is_court = is_court_view(gray_frame, template_gray);
    if (is_court) {
        consecutive_non_court_frames_ = 0;
    } else {
        ++consecutive_non_court_frames_;
    }

    if (consecutive_non_court_frames_ >= kNonCourtFramesThreshold && rally_state_.active()) {
        rally_state_.force_end();
        shuttlecock_tracker_->clear_trajectory();
    }
    if (!is_court) {
        return;  // non-court: frame NOT written to output, no jsonl record
    }

    ++detect_frame_count;

    const RoiCorners& roi_corners = ann.roi;
    const int x1 = roi_corners[0].x;
    const int y1 = roi_corners[0].y;
    // Python: roi = frame[y1:y2, x1:x2] (numpy clamps); Mat(Rect) asserts on
    // out-of-range — compute_expanded_roi already clamps, this is insurance.
    const int cx1 = std::clamp(x1, 0, frame.cols);
    const int cy1 = std::clamp(y1, 0, frame.rows);
    const int cx2 = std::clamp(roi_corners[1].x, 0, frame.cols);
    const int cy2 = std::clamp(roi_corners[1].y, 0, frame.rows);
    // Mat view into frame — same aliasing as the numpy slice: the ROI overlay
    // drawn below is visible to pose detection (Python behaves the same).
    const cv::Mat roi = frame(cv::Rect(cx1, cy1, std::max(0, cx2 - cx1),
                                       std::max(0, cy2 - cy1)));
    if (show_pose_roi_) {
        cv::rectangle(frame, roi_corners[0], roi_corners[1], cv::Scalar(255, 0, 0), 2);
        cv::putText(frame, "Pose ROI",
                    cv::Point(x1, std::max(24, y1 - 8)), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    }

    const auto pose_t0 = Clock::now();
    const PoseObs pose = pose_analyzer_->detect(roi, x1, y1, court_mapper_.get());
    const double pose_elapsed = elapsed_sec(pose_t0);

    const auto ball_t0 = Clock::now();
    const auto detected_ball = shuttlecock_tracker_->detect_ball(frame, 0.18, roi_corners);
    const double ball_elapsed = elapsed_sec(ball_t0);
    const auto ball_position =
        shuttlecock_tracker_->update_trajectory(detected_ball, roi_corners);
    // Python: ball_position is not None and ball_position != [0, 0]
    const bool ball_seen = !(ball_position[0] == 0.0 && ball_position[1] == 0.0);

    // rally = shuttlecock-activity segment (see system.py comment).
    const double now_sec = fps_ ? frame_count / fps_ : 0.0;
    const char* event = rally_state_.observe(now_sec, ball_seen);
    if (event && std::strcmp(event, "start") == 0) {  // nullptr = no event (Py None)
        player_tracker_->start_new_rally();
    } else if (event && std::strcmp(event, "end") == 0) {
        shuttlecock_tracker_->clear_trajectory();
    }

    const auto shuttle_t0 = Clock::now();
    shuttlecock_tracker_->handle_visualization(frame);
    const double shuttle_draw_elapsed = elapsed_sec(shuttle_t0);

    // ponytail: Python forwards raw [0,0] and nulls it at jsonl-serialization
    // time (_point_or_none zero_is_none); collapsing to nullopt here yields the
    // identical record — PlayerTracker only uses the ball for the jsonl row.
    std::optional<cv::Point2f> ball_opt;
    if (ball_seen) {
        ball_opt = cv::Point2f(static_cast<float>(ball_position[0]),
                               static_cast<float>(ball_position[1]));
    }

    // Python: player_tracker.update(...) — dots/trails for draw_players come
    // from the tracker's players()/history() image-space accessors afterwards.
    player_tracker_->update(frame_count, pose.people, ball_opt, detect_frame_count);

    // Python stats cadence: refresh at frame 1 (or when the cache is empty),
    // then every int(fps * 0.5) frames.
    if (frame_count == 1 || cached_movement_stats_.empty()) {
        cached_movement_stats_ = player_tracker_->get_player_movement_stats();
        stats_update_interval_frames_ = static_cast<int>(fps_ * 0.5);
    }
    if (frame_count - last_stats_update_frame_ >= stats_update_interval_frames_) {
        cached_movement_stats_ = player_tracker_->get_player_movement_stats();
        last_stats_update_frame_ = frame_count;
    }

    const bool should_log_performance =
        show_performance_stats_ && performance_log_interval_frames_ > 0 &&
        frame_count % performance_log_interval_frames_ == 0;

    const CourtHistory trail = player_tracker_->get_player_trajectories();

    const auto players_t0 = Clock::now();
    // Python draw_players: dots from tracker.players, trail from tracker.history
    // (both image px) — contract takes the accessors directly.
    skeleton_renderer_.draw(frame, pose_analyzer_->current_pose(),
                            player_tracker_->players(), player_tracker_->history());
    const double players_draw_elapsed = elapsed_sec(players_t0);

    double court_draw_elapsed = 0.0;
    if (show_court_trajectory_) {
        const auto court_t0 = Clock::now();
        court_trajectory_visualizer_.draw_overlay(frame, trail);
        court_draw_elapsed = elapsed_sec(court_t0);
    }

    if (show_player_stats_) {  // Python: stats_visualizer passed only when enabled
        stats_visualizer_->draw_player_stats(frame, cached_movement_stats_,
                                             rally_state_.count());
    }

    if (should_log_performance) {
        std::printf(
            "Frame %d: pose %.2fs, shuttlecock %.2fs, shuttle draw %.2fs, "
            "players draw %.2fs, court draw %.2fs\n",
            frame_count, pose_elapsed, ball_elapsed, shuttle_draw_elapsed,
            players_draw_elapsed, court_draw_elapsed);
    }

    if (show_display_) {
        cv::imshow("frame", frame);
        cv::waitKey(1);
    }
    video_writer_.write(frame);
    if (save_images_) {
        cv::imwrite((fs::path(images_save_dir_) / (std::to_string(frame_count) + ".png"))
                        .string(),
                    frame);
    }
}

void BadmintonAnalysisSystem::cleanup(cv::VideoCapture& cap) {
    // Python _cleanup: close jsonl -> release writer -> sleep(1) -> release cap
    // -> destroy windows -> audio mux.
    if (player_tracker_) {
        player_tracker_->close();  // Python's tracker.close also closes the writer;
        player_tracker_.reset();   // writer.close below is idempotent either way
    }
    if (detection_writer_) {
        detection_writer_->close();
        detection_writer_.reset();
    }
    if (video_writer_.isOpened()) {
        video_writer_.release();
        std::this_thread::sleep_for(std::chrono::seconds(1));  // Python time.sleep(1)
    }
    cap.release();
    if (show_display_) {
        cv::destroyAllWindows();
    }

    std::string last_error;
    const auto try_encode = [&](const std::optional<std::string>& audio) {
        gb::encode_compatible_mp4(temp_output_video_path_, output_video_path_, audio);
        gb::remove_file(temp_output_video_path_);  // Python cleanup_temp_files on success
    };
    const auto run_without_audio = [&]() {
        std::cout << "\nProcessing video without audio..." << std::endl;
        if (!fs::exists(temp_output_video_path_)) {
            throw std::runtime_error("Temporary video not found: " + temp_output_video_path_);
        }
        try_encode(std::nullopt);
        std::cout << "Video saved to: " << output_video_path_ << std::endl;
    };
    const auto run_with_audio = [&]() {
        std::cout << "\nProcessing video audio..." << std::endl;
        if (!has_audio_track(video_path_)) {
            std::cout << "No audio track detected; exporting video without audio."
                      << std::endl;
            run_without_audio();
            return;
        }
        if (!fs::exists(temp_output_video_path_)) {
            throw std::runtime_error("Temporary video not found: " + temp_output_video_path_);
        }
        try_encode(video_path_);
        std::cout << "Video with audio saved to: " << output_video_path_ << std::endl;
    };

    if (keep_audio_) {
        // Python process_video_with_audio: any failure falls back to -an encode.
        try {
            run_with_audio();
            return;
        } catch (const std::exception& exc) {
            last_error = exc.what();
            std::cout << "Audio merge failed: " << last_error << std::endl;
            std::cout << "Falling back to video without audio." << std::endl;
        }
    }
    try {
        run_without_audio();
        return;
    } catch (const std::exception& exc) {
        last_error = exc.what();
        std::cout << "Video processing failed: " << last_error << std::endl;
    }
    // Spec: total ffmpeg failure -> keep temp video, exit 3.
    throw CodedError(3, "ffmpeg export failed (temp video kept): " + last_error);
}

cv::Mat BadmintonAnalysisSystem::draw_court_roi(const cv::Mat& frame,
                                                const std::vector<cv::Point2f>& corners,
                                                const RoiCorners& roi_corners) const {
    CourtMapper mapper(corners);
    // Python draw_court_overlay returns a copy; the contract draws in place —
    // clone first so `frame` stays untouched, like the Python call site expects.
    cv::Mat overlay = frame.clone();
    mapper.draw_court_overlay(overlay);
    cv::rectangle(overlay, roi_corners[0], roi_corners[1], cv::Scalar(255, 0, 0), 2);
    return overlay;
    // ponytail: Python also reassigns self.court_mapper here; this method is
    // const (and its only Python call site is commented out), so no state churn.
}

void BadmintonAnalysisSystem::analyze_shuttlecock(
    const RoiCorners&, const std::vector<cv::Point2f>&) const {
    throw std::runtime_error(
        "Hit-point analysis is disabled until it is migrated to detections.jsonl.");
}

}  // namespace gb
