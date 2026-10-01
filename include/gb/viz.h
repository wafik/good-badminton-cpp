#pragma once
// Live overlays. Port sources: visualization/stats.py, player_pose.py (draw
// side), court_trajectory.py. Stats text: en/id via cv::putText (Latin).
// ponytail: zh stats text needs freetype (Python zh path uses PIL ImageFont)
// — out of scope per spec ("zh fonts via freetype later"); zh falls back to EN.
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include "gb/types.h"

namespace gb {

class StatsVisualizer {
public:
    // Python: StatsVisualizer(frame_width, frame_height, language='zh') -> use "en"
    StatsVisualizer(int frame_width, int frame_height, std::string language = "en");
    // Rally counter + upper/lower panels (movement_stats only needs those two).
    // Python queues every string while drawing the two panel backgrounds, then
    // batch-putText's them so text always lands on top of both bgs — same here.
    void draw_player_stats(cv::Mat& frame, const MovementStats& movement_stats,
                           int rally_count);

private:
    // Python _draw_text_batch item: (text, position, font_scale, color, thickness)
    struct TextItem {
        std::string text;
        cv::Point pos;
        double font_scale;
        cv::Scalar color;
        int thickness;
    };
    // Python _draw_player_panel: bg rect (alpha-blend) + queue 10 text lines.
    void draw_player_panel(cv::Mat& frame, const std::string& player_name,
                           const SlotStats& stats, int x_pos, int y_pos,
                           int panel_width, int panel_height,
                           const cv::Scalar& color, double font_scale,
                           int thickness, int line_height,
                           const cv::Scalar& bg_color, double bg_alpha,
                           std::vector<TextItem>& text_items);

    // Sizes from Python __init__ (scale vs 1920x1080 reference). language_
    // picks en/id texts; zh falls back to EN (freetype — see file top).
    std::string language_;
    int frame_width_ = 0;
    int frame_height_ = 0;
    double font_scale_ = 0.5;
    int thickness_ = 1;
    int line_height_ = 15;
    int panel_width_ = 180;
    int panel_height_ = 150;
    int margin_ = 5;
    static constexpr double kBackgroundAlpha = 0.5;  // Python background_alpha
};

// Skeletons (12 limbs) + player dots + fading trail.
// PoseDrawData.keypoints are FULL-FRAME (types.h): drawn as-is. PoseAnalyzer
// applies the ROI offset at store time and keeps offset_x/offset_y as
// information; Python stores ROI-relative and adds the offset while drawing —
// identical final pixels either way.
// C++ change vs contract draft: draft took a CourtHistory here, but Python
// draw_players paints pixel dots from tracker.players (image px) and the
// trail from tracker.history (image px); court_history is in METERS and would
// render in the top-left corner. Params match PlayerTracker::players()/
// history() accessors directly.
class PlayerSkeletonRenderer {
public:
    // Python PlayerPoseVisualizer.__init__ defaults. show_performance_stats
    // is not ported (timing prints live in system.py, as in Python).
    bool show_skeletons = true;
    bool show_player_trajectories = true;

    void draw(cv::Mat& frame, const std::optional<PoseDrawData>& pose,
              const std::map<std::string, std::optional<cv::Point2f>>& players,
              const std::map<std::string, std::deque<cv::Point2f>>& history);
};

// Mini court diagram top-right + fading trail dots (court-meters -> mini px).
class CourtTrajectoryVisualizer {
public:
    explicit CourtTrajectoryVisualizer(int width = 200, int height = 400);
    // Python draw_overlay returns frame; C++ mutates in place (same pixels).
    void draw_overlay(cv::Mat& frame, const CourtHistory& court_history);

private:
    cv::Mat create_court_overlay(int width, int height);
    int default_width_ = 200;
    int default_height_ = 400;
    int width_ = 200;
    int height_ = 400;
    cv::Mat court_overlay_;  // re-rendered when the frame-scale size changes
};

}  // namespace gb
