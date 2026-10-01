// Port of visualization/stats.py — EN path only (cv2.putText; zh path used
// PIL ImageFont in Python and needs freetype, out of scope for v1).
#include "gb/viz.h"

#include <algorithm>
#include <cstdio>

namespace gb {
namespace {

// Python f"{v:.2f}" — printf rounds the exact binary value, same as CPython.
std::string fmt2(double v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.2f", v);
    return buf;
}

// texts['en'] (stats.py)
const char* kRally = "Rally";
const char* kUpperPlayer = "Upper Player";
const char* kLowerPlayer = "Lower Player";
const char* kStats = "Stats";
const char* kCurrentSpeed = "Current Speed";
const char* kCurrentRally = "Current Rally";
const char* kMatchTotal = "Match Total";
const char* kDistance = "Distance";
const char* kAvgSpeed = "Avg Speed";
const char* kMaxSpeed = "Max Speed";
const char* kTotalDistance = "Total Distance";
const char* kUnitSpeed = "m/s";
const char* kUnitDistance = "m";

const cv::Scalar kWhite(255, 255, 255);
const cv::Scalar kBackgroundColor(0, 0, 0);  // Python background_color

}  // namespace

StatsVisualizer::StatsVisualizer(int frame_width, int frame_height,
                                 std::string language)
    : language_(std::move(language)),
      frame_width_(frame_width),
      frame_height_(frame_height) {
    // Python __init__ — sizes scale off a 1920x1080 reference.
    double scale_factor =
        2.0 * std::min(frame_width_ / 1920.0, frame_height_ / 1080.0);
    font_scale_ = std::max(0.4, 0.5 * scale_factor);
    thickness_ = std::max(1, static_cast<int>(1 * scale_factor));
    line_height_ = std::max(15, static_cast<int>(20 * scale_factor));
    panel_width_ = std::max(180, static_cast<int>(180 * scale_factor));
    panel_height_ = std::max(150, static_cast<int>(200 * scale_factor));
    margin_ = std::max(5, static_cast<int>(10 * scale_factor));
    // ponytail: language_ stored but ignored — only texts['en'] implemented.
}

void StatsVisualizer::draw_player_stats(cv::Mat& frame,
                                        const MovementStats& movement_stats,
                                        int rally_count) {
    if (frame.empty()) return;

    // Python: rally_pos_y = int(panel_height + frame_height * 0.1)
    int rally_pos_y =
        static_cast<int>(panel_height_ + frame_height_ * 0.1);

    // Queue: rally text first, then upper panel lines, then lower panel lines.
    std::vector<TextItem> text_items;
    text_items.push_back({std::string(kRally) + ": " + std::to_string(rally_count),
                          cv::Point(margin_, rally_pos_y),
                          font_scale_ * 1.5, cv::Scalar(0, 165, 255),
                          thickness_ + 2});

    // Python movement_stats.get('upper', {}) — missing key -> all-zero stats.
    auto slot = [&movement_stats](const char* key) -> const SlotStats& {
        static const SlotStats kEmpty{};
        auto it = movement_stats.find(key);
        return it == movement_stats.end() ? kEmpty : it->second;
    };

    draw_player_panel(frame, kUpperPlayer, slot("upper"), margin_,
                      static_cast<int>(frame_height_ * 0.05), panel_width_,
                      panel_height_, cv::Scalar(0, 255, 255), font_scale_,
                      thickness_, line_height_, kBackgroundColor,
                      kBackgroundAlpha, text_items);

    draw_player_panel(frame, kLowerPlayer, slot("lower"), margin_,
                      static_cast<int>(frame_height_ * 0.55), panel_width_,
                      panel_height_, cv::Scalar(255, 0, 255), font_scale_,
                      thickness_, line_height_, kBackgroundColor,
                      kBackgroundAlpha, text_items);

    // Python _draw_text_batch (EN): putText in queue order, LINE_AA.
    for (const auto& t : text_items) {
        cv::putText(frame, t.text, t.pos, cv::FONT_HERSHEY_SIMPLEX, t.font_scale,
                    t.color, t.thickness, cv::LINE_AA);
    }
}

void StatsVisualizer::draw_player_panel(cv::Mat& frame,
                                        const std::string& player_name,
                                        const SlotStats& stats, int x_pos,
                                        int y_pos, int panel_width,
                                        int panel_height, const cv::Scalar& color,
                                        double font_scale, int thickness,
                                        int line_height, const cv::Scalar& bg_color,
                                        double bg_alpha,
                                        std::vector<TextItem>& text_items) {
    // Python _draw_player_panel inner margin
    int margin = std::max(5, static_cast<int>(panel_width * 0.05));

    // Panel background: rect on a copy, alpha-blended into the frame.
    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay,
                  cv::Point(x_pos - margin, y_pos - margin),
                  cv::Point(x_pos + panel_width + margin,
                            y_pos + panel_height + margin),
                  bg_color, cv::FILLED);
    cv::addWeighted(overlay, bg_alpha, frame, 1.0 - bg_alpha, 0.0, frame);

    auto queue = [&](std::string text, cv::Point pos, double fs, cv::Scalar c,
                     int th) {
        text_items.push_back({std::move(text), pos, fs, c, th});
    };

    // 1. title (player color, bolder)
    queue(player_name + " " + kStats + ":", cv::Point(x_pos, y_pos),
          font_scale * 1.1, color, thickness + 1);
    // 2. current speed
    queue(std::string(kCurrentSpeed) + ": " + fmt2(stats.current_speed) + " " +
              kUnitSpeed,
          cv::Point(x_pos, y_pos + line_height), font_scale, kWhite, thickness);

    // 3..6. current rally block
    int y_rally = y_pos + 2 * line_height;
    queue(std::string(kCurrentRally) + ":", cv::Point(x_pos, y_rally),
          font_scale, kWhite, thickness);
    queue(std::string(" ") + kDistance + ": " + fmt2(stats.rally_distance) + " " +
              kUnitDistance,
          cv::Point(x_pos, y_rally + line_height), font_scale, kWhite, thickness);
    queue(std::string(" ") + kAvgSpeed + ": " + fmt2(stats.rally_avg_speed) +
              " " + kUnitSpeed,
          cv::Point(x_pos, y_rally + 2 * line_height), font_scale, kWhite,
          thickness);
    queue(std::string(" ") + kMaxSpeed + ": " + fmt2(stats.rally_max_speed) +
              " " + kUnitSpeed,
          cv::Point(x_pos, y_rally + 3 * line_height), font_scale, kWhite,
          thickness);

    // 7..10. match total block
    int y_match = y_rally + 4 * line_height;
    queue(std::string(kMatchTotal) + ":", cv::Point(x_pos, y_match), font_scale,
          kWhite, thickness);
    queue(std::string(" ") + kTotalDistance + ": " +
              fmt2(stats.match_distance) + " " + kUnitDistance,
          cv::Point(x_pos, y_match + line_height), font_scale, kWhite, thickness);
    queue(std::string(" ") + kAvgSpeed + ": " + fmt2(stats.match_avg_speed) +
              " " + kUnitSpeed,
          cv::Point(x_pos, y_match + 2 * line_height), font_scale, kWhite,
          thickness);
    queue(std::string(" ") + kMaxSpeed + ": " + fmt2(stats.match_max_speed) +
              " " + kUnitSpeed,
          cv::Point(x_pos, y_match + 3 * line_height), font_scale, kWhite,
          thickness);
}

}  // namespace gb
