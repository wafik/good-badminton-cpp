// Port of visualization/court_trajectory.py — mini court diagram (drawn once)
// + fading court-meter trail dots, blended top-right of the frame.
#include "gb/viz.h"

#include <algorithm>

namespace gb {
namespace {

// Python court line color (70, 62, 63)
const cv::Scalar kLineColor(70, 62, 63);

}  // namespace

CourtTrajectoryVisualizer::CourtTrajectoryVisualizer(int width, int height)
    : default_width_(width),
      default_height_(height),
      width_(width),
      height_(height) {
    court_overlay_ = create_court_overlay(width_, height_);
}

cv::Mat CourtTrajectoryVisualizer::create_court_overlay(int width, int height) {
    // Python _create_court_overlay — np.zeros((height, width, 3))
    cv::Mat court = cv::Mat::zeros(height, width, CV_8UC3);

    // Standard court dimensions (meters) — Python locals
    const double doubles_width = 6.10;
    const double court_length = 13.40;
    const double single_width = 0.46;
    const double service_line = 1.98;
    const double back_service = 0.76;
    const double line_width = 0.1;
    const double center_line_width = 0.1;

    const double scale_x = (width - 20) / doubles_width;
    const double scale_y = (height - 20) / court_length;
    const double scale = std::min(scale_x, scale_y);

    const int offset_x = static_cast<int>((width - doubles_width * scale) / 2);
    const int offset_y = static_cast<int>((height - court_length * scale) / 2);

    auto scale_point = [&](double x, double y) {
        return cv::Point(static_cast<int>(x * scale + offset_x),
                         static_cast<int>(y * scale + offset_y));
    };

    const int line_width_px = std::max(1, static_cast<int>(line_width * scale));
    const int center_line_width_px =
        std::max(1, static_cast<int>(center_line_width * scale));

    // Outer rectangle
    cv::rectangle(court, scale_point(0, 0),
                  scale_point(doubles_width, court_length), kLineColor,
                  line_width_px);
    // Singles sidelines
    cv::line(court, scale_point(single_width, 0),
             scale_point(single_width, court_length), kLineColor, line_width_px);
    cv::line(court, scale_point(doubles_width - single_width, 0),
             scale_point(doubles_width - single_width, court_length), kLineColor,
             line_width_px);
    // Center line — two segments with the net gap between them
    cv::line(court, scale_point(doubles_width / 2, 0),
             scale_point(doubles_width / 2, court_length / 2 - service_line),
             kLineColor, center_line_width_px);
    cv::line(court, scale_point(doubles_width / 2, court_length / 2 + service_line),
             scale_point(doubles_width / 2, court_length), kLineColor,
             center_line_width_px);
    // Net — dashed across the full width (dash 4, step 8 px)
    const int net_y =
        static_cast<int>(court_length / 2 * scale + offset_y);
    const int dash_length = 4;
    for (int x = 0; x < width; x += dash_length * 2) {
        const int end_x = std::min(x + dash_length, width);
        cv::line(court, cv::Point(x, net_y), cv::Point(end_x, net_y), kLineColor,
                 line_width_px);
    }
    // Short service lines (net ± service_line)
    cv::line(court, scale_point(0, court_length / 2 - service_line),
             scale_point(doubles_width, court_length / 2 - service_line),
             kLineColor, line_width_px);
    cv::line(court, scale_point(0, court_length / 2 + service_line),
             scale_point(doubles_width, court_length / 2 + service_line),
             kLineColor, line_width_px);
    // Back service lines
    cv::line(court, scale_point(0, back_service),
             scale_point(doubles_width, back_service), kLineColor, line_width_px);
    cv::line(court, scale_point(0, court_length - back_service),
             scale_point(doubles_width, court_length - back_service), kLineColor,
             line_width_px);

    return court;
}

void CourtTrajectoryVisualizer::draw_overlay(cv::Mat& frame,
                                             const CourtHistory& court_history) {
    // ponytail: Python wraps draw_overlay in try/except + print; the empty/
    // bounds guards below cover every failure mode it caught, no try needed.
    if (frame.empty()) return;
    const int frame_height = frame.rows;
    const int frame_width = frame.cols;

    // Python: min(w/1920, h/1080) * 1.5 — scaled up vs the reference court
    const double scale_factor =
        std::min(frame_width / 1920.0, frame_height / 1080.0) * 1.5;

    const int court_overlay_width =
        static_cast<int>(default_width_ * scale_factor);
    const int court_overlay_height =
        static_cast<int>(default_height_ * scale_factor);
    if (court_overlay_width != width_ || court_overlay_height != height_) {
        width_ = court_overlay_width;
        height_ = court_overlay_height;
        court_overlay_ = create_court_overlay(width_, height_);
    }

    // Python: overlay = court_overlay.copy()
    cv::Mat overlay = court_overlay_.clone();
    const int h = overlay.rows;
    const int w = overlay.cols;

    const int margin = std::max(5, static_cast<int>(10 * scale_factor));
    const int court_width = w - margin * 2;
    const int court_height = h - margin * 2;
    const int offset_x = margin;
    const int offset_y = margin;

    const double doubles_width = 6.10;
    const double court_length = 13.40;

    // Python slot order + colors: upper/upper2 cyan, lower/lower2 magenta
    struct SlotDraw {
        const char* name;
        cv::Scalar color;
    };
    const SlotDraw slots[] = {
        {"upper", cv::Scalar(0, 255, 255)},
        {"upper2", cv::Scalar(0, 255, 255)},
        {"lower", cv::Scalar(255, 0, 255)},
        {"lower2", cv::Scalar(255, 0, 255)},
    };

    const int radius_min = std::max(2, static_cast<int>(2 * scale_factor));
    const int radius_max = std::max(3, static_cast<int>(5 * scale_factor));

    for (const auto& sd : slots) {
        auto it = court_history.find(sd.name);
        if (it == court_history.end()) continue;
        const auto& hist = it->second;
        const size_t n = hist.size();
        if (n == 0) continue;
        // i enumerates ALL entries (Python includes None in len/index; they
        // just don't draw) — fade denominator must count them too.
        for (size_t i = 0; i < n; ++i) {
            const auto& pos = hist[i];
            if (!pos.has_value()) continue;  // Python: pos is not None
            const double x_norm = pos->x / doubles_width;
            const double y_norm = pos->y / court_length;
            const int x =
                static_cast<int>(x_norm * court_width + offset_x);
            const int y =
                static_cast<int>(y_norm * court_height + offset_y);
            if (x < 0 || x >= w || y < 0 || y >= h) continue;
            // Newer points bigger; no LINE_AA in Python here (plain circle)
            const int radius =
                n > 1 ? static_cast<int>(radius_min +
                                         (static_cast<double>(i) /
                                          static_cast<double>(n)) *
                                             (radius_max - radius_min))
                      : radius_min;
            cv::circle(overlay, cv::Point(x, y), radius, sd.color, -1);
        }
    }

    // Top-right placement; Python's roi.shape == overlay.shape recheck is
    // implied by these bounds (and by w/h coming from the overlay itself).
    const int padding = std::max(10, static_cast<int>(20 * scale_factor));
    if (frame_height >= padding + h && frame_width >= padding + w) {
        cv::Mat roi = frame(
            cv::Rect(frame_width - w - padding, padding, w, h));
        cv::addWeighted(overlay, 0.7, roi, 0.3, 0.0, roi);  // writes through
    }
}

}  // namespace gb
