#pragma once
// Court detection + mapping. Port sources: badminton_analysis/court/*.py
// Public signatures mirror Python; private members are yours to extend.
//
// Signature fixes vs the original hand-written header (Python truly differed):
//  - CourtMapper::draw_court_overlay returns mid_height (Python -> (overlay, mid));
//    still draws in place per this header's contract (Python drew on a copy).
//  - CourtMapper::court_to_image is single-point: Python transforms the batch
//    but returns only transformed[0][0].
//  - auto_detect_court_corners returns {corners, line_mask, debug} (Python
//    3-tuple); render_auto_court_preview takes optional corners + debug
//    (Python passes None for both on failure).
//  - resolve_court_corners returns {corners, roi, mid_height} (Python 3-tuple).
//  - compute_expanded_roi's image_shape is cv::Size(width, height) — OpenCV
//    order, i.e. mat.size(); Python's image.shape is (height, width, ...).
#include <array>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>
#include "gb/types.h"

namespace gb {

class CourtMapper {
public:
    // Python: CourtMapper(image_court_corners, court_dimensions=(6.1, 13.4))
    CourtMapper(const std::vector<cv::Point2f>& image_court_corners,
                const cv::Size2f& court_dimensions = cv::Size2f(6.1f, 13.4f));

    // Python: image_to_court(point) -> [x,y] rounded to 2 dp (never None for
    // a valid point; Python only returned [] for invalid input types).
    std::optional<cv::Point2f> image_to_court(const cv::Point2f& point) const;
    // Python: court_to_image(points) -> np.round(perspectiveTransform(...)[0][0], 2)
    // (only the first point of the batch is ever returned -> single-point API).
    cv::Point2f court_to_image(const cv::Point2f& point) const;

    // Draws service/net/singles overlay lines in place (in-memory only).
    // Python: (overlay_copy, mid_height) -> here the int return is mid_height;
    // clone the image first if you need Python's non-mutating behavior.
    int draw_court_overlay(cv::Mat& image) const;

    // Python: set by compute_court_overlay() during __init__.
    int mid_height() const { return mid_height_; }

    const std::vector<cv::Point2f>& corners() const { return corners_; }

private:
    // Python: compute_court_overlay()
    void compute_court_overlay();
    // Python: _line_to_image(start, end)
    std::pair<cv::Point2f, cv::Point2f> line_to_image(const cv::Point2f& start,
                                                      const cv::Point2f& end) const;

    std::vector<cv::Point2f> corners_;
    cv::Size2f dims_;
    cv::Mat matrix_;      // image -> court (CV_64F 3x3)
    cv::Mat inv_matrix_;  // court -> image (CV_64F 3x3)
    std::vector<std::pair<cv::Point2f, cv::Point2f>> vertical_lines_;
    std::vector<std::pair<cv::Point2f, cv::Point2f>> horizontal_lines_;
    int mid_height_ = 0;
};

// ---------------------------------------------------------------------------
// reference.py port (used by detector + preview).
// ---------------------------------------------------------------------------

inline constexpr double BADMINTON_COURT_WIDTH = 6.1;
inline constexpr double BADMINTON_COURT_LENGTH = 13.4;
inline constexpr double BADMINTON_SINGLES_MARGIN = 0.46;
inline constexpr double BADMINTON_BACK_SERVICE_OFFSET = 0.76;
inline constexpr double BADMINTON_SERVICE_LINE_FROM_NET = 1.98;

// Python: prepare_line_support() dict (distance_map + tolerance).
struct CourtLineSupport {
    cv::Mat distance_map;  // CV_32F, distance to nearest reference-line pixel
    double tolerance = 0.0;
};

// One projected reference line: Python (start, end, weight) tuple.
struct CourtProjectedLine {
    cv::Point2f start;
    cv::Point2f end;
    double weight = 0.0;
};

class BadmintonCourtReference {
public:
    BadmintonCourtReference();

    // Python: prepare_line_support(line_mask, image_shape) -> dict or None.
    // line_mask empty cv::Mat == Python None.
    std::optional<CourtLineSupport> prepare_line_support(const cv::Mat& line_mask,
                                                         cv::Size image_size) const;

    // Python: score_line_support(...) -> (score, details dict).
    std::pair<double, std::map<std::string, double>> score_line_support(
        const std::vector<cv::Point2f>& image_corners,
        const std::optional<CourtLineSupport>& line_support,
        cv::Size image_size) const;

    // Python: project_lines(image_corners) -> list of (start, end, weight).
    std::vector<CourtProjectedLine> project_lines(
        const std::vector<cv::Point2f>& image_corners) const;

private:
    // Python: _project_line(matrix, start, end)
    std::pair<cv::Point2f, cv::Point2f> project_line(const cv::Mat& matrix,
                                                     const cv::Point2f& start,
                                                     const cv::Point2f& end) const;
    // Python: _empty_details(tolerance)
    static std::map<std::string, double> empty_details(double tolerance);

    std::vector<cv::Point2f> court_corners_;
    std::vector<std::array<cv::Point2f, 2>> lines_;  // (start, end) per line
    std::vector<double> weights_;
};

// ---------------------------------------------------------------------------
// detector.py headless surface.
// ---------------------------------------------------------------------------

// Python: one Hough segment dict {"points","length","mid","angle"}.
struct CourtLineSegment {
    cv::Vec4i points{0, 0, 0, 0};  // x1, y1, x2, y2 (ints, as Python)
    double length = 0.0;
    cv::Point2f mid{0.f, 0.f};
    double angle = 0.0;
};

// Python: the debug dict {"horizontal","side","score","details"}.
// details empty == Python None; keys are the round(...,4) debug values.
struct CourtDetectDebug {
    std::vector<CourtLineSegment> horizontal;
    std::vector<CourtLineSegment> side;
    std::optional<double> score;
    std::map<std::string, double> details;
};

// Python: auto_detect_court_corners() -> (corners, line_mask, debug).
struct CourtDetection {
    std::optional<std::vector<cv::Point2f>> corners;  // nullopt == Python None
    cv::Mat line_mask;
    CourtDetectDebug debug;
};

// Python: resolve_court_corners() -> (corners, roi_corners, mid_height);
// all nullopt == (None, None, None).
struct CourtResolution {
    std::optional<std::vector<cv::Point2f>> corners;
    std::optional<RoiCorners> roi;
    std::optional<int> mid_height;
};

// Python: compute_expanded_roi(court_corners, image_shape).
// image_size: OpenCV order (width, height) == cv::Mat::size().
RoiCorners compute_expanded_roi(const std::vector<cv::Point2f>& court_corners,
                                cv::Size image_size);

// Python: auto_detect_court_corners(image) -> (corners or None, mask, debug)
CourtDetection auto_detect_court_corners(const cv::Mat& image);

// Python: render_auto_court_preview(image, corners, roi_corners=None, debug=None)
cv::Mat render_auto_court_preview(
    const cv::Mat& image,
    const std::optional<std::vector<cv::Point2f>>& corners,
    const std::optional<RoiCorners>& roi_corners = std::nullopt,
    const std::optional<CourtDetectDebug>& debug = std::nullopt);

// Python: resolve_court_corners(image, manual_corners=None) -> 3-tuple.
// Headless: auto-detect only; no interactive GUI in v1 (see spec).
CourtResolution resolve_court_corners(
    const cv::Mat& image,
    const std::optional<std::vector<cv::Point2f>>& manual_corners = std::nullopt);

}  // namespace gb
