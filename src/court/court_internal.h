#pragma once
// Internal helpers shared by detector.cpp / quad_score.cpp. Not installed.
// Mirrors private functions of badminton_analysis/court/detector.py.
#include <array>
#include <map>
#include <optional>
#include <vector>
#include <opencv2/opencv.hpp>
#include "gb/court.h"

namespace gb {
namespace court_detail {

// Python _line_angle: degrees(atan2(dy, dx)) % 180 (always in [0, 180)).
double line_angle(int x1, int y1, int x2, int y2);

// Python _line_intersection: None when |denom| < 1e-6.
std::optional<cv::Point2f> line_intersection(const cv::Vec4i& a, const cv::Vec4i& b);

// Python _inside_image(point, width, height, margin=0).
bool inside_image(const cv::Point2f& point, int width, int height, int margin = 0);

// Python _polygon_area: abs(cv2.contourArea(float32 pts)).
double polygon_area(const std::vector<cv::Point2f>& points);

// Python _is_convex_quad.
bool is_convex_quad(const std::vector<cv::Point2f>& points);

// Python _project_court_points(matrix, points).
std::vector<cv::Point2f> project_court_points(const cv::Mat& matrix,
                                              const std::vector<cv::Point2f>& points);

// Python _score_court_quad -> (score, details) or None.
struct QuadScore {
    double score = 0.0;
    std::map<std::string, double> details;
};
std::optional<QuadScore> score_court_quad(
    const std::vector<cv::Point2f>& corners, cv::Size image_size,
    const std::array<CourtLineSegment, 4>& lines,  // top, bottom, left, right
    const cv::Mat& line_mask,
    const std::vector<CourtLineSegment>& horizontal_lines,
    const BadmintonCourtReference* court_reference,
    const std::optional<CourtLineSupport>& line_support);

// Python _promote_far_baseline(lines, horizontal_lines, image_shape).
std::array<CourtLineSegment, 4> promote_far_baseline(
    const std::array<CourtLineSegment, 4>& lines,
    const std::vector<CourtLineSegment>& horizontal_lines, cv::Size image_size);

// Python _dedupe_lines(lines, orientation, max_count).
std::vector<CourtLineSegment> dedupe_lines(std::vector<CourtLineSegment> lines,
                                           bool horizontal, int max_count);

// Python twin of BadmintonCourtReference._sample_line. The caller computes
// `length` exactly as Python does (np.linalg.norm float32 in _sample_line,
// np.hypot float64 in _reference_segment_support) because int(sample_count)
// can flip at integer boundaries otherwise.
std::optional<std::vector<cv::Point>> sample_line_points(
    const cv::Point2f& start, const cv::Point2f& end, int width, int height,
    double length, double length_divisor);

// np.linalg.norm(end - start) for a float32 point pair (sqrt of float32 dot).
float norm_f32(const cv::Point2f& start, const cv::Point2f& end);

}  // namespace court_detail
}  // namespace gb
