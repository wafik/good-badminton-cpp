// Port of the scoring half of badminton_analysis/court/detector.py: the
// private helpers plus _score_court_quad (14 heuristic terms — term order and
// formulas must stay numerically identical to Python or quad selection flips).
#include "court_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace gb {
namespace court_detail {

double line_angle(int x1, int y1, int x2, int y2) {
    // Python: np.degrees(np.arctan2(dy, dx)) % 180 — Python % on a negative
    // dividend wraps into [0, 180); fmod does not, so adjust.
    double angle = std::atan2(static_cast<double>(y2 - y1), static_cast<double>(x2 - x1)) *
                   180.0 / CV_PI;
    angle = std::fmod(angle, 180.0);
    if (angle < 0.0) {
        angle += 180.0;
    }
    return angle;
}

std::optional<cv::Point2f> line_intersection(const cv::Vec4i& a, const cv::Vec4i& b) {
    const double x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3];
    const double x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];
    const double denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if (std::abs(denom) < 1e-6) {
        return std::nullopt;
    }
    const double px =
        ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom;
    const double py =
        ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom;
    // Python: np.array([px, py], dtype=np.float32)
    return cv::Point2f(static_cast<float>(px), static_cast<float>(py));
}

bool inside_image(const cv::Point2f& point, int width, int height, int margin) {
    return margin <= point.x && point.x <= width - 1 - margin &&
           margin <= point.y && point.y <= height - 1 - margin;
}

double polygon_area(const std::vector<cv::Point2f>& points) {
    // Python: abs(cv2.contourArea(np.array(points, float32)))
    return std::abs(cv::contourArea(points));
}

bool is_convex_quad(const std::vector<cv::Point2f>& points) {
    std::vector<cv::Point2f> hull;
    cv::convexHull(points, hull);
    if (hull.size() != 4) {
        return false;
    }
    // Python: cv2.isContourConvex(pts.astype(np.int32)) — truncates.
    std::vector<cv::Point> int_points;
    int_points.reserve(points.size());
    for (const auto& p : points) {
        int_points.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
    }
    return cv::isContourConvex(int_points);
}

std::vector<cv::Point2f> project_court_points(const cv::Mat& matrix,
                                             const std::vector<cv::Point2f>& points) {
    std::vector<cv::Point2f> out;
    cv::perspectiveTransform(points, out, matrix);
    return out;
}

namespace {

// np.round(p).astype(int): rint in float32, then integer cast.
int rint_i(float v) { return static_cast<int>(std::nearbyint(v)); }

// Python _badminton_line_alignment_score: rasterize the expected court-line
// model, dilate the detected line mask by 9x9, measure pixel coverage.
double badminton_line_alignment_score(const std::vector<cv::Point2f>& corners,
                                      const cv::Mat& line_mask) {
    const int height = line_mask.rows;
    const int width = line_mask.cols;
    const std::vector<cv::Point2f> court_points = {
        {0.f, 0.f}, {6.1f, 0.f}, {6.1f, 13.4f}, {0.f, 13.4f}};
    cv::Mat matrix = cv::getPerspectiveTransform(court_points, corners);

    cv::Mat expected = cv::Mat::zeros(height, width, CV_8U);
    const std::array<float, 7> horizontal_ys = {0.f, 0.76f, 4.72f, 6.7f, 8.68f, 12.64f, 13.4f};
    for (float y : horizontal_ys) {
        auto p = project_court_points(matrix, {{0.f, y}, {6.1f, y}});
        cv::line(expected, cv::Point(rint_i(p[0].x), rint_i(p[0].y)),
                 cv::Point(rint_i(p[1].x), rint_i(p[1].y)), cv::Scalar(255), 5);
    }
    const std::array<float, 5> vertical_xs = {0.f, 0.46f, 3.05f, 5.64f, 6.1f};
    for (float x : vertical_xs) {
        auto p = project_court_points(matrix, {{x, 0.f}, {x, 13.4f}});
        cv::line(expected, cv::Point(rint_i(p[0].x), rint_i(p[0].y)),
                 cv::Point(rint_i(p[1].x), rint_i(p[1].y)), cv::Scalar(255), 5);
    }
    const std::array<std::pair<cv::Point2f, cv::Point2f>, 2> center_segments = {{
        {cv::Point2f{3.05f, 0.f}, cv::Point2f{3.05f, 4.72f}},
        {cv::Point2f{3.05f, 8.68f}, cv::Point2f{3.05f, 13.4f}},
    }};
    for (const auto& seg : center_segments) {
        auto p = project_court_points(matrix, {seg.first, seg.second});
        cv::line(expected, cv::Point(rint_i(p[0].x), rint_i(p[0].y)),
                 cv::Point(rint_i(p[1].x), rint_i(p[1].y)), cv::Scalar(255), 5);
    }

    const int expected_pixels = cv::countNonZero(expected);
    if (expected_pixels == 0) {
        return 0.0;
    }

    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
    cv::Mat dilated_mask;
    cv::dilate(line_mask, dilated_mask, kernel);
    cv::Mat overlap;
    cv::bitwise_and(expected, dilated_mask, overlap);
    return static_cast<double>(cv::countNonZero(overlap)) / expected_pixels;
}

// Python _badminton_horizontal_pattern_score: nearest detected horizontal line
// per expected court line, weight-averaged in [0, 1].
double badminton_horizontal_pattern_score(const std::vector<cv::Point2f>& corners,
                                          const std::vector<CourtLineSegment>& horizontal_lines,
                                          cv::Size image_shape) {
    if (horizontal_lines.empty()) {
        return 0.0;
    }
    const int height = image_shape.height;
    const std::vector<cv::Point2f> court_points = {
        {0.f, 0.f}, {6.1f, 0.f}, {6.1f, 13.4f}, {0.f, 13.4f}};
    cv::Mat matrix = cv::getPerspectiveTransform(court_points, corners);

    const std::array<double, 7> expected_court_y = {0, 0.76, 4.72, 6.7, 8.68, 12.64, 13.4};
    const std::array<double, 7> weights = {5.0, 4.0, 1.6, 0.8, 1.6, 4.0, 5.0};
    std::vector<double> detected_y;
    detected_y.reserve(horizontal_lines.size());
    for (const auto& line : horizontal_lines) {
        detected_y.push_back(static_cast<double>(line.mid.y));
    }
    const double tolerance = std::max(12.0, height * 0.022);

    double weighted_score = 0.0;
    double total_weight = 0.0;
    for (size_t i = 0; i < expected_court_y.size(); ++i) {
        const float court_y = static_cast<float>(expected_court_y[i]);
        auto p = project_court_points(matrix, {{0.f, court_y}, {6.1f, court_y}});
        const double expected_y = (p[0].y + p[1].y) / 2.0;
        double nearest = std::numeric_limits<double>::max();
        for (double y : detected_y) {
            nearest = std::min(nearest, std::abs(y - expected_y));
        }
        weighted_score += std::max(0.0, 1.0 - nearest / tolerance) * weights[i];
        total_weight += weights[i];
    }
    return weighted_score / total_weight;
}

// Python _horizontal_segment_support / _side_segment_support: penalize a
// boundary line whose segment overshoots the quad edge.
double horizontal_segment_support(const CourtLineSegment& line, const cv::Point2f& p_left,
                                  const cv::Point2f& p_right, int image_width) {
    const double x1 = line.points[0], x2 = line.points[2];
    const double seg_min = std::min(x1, x2), seg_max = std::max(x1, x2);
    const double quad_min = std::min(static_cast<double>(p_left.x), static_cast<double>(p_right.x));
    const double quad_max = std::max(static_cast<double>(p_left.x), static_cast<double>(p_right.x));
    const double overshoot = std::max(0.0, seg_min - quad_min) + std::max(0.0, quad_max - seg_max);
    return std::max(0.0, 1.0 - overshoot / std::max(1.0, image_width * 0.20));
}

double side_segment_support(const CourtLineSegment& line, const cv::Point2f& p_top,
                            const cv::Point2f& p_bottom, int image_height) {
    const double y1 = line.points[1], y2 = line.points[3];
    const double seg_min = std::min(y1, y2), seg_max = std::max(y1, y2);
    const double quad_min = std::min(static_cast<double>(p_top.y), static_cast<double>(p_bottom.y));
    const double quad_max = std::max(static_cast<double>(p_top.y), static_cast<double>(p_bottom.y));
    const double overshoot = std::max(0.0, seg_min - quad_min) + std::max(0.0, quad_max - seg_max);
    return std::max(0.0, 1.0 - overshoot / std::max(1.0, image_height * 0.16));
}

// Python _reference_segment_support: sample the line against the distance map.
double reference_segment_support(const CourtLineSegment& line,
                                 const std::optional<CourtLineSupport>& line_support,
                                 cv::Size image_shape) {
    if (!line_support) {
        return 0.0;
    }
    const int height = image_shape.height;
    const int width = image_shape.width;
    const cv::Point2f start(static_cast<float>(line.points[0]), static_cast<float>(line.points[1]));
    const cv::Point2f end(static_cast<float>(line.points[2]), static_cast<float>(line.points[3]));
    // Python: float(np.hypot(x2 - x1, y2 - y1)) — ints, so float64 hypot.
    const double length =
        std::hypot(static_cast<double>(line.points[2] - line.points[0]),
                   static_cast<double>(line.points[3] - line.points[1]));
    if (length < 1.0) {
        return 0.0;
    }

    // Python's twin sampler uses length / 4 (reference.py uses / 5).
    auto samples = sample_line_points(start, end, width, height, length, 4.0);
    if (!samples) {
        return 0.0;  // also covers np.mean(valid) < 0.35
    }

    const cv::Mat& distance_map = line_support->distance_map;
    const double tolerance = line_support->tolerance;
    double score_sum = 0.0;
    int coverage_count = 0;
    for (const cv::Point& s : *samples) {
        const float dist = distance_map.at<float>(s.y, s.x);
        score_sum += 1.0 - std::min(std::max(static_cast<double>(dist) / tolerance, 0.0), 1.0);
        if (dist <= tolerance) {
            ++coverage_count;
        }
    }
    const double mean_score = score_sum / static_cast<double>(samples->size());
    const double coverage = static_cast<double>(coverage_count) / static_cast<double>(samples->size());
    return 0.65 * mean_score + 0.35 * coverage;
}

}  // namespace

std::array<CourtLineSegment, 4> promote_far_baseline(
    const std::array<CourtLineSegment, 4>& lines,
    const std::vector<CourtLineSegment>& horizontal_lines, cv::Size image_size) {
    const int height = image_size.height;
    const CourtLineSegment& top_line = lines[0];
    const CourtLineSegment& bottom_line = lines[1];
    const CourtLineSegment& left_line = lines[2];
    const CourtLineSegment& right_line = lines[3];
    const double top_y = static_cast<double>(top_line.mid.y);

    std::vector<const CourtLineSegment*> candidates;
    for (const auto& line : horizontal_lines) {
        const double candidate_y = static_cast<double>(line.mid.y);
        const double gap = top_y - candidate_y;
        if (gap <= height * 0.04 || gap >= height * 0.09) {
            continue;
        }
        if (candidate_y < height * 0.28) {
            continue;
        }
        if (line.length < top_line.length * 0.65) {
            continue;
        }
        // next_gap = min(gap of every horizontal line below top) or height
        bool any_below = false;
        double next_gap = static_cast<double>(height);
        for (const auto& other : horizontal_lines) {
            const double other_y = static_cast<double>(other.mid.y);
            if (other_y > top_y) {
                const double g = other_y - top_y;
                if (!any_below || g < next_gap) {
                    next_gap = g;
                    any_below = true;
                }
            }
        }
        if (gap > next_gap * 0.9) {
            continue;
        }
        candidates.push_back(&line);
    }

    if (candidates.empty()) {
        return lines;
    }
    // Python: max(candidates, key=length) — first maximal wins (also true of
    // std::max_element with a strict-greater comparison).
    const CourtLineSegment* promoted_top = candidates[0];
    for (const auto* c : candidates) {
        if (c->length > promoted_top->length) {
            promoted_top = c;
        }
    }
    return {*promoted_top, bottom_line, left_line, right_line};
}

std::vector<CourtLineSegment> dedupe_lines(std::vector<CourtLineSegment> lines,
                                           bool horizontal, int max_count) {
    // Python: sorted(lines, key=length, reverse=True) — stable.
    std::stable_sort(lines.begin(), lines.end(),
                     [](const CourtLineSegment& a, const CourtLineSegment& b) {
                         return a.length > b.length;
                     });

    std::vector<CourtLineSegment> selected;
    selected.reserve(static_cast<size_t>(max_count));
    for (const auto& line : lines) {
        bool duplicate = false;
        for (const auto& existing : selected) {
            bool same_band = false, same_angle = false;
            if (horizontal) {
                same_band = std::abs(line.mid.y - existing.mid.y) < 18.f;
                same_angle = std::abs(line.angle - existing.angle) < 8.0;
            } else {
                same_band = std::abs(line.mid.x - existing.mid.x) < 22.f;
                same_angle = std::abs(line.angle - existing.angle) < 10.0;
            }
            if (same_band && same_angle) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            selected.push_back(line);
        }
        if (static_cast<int>(selected.size()) >= max_count) {
            break;
        }
    }
    return selected;
}

std::optional<QuadScore> score_court_quad(
    const std::vector<cv::Point2f>& corners, cv::Size image_size,
    const std::array<CourtLineSegment, 4>& lines, const cv::Mat& line_mask,
    const std::vector<CourtLineSegment>& horizontal_lines,
    const BadmintonCourtReference* court_reference,
    const std::optional<CourtLineSupport>& line_support) {
    const int height = image_size.height;
    const int width = image_size.width;
    const cv::Point2f& top_left = corners[0];
    const cv::Point2f& top_right = corners[1];
    const cv::Point2f& bottom_right = corners[2];
    const cv::Point2f& bottom_left = corners[3];
    const CourtLineSegment& top_line = lines[0];
    const CourtLineSegment& bottom_line = lines[1];
    const CourtLineSegment& left_line = lines[2];
    const CourtLineSegment& right_line = lines[3];

    // --- validity gates -----------------------------------------------------
    for (const auto& point : corners) {
        if (!inside_image(point, width, height, /*margin=*/8)) {
            return std::nullopt;
        }
    }
    if (!is_convex_quad(corners)) {
        return std::nullopt;
    }

    const double area = polygon_area(corners);
    const double image_area = width * height;
    if (area < image_area * 0.12) {
        return std::nullopt;
    }

    const double top_width = std::hypot(static_cast<double>(top_right.x) - top_left.x,
                                        static_cast<double>(top_right.y) - top_left.y);
    const double bottom_width = std::hypot(static_cast<double>(bottom_right.x) - bottom_left.x,
                                           static_cast<double>(bottom_right.y) - bottom_left.y);
    const double left_height = std::hypot(static_cast<double>(bottom_left.x) - top_left.x,
                                          static_cast<double>(bottom_left.y) - top_left.y);
    const double right_height = std::hypot(static_cast<double>(bottom_right.x) - top_right.x,
                                           static_cast<double>(bottom_right.y) - top_right.y);
    const double avg_height = (left_height + right_height) / 2.0;
    if (std::min(std::min(top_width, bottom_width), avg_height) <= 1) {
        return std::nullopt;
    }
    if (avg_height < height * 0.34) {
        return std::nullopt;
    }
    if (bottom_width < top_width * 0.82) {
        return std::nullopt;
    }

    const double center_x = (top_left.x + top_right.x + bottom_right.x + bottom_left.x) / 4.0;
    const double center_y = (top_left.y + top_right.y + bottom_right.y + bottom_left.y) / 4.0;
    const double top_y = (top_left.y + top_right.y) / 2.0;
    const double bottom_y = (bottom_left.y + bottom_right.y) / 2.0;
    const double width_ratio = top_width / bottom_width;
    if (!(width * 0.42 <= center_x && center_x <= width * 0.76 &&
          height * 0.58 <= center_y && center_y <= height * 0.86)) {
        return std::nullopt;
    }
    if (top_y < height * 0.30 || bottom_y < height * 0.74) {
        return std::nullopt;
    }
    if (!(0.45 <= width_ratio && width_ratio <= 0.86)) {
        return std::nullopt;
    }

    float min_px = corners[0].x, max_px = corners[0].x;
    float min_py = corners[0].y, max_py = corners[0].y;
    for (const auto& p : corners) {
        min_px = std::min(min_px, p.x);
        max_px = std::max(max_px, p.x);
        min_py = std::min(min_py, p.y);
        max_py = std::max(max_py, p.y);
    }
    const double min_edge_distance = std::min(
        std::min(static_cast<double>(min_px), width - 1 - static_cast<double>(max_px)),
        std::min(static_cast<double>(min_py), height - 1 - static_cast<double>(max_py)));
    const double edge_limit = std::max(10.0, std::min(width, height) * 0.02);
    if (min_edge_distance < edge_limit) {
        return std::nullopt;
    }

    const double left_dx = bottom_left.x - top_left.x;
    const double right_dx = bottom_right.x - top_right.x;
    if (left_dx > width * 0.16 || right_dx < -width * 0.16) {
        return std::nullopt;
    }

    // --- 14 heuristic terms, summed in Python's exact order -----------------
    const double area_score = std::min(area / (image_area * 0.42), 1.0) * 44;
    const double height_score = std::min(avg_height / (height * 0.62), 1.0) * 22;
    const double perspective_score =
        (1.0 - std::min(std::abs(width_ratio - 0.66) / 0.22, 1.0)) * 16;
    const double center_x_score =
        (1.0 - std::min(std::abs(center_x - width * 0.50) / (width * 0.24), 1.0)) * 18;
    const double center_y_score =
        (1.0 - std::min(std::abs(center_y - height * 0.70) / (height * 0.20), 1.0)) * 8;
    const double bottom_score =
        std::min(std::max((bottom_y / height - 0.74) / 0.18, 0.0), 1.0) * 8;
    const double edge_score =
        std::min(min_edge_distance / (std::min(width, height) * 0.08), 1.0) * 6;
    const double line_score =
        std::min((top_line.length + bottom_line.length + left_line.length + right_line.length) /
                     (width * 2.0 + height),
                 1.0) * 6;
    const double horizontal_support =
        (horizontal_segment_support(top_line, top_left, top_right, width) +
         horizontal_segment_support(bottom_line, bottom_left, bottom_right, width)) /
        2.0;
    const double side_support =
        (side_segment_support(left_line, top_left, bottom_left, height) +
         side_segment_support(right_line, top_right, bottom_right, height)) /
        2.0;
    const double support_score = (horizontal_support * 0.55 + side_support * 0.45) * 24;

    const double top_span_min = std::min(static_cast<double>(top_line.points[0]),
                                         static_cast<double>(top_line.points[2]));
    const double top_span_max = std::max(static_cast<double>(top_line.points[0]),
                                         static_cast<double>(top_line.points[2]));
    const double bottom_span_min = std::min(static_cast<double>(bottom_line.points[0]),
                                            static_cast<double>(bottom_line.points[2]));
    const double bottom_span_max = std::max(static_cast<double>(bottom_line.points[0]),
                                            static_cast<double>(bottom_line.points[2]));
    const double top_endpoint_alignment =
        1.0 - std::min((std::abs(static_cast<double>(top_left.x) - top_span_min) +
                        std::abs(static_cast<double>(top_right.x) - top_span_max)) /
                           std::max(width * 0.34, 1.0),
                       1.0);
    const double bottom_endpoint_alignment =
        1.0 - std::min((std::abs(static_cast<double>(bottom_left.x) - bottom_span_min) +
                        std::abs(static_cast<double>(bottom_right.x) - bottom_span_max)) /
                           std::max(width * 0.34, 1.0),
                       1.0);
    const double endpoint_alignment_value = (top_endpoint_alignment + bottom_endpoint_alignment) / 2.0;
    const double endpoint_alignment_score = endpoint_alignment_value * 44;

    const double left_reference_support = reference_segment_support(left_line, line_support, image_size);
    const double right_reference_support = reference_segment_support(right_line, line_support, image_size);
    const double clean_side_support_value = std::min(left_reference_support, right_reference_support);
    const double clean_side_support_score = std::min(1.0, clean_side_support_value / 0.75) * 42;

    const double alignment_value = badminton_line_alignment_score(corners, line_mask);
    const double pattern_value =
        badminton_horizontal_pattern_score(corners, horizontal_lines, image_size);
    double reference_value = 0.0;
    std::map<std::string, double> reference_details;
    if (court_reference != nullptr) {
        auto scored = court_reference->score_line_support(corners, line_support, image_size);
        reference_value = scored.first;
        reference_details = std::move(scored.second);
    }

    const double alignment_score = alignment_value * 10;
    const double pattern_score = pattern_value * 110;
    const double reference_score = reference_value * 18;
    const double total_score = area_score + height_score + perspective_score + center_x_score +
                               center_y_score + bottom_score + edge_score + line_score +
                               support_score + endpoint_alignment_score +
                               clean_side_support_score + alignment_score + pattern_score +
                               reference_score;

    // Debug details: Python round(float(x), 4). ponytail: nearbyint(v*1e4)/1e4
    // instead of Python's correctly-rounded decimal round — debug-only keys,
    // never fed back into scoring; flips nothing (ceiling: display parity).
    auto round4 = [](double v) { return std::nearbyint(v * 1e4) / 1e4; };
    std::map<std::string, double> details = {
        {"alignment_score", round4(alignment_value)},
        {"horizontal_pattern_score", round4(pattern_value)},
        {"endpoint_alignment_score", round4(endpoint_alignment_value)},
        {"clean_side_support_score", round4(clean_side_support_value)},
    };
    // Python: details = {..., **reference_details}
    details.insert(reference_details.begin(), reference_details.end());

    return QuadScore{total_score, std::move(details)};
}

}  // namespace court_detail
}  // namespace gb
