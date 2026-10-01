// Port of badminton_analysis/court/reference.py — court model projection and
// line-support scoring. Keep formulas term-for-term identical to Python.
#include "gb/court.h"

#include "court_internal.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace gb {

BadmintonCourtReference::BadmintonCourtReference() {
    const double width = BADMINTON_COURT_WIDTH;
    const double length = BADMINTON_COURT_LENGTH;
    const double singles_margin = BADMINTON_SINGLES_MARGIN;
    const double net_y = length / 2.0;
    const double service_top = net_y - BADMINTON_SERVICE_LINE_FROM_NET;
    const double service_bottom = net_y + BADMINTON_SERVICE_LINE_FROM_NET;
    const double back_service_top = BADMINTON_BACK_SERVICE_OFFSET;
    const double back_service_bottom = length - BADMINTON_BACK_SERVICE_OFFSET;
    const double center_x = width / 2.0;

    court_corners_ = {
        {0.f, 0.f},
        {static_cast<float>(width), 0.f},
        {static_cast<float>(width), static_cast<float>(length)},
        {0.f, static_cast<float>(length)},
    };

    // Python self.lines: ((start), (end), weight)
    // ponytail: explicit seg() helper instead of a nested-brace initializer
    // list — MSVC rejects copy-list-init of std::array<Point2f,2> elements
    // here; output semantics identical (ceiling: none, same values).
    auto seg = [](const cv::Point2f& a, const cv::Point2f& b) {
        return std::array<cv::Point2f, 2>{a, b};
    };
    const float w = static_cast<float>(width);
    const float len = static_cast<float>(length);
    const float sm = static_cast<float>(singles_margin);
    const float bst = static_cast<float>(back_service_top);
    const float bsb = static_cast<float>(back_service_bottom);
    const float st = static_cast<float>(service_top);
    const float sb = static_cast<float>(service_bottom);
    const float cx = static_cast<float>(center_x);
    const float ny = static_cast<float>(net_y);
    lines_ = {
        seg({0.f, 0.f}, {w, 0.f}),
        seg({w, 0.f}, {w, len}),
        seg({w, len}, {0.f, len}),
        seg({0.f, len}, {0.f, 0.f}),
        seg({sm, 0.f}, {sm, len}),
        seg({w - sm, 0.f}, {w - sm, len}),
        seg({0.f, bst}, {w, bst}),
        seg({0.f, bsb}, {w, bsb}),
        seg({0.f, st}, {w, st}),
        seg({0.f, sb}, {w, sb}),
        seg({cx, 0.f}, {cx, st}),
        seg({cx, sb}, {cx, len}),
        seg({0.f, ny}, {w, ny}),
    };
    weights_ = {1.35, 1.35, 1.35, 1.35, 0.95, 0.95, 1.0, 1.0, 1.15, 1.15, 0.8, 0.8, 0.35};
}

std::optional<CourtLineSupport> BadmintonCourtReference::prepare_line_support(
    const cv::Mat& line_mask, cv::Size image_size) const {
    if (line_mask.empty()) {  // Python: if line_mask is None
        return std::nullopt;
    }

    const int height = image_size.height;
    const int width = image_size.width;
    cv::Mat support_mask;
    // Python: (line_mask > 0).astype(np.uint8) * 255 — THRESH_BINARY with
    // thresh=0, maxval=255 is identical for any input value.
    cv::threshold(line_mask, support_mask, 0, 255, cv::THRESH_BINARY);
    if (support_mask.rows != height || support_mask.cols != width) {
        cv::resize(support_mask, support_mask, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
    }

    CourtLineSupport out;
    cv::bitwise_not(support_mask, support_mask);  // Python: 255 - support_mask
    cv::distanceTransform(support_mask, out.distance_map, cv::DIST_L2, 3);
    out.tolerance = std::max(5.0, std::min(width, height) * 0.013);
    return out;
}

std::map<std::string, double> BadmintonCourtReference::empty_details(double tolerance) {
    return {
        {"reference_score", 0.0},
        {"reference_coverage", 0.0},
        {"reference_supported_lines", 0.0},
        {"reference_tolerance_px", std::nearbyint(tolerance * 100.0) / 100.0},  // round(..., 2)
    };
}

std::pair<double, std::map<std::string, double>> BadmintonCourtReference::score_line_support(
    const std::vector<cv::Point2f>& image_corners,
    const std::optional<CourtLineSupport>& line_support,
    cv::Size image_size) const {
    if (!line_support) {  // Python: line_support is None
        return {0.0, empty_details(0.0)};
    }
    if (image_corners.size() != 4) {  // Python: corners.shape != (4, 2)
        return {0.0, empty_details(0.0)};
    }

    const int height = image_size.height;
    const int width = image_size.width;
    cv::Mat matrix = cv::getPerspectiveTransform(court_corners_, image_corners);
    const cv::Mat& distance_map = line_support->distance_map;
    const double tolerance = line_support->tolerance;

    double weighted_score = 0.0;
    double total_weight = 0.0;
    std::vector<double> coverage_values;
    int supported_lines = 0;

    for (size_t i = 0; i < lines_.size(); ++i) {
        const auto& line = lines_[i];
        const double weight = weights_[i];
        auto [start_image, end_image] = project_line(matrix, line[0], line[1]);
        // Python: float(np.linalg.norm(end - start)) — float32 norm.
        auto samples = court_detail::sample_line_points(
            start_image, end_image, width, height,
            static_cast<double>(court_detail::norm_f32(start_image, end_image)), 5.0);
        if (!samples) {
            continue;
        }

        double line_score_sum = 0.0;
        int coverage_count = 0;
        for (const cv::Point& s : *samples) {
            const float dist = distance_map.at<float>(s.y, s.x);
            double line_score = 1.0 - std::min(std::max(static_cast<double>(dist) / tolerance, 0.0), 1.0);
            line_score_sum += line_score;
            if (dist <= tolerance) {
                ++coverage_count;
            }
        }
        const double line_score = line_score_sum / static_cast<double>(samples->size());
        const double coverage = static_cast<double>(coverage_count) / static_cast<double>(samples->size());
        weighted_score += weight * (0.68 * line_score + 0.32 * coverage);
        total_weight += weight;
        coverage_values.push_back(coverage);
        if (coverage >= 0.42) {
            ++supported_lines;
        }
    }

    if (total_weight <= 0) {
        return {0.0, empty_details(tolerance)};
    }

    double score = weighted_score / total_weight;
    score = std::min(std::max(score, 0.0), 1.0);  // np.clip(..., 0.0, 1.0)
    auto round4 = [](double v) { return std::nearbyint(v * 1e4) / 1e4; };
    const double mean_coverage =
        coverage_values.empty()
            ? 0.0
            : std::accumulate(coverage_values.begin(), coverage_values.end(), 0.0) /
                  static_cast<double>(coverage_values.size());
    std::map<std::string, double> details = {
        {"reference_score", round4(score)},
        {"reference_coverage", round4(mean_coverage)},
        {"reference_supported_lines", static_cast<double>(supported_lines)},
        {"reference_tolerance_px", std::nearbyint(tolerance * 100.0) / 100.0},
    };
    return {score, details};
}

std::vector<CourtProjectedLine> BadmintonCourtReference::project_lines(
    const std::vector<cv::Point2f>& image_corners) const {
    std::vector<CourtProjectedLine> out;
    if (image_corners.size() != 4) {  // Python: corners.shape != (4, 2)
        return out;
    }
    cv::Mat matrix = cv::getPerspectiveTransform(court_corners_, image_corners);
    out.reserve(lines_.size());
    for (size_t i = 0; i < lines_.size(); ++i) {
        auto [start_image, end_image] = project_line(matrix, lines_[i][0], lines_[i][1]);
        out.push_back({start_image, end_image, weights_[i]});
    }
    return out;
}

std::pair<cv::Point2f, cv::Point2f> BadmintonCourtReference::project_line(
    const cv::Mat& matrix, const cv::Point2f& start, const cv::Point2f& end) const {
    // Python: points = np.array([[start, end]], float32); perspectiveTransform(...)[0]
    std::vector<cv::Point2f> pts = {start, end};
    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(pts, projected, matrix);
    return {projected[0], projected[1]};
}

namespace court_detail {

float norm_f32(const cv::Point2f& start, const cv::Point2f& end) {
    // np.linalg.norm(float32 pair): float32 dx*dx + dy*dy, float32 sqrt.
    const float dx = end.x - start.x;
    const float dy = end.y - start.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Python BadmintonCourtReference._sample_line (shared with detector's twin
// sampler; length computed by the caller, length_divisor: 5 in reference.py,
// 4 in detector.py).
std::optional<std::vector<cv::Point>> sample_line_points(
    const cv::Point2f& start, const cv::Point2f& end, int width, int height,
    double length, double length_divisor) {
    // Python: if length < 1.0: return None (length computed by caller).
    if (length < 1.0) {
        return std::nullopt;
    }

    // np.linspace(float(s), float(e), sample_count) — exact same grid.
    const int sample_count = std::max(18, static_cast<int>(length / length_divisor));
    const double x1 = start.x, y1 = start.y, x2 = end.x, y2 = end.y;
    std::vector<double> xs(sample_count), ys(sample_count);
    if (sample_count == 1) {
        xs[0] = x1;
        ys[0] = y1;
    } else {
        const double step_x = (x2 - x1) / (sample_count - 1);
        const double step_y = (y2 - y1) / (sample_count - 1);
        for (int i = 0; i < sample_count; ++i) {
            xs[i] = x1 + i * step_x;
            ys[i] = y1 + i * step_y;
        }
    }

    int valid_count = 0;
    for (int i = 0; i < sample_count; ++i) {
        if (xs[i] >= 0 && xs[i] < width && ys[i] >= 0 && ys[i] < height) {
            ++valid_count;
        }
    }
    if (static_cast<double>(valid_count) / sample_count < 0.35) {  // np.mean(valid) < 0.35
        return std::nullopt;
    }

    std::vector<cv::Point> out;
    out.reserve(sample_count);
    for (int i = 0; i < sample_count; ++i) {
        if (!(xs[i] >= 0 && xs[i] < width && ys[i] >= 0 && ys[i] < height)) {
            continue;
        }
        // np.clip(np.rint(x), 0, width - 1); np.rint == nearbyint (half-to-even).
        const int ix = std::min(std::max(static_cast<int>(std::nearbyint(xs[i])), 0), width - 1);
        const int iy = std::min(std::max(static_cast<int>(std::nearbyint(ys[i])), 0), height - 1);
        out.emplace_back(ix, iy);
    }
    return out;
}
}  // namespace court_detail

}  // namespace gb
