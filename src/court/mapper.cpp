// Port of badminton_analysis/court/mapper.py (headless path): CourtMapper,
// compute_expanded_roi, resolve_court_corners. Skips the interactive
// annotate_court mouse GUI / draw_guidance / render_annotation_view /
// show_auto_roi and the __main__ demo.
#include "gb/court.h"

#include <algorithm>
#include <climits>
#include <cmath>

namespace gb {

namespace {

// np.round(value, 2) for float32 (nearbyint == np.rint, half-to-even).
float round2(float v) { return std::nearbyint(v * 100.f) / 100.f; }

}  // namespace

CourtMapper::CourtMapper(const std::vector<cv::Point2f>& image_court_corners,
                         const cv::Size2f& court_dimensions)
    : corners_(image_court_corners), dims_(court_dimensions) {
    // Python: court_points = [[0,0],[W,0],[W,H],[0,H]] float32;
    // matrix = getPerspectiveTransform(image_corners, court_points) (image->court);
    // inv_matrix = getPerspectiveTransform(court_points, image_corners).
    const float w = dims_.width;   // court_dimensions[0] = court width (x, meters)
    const float h = dims_.height;  // court_dimensions[1] = court length (y, meters)
    const std::vector<cv::Point2f> court_points = {
        {0.f, 0.f}, {w, 0.f}, {w, h}, {0.f, h}};
    matrix_ = cv::getPerspectiveTransform(corners_, court_points);
    inv_matrix_ = cv::getPerspectiveTransform(court_points, corners_);
    compute_court_overlay();
}

std::optional<cv::Point2f> CourtMapper::image_to_court(const cv::Point2f& point) const {
    // Python: perspectiveTransform(point.reshape(-1,1,2), matrix)[0][0],
    // np.round(..., 2). Python returned [] only for invalid input types,
    // which a cv::Point2f cannot express -> always has_value.
    std::vector<cv::Point2f> in = {point};
    std::vector<cv::Point2f> out;
    cv::perspectiveTransform(in, out, matrix_);
    return cv::Point2f(round2(out[0].x), round2(out[0].y));
}

cv::Point2f CourtMapper::court_to_image(const cv::Point2f& point) const {
    // Python transforms the whole batch but returns transformed[0][0] only.
    std::vector<cv::Point2f> in = {point};
    std::vector<cv::Point2f> out;
    cv::perspectiveTransform(in, out, inv_matrix_);
    return cv::Point2f(round2(out[0].x), round2(out[0].y));
}

std::pair<cv::Point2f, cv::Point2f> CourtMapper::line_to_image(
    const cv::Point2f& start, const cv::Point2f& end) const {
    return {court_to_image(start), court_to_image(end)};
}

void CourtMapper::compute_court_overlay() {
    const double width = dims_.width;
    const double length = dims_.height;
    const double scale_x = width / BADMINTON_COURT_WIDTH;
    const double scale_y = length / BADMINTON_COURT_LENGTH;
    const double singles_margin = BADMINTON_SINGLES_MARGIN * scale_x;
    const double net_y = length / 2.0;
    const double service_top = net_y - BADMINTON_SERVICE_LINE_FROM_NET * scale_y;
    const double service_bottom = net_y + BADMINTON_SERVICE_LINE_FROM_NET * scale_y;
    const double back_service_top = BADMINTON_BACK_SERVICE_OFFSET * scale_y;
    const double back_service_bottom = length - BADMINTON_BACK_SERVICE_OFFSET * scale_y;
    const double center_x = width / 2.0;

    auto f = [](double v) { return static_cast<float>(v); };

    vertical_lines_ = {
        line_to_image({f(singles_margin), 0.f}, {f(singles_margin), f(length)}),
        line_to_image({f(width - singles_margin), 0.f}, {f(width - singles_margin), f(length)}),
        line_to_image({f(center_x), 0.f}, {f(center_x), f(service_top)}),
        line_to_image({f(center_x), f(service_bottom)}, {f(center_x), f(length)}),
    };
    horizontal_lines_ = {
        line_to_image({0.f, f(back_service_top)}, {f(width), f(back_service_top)}),
        line_to_image({0.f, f(service_top)}, {f(width), f(service_top)}),
        line_to_image({0.f, f(net_y)}, {f(width), f(net_y)}),
        line_to_image({0.f, f(service_bottom)}, {f(width), f(service_bottom)}),
        line_to_image({0.f, f(back_service_bottom)}, {f(width), f(back_service_bottom)}),
    };

    // Python: mid_height = int((left_mid_image[1] + right_mid_image[1]) / 2)
    const cv::Point2f left_mid = court_to_image(cv::Point2f(0.f, f(net_y)));
    const cv::Point2f right_mid = court_to_image(cv::Point2f(f(width), f(net_y)));
    mid_height_ = static_cast<int>((left_mid.y + right_mid.y) / 2);  // int() truncates
}

int CourtMapper::draw_court_overlay(cv::Mat& image) const {
    // Python: overlay = image.copy(); draws on the copy and returns
    // (overlay, mid_height). Per court.h this draws in place; callers wanting
    // Python's non-mutating behavior should clone first.
    auto trunc_pt = [](const cv::Point2f& p) {
        return cv::Point(static_cast<int>(p.x), static_cast<int>(p.y));  // astype(int)
    };

    std::vector<cv::Point> quad;
    quad.reserve(corners_.size());
    for (const auto& p : corners_) {
        quad.push_back(trunc_pt(p));
    }
    cv::polylines(image, std::vector<std::vector<cv::Point>>{quad}, true,
                  cv::Scalar(0, 255, 0), 2);

    for (const auto& seg : vertical_lines_) {
        cv::line(image, trunc_pt(seg.first), trunc_pt(seg.second), cv::Scalar(0, 255, 0), 1);
    }
    for (const auto& seg : horizontal_lines_) {
        cv::line(image, trunc_pt(seg.first), trunc_pt(seg.second), cv::Scalar(0, 255, 0), 1);
    }
    return mid_height_;
}

RoiCorners compute_expanded_roi(const std::vector<cv::Point2f>& court_corners,
                                cv::Size image_size) {
    const int height = image_size.height;
    const int width = image_size.width;

    // Python: np.array(court_corners, dtype=np.int32) — truncates toward zero.
    int min_x = INT_MAX, max_x = INT_MIN, min_y = INT_MAX, max_y = INT_MIN;
    for (const auto& p : court_corners) {
        const int x = static_cast<int>(p.x);
        const int y = static_cast<int>(p.y);
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
    }

    const int court_width = max_x - min_x;
    const int pad_x = std::max(12, static_cast<int>(court_width * 0.08));

    const int x1 = std::max(0, min_x - pad_x);
    const int y1 = 0;
    const int x2 = std::min(width - 1, max_x + pad_x);
    const int y2 = height - 1;
    return RoiCorners{cv::Point2i(x1, y1), cv::Point2i(x2, y2)};
}

CourtResolution resolve_court_corners(
    const cv::Mat& image,
    const std::optional<std::vector<cv::Point2f>>& manual_corners) {
    CourtResolution fail;
    if (image.empty()) {  // Python: not isinstance(image, np.ndarray)
        return fail;
    }

    const int original_height = image.rows;
    const int original_width = image.cols;
    std::vector<cv::Point2f> corners;

    if (manual_corners && manual_corners->size() == 4) {
        // Python: corners = [(int(x), int(y)) for x, y in manual_corners]
        corners.reserve(4);
        for (const auto& p : *manual_corners) {
            corners.emplace_back(static_cast<float>(static_cast<int>(p.x)),
                                 static_cast<float>(static_cast<int>(p.y)));
        }
    } else {
        // Python: fixed_size = (1080, 720); auto-detect, scale back with int().
        cv::Mat base_image;
        cv::resize(image, base_image, cv::Size(1080, 720));
        CourtDetection detection = auto_detect_court_corners(base_image);
        if (!detection.corners) {
            return fail;
        }
        const double scale_x = static_cast<double>(original_width) / 1080.0;
        const double scale_y = static_cast<double>(original_height) / 720.0;
        corners.reserve(detection.corners->size());
        for (const auto& p : *detection.corners) {
            corners.emplace_back(
                static_cast<float>(static_cast<int>(p.x * scale_x)),
                static_cast<float>(static_cast<int>(p.y * scale_y)));
        }
    }

    CourtResolution out;
    out.roi = compute_expanded_roi(corners, image.size());  // Python: image.shape
    // Python builds a CourtMapper, draws on a *copy* (discarded) and reads
    // mid_height off the result; we read the precomputed value directly so
    // the caller's image stays unmodified (resolve never used the overlay).
    CourtMapper mapper(corners);
    out.corners = corners;
    out.mid_height = mapper.mid_height();
    return out;
}

}  // namespace gb
