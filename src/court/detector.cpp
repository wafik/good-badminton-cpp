// Port of badminton_analysis/court/detector.py (headless path): line masks,
// Hough segment extraction, quad search (auto_detect_court_corners) and
// render_auto_court_preview. Scoring lives in quad_score.cpp.
#include "gb/court.h"

#include "court_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

namespace gb {
namespace {

// Python (shared prefix of build_court_line_mask / build_reference_line_mask):
// green court region via HSV thresholds, largest connected component > 8% area.
cv::Mat green_court_region(const cv::Mat& image) {
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // green_mask = ((h>=35)&(h<=95)&(s>=30)&(v>=45)) * 255  (bounds inclusive)
    cv::Mat green_mask;
    cv::inRange(hsv, cv::Scalar(35, 30, 45), cv::Scalar(95, 255, 255), green_mask);
    cv::morphologyEx(green_mask, green_mask, cv::MORPH_OPEN,
                     cv::Mat::ones(5, 5, CV_8U), cv::Point(-1, -1), 1);
    cv::morphologyEx(green_mask, green_mask, cv::MORPH_CLOSE,
                     cv::Mat::ones(25, 25, CV_8U), cv::Point(-1, -1), 2);

    cv::Mat labels, stats, centroids;
    const int count =
        cv::connectedComponentsWithStats(green_mask, labels, stats, centroids, 8, CV_32S);
    if (count > 1) {
        const double image_area =
            static_cast<double>(image.rows) * static_cast<double>(image.cols);
        int court_idx = -1;
        int court_area = -1;
        for (int idx = 1; idx < count; ++idx) {  // Python range(1, count)
            const int area = stats.at<int>(idx, cv::CC_STAT_AREA);
            if (area > image_area * 0.08) {
                if (area > court_area) {  // Python max by CC_STAT_AREA
                    court_area = area;
                    court_idx = idx;
                }
            }
        }
        if (court_idx >= 0) {
            cv::compare(labels, court_idx, green_mask, cv::CMP_EQ);  // 0/255
        }
    }
    return green_mask;
}

// Python build_court_line_mask: white lines + Canny edges inside the court ROI.
cv::Mat build_court_line_mask(const cv::Mat& image) {
    cv::Mat green_mask = green_court_region(image);

    cv::Mat court_roi;
    const cv::Mat kernel17 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(17, 17));
    cv::dilate(green_mask, court_roi, kernel17);

    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    // white_mask = ((s <= 95) & (v >= 135)) * 255
    cv::Mat white_mask;
    cv::inRange(hsv, cv::Scalar(0, 0, 135), cv::Scalar(255, 95, 255), white_mask);
    cv::bitwise_and(white_mask, court_roi, white_mask);

    cv::Mat gray, blurred, edges;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0);
    cv::Canny(blurred, edges, 45, 135);
    cv::bitwise_and(edges, court_roi, edges);

    cv::Mat merged = white_mask | edges;
    cv::morphologyEx(merged, merged, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 5)),
                     cv::Point(-1, -1), 1);
    cv::morphologyEx(merged, merged, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)),
                     cv::Point(-1, -1), 1);
    return merged;
}

// Python build_reference_line_mask: white + yellow lines inside court ROI.
cv::Mat build_reference_line_mask(const cv::Mat& image) {
    cv::Mat green_mask = green_court_region(image);

    cv::Mat court_roi;
    const cv::Mat kernel13 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 13));
    cv::dilate(green_mask, court_roi, kernel13);

    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    cv::Mat white_mask, yellow_mask;
    // white_mask = ((s <= 90) & (v >= 145)) * 255
    cv::inRange(hsv, cv::Scalar(0, 0, 145), cv::Scalar(255, 90, 255), white_mask);
    // yellow_mask = ((h >= 12) & (h <= 42) & (s >= 45) & (v >= 120)) * 255
    cv::inRange(hsv, cv::Scalar(12, 45, 120), cv::Scalar(42, 255, 255), yellow_mask);

    cv::Mat line_mask;
    cv::bitwise_or(white_mask, yellow_mask, line_mask);
    cv::bitwise_and(line_mask, court_roi, line_mask);
    cv::morphologyEx(line_mask, line_mask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 3)),
                     cv::Point(-1, -1), 1);
    cv::morphologyEx(line_mask, line_mask, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)),
                     cv::Point(-1, -1), 1);
    return line_mask;
}

// Python detect_court_line_segments -> (horizontal, side, mask).
struct DetectedSegments {
    std::vector<CourtLineSegment> horizontal;
    std::vector<CourtLineSegment> side;
    cv::Mat mask;
};

DetectedSegments detect_court_line_segments(const cv::Mat& image) {
    const int height = image.rows;
    const int width = image.cols;
    cv::Mat mask = build_court_line_mask(image);

    const int min_line_length = std::max(46, static_cast<int>(std::min(width, height) * 0.10));
    const int max_gap = std::max(14, static_cast<int>(std::min(width, height) * 0.045));

    DetectedSegments out;
    out.mask = mask;
    std::vector<cv::Vec4i> raw_lines;
    cv::HoughLinesP(mask, raw_lines, 1, CV_PI / 180, 60, min_line_length, max_gap);
    if (raw_lines.empty()) {  // Python: raw_lines is None
        return out;
    }

    const int edge_margin = std::max(10, static_cast<int>(std::min(width, height) * 0.02));
    for (const cv::Vec4i& raw : raw_lines) {
        const int x1 = raw[0], y1 = raw[1], x2 = raw[2], y2 = raw[3];
        const double length = std::hypot(static_cast<double>(x2 - x1),
                                         static_cast<double>(y2 - y1));
        if (length < min_line_length) {
            continue;
        }
        if (std::min(x1, x2) <= edge_margin ||
            std::max(x1, x2) >= width - 1 - edge_margin ||
            std::min(y1, y2) <= edge_margin ||
            std::max(y1, y2) >= height - 1 - edge_margin) {
            continue;
        }

        CourtLineSegment segment;
        segment.points = cv::Vec4i(x1, y1, x2, y2);
        segment.length = length;
        segment.mid = cv::Point2f((x1 + x2) / 2.0f, (y1 + y2) / 2.0f);
        segment.angle = court_detail::line_angle(x1, y1, x2, y2);

        if (std::min(segment.angle, 180.0 - segment.angle) <= 16) {
            out.horizontal.push_back(segment);
        } else if (segment.angle >= 45 && segment.angle <= 135) {
            out.side.push_back(segment);
        }
    }

    out.horizontal = court_detail::dedupe_lines(std::move(out.horizontal), true, 14);
    out.side = court_detail::dedupe_lines(std::move(out.side), false, 18);
    return out;
}

}  // namespace

CourtDetection auto_detect_court_corners(const cv::Mat& image) {
    CourtDetection out;
    DetectedSegments segments = detect_court_line_segments(image);
    out.line_mask = segments.mask;
    out.debug.horizontal = segments.horizontal;
    out.debug.side = segments.side;
    // out.debug.score / details stay nullopt / empty == Python None.

    BadmintonCourtReference court_reference;
    cv::Mat reference_mask = build_reference_line_mask(image);
    std::optional<CourtLineSupport> line_support =
        court_reference.prepare_line_support(reference_mask, image.size());

    if (segments.horizontal.size() < 2 || segments.side.size() < 2) {
        return out;
    }

    const int height = image.rows;
    const int width = image.cols;

    // Python: sorted by mid[1] / mid[0] — stable sort.
    std::vector<CourtLineSegment> horizontals = segments.horizontal;
    std::stable_sort(horizontals.begin(), horizontals.end(),
                     [](const CourtLineSegment& a, const CourtLineSegment& b) {
                         return a.mid.y < b.mid.y;
                     });
    std::vector<CourtLineSegment> sides = segments.side;
    std::stable_sort(sides.begin(), sides.end(),
                     [](const CourtLineSegment& a, const CourtLineSegment& b) {
                         return a.mid.x < b.mid.x;
                     });

    struct Best {
        double score = 0.0;
        std::map<std::string, double> details;
        std::vector<cv::Point2f> corners;
        std::array<CourtLineSegment, 4> lines{};
    };
    std::optional<Best> best;

    for (size_t top_idx = 0; top_idx < horizontals.size(); ++top_idx) {
        const CourtLineSegment& top_line = horizontals[top_idx];
        for (size_t bottom_i = top_idx + 1; bottom_i < horizontals.size(); ++bottom_i) {
            const CourtLineSegment& bottom_line = horizontals[bottom_i];
            if (bottom_line.mid.y - top_line.mid.y < height * 0.28) {
                continue;
            }
            for (size_t left_idx = 0; left_idx < sides.size(); ++left_idx) {
                const CourtLineSegment& left_line = sides[left_idx];
                for (size_t right_i = left_idx + 1; right_i < sides.size(); ++right_i) {
                    const CourtLineSegment& right_line = sides[right_i];
                    if (right_line.mid.x - left_line.mid.x < width * 0.22) {
                        continue;
                    }

                    // Python order: TL, TR, BR, BL.
                    auto i_tl = court_detail::line_intersection(top_line.points, left_line.points);
                    auto i_tr = court_detail::line_intersection(top_line.points, right_line.points);
                    auto i_br = court_detail::line_intersection(bottom_line.points, right_line.points);
                    auto i_bl = court_detail::line_intersection(bottom_line.points, left_line.points);
                    if (!i_tl || !i_tr || !i_br || !i_bl) {
                        continue;
                    }

                    std::vector<cv::Point2f> corners = {*i_tl, *i_tr, *i_br, *i_bl};
                    const std::array<CourtLineSegment, 4> quad_lines = {
                        top_line, bottom_line, left_line, right_line};
                    auto scored = court_detail::score_court_quad(
                        corners, image.size(), quad_lines, segments.mask,
                        segments.horizontal, &court_reference, line_support);
                    if (!scored) {
                        continue;
                    }
                    // Python: strictly greater — first (loop-order) winner on ties.
                    if (!best || scored->score > best->score) {
                        Best b;
                        b.score = scored->score;
                        b.details = std::move(scored->details);
                        b.corners = corners;
                        b.lines = quad_lines;
                        best = std::move(b);
                    }
                }
            }
        }
    }

    if (!best) {
        return out;  // corners remain nullopt; debug score/details stay None
    }

    // Python: promote a far baseline top line, recompute corners, fall back to
    // the best quad if any promoted intersection is parallel (None).
    std::array<CourtLineSegment, 4> selected =
        court_detail::promote_far_baseline(best->lines, segments.horizontal, image.size());
    std::array<std::optional<cv::Point2f>, 4> inter = {
        court_detail::line_intersection(selected[0].points, selected[2].points),
        court_detail::line_intersection(selected[0].points, selected[3].points),
        court_detail::line_intersection(selected[1].points, selected[3].points),
        court_detail::line_intersection(selected[1].points, selected[2].points),
    };
    std::vector<cv::Point2f> corners_float;
    if (inter[0] && inter[1] && inter[2] && inter[3]) {
        corners_float = {*inter[0], *inter[1], *inter[2], *inter[3]};
    } else {
        corners_float = best->corners;
        selected = best->lines;
    }

    double final_score = best->score;
    std::map<std::string, double> final_details = best->details;
    auto final_scored = court_detail::score_court_quad(
        corners_float, image.size(), selected, segments.mask, segments.horizontal,
        &court_reference, line_support);
    if (final_scored) {
        final_score = final_scored->score;
        final_details = std::move(final_scored->details);
    }
    out.debug.score = final_score;
    out.debug.details = final_details;

    // Python: [(int(round(x)), int(round(y))) ...] — round == nearbyint (half-to-even).
    std::vector<cv::Point2f> corners;
    corners.reserve(4);
    for (const auto& p : corners_float) {
        corners.emplace_back(static_cast<float>(std::nearbyint(static_cast<double>(p.x))),
                             static_cast<float>(std::nearbyint(static_cast<double>(p.y))));
    }
    out.corners = corners;
    return out;
}

cv::Mat render_auto_court_preview(const cv::Mat& image,
                                  const std::optional<std::vector<cv::Point2f>>& corners,
                                  const std::optional<RoiCorners>& roi_corners,
                                  const std::optional<CourtDetectDebug>& debug) {
    cv::Mat preview = image.clone();

    auto round_i = [](const cv::Point2f& p) {
        // np.round(start).astype(int)
        return cv::Point(static_cast<int>(std::nearbyint(static_cast<double>(p.x))),
                         static_cast<int>(std::nearbyint(static_cast<double>(p.y))));
    };
    auto trunc_i = [](const cv::Point2f& p) {
        // np.array(corners, dtype=np.int32) — truncates toward zero
        return cv::Point(static_cast<int>(p.x), static_cast<int>(p.y));
    };

    if (corners && !corners->empty()) {  // Python: if corners:
        BadmintonCourtReference court_reference;
        for (const auto& projected : court_reference.project_lines(*corners)) {
            const cv::Scalar color =
                projected.weight >= 1.3 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 220, 255);
            const int thickness = projected.weight >= 1.3 ? 3 : 2;
            cv::line(preview, round_i(projected.start), round_i(projected.end), color,
                     thickness, cv::LINE_AA);
        }

        std::vector<cv::Point> quad;
        quad.reserve(corners->size());
        for (const auto& p : *corners) {
            quad.push_back(trunc_i(p));
        }
        cv::polylines(preview, std::vector<std::vector<cv::Point>>{quad}, true,
                      cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
        for (size_t idx = 0; idx < corners->size(); ++idx) {
            const cv::Point center = trunc_i((*corners)[idx]);
            cv::circle(preview, center, 6, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
            cv::putText(preview, std::to_string(idx + 1),
                        cv::Point(center.x + 8, center.y - 8), cv::FONT_HERSHEY_SIMPLEX,
                        0.75, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }
    } else if (debug) {  // Python: elif debug: — draw the candidate segments
        for (const auto& line : debug->horizontal) {
            cv::line(preview, cv::Point(line.points[0], line.points[1]),
                     cv::Point(line.points[2], line.points[3]), cv::Scalar(0, 220, 255), 2);
        }
        for (const auto& line : debug->side) {
            cv::line(preview, cv::Point(line.points[0], line.points[1]),
                     cv::Point(line.points[2], line.points[3]), cv::Scalar(255, 180, 0), 2);
        }
    }

    if (roi_corners) {
        cv::rectangle(preview, (*roi_corners)[0], (*roi_corners)[1], cv::Scalar(255, 0, 0), 3);
    }

    cv::rectangle(preview, cv::Point(0, 0), cv::Point(preview.cols, 44),
                  cv::Scalar(0, 0, 0), -1);
    std::string detail;
    if (debug && !debug->details.empty()) {
        auto it = debug->details.find("reference_score");
        if (it != debug->details.end()) {  // Python isinstance(reference_score, (int, float))
            char buf[40];
            std::snprintf(buf, sizeof(buf), " ref=%.2f", it->second);
            detail = buf;
        }
    }
    cv::putText(preview,
                "Auto court detection" + detail + ": Enter/Y accept, M/R/Esc manual",
                cv::Point(16, 29), cv::FONT_HERSHEY_SIMPLEX, 0.72,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    return preview;
}

}  // namespace gb
