// Port of visualization/player_pose.py detection side: detect_players,
// _normalize_people, _is_on_court + __init__ court_filter_margin default.
// (draw side lives in src/viz — PlayerSkeletonRenderer.)
// ponytail: detect_players' show_performance_stats timing print omitted — the
// contract PoseAnalyzer has no such flag and system.py passes False.
#include "gb/detect.h"

#include <utility>

#include "gb/court.h"

namespace gb {

namespace {

// Python _is_on_court: mapper None -> True; else map + bounds with margin.
// Court extents hardcoded 6.1 x 13.4 in Python (not the mapper's dims_).
bool is_on_court(const cv::Point2f& image_point, const CourtMapper* mapper,
                 double margin) {
    if (mapper == nullptr) return true;
    const std::optional<cv::Point2f> cp = mapper->image_to_court(image_point);
    if (!cp.has_value()) return false;
    return cp->x >= -margin && cp->x <= 6.1 + margin &&
           cp->y >= -margin && cp->y <= 13.4 + margin;
}

}  // namespace

PoseAnalyzer::PoseAnalyzer(YoloPoseProcessor& pose, double court_filter_margin)
    : pose_(pose), margin_(court_filter_margin) {}

void PoseAnalyzer::set_court_mapper(const CourtMapper* mapper) {
    stored_mapper_ = mapper;
}

PoseObs PoseAnalyzer::detect(const cv::Mat& roi, int x1, int y1,
                             const CourtMapper* mapper) {
    PoseObs obs;
    current_.reset();

    // Python: keypoints_all, _confidence_scores = processor.process_frame(roi)
    // — scores are discarded on the Python side too.
    const std::vector<PersonKeypoints> people = pose_.process_frame(roi);
    if (people.empty()) return obs;  // Python: keypoints_all is None path

    // Python: active_court_mapper = court_mapper or self.court_mapper
    const CourtMapper* active = mapper != nullptr ? mapper : stored_mapper_;

    std::vector<std::array<cv::Point2f, 17>> full_kps;
    full_kps.reserve(people.size());

    for (const PersonKeypoints& person : people) {
        // Python _normalize_people + (ndim,17,2) shape guard are a no-op here:
        // PersonKeypoints is always exactly 17 COCO keypoints.

        const cv::Point2f& lf = person.xy[15];
        const cv::Point2f& rf = person.xy[16];
        cv::Point2f mid;
        if (lf.x > 1 && lf.y > 1 && rf.x > 1 && rf.y > 1) {
            // feet midpoint, full-frame, +10 y (shoes below the ankle point)
            mid.x = ((lf.x + x1) + (rf.x + x1)) / 2;
            mid.y = ((lf.y + y1) + (rf.y + y1)) / 2 + 10;
        } else {
            const cv::Point2f& lhip = person.xy[11];
            const cv::Point2f& rhip = person.xy[12];
            if (lhip.x > 1 && lhip.y > 1 && rhip.x > 1 && rhip.y > 1) {
                // ponytail: fall back to hip midpoint when feet are missing
                // (cropped/occluded). Hip sits ~1m up so court coords are
                // slightly short — better than dropping the player entirely.
                mid.x = ((lhip.x + x1) + (rhip.x + x1)) / 2;
                mid.y = ((lhip.y + y1) + (rhip.y + y1)) / 2;
            } else {
                continue;
            }
        }
        if (!is_on_court(mid, active, margin_)) continue;

        PersonObs po;
        po.centroid = mid;

        // hands = wrists 9/10, int-truncated + full-frame offset (Python
        // int(lh[0] + x1) — truncation of the SUM, toward zero).
        const cv::Point2f& lh = person.xy[9];
        if (lh.x > 1 && lh.y > 1) {
            po.left_hand = cv::Point2f(
                static_cast<float>(static_cast<int>(lh.x + x1)),
                static_cast<float>(static_cast<int>(lh.y + y1)));
        }
        const cv::Point2f& rh = person.xy[10];
        if (rh.x > 1 && rh.y > 1) {
            po.right_hand = cv::Point2f(
                static_cast<float>(static_cast<int>(rh.x + x1)),
                static_cast<float>(static_cast<int>(rh.y + y1)));
        }

        obs.people.push_back(po);

        // current_pose_data: full-frame keypoints + informational offsets
        // (Python stores ROI-relative + applies offsets at draw time — same
        // rendered pixels; full-frame is what types.h/viz.h contract requires).
        std::array<cv::Point2f, 17> fk;
        for (int k = 0; k < 17; ++k) {
            fk[k] = cv::Point2f(person.xy[k].x + x1, person.xy[k].y + y1);
        }
        full_kps.push_back(fk);
    }

    // Python: current_pose_data set only when filtered_people non-empty.
    if (!obs.people.empty()) {
        current_ = PoseDrawData{std::move(full_kps), x1, y1};
    }
    return obs;
}

std::optional<PoseDrawData> PoseAnalyzer::current_pose() const {
    return current_;
}

}  // namespace gb
