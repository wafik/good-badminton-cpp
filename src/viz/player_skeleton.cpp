// Port of visualization/player_pose.py draw side: _draw_skeleton_on_frame +
// draw_players dots/trails (detect side lives in src/detect/pose_analyzer.cpp).
#include "gb/viz.h"

namespace gb {
namespace {

// Python PlayerPoseVisualizer.skeleton_connections — 12 COCO-17 limbs.
constexpr int kLimbs[12][2] = {
    {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 11},
    {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16},
};

// Python limb color (255,191,0), joint color (255,128,0).
const cv::Scalar kLimbColor(255, 191, 0);
const cv::Scalar kJointColor(255, 128, 0);
const cv::Scalar kUpperColor(0, 255, 255);
const cv::Scalar kLowerColor(255, 0, 255);

}  // namespace

void PlayerSkeletonRenderer::draw(
    cv::Mat& frame, const std::optional<PoseDrawData>& pose,
    const std::map<std::string, std::optional<cv::Point2f>>& players,
    const std::map<std::string, std::deque<cv::Point2f>>& history) {
    if (frame.empty()) return;

    // Python: if show_skeletons and current_pose_data is not None
    if (show_skeletons && pose) {
        for (const auto& person : pose->keypoints) {
            const int keypoint_count = static_cast<int>(person.size());
            for (const auto& limb : kLimbs) {
                const int a = limb[0];
                const int b = limb[1];
                if (a >= keypoint_count || b >= keypoint_count) continue;
                const cv::Point2f& p1 = person[a];
                const cv::Point2f& p2 = person[b];
                // Python: both endpoints gated on x>1 and y>1
                if (p1.x > 1 && p1.y > 1 && p2.x > 1 && p2.y > 1) {
                    // FULL-frame keypoints (types.h): Python adds offset_x/y
                    // here because it stores ROI-relative — detect already
                    // added them at store time, so draw as-is. int() truncates
                    // toward zero like Python's int().
                    cv::line(frame,
                             cv::Point(static_cast<int>(p1.x),
                                       static_cast<int>(p1.y)),
                             cv::Point(static_cast<int>(p2.x),
                                       static_cast<int>(p2.y)),
                             kLimbColor, 2, cv::LINE_AA);
                }
            }
            for (int i = 0; i < keypoint_count; ++i) {
                const cv::Point2f& p = person[i];
                if (p.x > 1 && p.y > 1) {
                    cv::circle(frame,
                               cv::Point(static_cast<int>(p.x),
                                         static_cast<int>(p.y)),
                               3, kJointColor, -1, cv::LINE_AA);
                }
            }
        }
    }

    // Python draw_players slot order: ["upper", "lower", "upper2", "lower2"]
    for (const auto& slot : SLOTS) {
        auto pit = players.find(slot);
        if (pit == players.end() || !pit->second.has_value()) continue;

        // Python: color = (0,255,255) if startswith("upper") else (255,0,255)
        const bool is_upper = slot.rfind("upper", 0) == 0;
        const cv::Scalar color = is_upper ? kUpperColor : kLowerColor;

        const cv::Point2f& cur = *pit->second;
        cv::circle(frame,
                   cv::Point(static_cast<int>(cur.x), static_cast<int>(cur.y)),
                   5, color, -1, cv::LINE_AA);

        if (!show_player_trajectories) continue;
        auto hit = history.find(slot);
        if (hit == history.end()) continue;
        const auto& hist = hit->second;
        const size_t n = hist.size();
        if (n == 0) continue;  // Python: loop over empty list draws nothing
        for (size_t i = 0; i < n; ++i) {
            // Python: radius = int(2 + (i / len(history)) * 3)
            const int radius = static_cast<int>(
                2 + (static_cast<double>(i) / static_cast<double>(n)) * 3);
            const cv::Point2f& pos = hist[i];
            // Python also skips pos is None — image history never holds None.
            cv::circle(frame,
                       cv::Point(static_cast<int>(pos.x),
                                 static_cast<int>(pos.y)),
                       radius, color, -1, cv::LINE_AA);
        }
    }
    // ponytail: Python draw_players then calls stats_visualizer.draw_player_stats
    // — split in C++ (system.cpp invokes StatsVisualizer separately), same order.
}

}  // namespace gb
