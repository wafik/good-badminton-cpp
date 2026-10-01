#pragma once
// Shared data types across modules. Contract owned by integration — do not
// edit from module code; extend your own module header instead.
#include <array>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace gb {

inline const std::vector<std::string> SLOTS = {"upper", "lower", "upper2", "lower2"};

// One pose observation in full-frame coords (ported from Python dicts keyed
// by centroid: parallel optionals == missing dict entries).
struct PersonObs {
    cv::Point2f centroid;
    std::optional<cv::Point2f> left_hand;
    std::optional<cv::Point2f> right_hand;
};

struct PoseObs {
    std::vector<PersonObs> people;
};

// Keypoint draw state (Python current_pose_data).
struct PoseDrawData {
    std::vector<std::array<cv::Point2f, 17>> keypoints;  // full-frame coords
    int offset_x = 0;
    int offset_y = 0;
};

// One player slot as written to detections.jsonl. Note: player image/court/
// hands are NEVER zero-null'd (only the shuttle is — see JsonlRecord.shuttle).
struct PlayerRecord {
    std::optional<cv::Point2f> image;
    std::optional<cv::Point2f> court;
    std::optional<double> speed;
    std::optional<cv::Point2f> left_hand;
    std::optional<cv::Point2f> right_hand;
};

struct SlotStats {
    double current_speed = 0;
    double rally_avg_speed = 0;
    double rally_max_speed = 0;
    double rally_distance = 0;
    double match_avg_speed = 0;
    double match_max_speed = 0;
    double match_distance = 0;
    int position_count = 0;
};

using MovementStats = std::map<std::string, SlotStats>;
using PlayersMap = std::map<std::string, PlayerRecord>;
// region -> history of court-meter positions (nullopt = no detection)
using CourtHistory = std::map<std::string, std::vector<std::optional<cv::Point2f>>>;

struct JsonlRecord {
    int frame = 0;
    std::optional<double> time_sec;  // null when fps unknown
    int detect_frame = 0;
    PlayersMap players;
    std::optional<cv::Point2f> shuttle;  // nullopt == null (incl. [0,0])
};

// ROI as Python roi_corners: [(x1,y1),(x2,y2)]
using RoiCorners = std::array<cv::Point2i, 2>;

}  // namespace gb
