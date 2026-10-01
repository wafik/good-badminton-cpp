// Port of badminton_analysis/tracking/player.py — PlayerTracker.
// Every constant/branch mirrors Python unless marked ponytail.
#include "gb/track.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "gb/io.h"

namespace gb {
namespace {

// Python round(x, n): correctly-rounded decimal, ties-to-even on the exact
// double. printf %.*f uses the same IEEE rule (round-to-nearest-even) and
// strtod re-converts to the nearest double — same result as Python round()
// for the magnitudes used here (speed <= 8, court metres, frame/fps).
// ponytail: diverges from Python only for |x| >= 1e300 (printf fixed format)
// — unreachable in this pipeline.
double py_round(double x, int ndigits) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%.*f", ndigits, x);
    return std::strtod(buf, nullptr);
}

// Python _initialize_player_record: all SLOTS -> empty record
// {image:None, court:None, speed:None, hands:{left:None, right:None}}.
PlayersMap empty_players_record() {
    PlayersMap record;
    for (const auto& slot : SLOTS) record.emplace(slot, PlayerRecord{});
    return record;
}

}  // namespace

// ponytail: no auto-close in the dtor — Python has no __del__ either;
// system.py closes detection_writer itself, player.close() mirrors Python.
PlayerTracker::~PlayerTracker() = default;

PlayerTracker::PlayerTracker(const std::vector<cv::Point2f>& corners, double threshold,
                             int history_size, JsonlDetectionWriter* detection_writer,
                             double fps)
    : threshold_(threshold),
      fps_(fps),
      max_frame_distance_(8.0 / fps),  // Python: 8.0/self.fps (raises on fps=0;
                                        // C++ yields inf instead — unreachable)
      history_size_(history_size),
      detection_writer_(detection_writer),
      court_mapper_(corners) {
    for (const auto& slot : SLOTS) {
        players_[slot] = std::nullopt;   // {slot: None}
        history_[slot] = {};             // deque(maxlen=history_size)
        court_history_[slot] = {};
        match_stats_[slot] = SlotAgg{};  // {total_distance:0, max_speed:0,
        rally_stats_[slot] = SlotAgg{};  //  total_frames:0}
        current_speed_[slot] = 0;
    }
}

// Python _point_or_none(point, zero_is_none=False): None -> None; x or y None
// -> None; (0,0) -> None only when zero_is_none; else [float(x), float(y)].
// ponytail: Point2f cannot express a None component, so the x/y-None branch
// is unreachable (Python's dict/tuple inputs make it defensive too).
std::optional<cv::Point2f> PlayerTracker::point_or_none(
    const std::optional<cv::Point2f>& point, bool zero_is_none) {
    if (!point) return std::nullopt;
    if (zero_is_none && point->x == 0.0f && point->y == 0.0f) return std::nullopt;
    return point;
}

PlayersMap PlayerTracker::update(int frame_index, const std::vector<PersonObs>& people,
                                 const std::optional<cv::Point2f>& ball_image_position,
                                 int detect_frame_count) {
    PlayersMap players_record = empty_players_record();

    for (const auto& slot : SLOTS) {
        if (players_[slot].has_value()) {
            match_stats_[slot].total_frames += 1;
            rally_stats_[slot].total_frames += 1;
        }
    }

    auto assigned = assign_to_slots(people);

    for (const auto& entry : assigned) {
        const std::string& region = entry.first;
        const PersonObs& obs = people[entry.second];
        try {
            // Python: left/right = dicts.get(centroid) inside the same try;
            // PersonObs carries the hands inline (missing dict key == nullopt).
            update_player_position(region, obs, players_record);
        } catch (const std::exception& exc) {
            // Python: print + traceback.print_exc(), continue next slot
            std::cerr << "Error processing player position: " << exc.what() << "\n";
        }
    }

    write_detection_record(frame_index, players_record, ball_image_position,
                           detect_frame_count);
    // Python returns self.players; contract returns the record — see header.
    return players_record;
}

// Python _assign_to_slots: match this frame's centroids to side slots by
// nearest neighbour. Two phases per side:
//  1) occupied slots claim their nearest remaining candidate (continuity),
//     gated by SLOT_GATE_PX;
//  2) only EMPTY slots take leftovers; primary slot (side, side2 order) takes
//     nearest-to-net first (max y for upper, min y for lower).
// Occupied-but-unmatched slots keep their stale pose (NO reacquire).
std::map<std::string, std::size_t> PlayerTracker::assign_to_slots(
    const std::vector<PersonObs>& people) const {
    std::map<std::string, std::size_t> assigned;

    struct Side {
        std::string name;
        std::vector<std::size_t> cands;
    };
    // Python sides dict: {"upper": ..., "lower": ...} — fixed iteration order
    std::vector<Side> sides{{"upper", {}}, {"lower", {}}};
    for (std::size_t i = 0; i < people.size(); ++i) {
        double y = people[i].centroid.y;
        if (y < threshold_)
            sides[0].cands.push_back(i);
        else
            sides[1].cands.push_back(i);
    }

    auto sq_dist = [&people](std::size_t idx, const cv::Point2f& pos) {
        double dx = static_cast<double>(people[idx].centroid.x) - pos.x;
        double dy = static_cast<double>(people[idx].centroid.y) - pos.y;
        return dx * dx + dy * dy;
    };

    for (const auto& side : sides) {
        if (side.cands.empty()) continue;
        const std::string slots[2] = {side.name, side.name + "2"};
        std::vector<std::size_t> remaining = side.cands;

        // occupied slots claim their nearest candidate first (continuity)
        for (const auto& slot : slots) {
            auto pos_it = players_.find(slot);
            // Python: self.players[slot] always exists (SLOTS pre-seeded)
            if (pos_it == players_.end() || !pos_it->second.has_value() ||
                remaining.empty())
                continue;
            const cv::Point2f pos = *pos_it->second;
            // Python min(...) — first minimal element wins ties
            std::size_t nearest = remaining[0];
            double best_d2 = sq_dist(nearest, pos);
            for (std::size_t idx : remaining) {
                double d2 = sq_dist(idx, pos);
                if (d2 < best_d2) {
                    nearest = idx;
                    best_d2 = d2;
                }
            }
            double dist = std::sqrt(best_d2);
            if (dist <= SLOT_GATE_PX) {
                assigned[slot] = nearest;
                remaining.erase(std::find(remaining.begin(), remaining.end(), nearest));
            }
        }

        // empty slots take leftovers; nearest-to-net first so slot0 keeps the
        // original "primary = closest to net" meaning. Occupied-but-unmatched
        // slots keep their last pose instead of grabbing a stranger.
        for (const auto& slot : slots) {
            if (assigned.count(slot) != 0) continue;
            auto pos_it = players_.find(slot);
            if (pos_it != players_.end() && pos_it->second.has_value()) continue;
            if (remaining.empty()) continue;
            std::size_t pick;
            if (slot == side.name) {
                // Python: max(remaining, y) for upper else min(remaining, y);
                // first extreme wins ties (strict comparisons below).
                pick = remaining[0];
                for (std::size_t idx : remaining) {
                    float y = people[idx].centroid.y;
                    float py = people[pick].centroid.y;
                    if (slot == "upper" ? (y > py) : (y < py)) pick = idx;
                }
            } else {
                pick = remaining[0];  // Python: remaining[0]
            }
            assigned[slot] = pick;
            remaining.erase(std::find(remaining.begin(), remaining.end(), pick));
        }
    }
    return assigned;
}

// Python _update_player_position
void PlayerTracker::update_player_position(const std::string& region,
                                           const PersonObs& obs,
                                           PlayersMap& players_record) {
    players_[region] = obs.centroid;
    push_capped(history_[region], obs.centroid);

    std::optional<cv::Point2f> court_position =
        court_mapper_.image_to_court(obs.centroid);
    push_capped(court_history_[region], court_position);

    PlayerRecord& player_record = players_record[region];
    player_record.image = point_or_none(obs.centroid);
    player_record.court = point_or_none(court_position);
    // Python: float(self.current_speed[region]) — always present (0 until a
    // stats fold); writes the capped/rounded value from the last stats call.
    player_record.speed = current_speed_[region];
    // Python: `if left_hand_pos:` — missing dict entry (None) skips the write;
    // detect_players only stores real wrist coords (>1px guard), never [0,0].
    if (obs.left_hand) player_record.left_hand = point_or_none(*obs.left_hand);
    if (obs.right_hand) player_record.right_hand = point_or_none(*obs.right_hand);
}

// Python _update_rally_and_match_stats
void PlayerTracker::update_rally_and_match_stats(const std::string& region,
                                                 double distance, double speed) {
    double capped_speed = py_round(std::min(speed, 8.0), 2);

    rally_stats_[region].total_distance += distance;
    rally_stats_[region].max_speed =
        std::max(rally_stats_[region].max_speed, capped_speed);
    current_speed_[region] = capped_speed;

    match_stats_[region].total_distance += distance;
    match_stats_[region].max_speed =
        std::max(match_stats_[region].max_speed, capped_speed);
    current_speed_[region] = capped_speed;  // Python assigns twice too
}

// Python write_detection_record: exact record semantics (schema_version is
// added by io — JsonlRecord has no such field; see report).
void PlayerTracker::write_detection_record(
    int frame_index, const PlayersMap& players_record,
    const std::optional<cv::Point2f>& ball_image_position, int detect_frame_count) {
    if (detection_writer_ == nullptr) return;

    JsonlRecord record;
    record.frame = frame_index;
    if (fps_ != 0.0) {
        record.time_sec = py_round(frame_index / fps_, 6);
    } else {
        record.time_sec = std::nullopt;  // Python: if self.fps else None
    }
    record.detect_frame = detect_frame_count;
    record.players = players_record;
    // Python: _point_or_none(ball_image_position, zero_is_none=True)
    record.shuttle = point_or_none(ball_image_position, /*zero_is_none=*/true);
    detection_writer_->write(record);
}

// Python start_new_rally: rally_stats only (match_stats/current_speed keep)
void PlayerTracker::start_new_rally() {
    for (const auto& slot : SLOTS) {
        rally_stats_[slot].total_distance = 0;
        rally_stats_[slot].max_speed = 0;
        rally_stats_[slot].total_frames = 0;
    }
}

// Python get_player_movement_stats — 0.5s window over the NON-None court
// history, sample_interval=5, gates distance>0.05 and < 8.0*frames/fps,
// folds accepted window distance into rally/match stats (side effect!).
MovementStats PlayerTracker::get_player_movement_stats() {
    MovementStats stats;
    for (const auto& region : SLOTS) {
        // Python: [pos for pos in list(deque) if pos is not None]
        std::vector<cv::Point2f> history;
        for (const auto& pos : court_history_[region]) {
            if (pos) history.push_back(*pos);
        }

        SlotStats region_stats;  // all-zero defaults
        region_stats.position_count = static_cast<int>(history.size());
        if (history.size() < 2) {
            stats[region] = region_stats;
            continue;
        }

        int current_time = static_cast<int>(history.size()) - 1;
        int window_start =
            std::max(0, current_time - static_cast<int>(fps_ / 2));  // int(fps/2)
        double half_second_total_distance = 0;
        int valid_frames = 0;
        double actual_time_span = 0;
        const int sample_interval = 5;

        std::vector<int> sample_points;
        if (current_time - window_start < sample_interval) {
            sample_points = {window_start, current_time};
        } else {
            for (int i = window_start; i <= current_time; i += sample_interval)
                sample_points.push_back(i);
            if (std::find(sample_points.begin(), sample_points.end(), current_time) ==
                sample_points.end()) {
                sample_points.push_back(current_time);
            }
        }

        for (std::size_t i = 0; i + 1 < sample_points.size(); ++i) {
            int idx1 = sample_points[i];
            int idx2 = sample_points[i + 1];
            const cv::Point2f& p1 = history[idx1];
            const cv::Point2f& p2 = history[idx2];
            double dx = static_cast<double>(p2.x) - p1.x;
            double dy = static_cast<double>(p2.y) - p1.y;
            double distance = std::sqrt(dx * dx + dy * dy);
            double time_span = (idx2 - idx1) / fps_;
            double max_possible_distance =
                max_frame_distance_ * (idx2 - idx1);  // 8.0 m/s * Δframes/fps

            if (distance > 0.05 && distance < max_possible_distance) {
                half_second_total_distance += distance;
                valid_frames += 1;
                actual_time_span += time_span;
            }
        }

        double current_speed = 0;
        if (valid_frames > 0 && actual_time_span > 0) {
            current_speed = half_second_total_distance / actual_time_span;
            update_rally_and_match_stats(region, half_second_total_distance,
                                         current_speed);
        }
        current_speed = std::min(current_speed, 8.0);

        double rally_distance = rally_stats_[region].total_distance;
        double rally_max_speed = rally_stats_[region].max_speed;
        int rally_frames = rally_stats_[region].total_frames;
        double rally_avg_speed = 0;
        if (rally_frames > 1 && fps_ > 0) {
            double rally_time = rally_frames / fps_;
            rally_avg_speed = rally_time > 0 ? rally_distance / rally_time : 0;
        }

        double match_distance = match_stats_[region].total_distance;
        double match_max_speed = match_stats_[region].max_speed;
        int match_frames = match_stats_[region].total_frames;
        double match_avg_speed = 0;
        if (match_frames > 1 && fps_ > 0) {
            double match_time = match_frames / fps_;
            match_avg_speed = match_time > 0 ? match_distance / match_time : 0;
        }

        region_stats.current_speed = py_round(current_speed, 2);
        region_stats.rally_avg_speed = py_round(rally_avg_speed, 2);
        region_stats.rally_max_speed = py_round(rally_max_speed, 2);
        region_stats.rally_distance = py_round(rally_distance, 2);
        region_stats.match_avg_speed = py_round(match_avg_speed, 2);
        region_stats.match_max_speed = py_round(match_max_speed, 2);
        region_stats.match_distance = py_round(match_distance, 2);
        stats[region] = region_stats;
    }
    return stats;
}

// Python get_player_trajectories returns self.history (image px, never None);
// the C++ contract asks for the court-history per slot instead — ponytail:
// Python's method has zero callers, the court trail is what viz consumes
// (court_trajectory overlay), and the image trail is exposed via history().
CourtHistory PlayerTracker::get_player_trajectories() const {
    CourtHistory out;
    for (const auto& entry : court_history_) {
        out[entry.first] =
            std::vector<std::optional<cv::Point2f>>(entry.second.begin(),
                                                   entry.second.end());
    }
    return out;
}

// Python close(): writer may be None
void PlayerTracker::close() {
    if (detection_writer_ != nullptr) detection_writer_->close();
}

}  // namespace gb
