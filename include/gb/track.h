#pragma once
// 4-slot player tracker + rally/match stats. Port source: tracking/player.py
#include <cstddef>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "gb/court.h"
#include "gb/types.h"

namespace gb {

class JsonlDetectionWriter;  // io module

class PlayerTracker {
public:
    // Python: PlayerTracker(corners, threshold=680, history_size=50,
    //                       detection_writer=None, fps=30)
    // system.py passes threshold=mid_height (625 for test4).
    // ponytail: contract signature has no default args — Python defaults live
    // on the caller in C++ (system.cpp passes all five explicitly).
    PlayerTracker(const std::vector<cv::Point2f>& corners, double threshold,
                  int history_size, JsonlDetectionWriter* detection_writer, double fps);
    ~PlayerTracker();  // no auto-close: Python has no __del__; system owns writer

    // Python: update(frame_index, centroids, ball_image_position,
    //   left_hand_positions, right_hand_positions, detect_frame_count)
    // hand dicts keyed by centroid -> PersonObs parallel optionals here.
    // Returns the per-frame players RECORD (Python returns self.players —
    // slot->centroid map, unused by system.py; see players() below).
    PlayersMap update(int frame_index, const std::vector<PersonObs>& people,
                      const std::optional<cv::Point2f>& ball_image_position,
                      int detect_frame_count);

    void start_new_rally();  // resets rally_stats only
    MovementStats get_player_movement_stats();  // side-effect: folds window
                                                // distance into rally/match stats
    CourtHistory get_player_trajectories() const;  // court_history per slot
    void close();

    // Python public attributes read by viz/system (draw_players uses
    // .players/.history, court overlay .court_history, stats interval .fps):
    // added for parity — non-breaking vs the base contract.
    double fps() const { return fps_; }
    const std::map<std::string, std::optional<cv::Point2f>>& players() const {
        return players_;
    }
    const std::map<std::string, std::deque<cv::Point2f>>& history() const {
        return history_;
    }
    const std::map<std::string, std::deque<std::optional<cv::Point2f>>>&
    court_history() const {
        return court_history_;
    }

private:
    // Python per-slot aggregate dicts {"total_distance","max_speed","total_frames"}
    struct SlotAgg {
        double total_distance = 0;
        double max_speed = 0;
        int total_frames = 0;
    };

    // Python class attr PlayerTracker.SLOT_GATE_PX
    static constexpr double SLOT_GATE_PX = 150.0;

    // Python _assign_to_slots: per side, occupied-first nearest neighbour with
    // SLOT_GATE_PX; returns region -> index into people (Python maps to the
    // centroid tuple itself).
    std::map<std::string, std::size_t> assign_to_slots(
        const std::vector<PersonObs>& people) const;
    void update_player_position(const std::string& region, const PersonObs& obs,
                                PlayersMap& players_record);
    void update_rally_and_match_stats(const std::string& region, double distance,
                                      double speed);
    void write_detection_record(int frame_index, const PlayersMap& players_record,
                                const std::optional<cv::Point2f>& ball_image_position,
                                int detect_frame_count);

    // Python _point_or_none(point, zero_is_none=False)
    static std::optional<cv::Point2f> point_or_none(
        const std::optional<cv::Point2f>& point, bool zero_is_none = false);

    // deque(maxlen=history_size): evict oldest when full (Python append on a
    // full deque drops from the left).
    template <typename T>
    void push_capped(std::deque<T>& d, T v) const {
        if (history_size_ <= 0) return;  // Python deque(maxlen=0) keeps nothing
        if (static_cast<int>(d.size()) >= history_size_) d.pop_front();
        d.push_back(std::move(v));
    }

    double threshold_;              // Python self.threshold (mid_height)
    double fps_;                    // Python self.fps
    double max_frame_distance_;     // Python self.max_frame_distance = 8.0/fps
    int history_size_;              // deque maxlen for both histories
    JsonlDetectionWriter* detection_writer_;  // non-owning; may be null
    std::map<std::string, std::optional<cv::Point2f>> players_;  // last centroid
    std::map<std::string, std::deque<cv::Point2f>> history_;  // image px trail
    // court-meter trail; nullopt = projection failed (Python may hold [])
    std::map<std::string, std::deque<std::optional<cv::Point2f>>> court_history_;
    std::map<std::string, SlotAgg> match_stats_;
    std::map<std::string, SlotAgg> rally_stats_;
    std::map<std::string, double> current_speed_;  // capped/rounded last speed
    CourtMapper court_mapper_;
};

}  // namespace gb
