#pragma once
// Shuttlecock-activity rally counter. Port of system.py RallyState —
// constants and semantics verified by tests/test_rally.cpp.
#include <vector>

namespace gb {

class RallyState {
public:
    // A rally = 3 shuttle detections within 2s; ends after 4s of silence
    // (test4 dead intervals >= 4.8s; shorter breaks merge).
    static constexpr int START_HITS = 3;
    static constexpr double HIT_WINDOW_SEC = 2.0;
    static constexpr double QUIET_SEC = 4.0;

    // Advance on one frame; returns "start", "end" or "" (None).
    // last-hit clock is frame_count/fps in system.py.
    const char* observe(double now_sec, bool ball_seen);
    // Court cut: end current rally without waiting for quiet.
    void force_end();

    int count() const { return count_; }
    bool active() const { return active_; }

private:
    bool active_ = false;
    int count_ = 0;
    // Python: list of hit times, only time-pruned (unbounded within the 2s
    // window — a fixed ring overflows when ball_seen every frame).
    std::vector<double> hits_;
    double last_hit_sec_ = -1.0;
    bool has_last_hit_ = false;
};

}  // namespace gb
