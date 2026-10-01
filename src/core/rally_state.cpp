#include "gb/rally_state.h"

namespace gb {

const char* RallyState::observe(double now_sec, bool ball_seen) {
    const char* event = nullptr;

    if (active_ && has_last_hit_ && now_sec - last_hit_sec_ > QUIET_SEC) {
        active_ = false;
        hits_.clear();
        event = "end";
    }

    if (ball_seen) {
        last_hit_sec_ = now_sec;
        has_last_hit_ = true;
        // Python: self._hits = [t for t in self._hits if now-t <= WINDOW];
        //         self._hits.append(now)
        std::vector<double> kept;
        kept.reserve(hits_.size() + 1);
        for (double h : hits_) {
            if (now_sec - h <= HIT_WINDOW_SEC) kept.push_back(h);
        }
        kept.push_back(now_sec);
        hits_ = std::move(kept);

        if (!active_ && static_cast<int>(hits_.size()) >= START_HITS) {
            active_ = true;
            ++count_;
            hits_.clear();
            event = "start";
        }
    }
    return event;
}

void RallyState::force_end() {
    active_ = false;
    hits_.clear();
}

}  // namespace gb
