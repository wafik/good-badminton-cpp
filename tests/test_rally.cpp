// runnable check for RallyState — build/test_rally.exe prints ok
// Port of Good-Badminton/test_rally.py (same scenarios, same expectations).
#include <cassert>
#include <cstdio>
#include <string>
#include "gb/rally_state.h"

using gb::RallyState;

int main() {
    // flicker (1-2 hits) never starts a rally
    RallyState s;
    assert(s.observe(0.0, true) == nullptr);
    assert(s.observe(0.1, true) == nullptr);
    assert(s.count() == 0 && !s.active());

    // 3rd hit within 2s starts it
    RallyState s1;
    assert(s1.observe(0.0, true) == nullptr);
    assert(s1.observe(0.1, true) == nullptr);
    const char* ev = s1.observe(0.2, true);
    assert(ev && std::string(ev) == "start");
    assert(s1.count() == 1 && s1.active());

    // sparse hits inside the 4s quiet keep it alive, no event
    assert(s1.observe(2.5, false) == nullptr);  // 2.3s silence < 4s
    assert(s1.observe(3.5, true) == nullptr);   // hit, stale hits pruned
    assert(s1.count() == 1);

    // >4s silence ends it
    ev = s1.observe(8.0, false);
    assert(ev && std::string(ev) == "end");  // 4.5s after last hit
    assert(!s1.active());

    // single/duo flickers during the break don't restart
    assert(s1.observe(9.0, true) == nullptr);
    assert(s1.observe(9.2, true) == nullptr);
    assert(s1.count() == 1);

    // 3rd hit starts rally #2
    ev = s1.observe(9.4, true);
    assert(ev && std::string(ev) == "start");
    assert(s1.count() == 2);

    // hits spaced beyond the 2s window never accumulate into a start
    RallyState s2;
    assert(s2.observe(0.0, true) == nullptr);
    assert(s2.observe(2.1, true) == nullptr);
    assert(s2.observe(4.2, true) == nullptr);
    assert(s2.count() == 0);

    // force_end (court cut) ends without counting
    RallyState s3;
    s3.observe(0.0, true);
    s3.observe(0.1, true);
    s3.observe(0.2, true);
    assert(s3.count() == 1);
    s3.force_end();
    assert(!s3.active());
    assert(s3.observe(1.0, false) == nullptr);
    assert(s3.count() == 1);

    std::printf("ok\n");
    return 0;
}
