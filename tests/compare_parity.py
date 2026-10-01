# Parity gate: compare two detections.jsonl (Python vs C++).
# python tests/compare_parity.py <python.jsonl> <cpp.jsonl>
# PASS criteria (spec):
#   - record coverage: >=99% of frames join (same court gate)
#   - player court coords within 1.0 m on >=95% of present slot-records
#   - shuttle image within 15 px on >=90% of rows where both see it
#   - rally count (same offline rule on each file) within +/-2
import json
import math
import sys
from collections import defaultdict


def load(path):
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            recs[r["frame"]] = r
    return recs


def rally_count(recs, start_hits=3, window=2.0, quiet=4.0):
    """Online-equivalent RallyState rule fed from jsonl rows."""
    active = False
    count = 0
    hits = []
    last_hit = None
    for r in sorted(recs.values(), key=lambda x: x["frame"]):
        t = r.get("time_sec")
        if t is None:
            continue
        sh = (r.get("shuttlecock") or {}).get("image")
        seen = sh is not None
        if active and last_hit is not None and t - last_hit > quiet:
            active = False
            hits = []
        if seen:
            last_hit = t
            hits = [h for h in hits if t - h <= window]
            hits.append(t)
            if not active and len(hits) >= start_hits:
                active = True
                count += 1
                hits = []
    return count


def main():
    a = load(sys.argv[1])   # python
    b = load(sys.argv[2])   # cpp
    frames_a, frames_b = set(a), set(b)
    common = frames_a & frames_b
    coverage = len(common) / max(1, len(frames_a | frames_b))

    slot_ok = slot_tot = 0
    shuttle_ok = shuttle_tot = 0
    detect_frame_mismatch = 0
    for f in common:
        ra, rb = a[f], b[f]
        if ra.get("detect_frame") != rb.get("detect_frame"):
            detect_frame_mismatch += 1
        for slot in ("upper", "lower", "upper2", "lower2"):
            ca = (ra["players"].get(slot) or {}).get("court")
            cb = (rb["players"].get(slot) or {}).get("court")
            if ca and cb:
                slot_tot += 1
                if math.dist(ca, cb) <= 1.0:
                    slot_ok += 1
        sa = (ra.get("shuttlecock") or {}).get("image")
        sb = (rb.get("shuttlecock") or {}).get("image")
        if sa and sb:
            shuttle_tot += 1
            if math.dist(sa, sb) <= 15:
                shuttle_ok += 1

    slot_pct = 100.0 * slot_ok / max(1, slot_tot)
    shuttle_pct = 100.0 * shuttle_ok / max(1, shuttle_tot)
    ra_count, rb_count = rally_count(a), rally_count(b)

    print(f"records: python={len(a)} cpp={len(b)} joined={len(common)} "
          f"coverage={coverage*100:.1f}% detect_frame_mismatch={detect_frame_mismatch}")
    print(f"court coords <=1.0m: {slot_ok}/{slot_tot} ({slot_pct:.1f}%)")
    print(f"shuttle <=15px: {shuttle_ok}/{shuttle_tot} ({shuttle_pct:.1f}%)")
    print(f"rally count: python={ra_count} cpp={rb_count} delta={abs(ra_count-rb_count)}")

    fails = []
    if coverage < 0.99:
        fails.append(f"coverage {coverage*100:.1f}% < 99%")
    if slot_pct < 95.0:
        fails.append(f"court {slot_pct:.1f}% < 95%")
    if shuttle_tot and shuttle_pct < 90.0:
        fails.append(f"shuttle {shuttle_pct:.1f}% < 90%")
    if abs(ra_count - rb_count) > 2:
        fails.append(f"rally delta {abs(ra_count-rb_count)} > 2")

    if fails:
        print("FAIL: " + "; ".join(fails))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
