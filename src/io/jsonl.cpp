// Port of badminton_analysis/data/writer.py (JsonlDetectionWriter + write_json).
// Key order matches Python dict insertion order — use nlohmann::ordered_json.
#include "gb/io.h"

#include <cmath>
#include <filesystem>
#include <stdexcept>

#include "nlohmann_json.hpp"

namespace gb {
namespace {

using ordered_json = nlohmann::ordered_json;

// data/writer.py: SCHEMA_VERSION = "1.0" (JsonlRecord has no schema field, so
// the writer injects it exactly like Python's write_detection_record did).
constexpr const char* kSchemaVersion = "1.0";

ordered_json point_json(const std::optional<cv::Point2f>& p) {
    if (!p) return nullptr;  // _point_or_none -> null
    return ordered_json::array({static_cast<double>(p->x), static_cast<double>(p->y)});
}

ordered_json empty_player_json() {
    // tracking/player.py _empty_player_record: image/court/speed null, hands
    // nested object with null left/right.
    ordered_json hands;
    hands["left"] = nullptr;
    hands["right"] = nullptr;
    ordered_json p;
    p["image"] = nullptr;
    p["court"] = nullptr;
    p["speed"] = nullptr;
    p["hands"] = hands;
    return p;
}

ordered_json player_json(const PlayerRecord& rec) {
    ordered_json hands;
    hands["left"] = point_json(rec.left_hand);
    hands["right"] = point_json(rec.right_hand);
    ordered_json p;
    p["image"] = point_json(rec.image);
    p["court"] = point_json(rec.court);
    if (rec.speed) {
        p["speed"] = *rec.speed;
    } else {
        p["speed"] = nullptr;
    }
    p["hands"] = hands;
    return p;
}

void ensure_parent_dir(const std::string& path) {
    std::error_code ec;
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent, ec);
}

}  // namespace

JsonlDetectionWriter::JsonlDetectionWriter(const std::string& path) : path_(path) {
    ensure_parent_dir(path_);  // writer.py: os.makedirs(os.path.dirname(path))
    file_.open(path_, std::ios::out);  // text mode, no BOM — Python open(..., "w", "utf-8")
    if (!file_.is_open()) {
        throw std::runtime_error("Unable to open detections file: " + path_);
    }
}

JsonlDetectionWriter::~JsonlDetectionWriter() { close(); }

void JsonlDetectionWriter::write(const JsonlRecord& record) {
    if (!file_.is_open()) return;

    ordered_json j;
    j["schema_version"] = kSchemaVersion;
    j["frame"] = record.frame;
    if (record.time_sec) {
        // writer.py: round(frame_index / fps, 6). nearbyint uses the default
        // FE_TONEAREST (ties-to-even) rounding, same intent as Python round().
        const double rounded = std::nearbyint(*record.time_sec * 1e6) / 1e6;
        j["time_sec"] = rounded;
    } else {
        j["time_sec"] = nullptr;
    }
    j["detect_frame"] = record.detect_frame;

    // players: always emit every slot in Python SLOTS order (Python
    // _initialize_player_record materializes all four with nulls).
    ordered_json players = ordered_json::object();
    for (const auto& slot : SLOTS) {
        const auto it = record.players.find(slot);
        players[slot] = (it != record.players.end()) ? player_json(it->second) : empty_player_json();
    }
    for (const auto& [slot, rec] : record.players) {  // defensive: extras keep their data
        if (!players.contains(slot)) players[slot] = player_json(rec);
    }
    j["players"] = players;

    ordered_json shuttle;
    shuttle["image"] = point_json(record.shuttle);  // zero->[0,0] mapping done upstream
    j["shuttlecock"] = shuttle;

    // writer.py: json.dumps(..., separators=(",", ":")) — compact; ensure_ascii
    // False to match Python's raw-UTF-8 output.
    file_ << j.dump(-1, ' ', false, ordered_json::error_handler_t::replace) << '\n';
}

void JsonlDetectionWriter::close() {
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

bool JsonlDetectionWriter::ok() const { return file_.is_open() && !file_.fail(); }

void write_json(const std::string& path, const std::string& json_text) {
    // writer.py write_json: makedirs + json.dump(indent=2) + "\n".
    // Text mode translates \n -> \r\n on Windows, same as Python.
    ensure_parent_dir(path);
    std::ofstream file(path, std::ios::out);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to write JSON file: " + path);
    }
    file << json_text << '\n';
}

}  // namespace gb
