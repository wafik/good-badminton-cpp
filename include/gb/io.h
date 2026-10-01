#pragma once
// Outputs: jsonl, metadata, temp video, ffmpeg mux. Port sources:
// data/writer.py, media/video_audio.py, system.py (_write_metadata/_setup_*)
#include <fstream>
#include <optional>
#include <string>
#include "gb/types.h"

namespace gb {

// Python: JsonlDetectionWriter(path) — one compact JSON object per line.
class JsonlDetectionWriter {
public:
    explicit JsonlDetectionWriter(const std::string& path);
    ~JsonlDetectionWriter();
    void write(const JsonlRecord& record);
    void close();
    bool ok() const;

private:
    std::string path_;
    // Text-mode stream on purpose: Python open(path, "w") writes \r\n on
    // Windows — keep byte parity (compare_parity.py json.loads tolerates it).
    std::ofstream file_;
};

// Python: write_json(path, payload) — pretty metadata.json
void write_json(const std::string& path, const std::string& json_text);

// Python: setup_video_writer(w, h, fps, temp_path) -> cv::VideoWriter mp4v
cv::VideoWriter setup_video_writer(int w, int h, double fps, const std::string& path);

// Python: has_audio_track(path) via ffprobe -print_format json -show_streams;
// ffprobe missing/fail -> try-and-assume-yes behavior of Python (return true
// on probe failure, mirroring Python's total-error default of True).
bool has_audio_track(const std::string& path);

// Python: encode_vscode_compatible_mp4 / process_video_with_audio /
// process_video_without_audio: libx264 crf20, aac 160k, +faststart,
// -shortest when audio present, fallback to -an re-encode on failure.
// Throws std::runtime_error with ffmpeg stderr tail on total failure.
void encode_compatible_mp4(const std::string& temp_video, const std::string& output,
                           const std::optional<std::string>& audio_source);

// Python: cleanup_temp_files — delete temp mp4v after successful mux.
void remove_file(const std::string& path);

}  // namespace gb
