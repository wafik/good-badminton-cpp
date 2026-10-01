// Port of media/video_audio.py: setup_video_writer + cleanup_temp_files.
#include "gb/io.h"

#include <filesystem>
#include <stdexcept>

namespace gb {

cv::VideoWriter setup_video_writer(int w, int h, double fps, const std::string& path) {
    // video_audio.py: makedirs(dirname) then VideoWriter mp4v; final export is
    // transcoded to H.264 by encode_compatible_mp4.
    std::error_code ec;
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent, ec);

    cv::VideoWriter writer(path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                           cv::Size(w, h));
    if (!writer.isOpened()) {
        throw std::runtime_error("Unable to create video writer: " + path);
    }
    return writer;
}

void remove_file(const std::string& path) {
    if (path.empty()) return;
    // video_audio.py cleanup_temp_files: best-effort delete + print on failure.
    // ponytail: failure is swallowed here (the caller logs mux outcomes anyway).
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

}  // namespace gb
