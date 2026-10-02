// CLI port of main.py (spec's minimal surface: input video, --out,
// --annotations + model overrides). Headless defaults: --display false.
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "system.h"

namespace fs = std::filesystem;

namespace {

void print_usage() {
    std::cout << "Usage: gb_cpp [input_video] [options]\n"
                 "  input_video                 input video path (default: videos/demo.mp4)\n"
                 "  --video-path PATH           same as the positional input\n"
                 "  --out DIR                   output directory (default: outputs/<name>)\n"
                 "  --annotations PATH          court annotations file\n"
                 "                              (default: <out>/court_annotations.txt)\n"
                 "  --template PATH             court template image (default: auto — a frame from the video)\n"
                 "  --ball-model PATH           shuttlecock ONNX\n"
                 "                              (default: Good-Badminton/weights/yolo11s-ball.onnx)\n"
                 "  --yolo-pose-model PATH      pose ONNX\n"
                 "                              (default: Good-Badminton/weights/yolo11n-pose-dyn.onnx)\n"
                 "  --pose-conf FLOAT           pose confidence threshold (default: 0.10;\n"
                 "                              Python parity is 0.15 — lower catches far-side players)\n"
                 "  --pose-imgsz INT            pose input size, multiple of 32 (default: 1600;\n"
                 "                              Python parity is 960 — 1600 needed for small far players)\n"
                 "  --audio true|false          keep original audio (default: true)\n"
                 "  --display true|false        show video window (default: false, headless)\n"
                 "  --language zh|en|id        stats panel language (default: zh; zh renders EN text,\n"
                 "                              id = teks Indonesia — both Latin via cv::putText)\n"
                 "  --performance-stats true|false  per-5s frame timings (default: true)\n"
                 "  --skeletons true|false     draw pose skeleton (default: true)\n"
                 "  --player-trajectories true|false  player dots+trails (default: true)\n"
                 "  --court-trajectory true|false     court trajectory overlay (default: true)\n"
                 "  --shuttlecock-trajectory true|false  shuttle trail (default: true)\n"
                 "  --player-stats true|false  stats panel (default: true)\n"
                 "  --pose-roi true|false       pose ROI overlay (default: true)\n"
                 "  --progress-json            emit one {\"frame\":N,\"total\":M} JSON line per frame\n"
                 "  -h, --help\n"
                 "Exit codes: 2 model load failure, 3 ffmpeg failure, 4 court annotation\n"
                 "failure, 1 other errors.\n";
}

bool parse_bool(const std::string& v, bool& out) {
    if (v == "true") { out = true; return true; }
    if (v == "false") { out = false; return true; }
    return false;
}

// Spec default: sibling Good-Badminton/weights/<file>, else the candidate
// used in the exit-2 message. Verified: weights/*.onnx exist next to the
// Python project directory.
std::string default_model_path(const char* argv0, const char* filename) {
    std::vector<fs::path> candidates;
    std::error_code ec;
    if (argv0 && *argv0) {
        const fs::path exe = fs::absolute(fs::path(argv0), ec);
        if (!ec) {
            const fs::path dir = exe.parent_path();
            candidates.push_back(dir / ".." / "Good-Badminton" / "weights" / filename);
            candidates.push_back(dir / ".." / ".." / "Good-Badminton" / "weights" / filename);
        }
    }
    const fs::path cwd = fs::current_path(ec);
    candidates.push_back(cwd / ".." / "Good-Badminton" / "weights" / filename);
    candidates.push_back(cwd / "weights" / filename);
    for (const auto& c : candidates) {
        std::error_code tec;
        if (fs::exists(c, tec)) return c.lexically_normal().string();
    }
    return candidates.front().lexically_normal().string();
}

}  // namespace

int main(int argc, char** argv) {
    gb::SystemOptions opts;
    bool got_video = false;
    std::string ball_arg;
    std::string pose_arg;
    bool bad_value = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto need = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\n";
                print_usage();
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            print_usage();
            return 0;
        } else if (a == "--video-path") {
            opts.video_path = need("--video-path");
            got_video = true;
        } else if (a == "--out") {
            opts.output_dir = need("--out");
        } else if (a == "--annotations") {
            opts.annotations_path = need("--annotations");
        } else if (a == "--template") {
            opts.template_path = need("--template");
        } else if (a == "--ball-model") {
            ball_arg = need("--ball-model");
        } else if (a == "--yolo-pose-model" || a == "--pose-model") {
            pose_arg = need("--yolo-pose-model");
        } else if (a == "--pose-conf") {
            try {
                opts.pose_conf = std::stod(need("--pose-conf"));
            } catch (const std::exception&) {
                bad_value = true;
            }
            if (opts.pose_conf <= 0.0 || opts.pose_conf > 1.0) bad_value = true;
        } else if (a == "--pose-imgsz") {
            try {
                opts.pose_imgsz = std::stoi(need("--pose-imgsz"));
            } catch (const std::exception&) {
                bad_value = true;
            }
            if (opts.pose_imgsz < 320 || opts.pose_imgsz > 4096 ||
                opts.pose_imgsz % 32 != 0) {
                bad_value = true;
            }
        } else if (a == "--audio") {
            if (!parse_bool(need("--audio"), opts.keep_audio)) bad_value = true;
        } else if (a == "--display") {
            if (!parse_bool(need("--display"), opts.show_display)) bad_value = true;
        } else if (a == "--language") {
            opts.language = need("--language");
            if (opts.language != "zh" && opts.language != "en" &&
                opts.language != "id") {
                bad_value = true;
            }
        } else if (a == "--performance-stats") {
            if (!parse_bool(need("--performance-stats"), opts.show_performance_stats)) {
                bad_value = true;
            }
        } else if (a == "--skeletons") {
            if (!parse_bool(need("--skeletons"), opts.show_skeletons)) bad_value = true;
        } else if (a == "--player-trajectories") {
            if (!parse_bool(need("--player-trajectories"), opts.show_player_trajectories)) bad_value = true;
        } else if (a == "--court-trajectory") {
            if (!parse_bool(need("--court-trajectory"), opts.show_court_trajectory)) bad_value = true;
        } else if (a == "--shuttlecock-trajectory") {
            if (!parse_bool(need("--shuttlecock-trajectory"), opts.show_shuttlecock_trajectory)) bad_value = true;
        } else if (a == "--player-stats") {
            if (!parse_bool(need("--player-stats"), opts.show_player_stats)) bad_value = true;
        } else if (a == "--pose-roi") {
            if (!parse_bool(need("--pose-roi"), opts.show_pose_roi)) bad_value = true;
        } else if (a == "--progress-json") {
            opts.progress_json = true;
        } else if (!a.empty() && a[0] == '-') {
            std::cerr << "Unknown option: " << a << "\n";
            print_usage();
            return 1;
        } else if (!got_video) {
            opts.video_path = a;
            got_video = true;
        } else {
            std::cerr << "Unexpected argument: " << a << "\n";
            print_usage();
            return 1;
        }
    }
    if (bad_value) {  // argparse would exit 2; exit 2 is reserved for model load
        std::cerr << "Invalid option value (see --help).\n";
        return 1;
    }

    if (ball_arg.empty()) opts.ball_model_path = default_model_path(argc > 0 ? argv[0] : nullptr, "yolo11s-ball.onnx");
    else opts.ball_model_path = ball_arg;
    if (pose_arg.empty()) opts.pose_model_path = default_model_path(argc > 0 ? argv[0] : nullptr, "yolo11n-pose-dyn.onnx");
    else opts.pose_model_path = pose_arg;

    try {
        gb::BadmintonAnalysisSystem system(std::move(opts));
        system.process_video();
    } catch (const gb::CodedError& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return e.code;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
