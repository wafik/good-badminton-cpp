// Port of media/video_audio.py: has_audio_track + encode_vscode_compatible_mp4
// (+ process_video_with_audio / process_video_without_audio semantics folded
// into encode_compatible_mp4's -map/-an selection and the caller's fallback).
#include "gb/io.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "nlohmann_json.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <cstdio>
#include <sys/wait.h>
#endif

namespace gb {
namespace {

struct RunResult {
    bool spawned = false;   // process image started (false -> exe missing)
    bool timed_out = false;
    int rc = -1;
    std::string output;     // merged stdout+stderr (Python captures both)
};

#ifdef _WIN32

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0);
    if (n <= 0) return {};
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

// Minimal CommandLineToArgvW-compatible quoting: wrap when needed; inner quotes
// doubled as \" — fine for file paths (no trailing backslash files here).
std::string quote_arg(const std::string& a) {
    if (!a.empty() && a.find_first_of(" \t\"") == std::string::npos) return a;
    std::string r;
    r.reserve(a.size() + 2);
    r += '"';
    for (const char c : a) {
        if (c == '"') r += "\\\"";
        else r += c;
    }
    r += '"';
    return r;
}

RunResult run_capture(const std::vector<std::string>& args, int timeout_ms) {
    RunResult res;
    if (args.empty()) return res;

    std::string cmd;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i) cmd += ' ';
        cmd += quote_arg(args[i]);
    }
    std::wstring wcmd = widen(cmd);
    std::vector<wchar_t> buf(wcmd.begin(), wcmd.end());
    buf.push_back(L'\0');

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE rd = nullptr;
    HANDLE wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) return res;
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);  // parent's read end stays private

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    si.hStdOutput = wr;
    si.hStdError = wr;   // merge: we only need the stderr tail for errors
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};

    // lpApplicationName=null -> first token resolved via PATH (like subprocess).
    const BOOL ok = CreateProcessW(nullptr, buf.data(), nullptr, nullptr, TRUE,
                                   CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    CloseHandle(wr);  // child holds the only write handle now -> EOF at exit
    if (!ok) {
        CloseHandle(rd);
        return res;  // spawned=false (ffmpeg/ffprobe not found)
    }
    res.spawned = true;

    // Drain concurrently so chatty children never fill the pipe and deadlock.
    std::string out;
    std::thread reader([&] {
        char b[8192];
        DWORD n = 0;
        while (ReadFile(rd, b, sizeof(b), &n, nullptr) && n > 0) out.append(b, b + n);
    });

    const DWORD wait = WaitForSingleObject(pi.hProcess, static_cast<DWORD>(timeout_ms));
    if (wait == WAIT_TIMEOUT) {
        res.timed_out = true;
        TerminateProcess(pi.hProcess, 1);
        WaitForSingleObject(pi.hProcess, 5000);
    }
    reader.join();  // termination closed the pipe -> ReadFile hit EOF
    CloseHandle(rd);

    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    res.rc = static_cast<int>(code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    res.output = std::move(out);
    return res;
}

#else  // POSIX shim

std::string quote_posix(const std::string& a) {
    std::string r = "'";
    for (const char c : a) {
        if (c == '\'') r += "'\\''";
        else r += c;
    }
    r += '\'';
    return r;
}

RunResult run_capture(const std::vector<std::string>& args, int timeout_ms) {
    (void)timeout_ms;  // ponytail: POSIX shim has no timeout (Windows target only)
    RunResult res;
    if (args.empty()) return res;
    std::string cmd;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i) cmd += ' ';
        cmd += quote_posix(args[i]);
    }
    cmd += " 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return res;  // spawned=false
    res.spawned = true;
    char b[8192];
    size_t n = 0;
    while ((n = fread(b, 1, sizeof(b), pipe)) > 0) res.output.append(b, n);
    const int rc = pclose(pipe);
    res.rc = (rc == -1) ? -1 : WEXITSTATUS(rc);
    return res;
}

#endif

std::string error_tail(std::string s) {
    // video_audio.py: result.stderr.strip()[-1000:]
    const auto first = s.find_first_not_of(" \t\r\n");
    const auto last = s.find_last_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    s = s.substr(first, last - first + 1);
    if (s.size() > 1000) s = s.substr(s.size() - 1000);
    return s;
}

}  // namespace

bool has_audio_track(const std::string& path) {
    // video_audio.py: ffprobe -v quiet -print_format json -show_streams.
    // Per io.h contract: any probe failure (missing exe / rc!=0 / bad JSON)
    // returns true, mirroring Python's outer-except default of True.
    // ponytail: Python's moviepy fallback (rc!=0 path) is dropped — moviepy is
    // a Python-only dependency; rc!=0 falls into the assume-yes default.
    const RunResult r = run_capture(
        {"ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path}, 10000);
    if (!r.spawned || r.timed_out || r.rc != 0) return true;

    try {
        const auto j = nlohmann::json::parse(r.output);
        if (!j.contains("streams") || !j["streams"].is_array()) return true;  // malformed
        for (const auto& stream : j["streams"]) {
            if (stream.contains("codec_type") && stream["codec_type"].is_string() &&
                stream["codec_type"].get<std::string>() == "audio") {
                return true;
            }
        }
        return false;
    } catch (...) {
        return true;  // Python: json.loads failure -> outer except -> True
    }
}

void encode_compatible_mp4(const std::string& temp_video, const std::string& output,
                           const std::optional<std::string>& audio_source) {
    namespace fs = std::filesystem;

    std::error_code ec;
    const fs::path out_path(output);
    if (out_path.has_parent_path()) fs::create_directories(out_path.parent_path(), ec);

    // video_audio.py: if abspath(input)==abspath(output) encode to a sibling
    // temp then os.replace — keeps in-place re-encodes atomic-ish.
    fs::path final_path = out_path;
    bool replace_at_end = false;
    {
        std::error_code ec1, ec2;
        if (fs::absolute(temp_video, ec1).lexically_normal() ==
            fs::absolute(output, ec2).lexically_normal()) {
            final_path = fs::path(output + ".h264.tmp.mp4");
            replace_at_end = true;
        }
    }

    std::vector<std::string> cmd = {"ffmpeg", "-y", "-i", temp_video};
    if (audio_source) {
        // MISSING -i here made audio_source an OUTPUT — ffmpeg -y truncated
        // the source video before failing the -map 1:a:0? check.
        cmd.push_back("-i");
        cmd.push_back(*audio_source);
    }
    cmd.push_back("-map");
    cmd.push_back("0:v:0");
    if (audio_source) {
        cmd.push_back("-map");
        cmd.push_back("1:a:0?");  // optional: source without audio still works
        cmd.push_back("-shortest");
    } else {
        cmd.push_back("-an");
    }
    const char* tail_args[] = {"-c:v",  "libx264", "-preset", "medium", "-crf",
                               "20",    "-pix_fmt", "yuv420p", "-c:a",  "aac",
                               "-b:a",  "160k",    "-movflags", "+faststart"};
    for (const char* a : tail_args) cmd.push_back(a);
    cmd.push_back(final_path.string());

    const RunResult r = run_capture(cmd, 180000);  // Python timeout=180
    if (!r.spawned) {
        throw std::runtime_error(
            "ffmpeg H.264 export failed: ffmpeg executable not found "
            "(install ffmpeg and add it to PATH)");
    }

    std::error_code fec;
    const bool exists = fs::exists(final_path, fec);
    std::uintmax_t size = 0;
    if (exists) size = fs::file_size(final_path, fec);
    const bool nonempty = exists && !fec && size > 0;

    if (r.timed_out || r.rc != 0 || !nonempty) {
        std::string tail = error_tail(r.output);
        if (tail.empty()) tail = "unknown ffmpeg error";
        if (r.timed_out) tail = "timeout after 180s: " + tail;
        throw std::runtime_error("ffmpeg H.264 export failed: " + tail);
    }

    if (replace_at_end) {
        // Python os.replace: overwrite destination atomically where possible.
        std::error_code rec;
        fs::remove(out_path, rec);
        rec.clear();
        fs::rename(final_path, out_path, rec);
        if (rec) {
            throw std::runtime_error("ffmpeg H.264 export failed: unable to move temp output "
                                     "into place: " +
                                     rec.message());
        }
    }
}

}  // namespace gb
