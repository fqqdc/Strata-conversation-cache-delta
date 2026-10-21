// Diagnostic: run the streaming restore's two passes (read pass + apply pass) over a session file with a
// no-op sink, and report whether the apply pass's payload hash still matches the file's trailer.
// A failure here on a file nobody is writing means the apply pass disagrees with the read pass on the
// file's own bytes - a code bug, not a transient bad read.  A pass means the file is consistent with the
// apply walk; an engine failure on it was a torn/transient read or a file changed between the passes.
//
// Build (from the repo root, with vcvars64):
//   cl /nologo /EHsc /std:c++20 /O2 /I include tools\check-session-apply.cpp src\core\conversation_file.cpp
//      /Fe:build\check-session-apply.exe
#include "strata/core/conversation_file.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace strata::core;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: check-session-apply <session file> [...]\n");
        return 2;
    }
    int failures = 0;
    for (int a = 1; a < argc; ++a) {
        const char* path = argv[a];
        unsigned char h[64];
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::printf("%s: cannot open\n", path); ++failures; continue; }
        f.read(reinterpret_cast<char*>(h), sizeof h);
        if (f.gcount() != (std::streamsize)sizeof h || std::memcmp(h, "STRSESS\1", 8) != 0) {
            std::printf("%s: not a session file\n", path);
            ++failures;
            continue;
        }
        SessionFileIdentity id;
        std::memcpy(&id.model, h + 16, 8);
        std::memcpy(&id.config, h + 24, 8);
        uint64_t payload = 0;
        std::memcpy(&payload, h + 32, 8);
        f.close();

        SavedConversation meta;
        size_t bytes = 0;
        std::string err;
        SessionStatus st;
        std::vector<std::array<uint64_t, 5>> parts;
        const auto t0 = std::chrono::steady_clock::now();
        const bool read_ok = session_file_read_streamed(path, id, meta, bytes, err, {}, &parts, &st);
        const auto t1 = std::chrono::steady_clock::now();
        const double read_s = std::chrono::duration<double>(t1 - t0).count();
        std::printf("%s\n  size %llu, payload %llu\n  read : %s (%.2f s, %zu layers) %s\n", path,
                    (unsigned long long)bytes, (unsigned long long)payload, read_ok ? "OK" : "FAIL", read_s,
                    parts.size(), err.c_str());
        if (!read_ok) { ++failures; continue; }

        uint64_t applied = 0;
        const SessionKvBlock sink = [&](size_t, size_t, uint64_t, const void*, size_t n) {
            applied += n;
            return true;
        };
        std::string err2;
        SessionStatus st2;
        const auto t2 = std::chrono::steady_clock::now();
        const bool apply_ok = session_file_apply_streamed(path, id, sink, parts, err2, &st2);
        const auto t3 = std::chrono::steady_clock::now();
        const double apply_s = std::chrono::duration<double>(t3 - t2).count();
        std::printf("  apply: %s (%.2f s, %llu K/V bytes) %s\n", apply_ok ? "OK" : "FAIL", apply_s,
                    (unsigned long long)applied, err2.c_str());
        if (!apply_ok) ++failures;
    }
    std::printf("\n%d file(s) did not pass\n", failures);
    return failures ? 1 : 0;
}
